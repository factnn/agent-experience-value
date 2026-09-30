"""Bounded online multi-turn GRPO engineering run; no allocation-effect claim.

Official TRL handles generation, environment masking, advantages and loss.
Subclass hooks only record evidence and timings; they do not change the update.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
# TRL 0.29 imports a public FSDPModule symbol introduced after torch 2.5.
# Alias the real 2.5 class from its old location; no dummy implementation.
# This runner is strictly single-device and never enables FSDP.
import torch.distributed.fsdp as torch_fsdp
if not hasattr(torch_fsdp, 'FSDPModule') and torch.__version__.startswith('2.5.'):
    from torch.distributed._composable.fsdp import FSDPModule
    torch_fsdp.FSDPModule = FSDPModule
from trl import GRPOConfig, GRPOTrainer
from rl_environment import ROOT, MessageEnv, dataset_rows, acceptance


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def fingerprint(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


class EvidenceTrainer(GRPOTrainer):
    def _generate_single_turn(self, prompts):
        result = super()._generate_single_turn(prompts)
        pids, cids, logps, extra = result
        # Count every sampled token, including suffixes later discarded by TRL's
        # multi-turn length cap. Retained trajectory tokens are not acquisition cost.
        append(self.evidence_dir / 'generation_calls.jsonl', {
            'policy_step': self.state.global_step, 'phase': self.evidence_phase,
            'prompt_tokens': [len(p) for p in pids],
            'sampled_tokens': [len(c) for c in cids],
            'prompt_ids': pids, 'completion_ids': cids,
        })
        self.sampled_tokens += sum(map(len, cids))
        return result

    def _generate(self, prompts):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = super()._generate(prompts)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pids, cids, masks, completions, n_tokens, logps, extra = result
        assert masks is not None, 'environment feedback mask missing'
        for c, mask in zip(cids, masks):
            assert len(c) == len(mask)
        self.last_generation = {'seconds': elapsed, 'policy_step': self.state.global_step,
                                'phase': self.evidence_phase}
        self.generation_seconds += elapsed
        for i, (p, c, mask, env) in enumerate(zip(pids, cids, masks, self.environments)):
            append(self.evidence_dir / 'rollouts.jsonl', {
                **self.last_generation, 'rollout_index': i, 'task_id': env.task['task_id'],
                'prompt_ids': p, 'completion_ids': c, 'model_token_mask': mask,
                'model_tokens': sum(mask), 'feedback_tokens': len(mask) - sum(mask),
                'completion': completions[i], 'events': env.events,
                'final_state': env._state(), 'reward': env._reward(),
                'env_seconds': env.env_seconds, 'tool_limit_exceeded': env.limit_exceeded,
                'ends_with_eos': c[-1] == self.eos_token_id,
                'policy_fingerprint': self.current_fingerprint})
        return result

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        mask = result.get('tool_mask')
        assert mask is not None, 'feedback mask not forwarded to loss'
        append(self.evidence_dir / 'groups.jsonl', {
            'policy_step': self.state.global_step, 'phase': self.evidence_phase,
            'task_ids': [x['task_id'] for x in inputs],
            'rewards': [e._reward() for e in self.environments],
            'advantages': result['advantages'].detach().cpu().tolist(),
            'model_tokens': int((mask * result['completion_mask']).sum()),
            'feedback_tokens': int(((1-mask) * result['completion_mask']).sum()),
            'old_logps_stored': result.get('old_per_token_logps') is not None,
            'logprob_rule': 'num_iterations=1; old policy is current detached policy at loss computation; no replay',
        })
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        torch.cuda.synchronize()
        before = time.perf_counter()
        generation_before = self.generation_seconds
        result = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.synchronize()
        total = time.perf_counter() - before
        self.train_compute_seconds += total - (self.generation_seconds - generation_before)
        return result


class UpdateEvidence(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.previous = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.previous = fingerprint(model)
        self.trainer.current_fingerprint = self.previous

    def on_step_end(self, args, state, control, model=None, **kwargs):
        current = fingerprint(model)
        append(self.trainer.evidence_dir / 'updates.jsonl', {
            'step': state.global_step, 'before_sha256': self.previous,
            'after_sha256': current, 'parameters_changed': current != self.previous})
        self.previous = current
        self.trainer.current_fingerprint = current

    def on_log(self, args, state, control, logs=None, **kwargs):
        append(self.trainer.evidence_dir / 'metrics.jsonl', {'step': state.global_step, **(logs or {})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='rl/smoke_001')
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--max-completion-length', type=int, default=768)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--enable-thinking', action='store_true')
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    config = vars(args) | {'trl': '0.29.0', 'algorithm': 'GRPO', 'purpose': 'engineering only',
                         'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         'pid': os.getpid(),
                         'source_sha256': {name: hashlib.sha256((ROOT/'pipeline'/name).read_bytes()).hexdigest()
                                           for name in ['train_rl_smoke.py', 'rl_environment.py']}}
    (out/'run.json').write_text(json.dumps(config, indent=2)+'\n')
    (out/'environment_acceptance.json').write_text(json.dumps(acceptance(), indent=2)+'\n')
    versions = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'trl', 'datasets', 'peft', 'accelerate', 'tokenizers']}
    (out/'versions.json').write_text(json.dumps(versions, indent=2)+'\n')
    set_seed(args.seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'smoke/model', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(ROOT/'smoke/model', dtype=torch.bfloat16,
                                               attn_implementation='sdpa', local_files_only=True)
    # apply_chat_template accepts padding_side as a Jinja kwarg in Transformers
    # 5.2; set the tokenizer property explicitly for decoder-only batched generation.
    tokenizer.padding_side = 'left'
    rows = dataset_rows()
    (out/'tasks.json').write_text(json.dumps(rows, indent=2)+'\n')
    training_args = GRPOConfig(
        output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4, max_steps=args.steps, max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=8, learning_rate=1e-5, lr_scheduler_type='constant',
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        beta=0.0, loss_type='grpo', scale_rewards='group', num_iterations=1,
        temperature=1.0, top_p=1.0, top_k=0, seed=args.seed, data_seed=args.seed,
        chat_template_kwargs={'enable_thinking': args.enable_thinking},
        logging_steps=1, save_strategy='no', report_to='none', disable_tqdm=True,
        dataloader_num_workers=0, use_vllm=False, mask_truncated_completions=False,
    )
    (out/'trainer_config.json').write_text(training_args.to_json_string())
    def reward(environments, **kwargs):
        return [env._reward() for env in environments]
    trainer = EvidenceTrainer(
        model=model, args=training_args, processing_class=tokenizer,
        train_dataset=Dataset.from_list(rows), reward_funcs=reward,
        environment_factory=MessageEnv,
        peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'),
    )
    assert not trainer.is_fsdp_enabled and trainer.accelerator.num_processes == 1
    trainer.evidence_dir = out
    trainer.evidence_phase = 'train'
    trainer.generation_seconds = 0.0
    trainer.sampled_tokens = 0
    trainer.train_compute_seconds = 0.0
    trainer.add_callback(UpdateEvidence(trainer))
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    train_generation = trainer.generation_seconds
    trainer.save_model(str(out/'adapter'))
    # Explicitly sample fresh trajectories from the final policy, without another update.
    trainer.evidence_phase = 'post_update_resample'
    trainer.model.eval()
    with torch.no_grad():
        trainer._generate_and_score_completions([json.loads(json.dumps(rows[0])) for _ in range(4)])
    updates = [json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
    groups = [json.loads(line) for line in (out/'groups.jsonl').read_text().splitlines()]
    changed = sum(x['parameters_changed'] for x in updates)
    wall = time.perf_counter() - start
    summary = {'status': 'closed_loop_verified' if changed else 'no_parameter_update_signal',
               'all_sampled_tokens_including_discarded': trainer.sampled_tokens,
               'optimizer_steps': trainer.state.global_step, 'steps_with_parameter_change': changed,
               'train_generation_and_environment_seconds': train_generation,
               'post_update_resample_seconds': trainer.generation_seconds - train_generation,
               'train_forward_backward_and_scoring_seconds': trainer.train_compute_seconds,
               'total_wall_seconds': wall, 'single_gpu_reserved_hours': wall/3600,
               'peak_allocated_gb': torch.cuda.max_memory_allocated()/2**30,
               'peak_reserved_gb': torch.cuda.max_memory_reserved()/2**30,
               'groups_with_reward_variance': sum(len(set(g['rewards'])) > 1 for g in groups if g['phase']=='train'),
               'training_metrics': result.metrics,
               'limitation': 'Engineering tasks only; no held-out transfer or allocation comparison. Timings overlap as described; no efficacy claim.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
