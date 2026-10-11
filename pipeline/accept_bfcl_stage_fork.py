"""Reproduce recorded nonzero-state updates in a fresh Qwen3/TRL trainer.

Engineering acceptance, not a new efficacy sample. No held-out evaluation.
No HF dataloader resume is claimed: task selection replaces its placeholder rows,
and local HF counters are explicitly mapped to the inherited learner position.
"""
import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import run_bfcl_intervention as r
from bfcl_dose_checkpoint import assert_equal, rng_state
from bfcl_stage_fork import validate_boundary, continuation_seed, learner_position


class ForkTrainer(r.InterventionTrainer):
    def _generate_and_score_completions(self, inputs):
        local = self.state.global_step
        assert len(inputs) == 4 and local == len(self.allocation_records)
        assert self._step == 4 * (self.source_step + local)
        choice = self.allocator.choose()
        selected = self.training_rows[choice['task_id']]
        inputs = [copy.deepcopy(selected) for _ in range(4)]
        seed = continuation_seed(self.protocol['seeds']['branch'], self.source_step, local)
        r.set_seed(seed)
        before = self.sampled_tokens
        # Bypass the step-zero InterventionTrainer allocator, retain official BFCL loss path.
        result = r.BFCLTrainer._generate_and_score_completions(self, inputs)
        rewards = [e.conversation.reward() for e in self.episodes]
        cost = self.sampled_tokens - before
        cumulative_step = self.source_step + local
        self.allocator.observe(choice['task_id'], rewards, cost,
                               f'AllTasks_dose:uniform:{cumulative_step}')
        row = choice | {'policy_step': local, 'cumulative_policy_step': cumulative_step,
                        'seed': seed, 'policy_fingerprint': self.current_fingerprint,
                        'rewards': rewards, 'sampled_tokens': cost,
                        'branch_cost_after_group': self.sampled_tokens,
                        'stop_reasons': [e.stop_reason for e in self.episodes]}
        self.allocation_records.append(row)
        r.append(self.evidence_dir / 'allocation.jsonl', row)
        return result


class RestoreFork(r.TrainerCallback):
    def __init__(self, trainer, acceptance):
        self.t = trainer
        self.a = acceptance

    def on_train_begin(self, args, state, control, **kwargs):
        from accelerate.optimizer import AcceleratedOptimizer
        t, a = self.t, self.a
        assert state.global_step == 0 and t._step == 0 and t._buffered_inputs is None
        assert isinstance(t.optimizer, AcceleratedOptimizer) and t.optimizer.scaler is None
        source = r.ROOT / a['source_checkpoint'] / 'components'
        manifest = json.loads((source / 'manifest.json').read_text())
        assert manifest['payload_sha256'] == a['source_payload_sha256']
        expected = r.provenance(t.protocol, r.ROOT / a['source_protocol']['path'])
        allocator, progress = r.restore_learning_state(source, t.model, t.optimizer.optimizer,
                                                       t.lr_scheduler, expected)
        t.source_step = validate_boundary(progress, t.protocol['algorithm'])
        assert t.source_step == a['source_step']
        assert allocator.rule == 'uniform' and allocator.pool == t.protocol['conditions']['AllTasks_dose']
        # No allocator reseed, pool change, or history reset in this replay control.
        t.allocator = allocator
        t.inherited_tokens = allocator.total_sampled_tokens
        assert t.inherited_tokens == manifest['cumulative_acquisition_tokens']
        assert t.inherited_tokens == progress['inherited_probe_tokens'] + progress['branch_sampled_tokens']
        t._step = progress['micro_step']
        assert t._buffered_inputs is None
        t.current_fingerprint = r.fingerprint(t.model)
        assert r.tensor_hash((n, p) for n, p in t.model.named_parameters() if p.requires_grad) == manifest['trainable_sha256']
        # Verify actual newly allocated optimizer tensors, scheduler, RNG and buffers.
        payload = r.torch.load(source / 'learning_state.pt', map_location='cpu', weights_only=False)
        assert_equal(payload['optimizer'], t.optimizer.optimizer.state_dict())
        assert_equal(payload['scheduler'], t.lr_scheduler.state_dict())
        assert_equal(payload['allocator'], allocator.state_dict())
        assert_equal(payload['buffers'], dict(t.model.named_buffers()))
        assert_equal(payload['module_training'], {n: m.training for n, m in t.model.named_modules()})
        saved_rng = payload['rng']
        assert_equal({'python': saved_rng['python'], 'numpy': saved_rng['numpy'],
                      'cpu': saved_rng['torch_cpu'], 'cuda': saved_rng['torch_cuda']}, rng_state())
        assert payload['optimizer']['state']
        assert any(r.torch.count_nonzero(v['exp_avg']).item() for v in payload['optimizer']['state'].values())
        del payload
        r.write(t.evidence_dir / 'restored.json', {
            'status': 'components_verified_in_new_trainer', 'source_step': t.source_step,
            'local_step': 0, 'source_payload_sha256': manifest['payload_sha256'],
            'policy_fingerprint': t.current_fingerprint, 'inherited_tokens': t.inherited_tokens,
            'optimizer_nonzero_moments': True, 'fresh_rollout_buffer': True,
            'source_micro_step': t._step, 'allocator_reset': False,
            'checks': ['parameters', 'optimizer', 'scheduler', 'allocator', 'buffers', 'modes', 'RNG']})

    def on_step_end(self, args, state, control, **kwargs):
        t = self.t
        assert len(t.allocation_records) == state.global_step
        assert t._step == 4 * (t.source_step + state.global_step)
        if time.perf_counter() - t.started >= self.a['soft_seconds']:
            control.should_training_stop = True
            t.stop_reason = 'soft_wall_limit_after_group'
        r.published_progress(t.evidence_dir, 'nonzero_fork_acceptance',
                             **learner_position(t.source_step, state.global_step),
                             new_sampled_tokens=t.sampled_tokens, inherited_tokens=t.inherited_tokens)
        return control


def make_trainer(p, out, tokenizer, model, acceptance):
    registry = r.load_tasks(p, list(p['training_registry']), 'train')
    rows = {key: {'task_id': key, 'prompt': r.initial_prompt(pair[0])} for key, pair in registry.items()}
    config = r.config_for(p, out)
    config.max_steps = acceptance['replay_updates']
    assert str(config.lr_scheduler_type).split('.')[-1].lower() == 'constant'
    r.write(out / 'trainer_config.json', json.loads(config.to_json_string()))
    a = p['algorithm']

    def reward(**kwargs):
        return [e.conversation.reward() for e in trainer.episodes]

    trainer = ForkTrainer(model=model, args=config, processing_class=tokenizer,
        train_dataset=r.Dataset.from_list(list(rows.values())), reward_funcs=reward,
        peft_config=r.LoraConfig(r=a['lora_r'], lora_alpha=a['lora_alpha'], lora_dropout=a['lora_dropout'],
                                target_modules=a['lora_targets'], task_type='CAUSAL_LM'))
    assert trainer.accelerator.num_processes == 1 and not trainer.is_fsdp_enabled and not trainer.tools
    trainer.registry = registry
    trainer.training_rows = rows
    trainer.evidence_dir = out
    trainer.protocol = p
    trainer.segment_cap = a['segment_cap']
    trainer.episode_limits = {'max_calls': a['max_calls'], 'max_segments': a['max_segments']}
    trainer.sampled_tokens = 0
    trainer.generation_seconds = trainer.train_compute_seconds = 0.
    trainer.allocation_records = []
    trainer.stop_reason = None
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    acceptance = json.loads(args.protocol.read_text())
    source_protocol = r.ROOT / acceptance['source_protocol']['path']
    assert r.digest(source_protocol) == acceptance['source_protocol']['sha256']
    for name, expected in acceptance['reference_sha256'].items():
        assert r.digest(r.ROOT / acceptance['reference_training'] / name) == expected
    p = json.loads(source_protocol.read_text())
    assert r.digest(r.ROOT / 'rl/bfcl_research_split_candidate/manifest.json') == p['manifest_sha256']
    assert r.digest(r.PKG / 'bfcl_eval/data/BFCL_v4_multi_turn_base.json') == p['data_sha256']
    assert r.torch.cuda.is_available() and r.torch.cuda.device_count() == 1
    out = r.ROOT / acceptance['output']
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    r.write(out / 'run.json', {'purpose': acceptance['purpose'], 'protocol_sha256': r.digest(args.protocol),
        'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r.ROOT, text=True).strip(),
        'source_sha256': {name: r.digest(r.ROOT / 'pipeline' / name) for name in
            ['accept_bfcl_stage_fork.py', 'bfcl_stage_fork.py', 'run_bfcl_intervention.py',
             'rl_learning_state.py', 'accept_bfcl_grpo.py', 'bfcl_dose_checkpoint.py']}})
    try:
        r.set_seed(p['seeds']['initialization'])
        r.torch.set_num_threads(4)
        tokenizer = r.AutoTokenizer.from_pretrained(r.ROOT / 'smoke/model', local_files_only=True)
        tokenizer.padding_side = 'left'
        model = r.AutoModelForCausalLM.from_pretrained(r.ROOT / 'smoke/model', dtype=r.torch.bfloat16,
                                                    attn_implementation='sdpa', local_files_only=True)
        trainer = make_trainer(p, out, tokenizer, model, acceptance)
        trainer.started = started
        trainer.add_callback(RestoreFork(trainer, acceptance))
        trainer.add_callback(r.UpdateEvidence(trainer))
        trainer.train()
        assert trainer._step % 4 == 0 and all(x.grad is None for x in trainer.model.parameters())
        position = learner_position(trainer.source_step, trainer.state.global_step)
        summary = {'status': 'complete' if trainer.state.global_step == acceptance['replay_updates'] else 'incomplete',
            **position, 'new_sampled_tokens': trainer.sampled_tokens, 'inherited_tokens': trainer.inherited_tokens,
            'final_fingerprint': r.fingerprint(trainer.model), 'wall_seconds': time.perf_counter() - started,
            'generation_seconds': trainer.generation_seconds, 'stop_reason': trainer.stop_reason,
            'limitation': 'Boundary fork engineering only; replay comparison still required.'}
        r.write(out / 'summary.json', summary)
    except Exception as error:
        r.write(out / 'failure.json', {'type': type(error).__name__, 'message': str(error),
            'wall_seconds': time.perf_counter() - started, 'partial_data_retained': True})
        raise


if __name__ == '__main__':
    main()
