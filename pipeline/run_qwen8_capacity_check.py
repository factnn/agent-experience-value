"""One preregistered native-Qwen3-8B generation/GRPO update and memory measurement."""
import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import run_bfcl_intervention as r
from bfcl_loss_scaling_contract import configure_single_normalization


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def stop_check(p):
    for name in [p['stop_file'], p['global_stop_file']]:
        if (r.ROOT / name).exists(): raise RuntimeError('Current user stop marker: ' + name)


def memory():
    return {k: float(fn()) / 2**20 for k, fn in {
        'allocated_mib': r.torch.cuda.memory_allocated, 'reserved_mib': r.torch.cuda.memory_reserved,
        'peak_allocated_mib': r.torch.cuda.max_memory_allocated, 'peak_reserved_mib': r.torch.cuda.max_memory_reserved}.items()}


class MeasuredTrainer(r.InterventionTrainer):
    def _generate_and_score_completions(self, inputs):
        stop_check(self.capacity_protocol)
        return super()._generate_and_score_completions(inputs)

    def _generate_single_turn(self, prompts):
        r.torch.cuda.synchronize(); r.torch.cuda.reset_peak_memory_stats()
        result = super()._generate_single_turn(prompts)
        r.torch.cuda.synchronize()
        self.generation_memory = memory()
        r.write(self.evidence_dir / 'generation_memory.json', self.generation_memory)
        r.torch.cuda.reset_peak_memory_stats()
        return result

    def training_step(self, *args, **kwargs):
        stop_check(self.capacity_protocol)
        assert self.accelerator.gradient_accumulation_steps == 1 and self.args.gradient_accumulation_steps == 4
        result = super().training_step(*args, **kwargs)
        assert self.current_gradient_accumulation_steps == 4
        r.append(self.evidence_dir / 'microsteps.jsonl', {'micro_step_after': self._step, 'loss': float(result),
            'accelerate_backward_divisor': 1, 'grpo_loss_divisor': 4, 'memory_after_backward': memory()})
        return result


class FreshOptimizer(r.TrainerCallback):
    def __init__(self, trainer): self.t = trainer
    def on_train_begin(self, args, state, control, **kwargs):
        t = self.t
        assert state.global_step == t._step == 0 and t._buffered_inputs is None
        assert not t.optimizer.optimizer.state and t.optimizer.scaler is None
        assert t.allocator.total_sampled_tokens == t.inherited_tokens == 0
        r.write(t.evidence_dir / 'initial_learning_state.json', {'optimizer_state_entries': 0,
            'policy_fingerprint': r.fingerprint(t.model), 'history_acquisition_tokens': 0,
            'scope': 'Fresh model-specific engineering state; not restored from or compared as a branch of the 4B parent.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True); args = parser.parse_args()
    assert digest(args.protocol) == args.protocol_sha256
    p = json.loads(args.protocol.read_text()); stop_check(p)
    out = r.ROOT / p['output']; out.mkdir(exist_ok=False); started = time.perf_counter()
    try:
        for name, sha in p['input_sha256'].items(): assert digest(r.ROOT / name) == sha, name
        source_path = r.ROOT / p['algorithm_protocol']['path']; assert digest(source_path) == p['algorithm_protocol']['sha256']
        source = json.loads(source_path.read_text()); a = source['algorithm']
        verify_path = r.ROOT / p['model_verification']['path']; assert digest(verify_path) == p['model_verification']['sha256']
        verification = json.loads(verify_path.read_text()); model_path = Path(verification['destination'])
        for row in verification['files']:
            path = model_path / row['path']; assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], row['path']
        assert verification['immutable_commit'] == p['model_revision'] and verification['parameter_count'] == 8190735360
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and r.torch.cuda.is_available() and r.torch.cuda.device_count() == 1
        r.torch.set_num_threads(4); r.torch.use_deterministic_algorithms(True, warn_only=False)
        r.torch.backends.cudnn.deterministic = True; r.torch.backends.cudnn.benchmark = False
        r.set_seed(p['initialization_seed']); stop_check(p)
        run = {'protocol_sha256': args.protocol_sha256, 'pid': os.getpid(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
            'started_unix': time.time(), 'model_revision': p['model_revision'], 'model_verification': p['model_verification'],
            'native_model_path': str(model_path), 'selected_task_id': p['task_selection']['selected_task_id'],
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r.ROOT, text=True).strip(),
            'target_evaluation': False, 'reserved21_accessed': False, 'maximum_updates': 1,
            'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets', 'tokenizers']}}
        r.write(out / 'run.json', run)
        tokenizer = r.AutoTokenizer.from_pretrained(model_path, local_files_only=True); tokenizer.padding_side = 'left'
        registry = r.load_tasks(source, sorted(source['training_registry']), 'train')
        rows = {key: {'task_id': key, 'prompt': r.initial_prompt(pair[0])} for key, pair in registry.items()}
        lengths = {key: len(tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)) for key, row in rows.items()}
        assert len(lengths) == p['task_selection']['training_task_count'] == 87
        r.write(out / 'training_prompt_lengths.json', lengths)
        selected = sorted(lengths, key=lambda key: (-lengths[key], key))[0]
        assert selected == p['task_selection']['selected_task_id'] and selected in source['training_registry']
        assert lengths[selected] == p['task_selection']['selected_initial_prompt_tokens']
        model = r.AutoModelForCausalLM.from_pretrained(model_path, dtype=r.torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
        q = copy.deepcopy(source); q['branch_max_updates'] = 1; q['branch_raw_token_budget'] = p['new_model_token_ceiling']
        q['seeds'] = {'initialization': p['initialization_seed'], 'branch': p['rollout_seed']}
        config = r.config_for(q, out)
        def reward(**kwargs): return [e.conversation.reward() for e in trainer.episodes]
        trainer = MeasuredTrainer(model=model, args=config, processing_class=tokenizer,
            train_dataset=r.Dataset.from_list(list(rows.values())), reward_funcs=reward,
            peft_config=r.LoraConfig(r=a['lora_r'], lora_alpha=a['lora_alpha'], lora_dropout=a['lora_dropout'], target_modules=a['lora_targets'], task_type='CAUSAL_LM'))
        configure_single_normalization(trainer)
        assert trainer.accelerator.num_processes == 1 and not trainer.is_fsdp_enabled and not trainer.tools
        trainer.registry = registry; trainer.training_rows = rows; trainer.evidence_dir = out; trainer.protocol = q
        trainer.capacity_protocol = p; trainer.segment_cap = a['segment_cap']
        trainer.episode_limits = {'max_calls': a['max_calls'], 'max_segments': a['max_segments']}
        trainer.allocator = r.CombinationAllocator(q['training_registry'], seed=p['allocator_seed'], rule='uniform', pool=[selected])
        trainer.inherited_tokens = 0; trainer.condition = 'capacity_check'; trainer.rule = 'uniform'; trainer.condition_index = 0
        trainer.sampled_tokens = 0; trainer.generation_seconds = 0.; trainer.train_compute_seconds = 0.
        trainer.allocation_records = []; trainer.stop_reason = None; trainer.started = started
        r.write(out / 'trainer_config.json', json.loads(config.to_json_string()))
        r.write(out / 'loaded_memory.json', memory())
        trainer.add_callback(FreshOptimizer(trainer)); trainer.add_callback(r.UpdateEvidence(trainer))
        stop_check(p); trainer.train(); r.torch.cuda.synchronize()
        assert trainer.state.global_step == 1 and trainer._step == 4 and len(trainer.allocation_records) == 1
        assert trainer.sampled_tokens <= p['new_model_token_ceiling'] and all(x.grad is None for x in trainer.model.parameters())
        calls = [json.loads(line) for line in (out / 'generation_calls.jsonl').read_text().splitlines()]
        assert sum(len(call['generated_ids']) for call in calls) == trainer.sampled_tokens
        summary = {'status': 'complete', 'scope': 'One development/training-task capacity measurement; no transfer, value ranking, or learning efficacy claim.',
            'model_revision': p['model_revision'], 'selected_task_id': selected, 'initial_prompt_tokens': lengths[selected],
            'optimizer_steps': 1, 'microsteps': 4, 'new_sampled_tokens': trainer.sampled_tokens,
            'input_token_positions_excluding_padding': sum(len(call['input_ids']) for call in calls),
            'external_tokens': sum(len(e.mask) - sum(e.mask) for e in trainer.episodes),
            'rewards': [e.conversation.reward() for e in trainer.episodes],
            'generation_seconds': trainer.generation_seconds, 'non_generation_training_seconds': trainer.train_compute_seconds,
            'wall_seconds': time.perf_counter() - started, 'generation_memory': trainer.generation_memory,
            'scoring_backward_optimizer_memory': memory(), 'final_fingerprint': r.fingerprint(trainer.model),
            'maximum_observed_input_length': max(len(call['input_ids']) for call in calls),
            'maximum_prompt_plus_completion_length': max(len(e.prompt_ids) + len(e.completion_ids) for e in trainer.episodes),
            'future_length_boundary': 'Observed task/trajectory lengths only; does not guarantee capacity for every possible future trajectory.',
            'target_evaluation_performed': False, 'reserved21_accessed': False, 'automatic_retry': False}
        r.write(out / 'summary.json', summary); print(json.dumps(summary), flush=True)
    except Exception as error:
        r.write(out / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'wall_seconds': time.perf_counter() - started,
            'partial_evidence_retained': True, 'automatic_retry': False, 'unrecorded_inflight_model_tokens_possible': True})
        raise


if __name__ == '__main__': main()
