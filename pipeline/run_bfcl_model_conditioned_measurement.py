"""One immutable E368 source/model case: forward likelihoods, no sampling/update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from bfcl_model_conditioned_contract import (SIGNALS, append, check_bytes, exclusive, feature_episodes,
    load_execution, ref, summarize_episodes)
from bfcl_model_conditioned_feature_inputs import feedback_inputs, saved_model_inputs
from bfcl_model_conditioned_likelihood import feedback_summary, reference_summary, token_logprobs
from rl_learning_state import tensor_hash
from train_rl_smoke import fingerprint, set_seed


def array_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


class ForwardJournal:
    def __init__(self, output, case):
        self.output, self.case = output, case
        self.started = self.completed = self.started_positions = self.completed_positions = 0
        self.forward_seconds = 0.

    def run(self, model, sequence, positions, trajectory, role):
        if not positions:
            raise ValueError('No forward call is needed for an empty scored target set')
        if len(sequence) > 13155:
            raise ValueError('Saved scored prefix exceeds the preregistered original-task bound')
        if (self.started + 1 > self.case['maximum_forward_calls']
                or self.started_positions + len(sequence) > self.case['forward_input_position_ceiling']):
            raise ValueError('Preregistered forward budget exhausted before admission')
        identity = {'forward_id': self.started, 'trajectory': trajectory, 'role': role,
                    'input_positions': len(sequence), 'scored_model_tokens': len(positions),
                    'input_ids_int32_sha256': array_sha(sequence, np.int32),
                    'target_positions_int32_sha256': array_sha(positions, np.int32)}
        self.started += 1
        self.started_positions += len(sequence)
        torch.cuda.synchronize()
        began = time.perf_counter()
        append(self.output / 'forward_attempts.jsonl', identity | {'state': 'started', 'unix': time.time()})
        try:
            values = token_logprobs(model, sequence, positions).numpy()
            torch.cuda.synchronize()
        except Exception as error:
            elapsed = time.perf_counter() - began
            self.forward_seconds += elapsed
            append(self.output / 'forward_attempts.jsonl', identity | {'state': 'failed', 'unix': time.time(),
                   'wall_seconds': elapsed, 'type': type(error).__name__, 'message': str(error),
                   'input_positions_are_issued_not_completed_work': True})
            raise
        elapsed = time.perf_counter() - began
        self.completed += 1
        self.completed_positions += len(sequence)
        self.forward_seconds += elapsed
        append(self.output / 'forward_attempts.jsonl', identity | {'state': 'completed', 'unix': time.time(),
               'wall_seconds': elapsed, 'float32_logprob_sha256': array_sha(values, np.float32),
               'cuda_allocated_bytes': torch.cuda.memory_allocated(),
               'cuda_reserved_bytes': torch.cuda.memory_reserved()})
        return values

    def cost(self):
        return {'forward_calls_started': self.started, 'forward_calls_completed': self.completed,
                'forward_input_positions_started': self.started_positions,
                'forward_input_positions_completed': self.completed_positions,
                'forward_wall_seconds': self.forward_seconds, 'new_model_tokens': 0,
                'new_environment_interactions': 0, 'optimizer_updates': 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execution', type=Path, required=True)
    ap.add_argument('--execution-sha256', required=True)
    ap.add_argument('--case', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    execution_ref = {'path': str(args.execution.resolve().relative_to(root)), 'sha256': args.execution_sha256}
    execution, design = load_execution(root, execution_ref)
    case = next(x for x in execution['cases'] if x['name'] == args.case)
    profile = design['profiles'][case['profile']]
    output = root / design['output_root'] / case['name']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'token_likelihoods').mkdir(exist_ok=False)
    began = time.perf_counter()
    journal = ForwardJournal(output, case)
    phase = 'input_admission'
    try:
        for reference in execution['model_metadata_assets'][case['profile']]:
            check_bytes(root, reference)
        own_plan, own_case, own = feature_episodes(root, profile['own_probe_plan'], case['own_case'], case['own_raw'])
        reference_plan, reference_case, references = feature_episodes(root, design['reference_probe_plan'],
                                                        case['reference_case'], case['reference_raw'])
        if own_case['task_ids'] != case['task_ids'] or reference_case['task_ids'] != case['task_ids']:
            raise ValueError('Source task pool changed')
        if case['profile'] == 'qwen4b' and (case['own_raw'] != case['reference_raw'] or own != references):
            raise ValueError('The shared4B factual/reference pass must use the exact same raw data')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Require one actually admitted visible GPU')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('Require the registered deterministic CUDA workspace')
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_seed(case['implementation_seed_no_sampling'])
        torch.cuda.reset_peak_memory_stats()
        phase = 'model_loading'
        tokenizer = AutoTokenizer.from_pretrained(profile['base_model_path'], local_files_only=True, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(profile['base_model_path'], dtype=torch.bfloat16,
                                                   attn_implementation='sdpa', local_files_only=True)
        model = PeftModel.from_pretrained(base, root / profile['policy_path'] / 'adapter',
                                          is_trainable=True).to('cuda:0').eval()
        phase = 'model_identity'
        before = fingerprint(model)
        frozen_before = tensor_hash((name, value) for name, value in model.named_parameters() if not value.requires_grad)
        if before != profile['policy_fingerprint'] or frozen_before != profile['frozen_base_tensor_sha256']:
            raise ValueError('Loaded trainable or full frozen-base state differs from registered learner')
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('Unexpected gradient at inference admission')
        exclusive(output / 'run.json', {'kind': 'forward_only_model_conditioned_source_measurement',
            'execution': execution_ref, 'design': execution['design'], 'case': case, 'pid': os.getpid(),
            'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'started_unix': time.time(),
            'model_revision': profile['base_revision'], 'native_model_path': profile['base_model_path'],
            'policy_fingerprint': before, 'frozen_base_tensor_sha256': frozen_before,
            'versions': execution['runtime_versions'], 'model_mode': 'eval + inference_mode',
            'target_evaluation': False, 'reserved21_accessed': False,
            'optimizer_updates': 0, 'new_environment_interactions': 0, 'new_model_tokens': 0})
        loading_seconds = time.perf_counter() - began
        phase = 'forward_scoring'
        results = []
        for key in case['task_ids']:
            for draw in range(4):
                own_row, reference_row = own[key][draw], references[key][draw]
                x = feedback_inputs(own_row, tokenizer)
                y = saved_model_inputs(reference_row)
                if not y['target_positions']:
                    raise ValueError('Fixed original4B reference unexpectedly has no model targets')
                identity = {'task_id': key, 'rollout_index': draw,
                            'own_draw_id': own_row['draw_id'], 'reference_draw_id': reference_row['draw_id']}
                eligible = x['feedback_sensitive_target_indices']
                factual = np.empty(0, dtype=np.float32)
                ablated = np.empty(0, dtype=np.float32)
                if case['profile'] == 'qwen4b' or eligible:
                    factual = journal.run(model, x['input_ids'], x['target_positions'], identity, 'own_factual')
                if eligible:
                    ablated = journal.run(model, x['ablated_input_ids'], x['target_positions'], identity, 'own_ablated')
                if case['profile'] == 'qwen4b':
                    reference_values = factual
                    reference_array, reference_positions = 'own_factual_logp', 'own_target_positions'
                else:
                    reference_values = journal.run(model, y['input_ids'], y['target_positions'], identity, 'fixed4b_reference')
                    reference_array, reference_positions = 'reference_logp', 'reference_target_positions'
                arrays = {'own_target_positions': np.asarray(x['target_positions'], dtype=np.int32),
                          'feedback_eligible_indices': np.asarray(eligible, dtype=np.int32),
                          'own_factual_logp': factual, 'own_ablated_logp': ablated}
                if case['profile'] != 'qwen4b':
                    arrays.update(reference_logp=reference_values,
                                  reference_target_positions=np.asarray(y['target_positions'], dtype=np.int32))
                archive = output / 'token_likelihoods' / (key + '_d' + str(draw) + '.npz')
                # Persist actual arrays before derived statistics/validity checks,
                # preserving evidence even if postprocessing subsequently fails.
                with archive.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                feedback = feedback_summary(factual, ablated, eligible) if eligible else {
                    'value': 0.0, 'eligible_model_tokens': 0, 'prefix_check_performed': False,
                    'reason': 'Definition gives0 when no own generated targets follow changed feedback.'}
                compatibility = reference_summary(reference_values)
                row = identity | {'array_file': ref(root, archive), 'reference_logprob_array': reference_array,
                    'reference_position_array': reference_positions, 'own_mask_audit': x['mask_audit'],
                    'own_input_ids_int32_sha256': array_sha(x['input_ids'], np.int32),
                    'ablated_input_ids_int32_sha256': array_sha(x['ablated_input_ids'], np.int32),
                    'reference_input_ids_int32_sha256': array_sha(y['input_ids'], np.int32),
                    'feedback_summary': feedback, 'compatibility_summary': compatibility,
                    'values': {SIGNALS[0]: feedback['value'], SIGNALS[1]: compatibility['value']},
                    'raw_reference_is_fixed4b': True, 'reward_or_grade_filtering': False}
                append(output / 'episode_scores.jsonl', row)
                results.append(row)
                with (output / 'progress.json').open('w') as stream:
                    json.dump({'phase': phase, 'episodes_complete': len(results),
                        'planned_episodes': 4 * len(case['task_ids']), 'cost': journal.cost(),
                        'pid': os.getpid()}, stream, indent=2)
                    stream.write('\n')
        phase = 'final_identity'
        after = fingerprint(model)
        frozen_after = tensor_hash((name, value) for name, value in model.named_parameters() if not value.requires_grad)
        if before != after or frozen_before != frozen_after or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Inference changed learner weights or created gradients')
        summary = summarize_episodes(case, results)
        phase = 'summary_write'
        exclusive(output / 'summary.json', summary | {'status': 'complete', 'case': case['name'],
            'source_group': case['source_group'], 'profile': case['profile'], 'episodes': len(results),
            'cost': journal.cost(), 'loading_wall_seconds': loading_seconds,
            'worker_wall_seconds': time.perf_counter() - began,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'trainable_before_sha256': before, 'trainable_after_sha256': after,
            'frozen_base_before_sha256': frozen_before, 'frozen_base_after_sha256': frozen_after,
            'gradients_all_none': True, 'target_evaluation': False, 'reserved21_accessed': False,
            'scope': 'Two fixed model-conditioned measurement proxies; no RL update or transfer-efficacy conclusion.'})
    except Exception as error:
        exclusive(output / 'failure.json', {'phase': phase, 'type': type(error).__name__, 'message': str(error),
            'classification': 'measurement_input_or_execution_failure_not_training_or_transfer_negative',
            'cost': journal.cost(), 'worker_wall_seconds': time.perf_counter() - began,
            'partial_evidence_retained': True, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
