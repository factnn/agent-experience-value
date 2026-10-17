"""CPU synthetic-array audit checks; no actual learner likelihoods or GPU work."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from audit_bfcl_model_conditioned_measurement import audit_case
from bfcl_model_conditioned_contract import SIGNALS, ref, summarize_episodes
from bfcl_model_conditioned_feature_inputs import feedback_inputs, saved_model_inputs
from bfcl_model_conditioned_likelihood import feedback_summary, reference_summary
from check_bfcl_model_conditioned_inputs import TinyTokenizer, episode
from run_bfcl_model_conditioned_measurement import array_sha


def dump(path, value):
    path.write_text(json.dumps(value) + '\n')


def one_profile(root, profile_name):
    case = {'name': 'fixture_' + profile_name, 'profile': profile_name, 'task_ids': ['fixture_task'],
            'source_group': 'fixture_source', 'own_case': 'own', 'reference_case': 'ref',
            'own_raw': {}, 'reference_raw': {}, 'maximum_forward_calls': 12,
            'forward_input_position_ceiling': 10000}
    profile = {'base_revision': 'synthetic', 'base_model_path': 'synthetic',
               'policy_fingerprint': 'fixture_adapter', 'frozen_base_tensor_sha256': 'fixture_base',
               'own_probe_plan': {}}
    design = {'output_root': 'rl/fixtures', 'profiles': {profile_name: profile}, 'reference_probe_plan': {}}
    execution_ref = {'path': 'synthetic_execution.json', 'sha256': 'synthetic_fixture_only'}
    execution = {'design': {'path': 'synthetic_design.json', 'sha256': 'synthetic_fixture_only'},
                 'cases': [case], 'runtime_versions': {}}
    output = root / design['output_root'] / case['name']
    (output / 'token_likelihoods').mkdir(parents=True)
    own, references = {'fixture_task': {}}, {'fixture_task': {}}
    rows, journal = [], []
    for draw in range(4):
        row = episode([{'role': 'tool', 'content': 'echo'}]) | {'draw_id': 'own:' + str(draw)}
        reference = copy.deepcopy(row)
        if profile_name != 'qwen4b':
            reference['draw_id'] = 'fixed4b:' + str(draw)
        own['fixture_task'][draw], references['fixture_task'][draw] = row, reference
        x, y = feedback_inputs(row, TinyTokenizer()), saved_model_inputs(reference)
        factual = np.asarray([-1., -2., -3., -4.], dtype=np.float32)
        ablated = np.asarray([-1., -2., -5., -8.], dtype=np.float32)
        fixed = factual if profile_name == 'qwen4b' else np.asarray([-2., -3., -4., -5.], dtype=np.float32)
        arrays = {'own_target_positions': np.asarray(x['target_positions'], dtype=np.int32),
                  'feedback_eligible_indices': np.asarray(x['feedback_sensitive_target_indices'], dtype=np.int32),
                  'own_factual_logp': factual, 'own_ablated_logp': ablated}
        if profile_name != 'qwen4b':
            arrays.update(reference_logp=fixed,
                          reference_target_positions=np.asarray(y['target_positions'], dtype=np.int32))
        archive = output / 'token_likelihoods' / ('fixture_task_d' + str(draw) + '.npz')
        with archive.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        identity = {'task_id': 'fixture_task', 'rollout_index': draw,
                    'own_draw_id': row['draw_id'], 'reference_draw_id': reference['draw_id']}
        feedback = feedback_summary(factual, ablated, x['feedback_sensitive_target_indices'])
        compatibility = reference_summary(fixed)
        scored = identity | {'array_file': ref(root, archive),
            'reference_logprob_array': 'own_factual_logp' if profile_name == 'qwen4b' else 'reference_logp',
            'reference_position_array': 'own_target_positions' if profile_name == 'qwen4b' else 'reference_target_positions',
            'own_mask_audit': x['mask_audit'],
            'own_input_ids_int32_sha256': array_sha(x['input_ids'], np.int32),
            'ablated_input_ids_int32_sha256': array_sha(x['ablated_input_ids'], np.int32),
            'reference_input_ids_int32_sha256': array_sha(y['input_ids'], np.int32),
            'feedback_summary': feedback, 'compatibility_summary': compatibility,
            'values': {SIGNALS[0]: feedback['value'], SIGNALS[1]: compatibility['value']},
            'raw_reference_is_fixed4b': True, 'reward_or_grade_filtering': False}
        rows.append(scored)
        calls = [('own_factual', x['input_ids'], x['target_positions'], factual),
                 ('own_ablated', x['ablated_input_ids'], x['target_positions'], ablated)]
        if profile_name != 'qwen4b':
            calls.append(('fixed4b_reference', y['input_ids'], y['target_positions'], fixed))
        for role, sequence, positions, values in calls:
            header = {'forward_id': len(journal) // 2, 'trajectory': identity, 'role': role,
                      'input_positions': len(sequence), 'scored_model_tokens': len(positions),
                      'input_ids_int32_sha256': array_sha(sequence, np.int32),
                      'target_positions_int32_sha256': array_sha(positions, np.int32)}
            journal.extend([header | {'state': 'started', 'unix': 10.},
                            header | {'state': 'completed', 'unix': 11., 'wall_seconds': .125,
                                      'float32_logprob_sha256': array_sha(values, np.float32)}])
    calls = len(journal) // 2
    positions = sum(row['input_positions'] for row in journal if row['state'] == 'started')
    cost = {'forward_calls_started': calls, 'forward_calls_completed': calls,
            'forward_input_positions_started': positions, 'forward_input_positions_completed': positions,
            'forward_wall_seconds': calls * .125, 'new_model_tokens': 0,
            'new_environment_interactions': 0, 'optimizer_updates': 0}
    summary = summarize_episodes(case, rows) | {'status': 'complete', 'case': case['name'],
        'profile': profile_name, 'source_group': case['source_group'], 'episodes': 4, 'cost': cost,
        'trainable_before_sha256': profile['policy_fingerprint'], 'trainable_after_sha256': profile['policy_fingerprint'],
        'frozen_base_before_sha256': profile['frozen_base_tensor_sha256'],
        'frozen_base_after_sha256': profile['frozen_base_tensor_sha256'],
        'gradients_all_none': True, 'target_evaluation': False, 'reserved21_accessed': False}
    dump(output / 'summary.json', summary)
    dump(output / 'run.json', {'execution': execution_ref, 'design': execution['design'], 'case': case,
        'model_revision': profile['base_revision'], 'native_model_path': profile['base_model_path'],
        'policy_fingerprint': profile['policy_fingerprint'], 'frozen_base_tensor_sha256': profile['frozen_base_tensor_sha256'],
        'versions': {}, 'new_model_tokens': 0, 'new_environment_interactions': 0, 'optimizer_updates': 0,
        'target_evaluation': False, 'reserved21_accessed': False})
    dump(output / 'supervisor.json', {'returncode': 0, 'hard_timeout': False, 'wall_seconds': 3.})
    (output / 'episode_scores.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (output / 'forward_attempts.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in journal))

    def source_loader(_root, _plan, source_name, _raw):
        return {}, {'task_ids': case['task_ids']}, own if source_name == 'own' else references

    with patch('audit_bfcl_model_conditioned_measurement.load_execution', return_value=(execution, design)), \
         patch('audit_bfcl_model_conditioned_measurement.feature_episodes', side_effect=source_loader):
        result = audit_case(root, execution_ref, case['name'], TinyTokenizer())
        assert result['cost']['forward_calls_completed'] == (8 if profile_name == 'qwen4b' else 12)
        assert result['source_features'][SIGNALS[0]] == 3.
        assert result['source_features'][SIGNALS[1]] == (-2.5 if profile_name == 'qwen4b' else -3.5)
        # Updating only the archive's SHA is insufficient: the audit must also
        # detect changed raw probabilities against saved scalar/journal data.
        archive = root / rows[0]['array_file']['path']
        with np.load(archive, allow_pickle=False) as data:
            changed = {name: data[name].copy() for name in data.files}
        changed['own_factual_logp'][-1] += .25
        with archive.open('wb') as stream:
            np.savez_compressed(stream, **changed)
        rows[0]['array_file'] = ref(root, archive)
        (output / 'episode_scores.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
        try:
            audit_case(root, execution_ref, case['name'], TinyTokenizer())
        except ValueError:
            pass
        else:
            raise AssertionError('Altered probability arrays accepted after SHA update')


def main():
    with tempfile.TemporaryDirectory() as folder:
        for profile in ['qwen4b', 'qwen8b']:
            one_profile(Path(folder), profile)
    print(json.dumps({'status': 'passed', 'checks': ['4B factual/reference pass shared once',
        '8B own versus fixed4B reference kept separate', 'raw array/position/mask/journal/cost replay',
        'altered probability arrays rejected even after archive SHA update'],
        'fixtures': 'Handwritten synthetic CPU arrays and mocked protocol/source provider; temporary files deleted.',
        'native_model_likelihoods_computed': False, 'gpu_jobs': 0, 'new_model_tokens': 0}))


if __name__ == '__main__':
    main()
