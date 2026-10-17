"""Replay E368 stored likelihood arrays and source/cost lineage without inference."""
import hashlib
import json

import numpy as np

from bfcl_model_conditioned_contract import (SIGNALS, check_bytes, feature_episodes, load_execution,
    read, ref, summarize_episodes)
from bfcl_model_conditioned_feature_inputs import feedback_inputs, saved_model_inputs
from bfcl_model_conditioned_likelihood import feedback_summary, reference_summary


def array_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def require_array(archive, name, dtype, expected=None, length=None):
    value = archive[name]
    if value.dtype != dtype or value.ndim != 1:
        raise ValueError('Wrong archived likelihood/position dtype or shape: ' + name)
    if expected is not None and not np.array_equal(value, np.asarray(expected, dtype=dtype)):
        raise ValueError('Archived source token positions differ: ' + name)
    if length is not None and len(value) != length:
        raise ValueError('Wrong archived score count: ' + name)
    return value


def audit_case(root, execution_ref, name, tokenizer):
    execution, design = load_execution(root, execution_ref)
    case = next(x for x in execution['cases'] if x['name'] == name)
    profile = design['profiles'][case['profile']]
    output = root / design['output_root'] / name
    run, summary, terminal = [read(output / f) for f in ['run.json', 'summary.json', 'supervisor.json']]
    if terminal['returncode'] != 0 or terminal['hard_timeout'] is not False or summary['status'] != 'complete':
        raise ValueError('Measurement worker is not complete/exited under its original hard bound')
    if (run['execution'] != execution_ref or run['design'] != execution['design'] or run['case'] != case
            or run['model_revision'] != profile['base_revision'] or run['native_model_path'] != profile['base_model_path']):
        raise ValueError('Worker model/execution/case identity differs')
    if any(run[key] != 0 for key in ['new_model_tokens', 'new_environment_interactions', 'optimizer_updates']):
        raise ValueError('Unexpected sampling or model learning in forward-only measurement')
    if run['target_evaluation'] is not False or run['reserved21_accessed'] is not False:
        raise ValueError('Unexpected target access')
    # Check each endpoint, including the complete frozen base rather than only
    # the adapter fingerprint, against the registered original learner.
    if (run['policy_fingerprint'] != profile['policy_fingerprint']
            or run['frozen_base_tensor_sha256'] != profile['frozen_base_tensor_sha256']
            or run['versions'] != execution['runtime_versions']):
        raise ValueError('Wrong initial learner state or runtime')
    if any(summary[k] != profile['policy_fingerprint'] for k in ['trainable_before_sha256', 'trainable_after_sha256']):
        raise ValueError('Wrong trainable state endpoint')
    if any(summary[k] != profile['frozen_base_tensor_sha256'] for k in ['frozen_base_before_sha256', 'frozen_base_after_sha256']):
        raise ValueError('Wrong complete frozen-base endpoint')
    if summary['gradients_all_none'] is not True:
        raise ValueError('Unexpected gradients')
    _, own_case, own = feature_episodes(root, profile['own_probe_plan'], case['own_case'], case['own_raw'])
    _, reference_case, references = feature_episodes(root, design['reference_probe_plan'], case['reference_case'], case['reference_raw'])
    if own_case['task_ids'] != case['task_ids'] or reference_case['task_ids'] != case['task_ids']:
        raise ValueError('Wrong source task membership')
    rows = [json.loads(line) for line in (output / 'episode_scores.jsonl').open()]
    expected_order = [(key, draw) for key in case['task_ids'] for draw in range(4)]
    if [(r['task_id'], r['rollout_index']) for r in rows] != expected_order:
        raise ValueError('Missing/reordered/unregistered feature trajectory')
    expected_forwards = []
    for row in rows:
        key, draw = row['task_id'], row['rollout_index']
        own_row, reference_row = own[key][draw], references[key][draw]
        x, y = feedback_inputs(own_row, tokenizer), saved_model_inputs(reference_row)
        if row['own_draw_id'] != own_row['draw_id'] or row['reference_draw_id'] != reference_row['draw_id']:
            raise ValueError('Wrong source draw identity')
        eligible = x['feedback_sensitive_target_indices']
        archive_path = check_bytes(root, row['array_file'])
        expected_path = output / 'token_likelihoods' / (key + '_d' + str(draw) + '.npz')
        if archive_path != expected_path:
            raise ValueError('Array evidence belongs to a different trajectory')
        with np.load(archive_path, allow_pickle=False) as archive:
            expected_keys = {'own_target_positions', 'feedback_eligible_indices', 'own_factual_logp', 'own_ablated_logp'}
            if case['profile'] != 'qwen4b':
                expected_keys |= {'reference_target_positions', 'reference_logp'}
            if set(archive.files) != expected_keys:
                raise ValueError('Unexpected/missing likelihood evidence arrays')
            require_array(archive, 'own_target_positions', np.int32, x['target_positions'])
            require_array(archive, 'feedback_eligible_indices', np.int32, eligible)
            factual = require_array(archive, 'own_factual_logp', np.float32,
                length=len(x['target_positions']) if eligible or case['profile'] == 'qwen4b' else 0)
            ablated = require_array(archive, 'own_ablated_logp', np.float32,
                length=len(x['target_positions']) if eligible else 0)
            if case['profile'] == 'qwen4b':
                if own_row != reference_row:
                    raise ValueError('The shared4B reference is not the same own trajectory')
                reference_key, position_key = 'own_factual_logp', 'own_target_positions'
            else:
                reference_key, position_key = 'reference_logp', 'reference_target_positions'
            if row['reference_logprob_array'] != reference_key or row['reference_position_array'] != position_key:
                raise ValueError('Incorrect reference-sharing provenance')
            require_array(archive, position_key, np.int32, y['target_positions'])
            reference = require_array(archive, reference_key, np.float32, length=len(y['target_positions']))
            feedback = feedback_summary(factual, ablated, eligible) if eligible else {
                'value': 0.0, 'eligible_model_tokens': 0, 'prefix_check_performed': False,
                'reason': 'Definition gives0 when no own generated targets follow changed feedback.'}
            compatibility = reference_summary(reference)
            if row['feedback_summary'] != feedback or row['compatibility_summary'] != compatibility:
                raise ValueError('Stored likelihood arrays and scalar measurements differ')
            if row['values'] != {SIGNALS[0]: feedback['value'], SIGNALS[1]: compatibility['value']}:
                raise ValueError('Scored feature values differ from raw arrays')
            identity = {k: row[k] for k in ['task_id', 'rollout_index', 'own_draw_id', 'reference_draw_id']}

            def expected(role, sequence, positions, values):
                expected_forwards.append({'forward_id': len(expected_forwards), 'trajectory': identity, 'role': role,
                    'input_positions': len(sequence), 'scored_model_tokens': len(positions),
                    'input_ids_int32_sha256': array_sha(sequence, np.int32),
                    'target_positions_int32_sha256': array_sha(positions, np.int32),
                    'float32_logprob_sha256': array_sha(values, np.float32)})

            if case['profile'] == 'qwen4b' or eligible:
                expected('own_factual', x['input_ids'], x['target_positions'], factual)
            if eligible:
                expected('own_ablated', x['ablated_input_ids'], x['target_positions'], ablated)
            if case['profile'] != 'qwen4b':
                expected('fixed4b_reference', y['input_ids'], y['target_positions'], reference)
        if (row['own_mask_audit'] != x['mask_audit']
                or row['own_input_ids_int32_sha256'] != array_sha(x['input_ids'], np.int32)
                or row['ablated_input_ids_int32_sha256'] != array_sha(x['ablated_input_ids'], np.int32)
                or row['reference_input_ids_int32_sha256'] != array_sha(y['input_ids'], np.int32)):
            raise ValueError('Counterfactual/source input token lineage differs')
        if row['raw_reference_is_fixed4b'] is not True or row['reward_or_grade_filtering'] is not False:
            raise ValueError('Wrong reference/filtering declaration')
    journal = [json.loads(line) for line in (output / 'forward_attempts.jsonl').open()]
    if len(journal) != 2 * len(expected_forwards):
        raise ValueError('Incomplete or repeated actual forward attempts')
    elapsed = 0.
    for i, expected in enumerate(expected_forwards):
        start, end = journal[2 * i:2 * i + 2]
        if start['state'] != 'started' or end['state'] != 'completed' or start['unix'] > end['unix']:
            raise ValueError('Forward journal is not a single completed attempt')
        if any(end.get(k) != value for k, value in expected.items()):
            raise ValueError('Archived probabilities and completed forward journal differ')
        if any(start.get(k) != value for k, value in expected.items() if k != 'float32_logprob_sha256'):
            raise ValueError('Forward start context/target/model identity differs')
        elapsed += end['wall_seconds']
    calls, positions = len(expected_forwards), sum(x['input_positions'] for x in expected_forwards)
    expected_cost = {'forward_calls_started': calls, 'forward_calls_completed': calls,
        'forward_input_positions_started': positions, 'forward_input_positions_completed': positions,
        'forward_wall_seconds': elapsed, 'new_model_tokens': 0, 'new_environment_interactions': 0, 'optimizer_updates': 0}
    if summary['cost'] != expected_cost or calls > case['maximum_forward_calls'] or positions > case['forward_input_position_ceiling']:
        raise ValueError('Actual forward cost differs or exceeds its fixed bounds')
    aggregated = summarize_episodes(case, rows)
    if any(summary[k] != value for k, value in aggregated.items()):
        raise ValueError('Equal-draw/equal-task source aggregation differs')
    if (summary['case'] != name or summary['profile'] != case['profile']
            or summary['source_group'] != case['source_group']
            or summary['episodes'] != 4 * len(case['task_ids'])
            or summary['target_evaluation'] is not False or summary['reserved21_accessed'] is not False):
        raise ValueError('Wrong source/model episode count')
    return {'status': 'passed', 'case': name, 'profile': case['profile'], 'source_group': case['source_group'],
        'source_features': aggregated['source_features'], 'episodes': len(rows),
        'cost': expected_cost | {'actual_supervised_worker_wall_seconds': terminal['wall_seconds']},
        'summary': ref(root, output / 'summary.json'), 'run': ref(root, output / 'run.json'),
        'episode_scores': ref(root, output / 'episode_scores.jsonl'),
        'forward_journal': ref(root, output / 'forward_attempts.jsonl'),
        'array_files': [r['array_file'] for r in rows], 'raw_probability_replay': 'exact',
        'optimizer_updates': 0, 'target_evaluation': False, 'scope': 'Measurement integrity only, no transfer-efficacy conclusion.'}
