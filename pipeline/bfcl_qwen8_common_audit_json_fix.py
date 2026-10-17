"""Finish the native8B common audit with exact JSON-compatible allocator checks.

The original frozen implementation and its failed control evidence stay intact.
Only tuple/list representation of the Python RNG state differs on JSON storage.
"""
import copy
import json

import torch

from bfcl_allocation import CombinationAllocator
from bfcl_qwen8_initial_probe import REVISION, digest, exclusive, pinned, ref
from report_bfcl_intervention import audit_sampling, lines, read
from rl_learning_state import tensor_hash
from supervise_bfcl_stage_fork import ROOT


def compare_allocation_state(replayed, payload, serialized):
    assert replayed == payload, 'Exact allocator replay differs from checkpoint'
    assert serialized == json.loads(json.dumps(payload)), 'JSON allocator values differ'
    restored = CombinationAllocator.from_state_dict(serialized).state_dict()
    assert restored == payload, 'Restored JSON allocator, including RNG, differs'
    return {'checkpoint_replay': 'exact', 'json_value_comparison': 'exact',
            'json_rng_restore_comparison': 'exact',
            'representation_change': 'Python RNG tuples serialize as JSON arrays only'}


def audit_common(p):
    out = ROOT / p['common_output']
    summary = read(out / 'summary.json')
    cost = audit_sampling(out)
    assert summary['status'] == 'complete' and summary['optimizer_updates'] == 0
    assert cost['new_model_tokens'] == summary['sampled_tokens'] <= 262144
    assert cost['rollouts'] == 32
    source = pinned(p['source_protocol'])
    manifest = read(out / 'components/manifest.json')
    assert digest(out / 'components/learning_state.pt') == manifest['payload_sha256']
    payload = torch.load(out / 'components/learning_state.pt', map_location='cpu', weights_only=False)
    assert payload['progress'] == manifest['progress']
    assert payload['progress']['global_step'] == payload['progress']['micro_step'] == 0
    assert not payload['optimizer']['state'] and payload['optimizer_type'] == 'AdamW'
    assert tensor_hash(payload['trainable'].items()) == manifest['trainable_sha256']
    assert payload['provenance'] == manifest['provenance']
    assert payload['provenance']['base_revision'] == REVISION
    assert payload['provenance']['intervention_protocol_sha256'] == p['source_protocol']['sha256']
    allocator = CombinationAllocator(source['training_registry'], seed=source['seeds']['initialization'])
    history = lines(out / 'probe_history.jsonl')
    assert len(history) == 8
    for i, row in enumerate(history):
        key = source['common_probe_task_ids'][i]
        choice = allocator.choose(forced=key)
        assert row['task_id'] == key and row['seed'] == source['seeds']['common_probe'] + 100 * i
        for k, v in choice.items():
            assert row[k] == v, k
        allocator.observe(key, row['rewards'], row['sampled_tokens'], f'probe:{i}')
    equivalence = compare_allocation_state(allocator.state_dict(), payload['allocator'],
                                          read(out / 'allocation_state.json'))
    assert allocator.total_sampled_tokens == manifest['cumulative_acquisition_tokens'] == cost['new_model_tokens']
    run = read(out / 'run.json')
    assert run['model_revision'] == REVISION and run['target_evaluation'] is False
    assert run['model_verification'] == p['model_verification'] and run['common_optimizer_updates'] == 0
    terminal = read(out / 'supervisor.json')
    assert terminal['returncode'] == 0 and terminal['hard_timeout'] is False
    result = {'status': 'passed', 'cost': cost, 'common_optimizer_updates': 0,
              'common_history_groups': 8, 'policy_fingerprint': summary['policy_fingerprint'],
              'learning_state': ref(out / 'components/learning_state.pt'),
              'manifest': ref(out / 'components/manifest.json'),
              'adapter_config': ref(out / 'adapter/adapter_config.json'),
              'adapter_weights': ref(out / 'adapter/adapter_model.safetensors'),
              'allocation_history_replay': 'exact', 'allocation_serialization': equivalence,
              'optimizer_initial_state_entries': 0, 'target_evaluation': False,
              'reserved21_accessed': False,
              'audit_implementation': ref(ROOT / 'pipeline/bfcl_qwen8_common_audit_json_fix.py'),
              'prior_control_failure': ref(ROOT / p['control_root'] / 'control_failure.json'),
              'scope': 'Native8B step0, own charged history and empty optimizer; no resampling or efficacy claim.'}
    exclusive(out / 'audit.json', result)
    return result, allocator.signals()


def self_test():
    allocator = CombinationAllocator({'task_a': 'source_a', 'task_b': 'source_b'}, seed=123)
    allocator.choose(forced='task_a')
    allocator.observe('task_a', [0, 1, 0, 1], 1234, 'probe:0')
    state = allocator.state_dict()
    serialized = json.loads(json.dumps(state))
    compare_allocation_state(state, state, serialized)
    rejected = []
    for name in ['rng_state', 'total_sampled_tokens', 'counts', 'extra_field']:
        bad = copy.deepcopy(serialized)
        if name == 'rng_state':
            bad[name][1][1] ^= 1
        elif name == 'counts':
            bad[name]['source_a'] += 1
        elif name == 'extra_field':
            bad[name] = 1
        else:
            bad[name] += 1
        try:
            compare_allocation_state(state, state, bad)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Changed allocator field accepted: ' + name)
    assert len(rejected) == 4
    return {'status': 'passed', 'tuple_list_equivalence': True, 'changed_fields_rejected': rejected,
            'new_model_tokens': 0, 'gpu_jobs': 0}


if __name__ == '__main__':
    print(json.dumps(self_test()))
