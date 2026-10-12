"""Audit full source-group acquisition traces and corrected learner checkpoints."""
import argparse
import hashlib
import math
from pathlib import Path
import torch
from bfcl_allocation import CombinationAllocator
from bfcl_dose_checkpoint import assert_equal
from bfcl_source_group_contract import validate_intervention, learner_progress
from report_bfcl_intervention import read, lines, audit_sampling
from rl_learning_state import tensor_hash
from supervise_bfcl_stage_fork import ROOT, write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_trace(job, summary, allocations, groups, updates, microsteps):
    steps = summary['branch_local_optimizer_steps']
    assert steps > 0 and len(allocations) == len(groups) == len(updates) == steps
    assert len(microsteps) == 4 * steps
    assert summary['stop_reason'] == 'token_budget_after_complete_group'
    assert summary['nominal_token_budget'] == job['token_budget']
    total = 0
    for i, (a, g, u) in enumerate(zip(allocations, groups, updates)):
        assert total < job['token_budget']
        assert a['policy_step'] == g['policy_step'] == i and u['step'] == i + 1
        assert a['cumulative_policy_step'] == job['source_step'] + i
        assert a['seed'] == job['rollout_seed_base'] + 100 * (job['source_step'] + i)
        assert a['task_id'] in job['task_pool'] and g['task_ids'] == [a['task_id']] * 4
        assert a['rewards'] == g['rewards'] and len(a['rewards']) == 4
        assert len(g['advantages']) == 4 and all(math.isfinite(v) for v in g['advantages'])
        assert (len(set(a['rewards'])) > 1) == any(abs(v) > 1e-8 for v in g['advantages'])
        assert a['probabilities'] == {t: 1 / len(job['task_pool']) for t in job['task_pool']}
        assert a['policy_fingerprint'] == u['before_sha256']
        if i:
            assert u['before_sha256'] == updates[i-1]['after_sha256']
        total += a['sampled_tokens']
        assert a['branch_cost_after_group'] == total
    assert job['token_budget'] <= total <= job['token_budget'] + 32768
    assert total == summary['new_sampled_tokens'] == summary['branch_sampled_tokens']
    assert updates[-1]['after_sha256'] == summary['final_fingerprint']
    for i, row in enumerate(microsteps):
        assert row['micro_step_after'] == 4 * job['source_step'] + i + 1
        assert row['local_optimizer_step'] == i // 4
        assert row['accelerate_backward_divisor'] == 1 and row['grpo_loss_divisor'] == 4
        assert math.isfinite(row['loss'])
    return total


def audit_job(job, job_sha):
    out = ROOT / job['output']
    summary, restored, run = [read(out / name) for name in ['summary.json', 'restored.json', 'run.json']]
    assert summary['status'] == 'complete' and not summary['evaluation_performed']
    assert run['job_sha256'] == job_sha and run['case_id'] == job['case_id']
    assert run['normalization'] == 'single' and run['deterministic_algorithms']
    protocol_path = ROOT / job['algorithm_protocol']['path']
    assert digest(protocol_path) == job['algorithm_protocol']['sha256'] == run['algorithm_protocol_sha256']
    protocol = read(protocol_path)
    validate_intervention(job, protocol['training_registry'])
    a, g, u, micro = [lines(out / name) for name in ['allocation.jsonl', 'groups.jsonl', 'updates.jsonl', 'microsteps.jsonl']]
    total = audit_trace(job, summary, a, g, u, micro)
    cost = audit_sampling(out)
    assert cost['rollouts'] == 4 * len(a) and cost['new_model_tokens'] == total
    rollouts = lines(out / 'rollouts.jsonl')
    calls = lines(out / 'generation_calls.jsonl')
    for i, allocation in enumerate(a):
        rows = [x for x in rollouts if x['policy_step'] == i]
        assert len(rows) == 4 and [x['rollout_index'] for x in rows] == list(range(4))
        assert [x['reward'] for x in rows] == allocation['rewards']
        assert all(x['task_id'] == allocation['task_id'] and x['policy_fingerprint'] == allocation['policy_fingerprint'] for x in rows)
        charged = sum(len(x['generated_ids']) for x in calls if x['policy_step'] == i)
        assert charged == allocation['sampled_tokens'] == g[i]['model_tokens_in_loss']
    source = ROOT / job['source_checkpoint'] / 'components'
    assert digest(source / 'learning_state.pt') == job['source_payload_sha256']
    prior = torch.load(source / 'learning_state.pt', map_location='cpu', weights_only=False)
    allocator = CombinationAllocator.from_state_dict(prior['allocator'], rule='uniform', pool=job['task_pool'])
    allocator.rng.seed(job['allocator_seed'])
    inherited = allocator.total_sampled_tokens
    assert restored['status'] == 'verified' and restored['history_preserved'] and restored['fresh_rollout_buffer']
    assert restored['source_payload_sha256'] == job['source_payload_sha256']
    assert restored['inherited_tokens'] == inherited and restored['source_step'] == job['source_step']
    assert restored['policy_fingerprint'] == a[0]['policy_fingerprint']
    assert restored['initial_signals'] == allocator.signals()
    assert restored['initial_probabilities'] == allocator.probabilities()
    for i, allocation in enumerate(a):
        choice = allocator.choose()
        for key, value in choice.items():
            assert value == allocation[key], key
        allocator.observe(choice['task_id'], allocation['rewards'], allocation['sampled_tokens'],
                          f"{job['case_id']}:{job['source_step']+i}")
    manifest = read(out / 'components/manifest.json')
    assert manifest == summary['final_manifest']
    assert digest(out / 'components/learning_state.pt') == manifest['payload_sha256']
    payload = torch.load(out / 'components/learning_state.pt', map_location='cpu', weights_only=False)
    assert_equal(allocator.state_dict(), payload['allocator'])
    progress = learner_progress(job['source_step'], len(a), inherited, total)
    assert progress == payload['progress'] == manifest['progress']
    for key, value in progress.items():
        assert summary[key] == value
    assert manifest['cumulative_acquisition_tokens'] == inherited + total
    assert manifest['frozen_base_sha256'] == prior['frozen_hash'] == payload['frozen_hash']
    assert manifest['trainable_sha256'] == tensor_hash(payload['trainable'].items())
    expected_provenance = {'base_revision': protocol['base_revision'],
        'task_manifest_sha256': protocol['manifest_sha256'],
        'algorithm_config_sha256': hashlib.sha256(__import__('json').dumps(protocol['algorithm'], sort_keys=True).encode()).hexdigest(),
        'intervention_protocol_sha256': digest(protocol_path)}
    assert payload['provenance'] == manifest['provenance'] == expected_provenance
    assert payload['optimizer_type'] == 'AdamW'
    assert all(int(v['step']) == job['source_step'] + len(a) for v in payload['optimizer']['state'].values())
    metrics = lines(out / 'metrics.jsonl')
    grad_rows = [x for x in metrics if 'grad_norm' in x]
    assert len(grad_rows) == len(a) and all(math.isfinite(x['grad_norm']) for x in grad_rows)
    return {'status': 'passed', 'case_id': job['case_id'], 'role': job['role'], 'job_sha256': job_sha,
            'optimizer_steps': len(a), 'fresh_nonzero_gradient_steps': [x['step'] for x in grad_rows if x['grad_norm'] > 0],
            'changed_parameter_steps': [x['step'] for x in u if x['parameters_changed']],
            'cost': cost, 'inherited_tokens_not_recharged': inherited,
            'full_allocator_replay': 'exact_match', 'final_payload_sha256': manifest['payload_sha256'],
            'scope': 'Training acquisition and learner-state audit only; no transfer G/V labels.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = read(args.job)
    try:
        result = audit_job(job, digest(args.job))
    except Exception as error:
        result = {'status': 'failed', 'type': type(error).__name__, 'message': str(error), 'partial_evidence_retained': True}
    write(ROOT / job['output'] / 'audit.json', result)
    print(__import__('json').dumps(result))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
