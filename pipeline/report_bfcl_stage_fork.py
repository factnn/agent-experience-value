"""Require exact recorded continuation; retain mismatches as evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from report_bfcl_intervention import ROOT, read, lines, audit_sampling


def compare_rows(reference, actual, fields, label):
    differences = []
    if len(reference) != len(actual):
        differences.append({'section': label, 'reference_count': len(reference), 'actual_count': len(actual)})
    for i, (left, right) in enumerate(zip(reference, actual)):
        changed = [key for key in fields if key not in left or key not in right or left[key] != right[key]]
        if changed:
            differences.append({'section': label, 'row': i, 'fields': changed})
    return differences


def audit(acceptance):
    start = acceptance['source_step']
    count = acceptance['replay_updates']
    source = ROOT / acceptance['reference_training']
    out = ROOT / acceptance['output']
    for name, expected in acceptance['reference_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
    restored = read(out / 'restored.json')
    summary = read(out / 'summary.json')
    assert summary['status'] == 'complete'
    assert restored['source_step'] == start and restored['local_step'] == 0
    assert restored['source_payload_sha256'] == acceptance['source_payload_sha256']
    assert restored['optimizer_nonzero_moments'] and restored['fresh_rollout_buffer']
    assert not restored['allocator_reset']
    assert summary['branch_local_optimizer_steps'] == count
    assert summary['cumulative_learner_optimizer_steps'] == start + count
    assert summary['micro_step'] == 4 * (start + count)
    assert summary['inherited_tokens'] == restored['inherited_tokens']
    manifest = read(ROOT / acceptance['source_checkpoint'] / 'components/manifest.json')
    assert summary['inherited_tokens'] == manifest['cumulative_acquisition_tokens']
    cost = audit_sampling(out)
    assert cost['new_model_tokens'] == summary['new_sampled_tokens']
    assert cost['rollouts'] == 4 * count
    differences = []
    fields = {
        'allocation': ['task_id', 'combination', 'phase', 'probabilities', 'signals_before',
                       'cost_before_choice', 'seed', 'policy_fingerprint', 'rewards', 'sampled_tokens', 'stop_reasons'],
        'groups': ['task_ids', 'rewards', 'advantages', 'model_tokens_in_loss', 'external_tokens_excluded'],
        'generation_calls': ['rollout_index', 'task_id', 'input_ids', 'generated_ids',
                             'generation_allowance', 'remaining_completion_budget'],
        'rollouts': ['task_id', 'enable_thinking', 'prompt_ids', 'completion_ids', 'model_token_mask',
                     'segments', 'bridges', 'completed_user_turns', 'total_user_turns', 'reward',
                     'stop_reason', 'rollout_index', 'policy_fingerprint'],
        'updates': ['before_sha256', 'after_sha256', 'parameters_changed']}
    for section, keys in fields.items():
        index = 'step' if section == 'updates' else 'policy_step'
        shift = 1 if section == 'updates' else 0
        old = [row for row in lines(source / (section + '.jsonl'))
               if start + shift <= row[index] < start + count + shift]
        new = lines(out / (section + '.jsonl'))
        differences += compare_rows(old, new, keys, section)
        if [r[index] - start for r in old] != [r[index] for r in new]:
            differences.append({'section': section, 'fields': ['learner/local step alignment']})
    allocations = lines(out / 'allocation.jsonl')
    assert len(allocations) == count
    assert [r['policy_step'] for r in allocations] == list(range(count))
    assert [r['cumulative_policy_step'] for r in allocations] == list(range(start, start + count))
    cumulative = 0
    for row in allocations:
        assert row['cost_before_choice'] == summary['inherited_tokens'] + cumulative
        cumulative += row['sampled_tokens']
        assert row['branch_cost_after_group'] == cumulative
    assert cumulative == cost['new_model_tokens'] <= acceptance['new_token_ceiling']
    updates = lines(out / 'updates.jsonl')
    expected = lines(source / 'updates.jsonl')[start + count - 1]['after_sha256']
    assert summary['final_fingerprint'] == updates[-1]['after_sha256']
    if summary['final_fingerprint'] != expected:
        differences.append({'section': 'endpoint', 'fields': ['final_fingerprint']})
    metrics = [r for r in lines(out / 'metrics.jsonl') if 'grad_norm' in r]
    reference_metrics = [r for r in lines(source / 'metrics.jsonl')
                         if 'grad_norm' in r and start < r['step'] <= start + count]
    differences += compare_rows(reference_metrics, metrics, ['grad_norm'], 'gradient_norm')
    fresh = [r['step'] for r in metrics if r['grad_norm'] > 0]
    if not fresh:
        differences.append({'section': 'learning_signal', 'fields': ['no fresh nonzero gradient observed']})
    return {'status': 'passed' if not differences else 'mismatch_requires_review',
            'source_step': start, 'replayed_updates': count,
            'inherited_acquisition_tokens': summary['inherited_tokens'], 'new_cost': cost,
            'fresh_gradient_local_steps': fresh, 'exact_replay': not differences,
            'differences': differences, 'final_fingerprint': summary['final_fingerprint'],
            'scope': 'Fresh trainer boundary fork with nonzero Adam state; no nested fork, arbitrary dataloader resume, or efficacy claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    acceptance = read(args.protocol)
    out = ROOT / acceptance['output']
    try:
        result = audit(acceptance)
    except Exception as error:
        result = {'status': 'failed_or_incomplete', 'error_type': type(error).__name__,
                  'message': str(error), 'partial_evidence_retained': True}
    (out / 'replay_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
