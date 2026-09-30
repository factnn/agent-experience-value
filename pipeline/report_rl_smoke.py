"""Audit saved online-policy provenance and summarize engineering costs."""
import argparse
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def audit(path):
    summary = json.loads((path/'summary.json').read_text())
    updates = read_rows(path/'updates.jsonl')
    groups = read_rows(path/'groups.jsonl')
    trajectories = read_rows(path/'rollouts.jsonl')
    generations = read_rows(path/'generation_calls.jsonl')
    metrics = read_rows(path/'metrics.jsonl')
    nonzero_gradient_steps = [m['step'] for m in metrics if float(m.get('grad_norm', 0)) > 0]
    assert len(updates) == summary['optimizer_steps']
    assert len(trajectories) == len(groups)*4
    fingerprints = {u['step']: u['after_sha256'] for u in updates}
    fingerprints[0] = updates[0]['before_sha256']
    for t in trajectories:
        assert t['policy_fingerprint'] == fingerprints[t['policy_step']]
        assert len(t['completion_ids']) == len(t['model_token_mask'])
        assert set(t['model_token_mask']) <= {0, 1}
        assert t['model_tokens'] == sum(t['model_token_mask'])
    for g in groups:
        batch = [t for t in trajectories if (t['policy_step'], t['phase']) == (g['policy_step'], g['phase'])]
        assert [t['reward'] for t in batch] == g['rewards']
        assert [t['task_id'] for t in batch] == g['task_ids']
    post = [t for t in trajectories if t['phase'] == 'post_update_resample']
    assert len(post) == 4 and all(t['policy_step'] == len(updates) for t in post)
    train = [t for t in trajectories if t['phase'] == 'train']
    kept_tokens = sum(t['model_tokens'] for t in trajectories)
    sampled = sum(sum(g['sampled_tokens']) for g in generations) if generations else None
    if sampled is not None:
        assert sampled == summary['all_sampled_tokens_including_discarded']
        assert sampled >= kept_tokens
    return {'run': path.name, 'audit': 'passed', 'closed_loop_status': summary['status'],
            'train_successes': sum(t['reward'] for t in train), 'train_episodes': len(train),
            'post_update_successes': sum(t['reward'] for t in post), 'post_update_episodes': len(post),
            'parameter_change_steps': sum(u['parameters_changed'] for u in updates),
            'nonzero_gradient_steps': nonzero_gradient_steps,
            'optimizer_note': 'Adam momentum can move parameters on later zero-gradient groups; changed hashes alone do not prove fresh learning signal.',
            'groups_with_variance': summary['groups_with_reward_variance'],
            'sampled_tokens_including_discarded': sampled, 'retained_model_tokens': kept_tokens,
            'discarded_sampled_tokens': sampled - kept_tokens if sampled is not None else None,
            'retained_feedback_tokens': sum(t['feedback_tokens'] for t in trajectories),
            'environment_calls': sum(len(t['events']) for t in trajectories),
            'environment_seconds': sum(t['env_seconds'] for t in trajectories),
            'wall_seconds': summary['total_wall_seconds'],
            'peak_allocated_gb': summary['peak_allocated_gb'],
            'interpretation': 'Engineering only. No held-out efficacy, allocation effect, or cross-seed evidence.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    args = ap.parse_args()
    for path in args.runs:
        report = audit(path)
        (path/'audit_report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
