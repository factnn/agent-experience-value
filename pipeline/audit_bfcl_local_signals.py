"""Retrospective calibration of logged pre-acquisition signals; no new inference.

Local reward variation is not transfer value. Policy-dependent acquisition and
shared ancestry preclude treating these decision rows as independent samples.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from bfcl_allocation import CombinationAllocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'rl/bfcl_local_signal_audit_20261011/analysis'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def beta_mixed_probability(successes, observations, group_size=4):
    alpha, beta = successes + 1, observations - successes + 1
    if not 0 <= successes <= observations or group_size < 1:
        raise ValueError('Invalid history or group size')
    def moment(parameter):
        value = 1.
        for k in range(group_size):
            value *= (parameter + k) / (alpha + beta + k)
        return value
    return 1 - moment(alpha) - moment(beta)


def auc(rows, field):
    positive = [r[field] for r in rows if r['mixed_reward']]
    negative = [r[field] for r in rows if not r['mixed_reward']]
    if not positive or not negative:
        return None
    # Symmetric p and 1-p yield equal Frontier mathematically, but can differ by
    # floating-point roundoff. Do not treat a 1e-16 difference as ranking evidence.
    return sum((x - y > 1e-12) + .5 * (abs(x - y) <= 1e-12)
               for x in positive for y in negative) / (len(positive) * len(negative))


def summarize(rows):
    return {'groups': len(rows), 'mixed_groups': sum(r['mixed_reward'] for r in rows),
            'mixed_rate': mean(r['mixed_reward'] for r in rows),
            'new_model_tokens_in_archived_groups': sum(r['sampled_tokens'] for r in rows),
            'mean_forecast_mixed_iid': mean(r['mixed_iid'] for r in rows),
            'mean_forecast_mixed_beta': mean(r['mixed_beta'] for r in rows),
            'mixed_brier_iid': mean((r['mixed_iid'] - r['mixed_reward']) ** 2 for r in rows),
            'mixed_brier_beta': mean((r['mixed_beta'] - r['mixed_reward']) ** 2 for r in rows),
            'mixed_brier_constant_half': mean((.5 - r['mixed_reward']) ** 2 for r in rows),
            'frontier_auc_mixed_descriptive': auc(rows, 'frontier'),
            'coverage_auc_mixed_descriptive': auc(rows, 'coverage_deficit'),
            'posterior_success_brier_per_episode': mean(
                mean((r['posterior_success'] - reward) ** 2 for reward in r['rewards']) for r in rows)}


def main():
    sources = [ROOT / 'rl/bfcl_intervention_followup_20261009' / f'train_{seed}_{rule}'
               for seed in range(2) for rule in ['uniform', 'frontier', 'coverage']]
    sources += [ROOT / 'rl/bfcl_intervention_dose_20261010/train']
    sources += [ROOT / 'rl/bfcl_intervention_dose_repeats_20261011' / f'train_repeat_{i}' for i in range(1, 4)]
    all_rows, trajectory_rows, provenance = [], [], []
    for source in sources:
        summary, restored = read(source / 'summary.json'), read(source / 'restored.json')
        assert summary['status'] == 'complete'
        allocations, groups, updates = [lines(source / f'{name}.jsonl') for name in ['allocation', 'groups', 'updates']]
        assert len(allocations) == len(groups) == len(updates) == summary['optimizer_steps']
        allocator = CombinationAllocator.from_state_dict(restored['inherited_allocator_state'],
            rule=summary.get('rule', 'uniform'), pool=list(allocations[0]['probabilities']))
        metrics = {r['step']: r['grad_norm'] for r in lines(source / 'metrics.jsonl') if 'grad_norm' in r}
        rows = []
        for index, (allocation, group, update) in enumerate(zip(allocations, groups, updates)):
            assert allocation['policy_step'] == group['policy_step'] == index and update['step'] == index + 1
            assert allocation['signals_before'] == allocator.signals()
            assert allocation['probabilities'] == allocator.probabilities()
            assert allocation['cost_before_choice'] == allocator.total_sampled_tokens
            assert allocation['rewards'] == group['rewards'] and len(group['rewards']) == 4
            signal = allocation['signals_before'][allocation['combination']]
            p = signal['posterior_success']
            n, successes = signal['observed_episodes'], signal['successes_in_window']
            assert p == (successes + 1) / (n + 2) and signal['frontier'] == 4 * p * (1 - p)
            mixed = len(set(group['rewards'])) > 1
            assert mixed == any(abs(x) > 1e-8 for x in group['advantages'])
            assert mixed == (metrics[index + 1] > 0)
            key = {'task_id': allocation['task_id'], 'seed': allocation['seed'],
                   'policy_fingerprint': allocation['policy_fingerprint']}
            row = {'trajectory': str(source.relative_to(ROOT)), 'policy_step': index,
                   'combination': allocation['combination'], **key, **signal,
                   'mixed_reward': int(mixed), 'rewards': group['rewards'],
                   'mixed_iid': 1 - p ** 4 - (1 - p) ** 4,
                   'mixed_beta': beta_mixed_probability(successes, n),
                   'sampled_tokens': allocation['sampled_tokens'], 'grad_norm': metrics[index + 1],
                   'parameters_changed': update['parameters_changed'],
                   'sampling_context_sha256': hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()}
            rows.append(row)
            allocator.choose(forced=allocation['task_id'])
            allocator.observe(allocation['task_id'], group['rewards'], allocation['sampled_tokens'], f'audit:{index}')
        all_rows.extend(rows)
        trajectory_rows.append({'path': str(source.relative_to(ROOT)), 'rule': allocator.rule, **summarize(rows)})
        provenance.append({'path': str(source.relative_to(ROOT)), 'sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ['summary.json', 'restored.json', 'allocation.jsonl', 'groups.jsonl', 'updates.jsonl', 'metrics.jsonl']}})
    cold = [r for r in all_rows if r['observed_episodes'] == 0]
    warm = [r for r in all_rows if r['observed_episodes'] > 0]
    contexts = Counter(r['sampling_context_sha256'] for r in all_rows)
    result = {'status': 'complete', 'analysis_type': 'retrospective_local_signal_calibration',
        'analysis_revision': 2, 'auc_tie_tolerance': 1e-12,
        'revision_note': 'Treat mathematically symmetric Frontier scores as ties despite floating-point roundoff; first observer snapshot retained in event history.',
        'new_model_calls': 0, 'new_sampled_tokens': 0, 'predictor_fitted': False,
        'feature_timing_audit': 'reconstructed allocator histories/probabilities/costs match every pre-acquisition row',
        'all_groups': summarize(all_rows), 'cold_history': summarize(cold), 'observed_history': summarize(warm),
        'trajectories': trajectory_rows, 'distinct_sampling_contexts': len(contexts),
        'repeated_sampling_context_counts': [n for n in contexts.values() if n > 1], 'sources': provenance,
        'limits': ['Selected acquisition groups only, not an unbiased estimate over every available task.',
                   'Shared ancestor and repeated sampling contexts; rows/seeds are not independent population samples.',
                   'IID Bernoulli and Beta-binomial forecasts are retrospective diagnostics, not preregistered predictive tests.',
                   'Combination history aggregates heterogeneous tasks and changing learner policies.',
                   'Local mixed rewards/nonzero advantages do not measure ID or transfer benefit. No efficacy conclusion.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'decision_rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in all_rows))
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['all_groups', 'cold_history', 'observed_history', 'distinct_sampling_contexts']}))


if __name__ == '__main__':
    main()
