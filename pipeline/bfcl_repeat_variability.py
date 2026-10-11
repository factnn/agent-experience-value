"""Descriptive seed contrasts, not population variance component estimates."""
from statistics import mean, stdev


def describe_variability(rows, baseline, training_names, evaluation_names, thresholds):
    assert len(evaluation_names) == 2 and len(set(evaluation_names)) == 2
    assert len(training_names) >= 2 and len(set(training_names)) == len(training_names)
    e0, e1 = evaluation_names
    base = {}
    for name in evaluation_names:
        outcomes = baseline[name]['outcomes']
        base[name] = {r['task_id']: r['rewards'][0] for r in outcomes}
        assert len(base[name]) == len(outcomes) and base[name]
        assert all(v in (0, 1) for v in base[name].values())
    assert base[e0].keys() == base[e1].keys()
    n = len(base[e0])
    baseline_delta = sum(base[e1].values()) - sum(base[e0].values())
    indexed = {}
    for r in rows:
        key = (r['training_repeat'], r['threshold'], r['evaluation_repeat'])
        assert key not in indexed, 'duplicate cell'
        indexed[key] = r
    repeats = sorted({r['training_repeat'] for r in rows})
    assert set(training_names).issubset(repeats)
    assert len(indexed) == len(repeats) * len(thresholds) * 2, 'incomplete matrix'
    fixed_policy = []
    for repeat in repeats:
        for threshold in thresholds:
            a, b = (indexed[(repeat, threshold, e)] for e in evaluation_names)
            assert a['policy_fingerprint'] == b['policy_fingerprint']
            paired = []
            for r, name in [(a, e0), (b, e1)]:
                outcomes = {x['task_id']: x for x in r['paired_outcomes']}
                assert len(outcomes) == len(r['paired_outcomes']) == n
                assert outcomes.keys() == base[name].keys()
                assert all(x['source'] == base[name][task] and x['checkpoint'] in (0, 1)
                           for task, x in outcomes.items())
                assert sum(x['checkpoint'] for x in outcomes.values()) == r['successes']
                assert abs(r['G'] - (r['successes'] - sum(base[name].values())) / n) < 1e-12
                assert r['prospective_training_repeat'] == (repeat in training_names)
                paired.append(outcomes)
            deltas = [r['successes'] - sum(base[e].values()) for r, e in [(a, e0), (b, e1)]]
            policy_delta = b['successes'] - a['successes']
            assert deltas[1] - deltas[0] == policy_delta - baseline_delta
            fixed_policy.append({
                'training_repeat': repeat, 'threshold': threshold,
                'prospective_training_repeat': repeat in training_names,
                'policy_fingerprint': a['policy_fingerprint'],
                'successes_by_eval': {e0: a['successes'], e1: b['successes']},
                'G_task_counts_by_eval': dict(zip(evaluation_names, deltas)),
                'policy_success_delta_e1_minus_e0': policy_delta,
                'G_task_count_delta_e1_minus_e0': deltas[1] - deltas[0],
                'strict_G_sign_flip': deltas[0] * deltas[1] < 0,
                'policy_outcome_disagreements': sum(paired[0][t]['checkpoint'] != paired[1][t]['checkpoint'] for t in base[e0]),
                'paired_gain_disagreements': sum(
                    paired[0][t]['checkpoint'] - base[e0][t] != paired[1][t]['checkpoint'] - base[e1][t]
                    for t in base[e0]),
            })
    across_training = []
    for threshold in thresholds:
        values = {name: mean(indexed[(name, threshold, e)]['G'] for e in evaluation_names)
                  for name in training_names}
        across_training.append({'threshold': threshold, 'G_mean_over_two_eval_seeds_by_training_repeat': values,
                                'conditional_G_mean': mean(values.values()),
                                'conditional_G_range': [min(values.values()), max(values.values())],
                                'conditional_G_sample_sd': stdev(values.values())})
    return {
        'analysis_status': 'supplementary descriptive analysis added after partial panels were observed',
        'evaluation_order': evaluation_names, 'tasks': n,
        'baseline_success_delta_e1_minus_e0': baseline_delta,
        'baseline_outcome_disagreements': sum(base[e0][t] != base[e1][t] for t in base[e0]),
        'fixed_policy_evaluation_contrasts': fixed_policy,
        'prospective_training_contrasts_averaged_over_evaluation': across_training,
        'limits': [
            'Two evaluation seeds on shared tasks; contrasts are descriptive, not a population noise estimate.',
            'Training sample SD retains residual evaluation noise and is not a pure training variance component.',
            'Shared baseline/history and repeated tasks prevent treating cells as independent samples.',
            'Discovery is reported separately and excluded from prospective training summaries.',
            'Dose labels may alias one policy; no pooling of dose rows as independent observations.',
            'No confidence interval, significance test, best-dose selection or transfer claim.',
        ],
    }
