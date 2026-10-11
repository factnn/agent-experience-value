"""Task-balanced probe statistics; no model calls, fitting, or efficacy labels.

Unbiased moment claims require independent Bernoulli draws at a fixed policy,
including independence across tasks. Keep negative variance estimates visible.
"""
from collections import defaultdict
from math import prod


def task_statistics(rewards):
    if len(rewards) < 4 or any(type(x) not in (int, float) or x not in (0, 1) for x in rewards):
        raise ValueError('At least four binary rewards required')
    n, k = len(rewards), sum(rewards)
    p = k / n
    p2 = k * (k - 1) / (n * (n - 1))
    denominator = prod(n - i for i in range(4))
    mixed = 1 - prod(k - i for i in range(4)) / denominator - prod(n - k - i for i in range(4)) / denominator
    return {'n': n, 'successes': k, 'p_hat': p, 'p_squared_unbiased': p2,
            'within_uncertainty_unbiased': p - p2,
            'four_draw_mixed_probability_unbiased': mixed}


def source_statistics(task_rewards):
    if not task_rewards:
        raise ValueError('Empty source group')
    tasks = {key: task_statistics(value) for key, value in sorted(task_rewards.items())}
    m = len(tasks)
    ps = [x['p_hat'] for x in tasks.values()]
    mean_p = sum(ps) / m
    mean_p2 = sum(x['p_squared_unbiased'] for x in tasks.values()) / m
    # Cross-task products and within-task factorial moments estimate (mean p)^2.
    mean_squared = (m * mean_p2 + 2 * sum(ps[i] * ps[j] for i in range(m) for j in range(i))) / (m * m)
    within = mean_p - mean_p2
    between = mean_p2 - mean_squared
    pooled = mean_p - mean_squared
    return {'task_count': m, 'tasks': tasks, 'task_uniform_success': mean_p,
            'within_uncertainty_unbiased': within,
            'between_task_variance_unbiased': between,
            'pooled_uncertainty_unbiased': pooled,
            'decomposition_residual': pooled - within - between,
            'four_draw_mixed_probability_unbiased': sum(x['four_draw_mixed_probability_unbiased'] for x in tasks.values()) / m,
            'pooled_plugin_frontier': 4 * mean_p * (1 - mean_p),
            'pooled_plugin_mixed_probability': 1 - mean_p**4 - (1 - mean_p)**4,
            'negative_between_estimate_preserved': between < 0,
            'weighting': 'uniform over tasks, regardless of task observation counts'}


def validate_panel(rows):
    if not rows:
        raise ValueError('Empty panel')
    required = {'task_id', 'source_group', 'policy_fingerprint', 'stage', 'draw_id', 'rewards'}
    if any(not required.issubset(row) for row in rows):
        raise ValueError('Missing probe provenance')
    if len({(r['stage'], r['policy_fingerprint']) for r in rows}) != 1:
        raise ValueError('Panel mixes policies or stages')
    ids = [r['draw_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated acquisition draw')
    tasks = defaultdict(list)
    group_for_task = {}
    for row in rows:
        if len(row['rewards']) != 4:
            raise ValueError('Expected a complete four-rollout acquisition group')
        task_statistics(row['rewards'])
        key = row['task_id']
        if key in group_for_task and group_for_task[key] != row['source_group']:
            raise ValueError('Task changes source group')
        group_for_task[key] = row['source_group']
        tasks[key].extend(row['rewards'])
    if len({len(x) for x in tasks.values()}) != 1:
        raise ValueError('Probe panel is not task-balanced')
    return tasks, group_for_task


def compare_independent_panels(feature_rows, validation_rows):
    """Guard independent acquisition identities; no claim seeds alone ensure IID."""
    f, fg = validate_panel(feature_rows)
    v, vg = validate_panel(validation_rows)
    if fg != vg:
        raise ValueError('Feature and validation task/source panels differ')
    if (feature_rows[0]['stage'], feature_rows[0]['policy_fingerprint']) != (validation_rows[0]['stage'], validation_rows[0]['policy_fingerprint']):
        raise ValueError('Feature and validation policies differ')
    if {r['draw_id'] for r in feature_rows} & {r['draw_id'] for r in validation_rows}:
        raise ValueError('Feature and validation acquisitions overlap')
    results = {}
    for group in sorted(set(fg.values())):
        selected = [task for task in f if fg[task] == group]
        observed = [int(0 < sum(r['rewards']) < 4) for r in validation_rows if r['source_group'] == group]
        results[group] = {'features': source_statistics({t: f[t] for t in selected}),
                          'validation': source_statistics({t: v[t] for t in selected}),
                          'validation_observed_group_mixed_rate': sum(observed) / len(observed)}
    return results
