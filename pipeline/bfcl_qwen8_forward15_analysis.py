"""Fixed-source native8B G/V and unchanged-forecast analysis, without fitting.

Pure CPU arithmetic. Callers must first replay all15 training and32 target
panels and verify the actual prospective seals. This module admits no work.
"""
from copy import deepcopy
from itertools import combinations
import math
from statistics import mean

import numpy as np

from bfcl_transfer_labels import aggregate_labels
from bfcl_stage1_forward15_analysis import stability


def registry(master, jobs, analysis):
    groups = analysis['source_split']['heldout']
    if (master['stage'] != 0 or master['training_repeats'] != 3
            or len(jobs) != 15 or len(groups) != 4 or len(set(groups)) != 4
            or [j['case_id'] for j in jobs] != [e['name'] for e in master['jobs']]):
        raise ValueError('Require the registered native8B15-source/control identities')
    result = {'common': {'role': 'baseline', 'training_repeat': None, 'source_group': None}}
    cells = set()
    for job in jobs:
        group, repeat, case = job['source_group'], job['training_repeat'], job['case_id']
        if (job['stage'] != 0 or type(repeat) is not int or repeat not in range(3)
                or group not in [None, *groups] or (group, repeat) in cells or case in result):
            raise ValueError('Missing, repeated or substituted native8B source/control cell')
        if group is None and case != 'q8_s0_uniform_r' + str(repeat):
            raise ValueError('Original native8B Uniform identity changed')
        cells.add((group, repeat))
        result[case] = {'role': 'control' if group is None else 'heldout',
                        'training_repeat': repeat, 'source_group': group}
    if cells != {(g, r) for g in [None, *groups] for r in range(3)}:
        raise ValueError('Incomplete native8B15 matrix')
    return result


def labels(panels, source_registry, analysis):
    view = deepcopy(analysis)
    view['source_split']['all_groups'] = list(analysis['source_split']['heldout'])
    result = aggregate_labels(panels, source_registry, view, 'all')
    if result['panel_count'] != 32 or len(result['groups']) != 4:
        raise ValueError('Every original native8B32 panel is required')
    result.update(phase='forward_validation', batch_mode='native8b_forward_only_15',
                  within_native8b_fitting_performed=False, original_full104_complete=False)
    return result


def error_scores(points, observed):
    if set(points) != set(observed) or len(points) != 4:
        raise ValueError('Score exactly the four fixed, tested sources')
    names = list(next(iter(points.values())))
    if not names or any(set(points[g]) != set(names) for g in points):
        raise ValueError('A model cannot omit an unfavorable tested source')
    if any(not math.isfinite(x) for g in points for x in [observed[g], *points[g].values()]):
        raise ValueError('Nonfinite measured gain or unchanged point prediction')
    result = {}
    for name in names:
        errors = {g: points[g][name] - observed[g] for g in points}
        bias = mean(errors.values())
        mse = mean(e * e for e in errors.values())
        centered = mean((e - bias) ** 2 for e in errors.values())
        if abs(mse - bias * bias - centered) > 1e-12:
            raise ValueError('Descriptive MSE decomposition failed')
        result[name] = {'MSE': mse, 'RMSE': math.sqrt(mse), 'mean_signed_error': bias,
            'squared_mean_signed_error': bias * bias, 'centered_error_MSE': centered,
            'signed_errors': errors, 'squared_errors': {g: e * e for g, e in errors.items()},
            'intercept_corrected': False, 'prediction_refitted': False}
    baseline = 'original_five/fit_mean'
    if baseline not in result or 'zero_gain_reference' not in result:
        raise ValueError('Both frozen old4B mean and zero-gain reference are mandatory')
    for name in names:
        row = result[name]
        row['MSE_minus_frozen_mean'] = row['MSE'] - result[baseline]['MSE']
        row['MSE_minus_zero_reference'] = row['MSE'] - result['zero_gain_reference']['MSE']
        row['sources_with_lower_squared_error_than_mean'] = [g for g in points
            if row['squared_errors'][g] < result[baseline]['squared_errors'][g]]
    return result


def prediction_views(points, measured):
    full = error_scores(points, measured['G_labels'])
    training = {}
    for repeat in range(3):
        observed = {g: next(r for r in measured['groups'][g]['training_repeats']
                    if r['training_repeat'] == repeat)['G'] for g in points}
        training[str(repeat)] = error_scores(points, observed)
    evaluation = {}
    for repeat in range(2):
        observed = {g: mean(next(p for p in r['panels'] if p['evaluation_repeat'] == repeat)['G']
                    for r in measured['groups'][g]['training_repeats']) for g in points}
        evaluation[str(repeat)] = error_scores(points, observed)
    return {'full_four_source_scores': full, 'by_training_repeat': training,
        'by_evaluation_repeat': evaluation, 'prediction_endpoint': 'G',
        'V_silently_substituted_as_prediction_label': False,
        'scope': 'Matched views share forecasts, tasks and ancestor; not independent replications or a test of a broad source population.'}


def outcome_cube(outcomes, source_registry, evaluation):
    cases = list(source_registry)
    strata = list(dict.fromkeys(evaluation['registry'].values()))
    tasks = {s: [t for t, value in evaluation['registry'].items() if value == s] for s in strata}
    if len(cases) != 16 or len(strata) != 4 or any(len(t) != 8 for t in tasks.values()):
        raise ValueError('Require the original16 policies and four equal eight-task strata')
    if set(outcomes) != {(case, e) for case in cases for e in range(2)}:
        raise ValueError('Raw paired outcomes are incomplete or repeated')
    values = np.empty((16, 2, 4, 8), dtype=np.float64)
    for i, case in enumerate(cases):
        for e in range(2):
            rows = outcomes[case, e]
            by_task = {r['task_id']: r for r in rows}
            if len(rows) != 32 or len(by_task) != 32 or set(by_task) != set(evaluation['registry']):
                raise ValueError('Every target task must be retained exactly once per panel')
            for s, stratum in enumerate(strata):
                for t, task in enumerate(tasks[stratum]):
                    row = by_task[task]
                    if row['stratum'] != stratum or len(row['rewards']) != 1 or row['rewards'][0] not in (0, 1):
                        raise ValueError('Only audited terminal-success outcomes enter G/V')
                    values[i, e, s, t] = row['rewards'][0]
    return cases, strata, tasks, values


def paired_resampling(outcomes, source_registry, evaluation, measured, points, specification):
    if (specification['replicates'] != 10000 or specification['seed'] != 2026101701
            or specification['source_groups_resampled'] is not False
            or specification['frozen_predictions_refitted'] is not False):
        raise ValueError('Use the frozen E369 resampling count, seed and scope')
    cases, strata, tasks, values = outcome_cube(outcomes, source_registry, evaluation)
    groups = measured['source_groups']
    source_indices = np.asarray([[next(i for i, c in enumerate(cases)
        if source_registry[c]['source_group'] == g and source_registry[c]['training_repeat'] == r)
        for r in range(3)] for g in groups])
    uniform_indices = np.asarray([next(i for i, c in enumerate(cases)
        if source_registry[c]['role'] == 'control' and source_registry[c]['training_repeat'] == r)
        for r in range(3)])
    baseline_index = cases.index('common')
    original = values.mean(axis=(2, 3))
    actual_G = original[source_indices].mean(axis=(1, 2)) - original[baseline_index].mean()
    actual_V = original[source_indices].mean(axis=(1, 2)) - original[uniform_indices].mean()
    if (not np.allclose(actual_G, [measured['G_labels'][g] for g in groups], atol=1e-12, rtol=0)
            or not np.allclose(actual_V, [measured['V_labels'][g] for g in groups], atol=1e-12, rtol=0)):
        raise ValueError('Task-level data do not reproduce the original paired macro G/V')
    count = specification['replicates']
    rng = np.random.default_rng(specification['seed'])
    task_draws = np.empty((count, 4, 8), dtype=np.int64)
    train_draws = np.empty((count, 3), dtype=np.int64)
    eval_draws = np.empty((count, 2), dtype=np.int64)
    for b in range(count):
        task_draws[b] = rng.integers(0, 8, size=(4, 8))
        train_draws[b] = rng.integers(0, 3, size=3)
        eval_draws[b] = rng.integers(0, 2, size=2)
    draws = np.take_along_axis(values[None, ...], task_draws[:, None, None, :, :], axis=4).mean(axis=(3, 4))
    policy_scores = np.take_along_axis(draws, eval_draws[:, None, :], axis=2).mean(axis=2)
    source_scores = np.take_along_axis(policy_scores[:, source_indices], train_draws[:, None, :], axis=2).mean(axis=2)
    control_scores = np.take_along_axis(policy_scores[:, uniform_indices], train_draws, axis=1).mean(axis=1)
    G = source_scores - policy_scores[:, baseline_index, None]
    V = source_scores - control_scores[:, None]
    names = list(points[groups[0]])
    predictions = np.asarray([[points[g][name] for name in names] for g in groups])
    mse = ((predictions[None, ...] - G[:, :, None]) ** 2).mean(axis=1)

    def interval(array):
        lo, mid, hi = np.quantile(array, [0.025, 0.5, 0.975])
        return {'q025': float(lo), 'median': float(mid), 'q975': float(hi)}

    pairs = []
    for a, b in combinations(range(4), 2):
        difference = G[:, a] - G[:, b]
        if not np.allclose(difference, V[:, a] - V[:, b], atol=1e-12, rtol=0):
            raise ValueError('Pairing lost: G/V source contrasts must be the same comparison')
        pairs.append({'left': groups[a], 'right': groups[b], 'paired_G_and_V_difference': interval(difference)})
    mean_index, zero_index = names.index('original_five/fit_mean'), names.index('zero_gain_reference')
    return {'replicates': count, 'seed': specification['seed'],
        'draw_order': 'For each replicate:4x8 tasks within ordered strata,3 train indices,2 evaluation indices; NumPy default_rng.',
        'strata_order': strata, 'tasks_within_stratum': tasks, 'source_groups_resampled': False,
        'all_policies_share_task_train_and_evaluation_draws': True, 'frozen_predictions_refitted': False,
        'groups': {g: {'G': interval(G[:, i]), 'V': interval(V[:, i])} for i, g in enumerate(groups)},
        'mean_G': interval(G.mean(axis=1)), 'mean_V': interval(V.mean(axis=1)),
        'paired_source_contrasts': pairs,
        'models': {name: {'MSE': interval(mse[:, i]),
            'MSE_minus_frozen_mean': interval(mse[:, i] - mse[:, mean_index]),
            'MSE_minus_zero_reference': interval(mse[:, i] - mse[:, zero_index])}
            for i, name in enumerate(names)},
        'interval_interpretation': 'Central95% conditional fixed-set sensitivity ranges; not independent-model, ancestor or broad source-population confidence intervals.'}
