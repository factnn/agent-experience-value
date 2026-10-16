"""Paired forward15/32 labels and scoring of unchanged, already-sealed forecasts.

No fitting, source selection, normalization changes, or target acquisition.
Callers must supply complete freshly audited panels and verify publication.
"""
from copy import deepcopy
import math
from statistics import mean
from itertools import combinations

from bfcl_transfer_labels import aggregate_labels
from bfcl_transfer_predictor import SIGNALS, score_heldout


def registry(master, training_jobs, analysis):
    groups = analysis['source_split']['heldout']
    if (master['stage'] != 1 or master['batch_mode'] != 'forward_only_15'
            or master['training_jobs'] != 15 or master['training_repeats'] != 3
            or len(groups) != 4 or len(set(groups)) != 4 or len(training_jobs) != 15):
        raise ValueError('Require the approved15 branches and original four heldout sources')
    if [j['case_id'] for j in training_jobs] != [e['name'] for e in master['jobs']]:
        raise ValueError('Wrong original training identities/order')
    result = {'common': {'role': 'baseline', 'training_repeat': None, 'source_group': None}}
    cells = set()
    for job in training_jobs:
        group, repeat = job['source_group'], job['training_repeat']
        if job['stage'] != 1 or type(repeat) is not int or repeat not in range(3):
            raise ValueError('Wrong stage or training repeat')
        if group is None:
            if job['case_id'] != 's1_uniform_r'+str(repeat):
                raise ValueError('Control identity changed')
            role = 'control'
        elif group in groups:
            role = 'heldout'
        else:
            raise ValueError('Deferred fit-source branch entered forward15 analysis')
        cell = group, repeat
        if cell in cells or job['case_id'] in result:
            raise ValueError('Duplicate training source/control cell')
        cells.add(cell)
        result[job['case_id']] = {'role': role, 'training_repeat': repeat, 'source_group': group}
    if cells != {(g, r) for g in [None, *groups] for r in range(3)}:
        raise ValueError('Incomplete approved15 source/control matrix')
    return result


def labels(panels, source_registry, analysis):
    # Reuse the original exact pairedG/V arithmetic. Only the in-memory group
    # view changes; the original16-source analysis contract stays untouched.
    view = deepcopy(analysis)
    view['source_split']['all_groups'] = list(analysis['source_split']['heldout'])
    result = aggregate_labels(panels, source_registry, view, 'all')
    if result['panel_count'] != 32 or len(result['groups']) != 4:
        raise ValueError('Require every approved32 panel, without dropping/imputing')
    result.update(stage=1, phase='forward_validation', batch_mode='forward_only_15',
                  within_late_fitting_performed=False, original_full104_complete=False)
    return result


def score(predictions, h1_predictions, measured, analysis):
    groups = analysis['source_split']['heldout']
    if (measured['stage'] != 1 or measured['phase'] != 'forward_validation'
            or measured['source_groups'] != groups or set(predictions) != set(groups)
            or set(h1_predictions) != set(groups)):
        raise ValueError('Complete four-source forward forecasts and labels required')
    if any(set(predictions[g]) != {'fit_mean', *SIGNALS} for g in groups):
        raise ValueError('Original five-model registry changed')
    if any(not math.isfinite(v) for g in groups for v in [*predictions[g].values(), h1_predictions[g]]):
        raise ValueError('Nonfinite frozen point prediction')
    original = score_heldout(predictions, measured['G_labels'], analysis)
    h1_losses = {g: (h1_predictions[g]-measured['G_labels'][g])**2 for g in groups}
    h1_mse = mean(h1_losses.values())
    by_training_repeat = {}
    for repeat in range(3):
        y = {g: next(r for r in measured['groups'][g]['training_repeats']
                     if r['training_repeat'] == repeat)['G'] for g in groups}
        current = score_heldout(predictions, y, analysis)
        current['separate_H1_MSE'] = mean((h1_predictions[g]-y[g])**2 for g in groups)
        by_training_repeat[str(repeat)] = current
    by_evaluation_repeat = {}
    for seed in range(2):
        y = {g: mean(next(p for p in r['panels'] if p['evaluation_repeat'] == seed)['G']
                     for r in measured['groups'][g]['training_repeats']) for g in groups}
        by_evaluation_repeat[str(seed)] = score_heldout(predictions, y, analysis)
    rows = [{'source_group': g, 'G': measured['G_labels'][g], 'V': measured['V_labels'][g],
             'frozen_predictions_G': predictions[g], 'separate_H1_prediction_G': h1_predictions[g],
             'original_five_squared_errors': original['squared_errors'][g], 'separate_H1_squared_error': h1_losses[g]}
            for g in groups]
    return {'original_five': original,
            'separate_H1': {'MSE': h1_mse, 'squared_errors': h1_losses,
                'primary_H1_minus_early_mean': h1_mse-original['MSE']['fit_mean'],
                'secondary_H1_minus_early_task': h1_mse-original['MSE']['primary_task_beta']},
            'by_training_repeat': by_training_repeat, 'by_evaluation_repeat': by_evaluation_repeat,
            'source_rows': rows, 'model_refitted': False,
            'scope': 'Only four original sources, fixed early forecasts on late learner. Repeat views share predictions/targets/ancestor and are sensitivity analyses, not independent replications. MSE reduction is not evidence that higher signal should be sampled.'}


def stability(measured):
    groups = measured['source_groups']
    result = {}
    for group in groups:
        repeats = measured['groups'][group]['training_repeats']
        result[group] = {
            'G_by_training_repeat': [r['G'] for r in repeats],
            'V_by_training_repeat': [r['V'] for r in repeats],
            'G_positive_all_three': all(r['G'] > 0 for r in repeats),
            'V_positive_all_three': all(r['V'] > 0 for r in repeats),
            'G_training_repeat_range': max(r['G'] for r in repeats)-min(r['G'] for r in repeats),
            'paired_evaluation_seed_G_difference_by_training_repeat': [r['panels'][1]['G']-r['panels'][0]['G'] for r in repeats]}
    pairs = []
    for a, b in combinations(groups, 2):
        left = {r['training_repeat']: r for r in measured['groups'][a]['training_repeats']}
        right = {r['training_repeat']: r for r in measured['groups'][b]['training_repeats']}
        differences = [left[r]['G']-right[r]['G'] for r in range(3)]
        # Corresponding Uniform controls and the common start cancel in both
        # contrasts: this is ONE comparison, never two independent G/V tests.
        assert all(abs(d-(left[r]['V']-right[r]['V'])) < 1e-12 for r, d in enumerate(differences))
        pairs.append({'left': a, 'right': b, 'paired_G_and_V_difference_by_training_repeat': differences,
                      'strict_same_direction_all_three': all(d > 0 for d in differences) or all(d < 0 for d in differences)})
    return {'groups': result, 'matched_source_pairs': pairs,
            'scope': 'Descriptive matched sensitivities over the same32 panels; no independent-row CI, p-value, new decision threshold, or general16-source late ranking.'}
