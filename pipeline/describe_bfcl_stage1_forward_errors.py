"""Describe the errors of unchanged forecasts after the complete15/32 report.

This is an algebraic supplement requested in gpt21, not a recalibration,
refit, new primary score, independent replication, or causal decomposition.
It reads existing audited reports and exclusively creates a new output file.
"""
import argparse
import datetime
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values)


def error_components(predictions, observed, groups):
    if len(groups) != 4 or len(set(groups)) != 4 or set(predictions) != set(groups) or set(observed) != set(groups):
        raise ValueError('Require the same complete four sources, without dropping or imputing')
    if any(not math.isfinite(x) for x in [*predictions.values(), *observed.values()]):
        raise ValueError('Nonfinite prediction or observed gain')
    errors = {g: predictions[g]-observed[g] for g in groups}
    bias = mean(errors.values())
    squared_bias = bias*bias
    centered = mean((e-bias)**2 for e in errors.values())
    mse = mean(e*e for e in errors.values())
    pairs = [{'left': a, 'right': b,
              'predicted_G_difference': predictions[a]-predictions[b],
              'observed_G_difference': observed[a]-observed[b],
              'error_difference': errors[a]-errors[b]}
             for a, b in combinations(groups, 2)]
    pair_identity = math.fsum(p['error_difference']**2 for p in pairs)/(len(groups)**2)
    if not math.isclose(mse, squared_bias+centered, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('MSE algebra failed')
    if not math.isclose(centered, pair_identity, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('Centered/pairwise algebra failed')
    return {'MSE_original': mse, 'mean_signed_error': bias,
            'overall_squared_bias': squared_bias, 'centered_between_source_error': centered,
            'source_rows': [{'source_group': g, 'frozen_prediction_G': predictions[g],
                             'observed_G': observed[g], 'signed_error': errors[g],
                             'centered_error': errors[g]-bias} for g in groups],
            'matched_source_pairs': pairs,
            'centered_error_equals_sum_six_squared_pair_errors_divided_by_16': pair_identity,
            'units': 'G and signed errors are proportions; squared components are proportion squared.'}


def original_models(predictions, observed, groups, scores):
    if set(predictions) != set(groups):
        raise ValueError('Frozen predictions differ from original heldout source set')
    names = list(predictions[groups[0]])
    if len(names) != 5 or 'fit_mean' not in names or any(set(predictions[g]) != set(names) for g in groups):
        raise ValueError('Original five-model registry changed')
    result = {}
    for name in names:
        row = error_components({g: predictions[g][name] for g in groups}, observed, groups)
        if not math.isclose(row['MSE_original'], scores['MSE'][name], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('Supplement differs from original primary scoring: '+name)
        result[name] = row
    baseline = result['fit_mean']
    for row in result.values():
        row['differences_from_frozen_mean_baseline'] = {
            key: row[key]-baseline[key] for key in
            ('MSE_original', 'overall_squared_bias', 'centered_between_source_error')}
    return result


def build(root, report_path, expected_sha):
    if sha(report_path) != expected_sha:
        raise ValueError('Full scientific report bytes changed')
    late = json.loads(report_path.read_text())
    if (late['status'] != 'passed' or late['stage'] != 1 or late['batch_mode'] != 'forward_only_15'
            or late['raw_replay'] != {'training_jobs': 15, 'target_panels': 32, 'saved_audits_equal_fresh': True}
            or late['labels']['panel_count'] != 32 or late['within_late_fitting_performed'] is not False
            or late['reserved21_accessed'] is not False):
        raise ValueError('Require the complete original15/32 audited scientific report')
    groups = late['labels']['source_groups']
    actual = late['labels']['G_labels']
    early_ref = late['evidence']['early_report'];early_path = root/early_ref['path']
    if sha(early_path) != early_ref['sha256']:
        raise ValueError('Original stage0 report changed')
    early = json.loads(early_path.read_text())
    if early['status'] != 'passed' or early['stage'] != 0 or early['labels']['panel_count'] != 104:
        raise ValueError('Require the original complete51/104 report')
    early_y = {g: early['labels']['G_labels'][g] for g in groups}
    early_results = original_models(early['sealed_predictions']['point_predictions'], early_y, groups,
                                    early['prediction_results']['primary'])
    late_results = original_models(late['sealed_predictions']['point_predictions'], actual, groups,
                                   late['prediction_results']['original_five'])
    if set(early_results) != set(late_results):
        raise ValueError('Early and late original predictor identities differ')
    h1 = error_components(late['sealed_separate_H1_predictions']['point_predictions'], actual, groups)
    if not math.isclose(h1['MSE_original'], late['prediction_results']['separate_H1']['MSE'], rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError('Separate H1 original scoring changed')
    h1['differences_from_frozen_mean_baseline'] = {
        k: h1[k]-late_results['fit_mean'][k] for k in
        ('MSE_original', 'overall_squared_bias', 'centered_between_source_error')}
    change = {g: actual[g]-early_y[g] for g in groups};average_change = mean(change.values())
    return {'kind': 'descriptive_gpt21_error_decomposition', 'computed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_groups': groups, 'early_original_five': early_results, 'late_original_five': late_results,
            'late_separate_H1': h1,
            'same_four_source_observed_G_changes': {
                'mean_late_G_minus_early_G': average_change,
                'rows': [{'source_group': g, 'early_G': early_y[g], 'late_G': actual[g],
                          'G_change': change[g], 'G_change_minus_four_source_mean_change': change[g]-average_change}
                         for g in groups]},
            'original_main_scores_unchanged': True, 'recalibrated_predictions_created': False,
            'new_scientific_tokens': 0, 'new_gpu_jobs': 0,
            'sources': [{'path': str(report_path.relative_to(root)), 'sha256': expected_sha}, early_ref],
            'analysis_code_sha256': sha(Path(__file__)),
            'limitations': [
                'Four source means use original equal source weighting. Shared tasks, ancestors, controls and repeats remain dependent.',
                'The two nonnegative components are an exact error identity. They do not identify causes, eliminate label noise, or establish a stable ordering.',
                'Six matched pair rows encode the same centered error component and are not six independent replications.',
                'No intercept, slope, scaling, feature, forecast, seed, source or original main score was changed.',
                'Stage differences in G also reflect uncertainty and the specified full learner state; they do not establish that the later parent is stronger.',
                'H1 is separate from the original five. No retrospective early H1 forecast is presented as prospectively sealed.',
                'This supplement does not authorize late refitting, matrix expansion, reserved21 access or another experiment.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/share/project/zpy/RL/agent_experience_project_20260928'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();root = args.root.resolve()
    source = (root/args.report).resolve();output = (root/args.output).resolve()
    if not source.is_relative_to(root/'rl') or not output.is_relative_to(root/'rl') or output.exists():
        raise ValueError('Read a project report and exclusively create a separate new rl file')
    result = build(root, source, args.report_sha256)
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False);handle.write('\n')
    print(json.dumps({'output': str(output.relative_to(root)), 'sha256': sha(output), 'new_scientific_tokens': 0}))


if __name__ == '__main__': main()
