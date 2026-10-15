"""Bounded post-hoc diagnostics of frozen stage0 evidence; no fitting or sampling."""
import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from bfcl_probe_forecast import beta_mixed, forecast, score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def near(a, b):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'Diagnostic identity failed: {a} != {b}')


def covariance(x, y):
    return mean((a-mean(x))*(b-mean(y)) for a, b in zip(x, y))


def pearson(x, y):
    scale = math.sqrt(covariance(x, x)*covariance(y, y))
    return covariance(x, y)/scale if scale else None


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [None]*len(values)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start+1+end)/2
        start = end
    return result


def decomposition(predicted, observed):
    p, y = mean(predicted), mean(observed)
    bias_squared = (p-y)**2
    vp, vy, cov = covariance(predicted, predicted), covariance(observed, observed), covariance(predicted, observed)
    mse = mean((a-b)**2 for a, b in zip(predicted, observed))
    near(mse, bias_squared+vp+vy-2*cov)
    return {'mean_prediction': p, 'mean_observed': y, 'mean_bias': p-y,
            'squared_mean_bias': bias_squared, 'prediction_variance': vp,
            'observation_variance': vy, 'prediction_observation_covariance': cov,
            'centered_error': vp+vy-2*cov, 'MSE': mse,
            'Pearson': pearson(predicted, observed),
            'Spearman_with_average_tie_ranks': pearson(ranks(predicted), ranks(observed))}


def describe(root):
    root = root.resolve()
    sources = {}

    def pinned(name, expected=None):
        value = sha(root/name)
        if expected is not None and value != expected:
            raise ValueError('Changed frozen diagnostic input: '+name)
        sources[name] = value
        return read(root/name)

    report = pinned('rl/bfcl_source_stage0_transfer_prediction_20261015/audit_status.json',
                    'c26424eee244968470b478a32ad6bf4b96d5563197972bc04331d6bc753bb992')
    features = pinned('rl/bfcl_source_stage0_features_20261012.json',
                      '1414884d546fbfbb56825b6069afe45ead9afa965b10a6763a887e0a19a05f7e')
    master_ref = report['evidence']['execution_protocol']
    master = pinned(master_ref['path'], master_ref['sha256'])
    analysis_ref = master['analysis_contract']
    analysis = pinned(analysis_ref['path'], analysis_ref['sha256'])
    protocol_ref = features['source_protocol']
    protocol = pinned(protocol_ref['path'], protocol_ref['sha256'])
    probe_batch = pinned(protocol['output_root']+'/audit_status.json')
    if probe_batch['status'] != 'passed' or probe_batch['complete_cases'] != 16:
        raise ValueError('Require original complete local calibration evidence')
    feature_map = {row['source_group']: row for row in features['rows']}
    group_order = analysis['source_split']['all_groups']
    if set(feature_map) != set(group_order) or set(report['labels']['groups']) != set(group_order):
        raise ValueError('Missing original source group')

    repeat_values = [[report['labels']['groups'][g]['training_repeats'][rr]['G']
                      for g in group_order] for rr in range(3)]
    stability = []
    for a, b in itertools.combinations(range(3), 2):
        stability.append({'training_repeats': [a, b],
                          'Spearman_with_average_tie_ranks': pearson(ranks(repeat_values[a]), ranks(repeat_values[b])),
                          'Pearson_G': pearson(repeat_values[a], repeat_values[b])})

    local_rows = []
    task_rows = []
    for case in protocol['cases']:
        group = case['source_group']
        directory = root/protocol['output_root']/case['name']
        paths = {n: str((directory/n).relative_to(root)) for n in ['feature.jsonl', 'validation.jsonl', 'predictions.json', 'audit.json']}
        raw = {}
        for phase in ('feature', 'validation'):
            name = paths[phase+'.jsonl']
            sources[name] = sha(root/name)
            raw[phase] = [json.loads(line) for line in (root/name).read_text().splitlines()]
            if [r['task_id'] for r in raw[phase]] != case['task_ids'] or [r['seed'] for r in raw[phase]] != case['seeds'][phase]:
                raise ValueError('Changed ordered original local panel')
            if any(r['stage'] != 0 or r['phase'] != phase or r['source_group'] != group or
                   r['policy_fingerprint'] != features['policy_fingerprint'] for r in raw[phase]):
                raise ValueError('Wrong stage/policy/source in local panel')
        if sources[paths['feature.jsonl']] != feature_map[group]['feature_sha256']:
            raise ValueError('Feature bytes differ from frozen transfer feature inventory')
        predicted = pinned(paths['predictions.json'], feature_map[group]['prediction_sha256'])
        if forecast(raw['feature'], case['historical_signal']) != predicted:
            raise ValueError('Original feature-only prediction replay differs')
        measured = score(predicted, raw['validation'])
        saved = pinned(paths['audit.json'])
        batch_row = next(row for row in probe_batch['rows'] if row['case'] == case['name'])
        if saved['status'] != 'passed' or saved['rows'][0]['scores'] != measured or batch_row['scores'] != measured:
            raise ValueError('Independent validation score differs from original completed audit')
        observed = [measured['observed_mixed'][t] for t in case['task_ids']]
        models = {}
        for model in ('task_beta', 'current_pooled_beta'):
            p = [predicted['task_predictions'][t][model] for t in case['task_ids']]
            models[model] = decomposition(p, observed)
            near(models[model]['MSE'], measured['Brier'][model])
        for index, t in enumerate(case['task_ids']):
            task_rows.append({'source_group': group, 'task_id': t,
                              'feature_successes_of4': sum(raw['feature'][index]['rewards']),
                              'validation_mixed_of4': observed[index],
                              'task_beta': predicted['task_predictions'][t]['task_beta'],
                              'current_pooled_beta': predicted['task_predictions'][t]['current_pooled_beta']})
        local_rows.append({'source_group': group, 'task_count': len(observed), 'models': models})

    macro = {}
    for model in ('task_beta', 'current_pooled_beta'):
        p = [r['models'][model]['mean_prediction'] for r in local_rows]
        y = [r['models'][model]['mean_observed'] for r in local_rows]
        between = decomposition(p, y)
        within = mean(r['models'][model]['centered_error'] for r in local_rows)
        brier = mean(r['models'][model]['MSE'] for r in local_rows)
        near(brier, between['MSE']+within)
        macro[model] = {'macro_task_Brier': brier, 'between_source_means': between,
                        'mean_within_source_centered_error': within}
    task, pooled = macro['task_beta'], macro['current_pooled_beta']
    delta = {'macro_task_Brier': task['macro_task_Brier']-pooled['macro_task_Brier'],
             'global_mean_bias_squared_component': task['between_source_means']['squared_mean_bias']-pooled['between_source_means']['squared_mean_bias'],
             'centered_between_source_component': task['between_source_means']['centered_error']-pooled['between_source_means']['centered_error'],
             'within_source_component': task['mean_within_source_centered_error']-pooled['mean_within_source_centered_error']}
    near(delta['macro_task_Brier'], sum(v for k, v in delta.items() if k != 'macro_task_Brier'))

    coverage = []
    late_ref = {'path': 'rl/BFCL_STAGE1_PROBE_SSH_EIGHT_EXECUTION_20261015.json',
                'sha256': '70a5a135ac1aa6b29c25cfe25296323b2398abe621043dbd354ec3852fdfb206'}
    late = pinned(late_ref['path'], late_ref['sha256'])
    late_map = {case['source_group']: case['historical_signal']['coverage_deficit'] for case in late['cases']}
    for group in group_order:
        contains = bool(set(group.split('+')) & {'TravelAPI', 'VehicleControlAPI'})
        proxy = .5+.5*contains
        coverage.append({'source_group': group, 'contains_TravelAPI_or_VehicleControlAPI': contains,
                         'stage0_coverage': feature_map[group]['historical_coverage'],
                         'identity_formula': proxy,
                         'stage0_exact_identity': feature_map[group]['historical_coverage'] == proxy,
                         'registered_stage1_historical_coverage': late_map[group],
                         'stage1_same_formula_still_holds': late_map[group] == proxy})
    fit_groups = analysis['source_split']['fit']
    fit_x = [feature_map[g]['primary_task_beta'] for g in fit_groups]
    fit_y = [report['labels']['groups'][g]['G'] for g in fit_groups]
    x_std = math.sqrt(covariance(fit_x, fit_x))
    z = [(x-mean(fit_x))/x_std for x in fit_x]
    near(sum(v*v for v in z), 12)
    ridge_slope = sum(a*(b-mean(fit_y)) for a, b in zip(z, fit_y))/13
    near(ridge_slope, report['sealed_models']['models']['primary_task_beta']['slope'])
    sources['pipeline/bfcl_probe_forecast.py'] = sha(root/'pipeline/bfcl_probe_forecast.py')
    sources['pipeline/bfcl_task_probe_statistics.py'] = sha(root/'pipeline/bfcl_task_probe_statistics.py')
    return {'kind': 'post_completion_stage0_proxy_and_value_diagnostics',
            'computed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'analysis_code_sha256': sha(Path(__file__)), 'source_sha256': sources,
            'source_order': group_order, 'source_G_by_training_repeat': repeat_values,
            'source_ranks_by_training_repeat': [ranks(x) for x in repeat_values],
            'repeat_rank_stability': stability,
            'task_beta_fit_only_diagnostic': {'Pearson_signal_vs_G': pearson(fit_x, fit_y),
                                             'standardized_ridge_slope': ridge_slope,
                                             'lambda': 1, 'OLS_slope_multiplier': 12/13,
                                             'alternative_model_fitted': False},
            'Beta_1_1_prediction_by_feature_successes_of4': {str(k): beta_mixed(k, 4) for k in range(5)},
            'local_proxy_macro': macro, 'task_minus_pooled_Brier_decomposition': delta,
            'local_source_rows': local_rows, 'local_task_rows': task_rows,
            'coverage_tool_identity': {'all16_stage0_exact_formula': all(r['stage0_exact_identity'] for r in coverage),
                                       'stage1_formula_mismatch_count': sum(not r['stage1_same_formula_still_holds'] for r in coverage),
                                       'rows': coverage,
                                       'late_information_used': 'Only historical signals in the already frozen stage1 protocol; no stage1 feature, validation or target outcomes read.'},
            'checks': {'all16_and87_local_tasks': len(local_rows) == 16 and len(task_rows) == 87,
                       'original_feature_predictions_and_independent_validation_scores_replayed': True,
                       'Brier_decomposition_exact_within_1e_12': True,
                       'frozen_task_beta_coefficient_reconstructed_without_refit': True},
            'interpretation_limits': [
                'Post-hoc descriptions after stage0 labels; no new primary endpoint, predictor, feature selection, clipping, seed or retry.',
                'Training repeats are random repeats, not learner stages. All16 ranks share32target tasks, common ancestry and controls; correlations are not independent population tests.',
                'Local Brier identity is an algebraic bias/dispersion/covariance decomposition, not an estimated reliability/resolution decomposition. One independent four-rollout validation group/task is a noisy observation, not latent learnability ground truth.',
                'Source-level correlations use all16 empirical local validation mixed rates with equal source weights. They are not transfer-prediction validation or a fitted calibration transform.',
                'Stage0 exact Coverage/tool identity prevents identifying their mechanisms. Changed late historical values can break that exact identity but one fixed descendant state does not guarantee causal identification.',
                'No population confidence interval or p-value is claimed. Existing matched source/task contrasts remain the paired uncertainty evidence; measurement limits versus a true weak relation remain unresolved.',
                'G/V and original model scoring stay frozen. No new ID evidence or successful-versus-failed trajectory intervention; short-run results do not imply long-horizon outcomes.'
            ],
            'validation_rewards_used_only_for_posthoc_local_diagnosis': True,
            'validation_rewards_used_to_modify_frozen_transfer_features': False,
            'new_scientific_tokens': 0, 'new_gpu_jobs': 0, 'goal_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = describe(args.root)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ['repeat_rank_stability', 'task_beta_fit_only_diagnostic',
                                           'local_proxy_macro', 'task_minus_pooled_Brier_decomposition', 'checks']}, indent=2))
