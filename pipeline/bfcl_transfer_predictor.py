"""Fixed univariate ridge predictors for the preregistered source-group split.

No dependency on validation probe rewards or heldout transfer labels. A caller
must bind these inputs to audited artifacts and publish predictions before eval.
"""
import math
from statistics import mean

SIGNALS = ('primary_task_beta', 'current_pooled_beta', 'historical_frontier', 'historical_coverage')


def feature_rows(features, contract):
    if features.get('validation_labels_used') is not False:
        raise ValueError('Only declared feature-panel measurements are accepted')
    rows = features['rows']
    expected = set(contract['source_split']['all_groups'])
    if len(rows) != len(expected) or {r['source_group'] for r in rows} != expected:
        raise ValueError('Require exactly one row per registered source group')
    for row in rows:
        if any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) for k in SIGNALS):
            raise ValueError('Missing or nonfinite source signal')
    return {r['source_group']: r for r in rows}


def fit_models(features, fit_labels, contract):
    rows = feature_rows(features, contract)
    groups = contract['source_split']['fit']
    if set(fit_labels) != set(groups):
        raise ValueError('Fit input must contain exactly the registered fit groups; no heldout labels')
    if any(not isinstance(y, (int, float)) or not math.isfinite(y) or not -1 <= y <= 1 for y in fit_labels.values()):
        raise ValueError('G must be a finite success-rate difference in [-1,1]')
    if contract['fitting']['lambda'] != 1.0:
        raise ValueError('Only fixed lambda=1 analysis is implemented')
    if contract['features']['allowed_models'] != ['fit_mean', *SIGNALS]:
        raise ValueError('Analysis model registry changed')
    y = [fit_labels[g] for g in groups]
    intercept = mean(y)
    models = {'fit_mean': {'intercept': intercept}}
    for signal in SIGNALS:
        x = [rows[g][signal] for g in groups]
        location = mean(x)
        scale = math.sqrt(mean((v-location)**2 for v in x))
        z = [(v-location)/scale for v in x] if scale else [0.0]*len(x)
        slope = sum(a*(b-intercept) for a, b in zip(z, y)) / (sum(a*a for a in z) + 1.0)
        models[signal] = {'intercept': intercept, 'slope': slope, 'fit_mean': location,
                          'fit_population_std': scale, 'lambda': 1.0}
    return {'models': models, 'fit_groups': list(groups), 'fit_stage': features['stage'],
            'fit_policy_fingerprint': features['policy_fingerprint'],
            'heldout_groups': contract['source_split']['heldout'],
            'feature_selection': False, 'prediction_clipping': False}


def predict_models(fitted, features, contract):
    rows = feature_rows(features, contract)
    if fitted['fit_groups'] != contract['source_split']['fit'] or fitted['heldout_groups'] != contract['source_split']['heldout']:
        raise ValueError('Bound source-group split differs')
    result = {}
    for group, row in rows.items():
        prediction = {'fit_mean': fitted['models']['fit_mean']['intercept']}
        for signal in SIGNALS:
            model = fitted['models'][signal]
            scale = model['fit_population_std']
            z = (row[signal] - model['fit_mean'])/scale if scale else 0.0
            prediction[signal] = model['intercept'] + model['slope']*z
        result[group] = prediction
    return result


def score_heldout(predictions, heldout_labels, contract):
    groups = contract['source_split']['heldout']
    if set(heldout_labels) != set(groups):
        raise ValueError('Require the entire heldout panel, without fit-source labels')
    if any(not math.isfinite(v) or not -1 <= v <= 1 for v in heldout_labels.values()):
        raise ValueError('Invalid heldout G')
    losses = {g: {m: (predictions[g][m] - heldout_labels[g])**2 for m in ['fit_mean', *SIGNALS]} for g in groups}
    mse = {m: mean(losses[g][m] for g in groups) for m in ['fit_mean', *SIGNALS]}
    return {'heldout_groups': list(groups), 'squared_errors': losses, 'MSE': mse,
            'primary_difference': mse['primary_task_beta'] - mse['fit_mean'],
            'mechanism_secondary_difference': mse['primary_task_beta'] - mse['current_pooled_beta'],
            'scope': 'Finite heldout-source contrast; shared-state/target dependence, no population significance claim.'}
