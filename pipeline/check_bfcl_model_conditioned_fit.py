"""Analytic CPU checks for signed frozen normalization and label separation."""
import copy
import json
import math

from bfcl_model_conditioned_contract import SIGNALS
from seal_bfcl_model_conditioned_predictions import fit_and_predict


def main():
    groups = ['cpu_fixture_%02d' % i for i in range(16)]
    fit, held = groups[:12], groups[12:]
    analysis = {'source_split': {'all_groups': groups, 'fit': fit, 'heldout': held}}
    profiles = {'qwen4b': {'policy_fingerprint': 'synthetic4b'},
                'qwen8b': {'policy_fingerprint': 'synthetic8b', 'base_revision': 'synthetic'}}
    design = {'profiles': profiles}
    data = {profile: {} for profile in profiles}
    for i, group in enumerate(groups):
        data['qwen4b'][group] = {SIGNALS[0]: float(i), SIGNALS[1]: -2.}
        data['qwen8b'][group] = {SIGNALS[0]: float(i + 3), SIGNALS[1]: -3. - .1 * i}
    features = {'feature_phase_only': True, 'target_labels_used': False,
                'reward_or_grade_filtering': False, 'model_updates': 0, 'profiles': profiles, 'groups': data}
    sigma = math.sqrt(143 / 12)
    labels = {group: .03 - .01 * ((i - 5.5) / sigma) for i, group in enumerate(groups)}
    report = {'status': 'passed', 'labels': {'panel_count': 104, 'G_labels': labels}}
    registry, points = fit_and_predict(design, features, analysis, report)
    model = registry['models'][SIGNALS[0]]
    assert abs(model['fit_mean'] - 5.5) < 1e-14
    assert abs(model['fit_population_std'] - sigma) < 1e-14
    assert abs(model['intercept'] - .03) < 1e-14
    assert abs(model['slope'] - (-.01 * 12 / 13)) < 1e-14
    assert registry['models'][SIGNALS[1]]['slope'] == 0.
    assert registry['models'][SIGNALS[1]]['fit_population_std'] == 0.
    assert all(abs(values[SIGNALS[1]] - .03) < 1e-14 for values in points['points'].values())
    changed_report = copy.deepcopy(report)
    for group in held:
        changed_report['labels']['G_labels'][group] = 999.
    assert fit_and_predict(design, features, analysis, changed_report) == (registry, points)
    shifted = copy.deepcopy(features)
    shifted['groups']['qwen8b'][groups[0]][SIGNALS[0]] = 1e6
    second_registry, second_points = fit_and_predict(design, shifted, analysis, report)
    assert second_registry == registry
    assert second_points['points'][groups[0]][SIGNALS[0]] < -1.
    assert len(points['standardized_source_rows']) == 32
    print(json.dumps({'status': 'passed', 'checks': ['signed ridge equals analytic slope',
        'population normalization frozen on4Bfit12', 'historical heldout labels do not enter fitting',
        '8B feature changes do not refit coefficients', 'zero4Bscale remains zero despite8Bvariation',
        'unbounded negative forecasts retained', 'all16x2 actual-calculation rows'],
        'fixtures': 'Analytic synthetic source features/labels only, not native8B point forecasts',
        'native_point_predictions_created': False, 'gpu_jobs': 0, 'new_model_tokens': 0}))


if __name__ == '__main__':
    main()
