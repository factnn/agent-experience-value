"""Describe completed fit inputs and the sealed predictor; never refit or read heldout outcomes."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

REFERENCES = {
    'rl/bfcl_source_stage0_prediction_seal_20261015/receipt.json': 'c19ddc54e26aff0bc7579d0e185cd2bafad075ae928c0e58a6fef5f094cc2c15',
    'rl/bfcl_source_stage0_prediction_seal_20261015/models.json': 'de1eb70bf8752865a74a725f9c38976eeb0b0d0795d6a731ce26cff1be5e9644',
    'rl/bfcl_source_stage0_prediction_seal_20261015/predictions.json': 'fbc7514333a0842b2f0bae4950761d392d846648175b75bba4d4fc5d3feb86ff',
    'rl/bfcl_source_stage0_prediction_seal_20261015/fit_labels.json': '306d243adb9008fa02e1e09524bf7c43d78aad1229961bbc6e1e47843ef84071',
    'rl/bfcl_source_stage0_prediction_seal_20261015/publication.json': 'e63b63c991f00c275d22023ddde6ddadb52b65ca99698b29294b9cba35014de6',
    'rl/bfcl_source_stage0_features_20261012.json': '1414884d546fbfbb56825b6069afe45ead9afa965b10a6763a887e0a19a05f7e',
    'rl/BFCL_SOURCE_GROUP_ANALYSIS_CONTRACT_20261011.json': 'eeee267b26ccfa420faf3325fa02bb068b5542c3cedd7f60b73ebea43f391f3d',
    'rl/bfcl_source_stage0_train_20261012/training_cost_and_exposure.json': '0ecab5893d9de7d14a7efa9d297c3ab0b228a03b2a27acbff7e055e93500800b',
    'rl/bfcl_source_stage0_fit_tail_eight_20261015/execution_chain_review.json': '8593b90ea423c8faa2eeda98c5511fbcd7a33258de69724d88ea5af5991eceff',
}
SIGNALS = ('primary_task_beta', 'current_pooled_beta', 'historical_frontier', 'historical_coverage')


def describe(root, bundle_path=None):
    bundle = json.loads(bundle_path.read_text()) if bundle_path else None
    inputs = {}
    for path, expected in REFERENCES.items():
        raw = bundle['files'][path]['text'].encode('utf-8') if bundle else (root/path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Completed/sealed input changed: '+path)
        inputs[Path(path).name] = json.loads(raw)
    labels = inputs['fit_labels.json']
    fitted = inputs['models.json']
    forecasts = inputs['predictions.json']
    features = inputs['bfcl_source_stage0_features_20261012.json']
    contract = inputs['BFCL_SOURCE_GROUP_ANALYSIS_CONTRACT_20261011.json']
    training = inputs['training_cost_and_exposure.json']
    chain = inputs['execution_chain_review.json']
    groups = contract['source_split']['fit']
    assert labels['phase'] == 'fit' and labels['panel_count'] == chain['panel_count'] == 80
    assert len(groups) == 12 and fitted['fit_groups'] == labels['source_groups'] == groups
    assert features['validation_labels_used'] is False
    assert forecasts['heldout_labels_used'] is False and forecasts['target'] == 'G'
    assert fitted['feature_selection'] is False and fitted['prediction_clipping'] is False
    assert list(fitted['models']) == ['fit_mean', *SIGNALS]
    feature_rows = {row['source_group']: row for row in features['rows']}
    jobs = {row['case']: row for row in training['jobs']}
    panels = {(row['policy_case'], row['evaluation_repeat']): row for row in chain['panels']}
    model_predictions = {}
    for group in contract['source_split']['all_groups']:
        values = {'fit_mean': fitted['models']['fit_mean']['intercept']}
        for signal in SIGNALS:
            model = fitted['models'][signal]
            assert model['lambda'] == 1.0
            scale = model['fit_population_std']
            z = (feature_rows[group][signal]-model['fit_mean'])/scale if scale else 0.0
            values[signal] = model['intercept']+model['slope']*z
        model_predictions[group] = values
    # Reuse original coefficients and normalization, not a new fit or selection.
    assert {g: model_predictions[g] for g in forecasts['heldout_groups']} == forecasts['point_predictions']
    rows = []
    for group in groups:
        row = labels['groups'][group]
        assert len(row['training_repeats']) == 3
        repeats = []
        strata = {s: {'G': [], 'V': []} for s in contract['target_panel']['registry'].values()}
        for repeat in row['training_repeats']:
            r = repeat['training_repeat']
            assert [p['evaluation_repeat'] for p in repeat['panels']] == [0, 1]
            branch = jobs[repeat['policy_case']]
            assert branch['group'] == group and branch['repeat'] == r
            assert math.isclose(repeat['G'], mean(p['G'] for p in repeat['panels']), abs_tol=1e-12)
            assert math.isclose(repeat['V'], mean(p['V'] for p in repeat['panels']), abs_tol=1e-12)
            controls = [j['case'] for j in training['jobs'] if j['role'] == 'uniform_control' and j['repeat'] == r]
            assert len(controls) == 1
            for p in repeat['panels']:
                e = p['evaluation_repeat']
                source = panels[repeat['policy_case'], e]['strata']
                common = panels['common', e]['strata']
                uniform = panels[controls[0], e]['strata']
                for s in strata:
                    strata[s]['G'].append(source[s]-common[s])
                    strata[s]['V'].append(source[s]-uniform[s])
            repeats.append({
                'training_repeat': r, 'policy_case': repeat['policy_case'], 'G': repeat['G'], 'V': repeat['V'],
                'evaluation_panels': repeat['panels'],
                'training_new_tokens': branch['new_tokens'], 'optimizer_steps': branch['updates'],
                'fresh_nonzero_gradient_steps': branch['fresh_steps'], 'changed_parameter_steps': branch['changed_steps'],
            })
        stratum_means = {s: {metric: mean(values) for metric, values in metrics.items()} for s, metrics in strata.items()}
        for metric in ('G', 'V'):
            assert math.isclose(row[metric], mean(rep[metric] for rep in repeats), abs_tol=1e-12)
            assert math.isclose(row[metric], mean(value[metric] for value in stratum_means.values()), abs_tol=1e-12)
        rows.append({
            'source_group': group, 'G': row['G'], 'V': row['V'], 'training_repeats': repeats,
            'G_repeat_range': [min(rep['G'] for rep in repeats), max(rep['G'] for rep in repeats)],
            'V_repeat_range': [min(rep['V'] for rep in repeats), max(rep['V'] for rep in repeats)],
            'features': {signal: feature_rows[group][signal] for signal in SIGNALS},
            'original_model_fit_predictions': model_predictions[group], 'strata': stratum_means,
        })
    uniform = labels['uniform_G_by_training_repeat']
    uniform_mean = mean(row['G'] for row in uniform.values())
    for row in rows:
        assert math.isclose(row['G']-row['V'], uniform_mean, abs_tol=1e-12)
    def signs(metric):
        return dict(Counter('positive' if row[metric] > 1e-12 else 'negative' if row[metric] < -1e-12 else 'zero' for row in rows))
    positive_v = [{'source_group': row['source_group'], 'mean_V': row['V'],
                   'V_by_training_repeat': [rep['V'] for rep in row['training_repeats']]} for row in rows if row['V'] > 1e-12]
    summary = {
        'fit_source_conditions': 12, 'fit_panels': 80, 'source_endpoint_panels': 72,
        'training_repeats_per_group': 3, 'evaluation_repeats_per_training_repeat': 2,
        'baseline_success_by_evaluation_repeat': labels['baseline_success_by_evaluation_repeat'],
        'mean_fit_G': mean(row['G'] for row in rows), 'mean_fit_V': mean(row['V'] for row in rows),
        'uniform_mean_G': uniform_mean, 'uniform_G_by_training_repeat': uniform,
        'G_sign_counts': signs('G'), 'V_sign_counts': signs('V'),
        'groups_G_positive_in_all_three_training_repeats': sum(all(rep['G'] > 1e-12 for rep in row['training_repeats']) for row in rows),
        'groups_V_positive_in_all_three_training_repeats': sum(all(rep['V'] > 1e-12 for rep in row['training_repeats']) for row in rows),
        'positive_mean_V_groups': positive_v,
        'maximum_absolute_paired_eval_G_difference': max(abs(rep['evaluation_panels'][0]['G']-rep['evaluation_panels'][1]['G']) for row in rows for rep in row['training_repeats']),
        'fit_group_G_range': [min(row['G'] for row in rows), max(row['G'] for row in rows)],
        'fit_group_V_range': [min(row['V'] for row in rows), max(row['V'] for row in rows)],
        'stratum_mean_G_V': {s: {m: mean(row['strata'][s][m] for row in rows) for m in ('G', 'V')} for s in rows[0]['strata']},
    }
    return {
        'kind': 'stage0_complete_fit_descriptive_report', 'computed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Complete12 fit-source conditions only. No heldout outcomes, no refit, no confirmatory heldout loss or population CI.',
        'source_files': [{'path': p, 'sha256': h} for p, h in REFERENCES.items()],
        'analysis_code': {'path': 'pipeline/report_bfcl_fit_description.py', 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'summary': summary, 'source_rows': rows, 'original_sealed_models': fitted,
        'original_heldout_point_predictions': forecasts,
        'descriptive_in_fit_MSE_not_validation': {name: mean((row['original_model_fit_predictions'][name]-row['G'])**2 for row in rows) for name in fitted['models']},
        'closed_cost_accounts': {'training_51': training['totals'], 'fit80_new_model_tokens': chain['new_model_tokens'],
            'fit80_cost_by_role': chain['cost_by_role'], 'training_overshoot_distribution': training['overshoot_distribution'],
            'training_branch_extrema': training['branch_extrema'], 'shared_ancestor_once_separate': training['accounting']['shared_ancestor_acquisition_tokens_counted_once'],
            'excluded': ['running heldout costs', 'opportunity probe acquisitions', 'other historical experiments and diagnostics'],
            'training_four_unchanged_parameter_cases': training['zero_changed_cases']},
        'interpretation_limits': [
            'Positive mean fitG is not positiveV; only one fit source has positive meanV and its three repeatV values include a negative one.',
            'At the same stage and fixed repeat weighting, groupV is groupG minus the same averaged UniformG; their rankings are not independent replications.',
            'The four nonbaseline slopes are the original positive in-fit ridge slopes. In-fit error improvement is not new-source validation or a sampling-policy recommendation.',
            'Repeat ranges are descriptive, not confidence intervals. Tasks, seeds, source primitives and the shared ancestor/baseline induce dependence.',
            'Actual token dose and optimizer/fresh-gradient/parameter-change counts differ; nominal budget does not imply equal compute. No gain/token normalization removes this confounding.',
            'Evaluation0 optimizer updates are by design, distinct from completed RL optimizer updates or unchanged-parameter training branches.',
            'Four heldout source conditions, fixed late-stage parent, other mechanisms, scale/family, ID and target/environment extrapolation remain incomplete.',
        ],
        'heldout_outcomes_used': False, 'validation_probe_rewards_used': False,
        'predictor_refitted': False, 'new_scientific_tokens': 0, 'goal_complete': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = describe(args.root.resolve(), args.bundle)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'summary': result['summary'], 'original_models': result['original_sealed_models']['models'],
                      'output': str(args.output), 'heldout_outcomes_used': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
