import copy
import unittest
from bfcl_transfer_labels import aggregate_labels
from report_bfcl_transfer_prediction import (prediction_results, cost_accounting, markdown,
                                           coefficient_rows, source_prediction_rows)
from bfcl_transfer_predictor import SIGNALS, fit_models, predict_models
import test_bfcl_transfer_labels as label_fixtures


class PredictionReportTests(unittest.TestCase):
    def fixture(self):
        master, analysis, features, jobs, registry, panels = label_fixtures.TransferLabelsTests().fixture()
        for case, row in registry.items():
            if row['role'] != 'heldout':
                continue
            for e, base in enumerate([.25, .5]):
                score = base + (row['training_repeat']-1)*.125
                panels.append({'policy_case': case, 'evaluation_repeat': e, 'role': row['role'],
                    'status': 'passed', 'macro_success': score,
                    'stratum_success': {s: score for s in set(analysis['target_panel']['registry'].values())}})
        for panel in panels:
            panel['cost'] = {'new_model_tokens': 10}
        labels = aggregate_labels(panels, registry, analysis, 'all')
        predictions = {g: {m: .125 if m == 'fit_mean' else 0.0 for m in analysis['features']['allowed_models']}
                       for g in analysis['source_split']['heldout']}
        training_rows = [{'case_id': j['case_id'], 'role': j['role'], 'training_repeat': j['training_repeat'],
                          'new_sampled_tokens': 65536+j['training_repeat']*512, 'nominal_token_budget': 65536}
                         for j in jobs]
        return master, analysis, labels, predictions, training_rows, panels

    def test_same_prediction_across_repeats_and_gain_separation(self):
        _, analysis, labels, predictions, _, _ = self.fixture()
        original = copy.deepcopy(predictions)
        result = prediction_results(labels, predictions, analysis)
        self.assertEqual(result['primary']['MSE']['primary_task_beta'], 0)
        self.assertEqual(result['primary']['primary_difference'], -.015625)
        self.assertEqual([r['primary_difference'] for r in result['same_sealed_predictions_by_training_repeat']],
                         [-.046875, -.015625, .015625])
        self.assertEqual(result['primary_contrast_negative_training_repeats'], 2)
        for row in result['gains_separate_from_prediction_accuracy'].values():
            self.assertFalse(row['positive_G_mean'])
            self.assertTrue(row['positive_V_mean'])
            self.assertEqual(row['positive_G_training_repeats'], 1)
        self.assertEqual(predictions, original)
        # Negative primary contrast and positive V do not imply a positive average G.

    def test_complete_matrix_and_models_required(self):
        _, analysis, labels, predictions, _, _ = self.fixture()
        bad = copy.deepcopy(predictions)
        bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):
            prediction_results(labels, bad, analysis)
        bad = copy.deepcopy(predictions)
        bad[next(iter(bad))]['posthoc_winner'] = 0
        with self.assertRaises(ValueError):
            prediction_results(labels, bad, analysis)
        bad = copy.deepcopy(predictions)
        bad[next(iter(bad))]['primary_task_beta'] = float('nan')
        with self.assertRaises(ValueError):
            prediction_results(labels, bad, analysis)
        with self.assertRaises(ValueError):
            prediction_results(labels | {'panel_count': 103}, predictions, analysis)
        bad = copy.deepcopy(labels)
        bad['groups'][analysis['source_split']['heldout'][0]]['training_repeats'].pop()
        with self.assertRaises(ValueError):
            prediction_results(bad, predictions, analysis)

    def test_costs_once_and_complete_readable_report(self):
        master, analysis, labels, predictions, training, panels = self.fixture()
        costs = cost_accounting(master, training, panels)
        self.assertEqual(costs['training_new_tokens'], 17*(65536+66048+66560))
        self.assertEqual(costs['evaluation_new_tokens'], 1040)
        self.assertEqual(costs['stage0_total_with_shared_acquisition_once'],
                         costs['training_new_tokens']+1040+90122+2044472)
        self.assertTrue(all(row['token_ratio_to_same_repeat_uniform'] == 1 for row in costs['training_rows']))
        for rows in [training[:-1], training+[training[0]]]:
            with self.assertRaises(ValueError):
                cost_accounting(master, rows, panels)
        for rows in [panels[:-1], panels+[panels[0]], panels[:-1]+[panels[0]]]:
            with self.assertRaises(ValueError):
                cost_accounting(master, training, rows)
        features = label_fixtures.TransferLabelsTests().fixture()[2]
        fitted = fit_models(features, {g: labels['groups'][g]['G'] for g in analysis['source_split']['fit']}, analysis)
        all_predictions = predict_models(fitted, features, analysis)
        predictions = {g: all_predictions[g] for g in analysis['source_split']['heldout']}
        report = {'prediction_results': prediction_results(labels, predictions, analysis), 'labels': labels,
                  'sealed_predictions': {'heldout_groups': analysis['source_split']['heldout']}, 'costs': costs,
                  'sealed_models': fitted, 'coefficient_directions': coefficient_rows(fitted),
                  'source_predictions': source_prediction_rows(labels, fitted, features, analysis, predictions)}
        text = markdown(report)
        for group in analysis['source_split']['all_groups']:
            self.assertIn('| '+group+' |', text)
        for model in analysis['features']['allowed_models']:
            self.assertIn('| '+model+' |', text)
        self.assertIn('不是总体显著性检验', text)
        self.assertIn('无新ID对照', text)

    def test_negative_fit_can_predict_better_without_supporting_high_scores(self):
        master, analysis, labels, _, training, panels = self.fixture()
        features = copy.deepcopy(label_fixtures.TransferLabelsTests().fixture()[2])
        features['policy_fingerprint'] = 'synthetic-direction-check'
        for i, row in enumerate(features['rows']):
            x = (i + .5) / len(features['rows'])
            row.update(primary_task_beta=x, current_pooled_beta=1-x,
                       historical_frontier=.5, historical_coverage=.25)
            gain = -.4*(x-.5)
            label = labels['groups'][row['source_group']]
            label.update(G=gain, V=gain+.125)
            for repeat in label['training_repeats']:
                repeat['G'] = gain + (repeat['training_repeat']-1)/64
                repeat['V'] = repeat['G']+.125
        fitted = fit_models(features, {g: labels['groups'][g]['G'] for g in analysis['source_split']['fit']}, analysis)
        original_fit = copy.deepcopy(fitted)
        all_predictions = predict_models(fitted, features, analysis)
        sealed = {g: all_predictions[g] for g in analysis['source_split']['heldout']}
        scores = prediction_results(labels, sealed, analysis)
        self.assertLess(scores['primary']['primary_difference'], 0)
        model = fitted['models']['primary_task_beta']
        # Sum(z^2)=12: a linear negative relation is shrunk by 12/(12+1).
        self.assertAlmostEqual(model['slope'], -.4*model['fit_population_std']*12/13)
        directions = {r['model']: r['direction'] for r in coefficient_rows(fitted)}
        self.assertEqual(directions, {'fit_mean': 'constant', 'primary_task_beta': 'negative',
                                     'current_pooled_beta': 'positive', 'historical_frontier': 'zero',
                                     'historical_coverage': 'zero'})
        rows = source_prediction_rows(labels, fitted, features, analysis, sealed)
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(r['split'] == 'heldout' for r in rows), 4)
        altered_labels = copy.deepcopy(labels)
        altered_labels['groups'][rows[0]['source_group']]['G'] += .01
        changed = source_prediction_rows(altered_labels, fitted, features, analysis, sealed)
        self.assertEqual([r['predictions'] for r in rows], [r['predictions'] for r in changed])
        self.assertEqual(fitted, original_fit)
        tampered = copy.deepcopy(sealed)
        tampered[next(iter(tampered))]['primary_task_beta'] += .01
        with self.assertRaises(ValueError):
            source_prediction_rows(labels, fitted, features, analysis, tampered)
        report = {'prediction_results': scores, 'labels': labels, 'sealed_models': fitted,
                  'coefficient_directions': coefficient_rows(fitted), 'source_predictions': rows,
                  'sealed_predictions': {'heldout_groups': analysis['source_split']['heldout']},
                  'costs': cost_accounting(master, training, panels)}
        text = markdown(report)
        self.assertIn('负斜率', text)
        self.assertIn('不支持高分来源更值得学习', text)
        self.assertIn('fit行是拟合内描述', text)

    def test_coefficient_display_refuses_missing_nonfinite_or_invalid_models(self):
        _, analysis, labels, _, _, _ = self.fixture()
        features = label_fixtures.TransferLabelsTests().fixture()[2]
        fitted = fit_models(features, {g: labels['groups'][g]['G'] for g in analysis['source_split']['fit']}, analysis)
        bad = copy.deepcopy(fitted)
        bad['models'].pop(SIGNALS[0])
        with self.assertRaises(ValueError):
            coefficient_rows(bad)
        bad = copy.deepcopy(fitted)
        bad['models'][SIGNALS[0]]['slope'] = float('nan')
        with self.assertRaises(ValueError):
            coefficient_rows(bad)
        bad = copy.deepcopy(fitted)
        bad['models'][SIGNALS[0]].update(fit_population_std=0, slope=.1)
        with self.assertRaises(ValueError):
            coefficient_rows(bad)


if __name__ == '__main__':
    unittest.main()
