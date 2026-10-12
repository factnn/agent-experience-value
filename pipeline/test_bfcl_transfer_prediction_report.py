import copy
import unittest
from bfcl_transfer_labels import aggregate_labels
from report_bfcl_transfer_prediction import prediction_results, cost_accounting, markdown
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
        report = {'prediction_results': prediction_results(labels, predictions, analysis), 'labels': labels,
                  'sealed_predictions': {'heldout_groups': analysis['source_split']['heldout']}, 'costs': costs}
        text = markdown(report)
        for group in analysis['source_split']['all_groups']:
            self.assertIn('| '+group+' |', text)
        for model in analysis['features']['allowed_models']:
            self.assertIn('| '+model+' |', text)
        self.assertIn('不是总体显著性检验', text)
        self.assertIn('无新ID对照', text)


if __name__ == '__main__':
    unittest.main()
