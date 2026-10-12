import copy
import unittest
from bfcl_transfer_predictor import SIGNALS, fit_models, predict_models, score_heldout


class TransferPredictorTests(unittest.TestCase):
    def fixture(self):
        contract = {'source_split': {'all_groups': ['a', 'b', 'c', 'd'], 'fit': ['a', 'b', 'c'], 'heldout': ['d']},
                    'fitting': {'lambda': 1.0}, 'features': {'allowed_models': ['fit_mean', *SIGNALS]}}
        features = {'validation_labels_used': False, 'stage': 0, 'policy_fingerprint': 'fixture',
                    'rows': [{'source_group': g, **{s: i for s in SIGNALS}} for i, g in enumerate(['a', 'b', 'c', 'd'])]}
        return features, {'a': -.1, 'b': 0, 'c': .1}, contract

    def test_closed_form_and_no_test_standardization(self):
        features, labels, contract = self.fixture()
        fit = fit_models(features, labels, contract)
        prediction = predict_models(fit, features, contract)
        self.assertAlmostEqual(prediction['d']['primary_task_beta'], .15)
        altered = copy.deepcopy(features)
        altered['rows'][-1]['primary_task_beta'] = 100
        self.assertEqual(fit_models(altered, labels, contract), fit)
        with self.assertRaises(ValueError):
            fit_models(features, labels | {'d': .5}, contract)
        with self.assertRaises(ValueError):
            score_heldout(prediction, labels | {'d': .2}, contract)
        self.assertLess(score_heldout(prediction, {'d': .2}, contract)['primary_difference'], 0)

    def test_late_stage_keeps_early_normalization_and_constant_features(self):
        features, labels, contract = self.fixture()
        for row in features['rows']:
            row['historical_coverage'] = 1
        fit = fit_models(features, labels, contract)
        late = copy.deepcopy(features)
        late.update(stage=1, policy_fingerprint='later-policy')
        for row in late['rows']:
            row['primary_task_beta'] += 2
            row['historical_coverage'] = 5
        prediction = predict_models(fit, late, contract)
        self.assertAlmostEqual(prediction['d']['primary_task_beta'], .3)
        self.assertEqual(prediction['d']['historical_coverage'], 0)
        self.assertEqual(fit['fit_stage'], 0)
        with self.assertRaises(ValueError):
            fit_models(features | {'validation_labels_used': True}, labels, contract)


if __name__ == '__main__':
    unittest.main()
