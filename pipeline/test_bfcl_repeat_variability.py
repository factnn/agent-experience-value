"""Check shared-baseline cancellation and rejection of invalid seed comparisons."""
import copy
import unittest
from bfcl_repeat_variability import describe_variability


class VariabilityTests(unittest.TestCase):
    def setUp(self):
        self.base_values = {'e0': [1, 0, 0], 'e1': [0, 1, 0]}
        self.base = {e: {'outcomes': [{'task_id': str(i), 'rewards': [v]} for i, v in enumerate(vs)]}
                     for e, vs in self.base_values.items()}
        policies = {'r1': {'e0': [1, 1, 0], 'e1': [0, 0, 0]},
                    'r2': self.base_values, 'discovery': {'e0': [1, 1, 1], 'e1': [1, 1, 1]}}
        self.rows = []
        for repeat, panels in policies.items():
            for e, values in panels.items():
                self.rows.append({'training_repeat': repeat, 'threshold': 32, 'evaluation_repeat': e,
                    'prospective_training_repeat': repeat != 'discovery', 'policy_fingerprint': repeat,
                    'successes': sum(values), 'G': (sum(values) - 1) / 3,
                    'paired_outcomes': [{'task_id': str(i), 'source': self.base_values[e][i], 'checkpoint': v}
                                        for i, v in enumerate(values)]})

    def describe(self, rows=None):
        return describe_variability(self.rows if rows is None else rows, self.base, ['r1', 'r2'], ['e0', 'e1'], [32])

    def test_shared_baseline_and_discovery_exclusion(self):
        result = self.describe()
        self.assertEqual(result['baseline_outcome_disagreements'], 2)
        fixed = {r['training_repeat']: r for r in result['fixed_policy_evaluation_contrasts']}
        self.assertTrue(fixed['r1']['strict_G_sign_flip'])
        self.assertEqual(fixed['r1']['G_task_count_delta_e1_minus_e0'], -2)
        self.assertEqual(fixed['r2']['policy_outcome_disagreements'], 2)
        self.assertEqual(fixed['r2']['paired_gain_disagreements'], 0)
        training = result['prospective_training_contrasts_averaged_over_evaluation'][0]
        self.assertEqual(training['conditional_G_range'], [0, 0])
        self.assertEqual(training['conditional_G_sample_sd'], 0)

    def test_incomplete_duplicate_and_wrong_policy_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaises(AssertionError): self.describe(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]['policy_fingerprint'] = 'different'
        with self.assertRaises(AssertionError): self.describe(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]['paired_outcomes'][0]['source'] = 0
        with self.assertRaises(AssertionError): self.describe(rows)


if __name__ == '__main__':
    unittest.main()
