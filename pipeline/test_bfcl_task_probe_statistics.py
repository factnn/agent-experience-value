import copy
import itertools
import unittest
from bfcl_task_probe_statistics import source_statistics, compare_independent_panels


class TaskProbeStatisticsTests(unittest.TestCase):
    def test_deterministic_task_mixture_is_not_within_task_opportunity(self):
        s = source_statistics({'easy': [1]*4, 'hard': [0]*4})
        self.assertEqual(s['within_uncertainty_unbiased'], 0)
        self.assertEqual(s['between_task_variance_unbiased'], .25)
        self.assertEqual(s['four_draw_mixed_probability_unbiased'], 0)
        self.assertEqual(s['pooled_plugin_mixed_probability'], .875)

    def test_exact_expectations_over_all_256_binary_samples(self):
        expected = {key: 0. for key in ['within_uncertainty_unbiased', 'between_task_variance_unbiased', 'pooled_uncertainty_unbiased', 'four_draw_mixed_probability_unbiased']}
        for bits in itertools.product([0, 1], repeat=8):
            probability = 1.
            for index, bit in enumerate(bits):
                p = .2 if index < 4 else .8
                probability *= p if bit else 1-p
            s = source_statistics({'a': list(bits[:4]), 'b': list(bits[4:])})
            self.assertAlmostEqual(s['decomposition_residual'], 0)
            for key in expected:
                expected[key] += probability*s[key]
        self.assertAlmostEqual(expected['within_uncertainty_unbiased'], .16)
        self.assertAlmostEqual(expected['between_task_variance_unbiased'], .09)
        self.assertAlmostEqual(expected['pooled_uncertainty_unbiased'], .25)
        self.assertAlmostEqual(expected['four_draw_mixed_probability_unbiased'], 1-.2**4-.8**4)

    def test_negative_variance_estimate_remains_visible(self):
        s = source_statistics({'a': [0, 0, 1, 1], 'b': [0, 0, 1, 1]})
        self.assertLess(s['between_task_variance_unbiased'], 0)
        self.assertTrue(s['negative_between_estimate_preserved'])
        one = source_statistics({'only': [0, 0, 1, 1]})
        self.assertEqual(one['between_task_variance_unbiased'], 0)

    def test_independent_panel_contract_rejects_leakage_and_drift(self):
        a = [dict(task_id=t, source_group='g', stage=0, policy_fingerprint='f', draw_id='feature:'+t, rewards=[0, 1, 0, 1]) for t in ['a', 'b']]
        b = copy.deepcopy(a)
        for r in b: r['draw_id'] = 'validation:'+r['task_id']
        result = compare_independent_panels(a, b)
        self.assertEqual(result['g']['validation_observed_group_mixed_rate'], 1)
        for bad in [a, b[:1], b + [dict(b[0], draw_id='extra')], [dict(r, policy_fingerprint='changed') for r in b], [dict(b[0], rewards=[0, .5, 0, 1]), b[1]]]:
            with self.assertRaises(ValueError): compare_independent_panels(a, bad)


if __name__ == '__main__': unittest.main()
