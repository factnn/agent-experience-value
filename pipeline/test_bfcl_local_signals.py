import unittest
from audit_bfcl_local_signals import beta_mixed_probability, auc


class LocalSignalTests(unittest.TestCase):
    def test_prior_predictive_and_symmetry(self):
        self.assertAlmostEqual(beta_mixed_probability(0, 0), .6)
        self.assertAlmostEqual(beta_mixed_probability(1, 4), beta_mixed_probability(3, 4))
        self.assertAlmostEqual(beta_mixed_probability(2, 4, 1), 0)
        with self.assertRaises(ValueError): beta_mixed_probability(5, 4)

    def test_auc_ties_and_no_class(self):
        rows = [{'mixed_reward': 1, 'x': .5}, {'mixed_reward': 0, 'x': .5}]
        self.assertEqual(auc(rows, 'x'), .5)
        rows[0]['x'] = .5 + 1e-15
        self.assertEqual(auc(rows, 'x'), .5)
        rows[0]['x'] = .7
        self.assertEqual(auc(rows, 'x'), 1)
        self.assertIsNone(auc(rows[:1], 'x'))


if __name__ == '__main__': unittest.main()
