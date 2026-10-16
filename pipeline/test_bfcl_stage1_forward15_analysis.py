"""CPU-only arithmetic/scope checks; fixtures are not scientific measurements."""
import unittest
from copy import deepcopy

from bfcl_stage1_forward15_analysis import registry, labels, score, stability
from bfcl_transfer_predictor import SIGNALS


class ForwardArithmetic(unittest.TestCase):
    def setUp(self):
        self.groups = ['a', 'b', 'c', 'd']
        self.analysis = {'source_split': {'heldout': self.groups, 'all_groups': [*self.groups, 'deferred'], 'fit': ['deferred']},
                         'target_panel': {'registry': {str(i): str(i) for i in range(4)}}}
        self.jobs = []
        for r in range(3):
            for g in [None, *self.groups]:
                self.jobs.append({'stage': 1, 'source_group': g, 'training_repeat': r,
                                  'case_id': 's1_uniform_r'+str(r) if g is None else 's1_'+g+'_r'+str(r)})
        self.master = {'stage': 1, 'batch_mode': 'forward_only_15', 'training_jobs': 15,
                       'training_repeats': 3, 'jobs': [{'name': j['case_id']} for j in self.jobs]}
        self.reg = registry(self.master, self.jobs, self.analysis)
        self.panels = []
        for case, row in self.reg.items():
            for e, baseline in enumerate([0.4, 0.6]):
                value = baseline if row['role'] == 'baseline' else baseline-0.1 if row['role'] == 'control' else baseline-0.05
                self.panels.append({'status': 'passed', 'policy_case': case, 'evaluation_repeat': e,
                                    'role': row['role'], 'macro_success': value,
                                    'stratum_success': {str(i): value for i in range(4)}})

    def test_negative_G_positive_V_and_original_contract_unchanged(self):
        before = deepcopy(self.analysis)
        measured = labels(self.panels, self.reg, self.analysis)
        self.assertEqual(self.analysis, before)
        self.assertEqual(measured['stage'], 1)
        self.assertEqual(measured['panel_count'], 32)
        for group in self.groups:
            self.assertAlmostEqual(measured['G_labels'][group], -0.05)
            self.assertAlmostEqual(measured['V_labels'][group], 0.05)
        view = stability(measured)
        self.assertEqual(len(view['matched_source_pairs']), 6)
        self.assertTrue(all(not row['G_positive_all_three'] and row['V_positive_all_three'] for row in view['groups'].values()))

    def test_fixed_predictions_error_and_dependent_repeat_views(self):
        measured = labels(self.panels, self.reg, self.analysis)
        predictions = {g: {'fit_mean': 0.0, **{s: -0.05 for s in SIGNALS}} for g in self.groups}
        result = score(predictions, {g: -0.05 for g in self.groups}, measured, self.analysis)
        self.assertAlmostEqual(result['original_five']['MSE']['fit_mean'], 0.0025)
        self.assertAlmostEqual(result['original_five']['primary_difference'], -0.0025)
        self.assertAlmostEqual(result['separate_H1']['primary_H1_minus_early_mean'], -0.0025)
        self.assertEqual(len(result['by_training_repeat']), 3)
        self.assertEqual(len(result['by_evaluation_repeat']), 2)
        self.assertFalse(result['model_refitted'])

    def test_incomplete_or_other_scope_is_rejected(self):
        with self.assertRaises(ValueError): labels(self.panels[:-1], self.reg, self.analysis)
        with self.assertRaises(ValueError): labels([*self.panels, self.panels[0]], self.reg, self.analysis)
        altered = deepcopy(self.jobs); altered[0]['stage'] = 0
        with self.assertRaises(ValueError): registry(self.master, altered, self.analysis)
        altered = deepcopy(self.jobs); altered[1]['source_group'] = 'deferred'
        with self.assertRaises(ValueError): registry(self.master, altered, self.analysis)


if __name__ == '__main__': unittest.main(verbosity=2)
