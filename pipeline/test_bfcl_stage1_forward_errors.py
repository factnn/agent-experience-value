import json
from pathlib import Path
import tempfile
import unittest

from describe_bfcl_stage1_forward_errors import build, error_components, sha


class ErrorDescriptionTests(unittest.TestCase):
    def test_shared_offset_preserves_all_source_contrasts(self):
        observed = dict(zip('abcd', [.01, .02, .03, .04]))
        prediction = {g: y+.03 for g, y in observed.items()}
        result = error_components(prediction, observed, list('abcd'))
        self.assertAlmostEqual(result['MSE_original'], .0009)
        self.assertAlmostEqual(result['mean_signed_error'], .03)
        self.assertAlmostEqual(result['overall_squared_bias'], .0009)
        self.assertAlmostEqual(result['centered_between_source_error'], 0)
        for pair in result['matched_source_pairs']:
            self.assertAlmostEqual(pair['predicted_G_difference'], pair['observed_G_difference'])

    def test_zero_mean_error_can_hide_reversed_source_order(self):
        observed = dict(zip('abcd', [.01, .02, .03, .04]))
        prediction = dict(zip('abcd', [.04, .03, .02, .01]))
        result = error_components(prediction, observed, list('abcd'))
        self.assertAlmostEqual(result['mean_signed_error'], 0)
        self.assertAlmostEqual(result['overall_squared_bias'], 0)
        self.assertAlmostEqual(result['MSE_original'], .0005)
        self.assertAlmostEqual(result['centered_between_source_error'], .0005)
        for pair in result['matched_source_pairs']:
            self.assertLess(pair['predicted_G_difference']*pair['observed_G_difference'], 0)

    def test_incomplete_scientific_matrix_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);report = root/'partial.json'
            report.write_text(json.dumps({'status': 'passed', 'stage': 1, 'batch_mode': 'forward_only_15',
                                         'raw_replay': {'training_jobs': 15, 'target_panels': 31, 'saved_audits_equal_fresh': True}}))
            with self.assertRaisesRegex(ValueError, 'complete original15/32'):
                build(root, report, sha(report))
        with self.assertRaisesRegex(ValueError, 'complete four sources'):
            error_components({'a': 0, 'b': 0, 'c': 0}, {'a': 0, 'b': 0, 'c': 0}, list('abc'))


if __name__ == '__main__': unittest.main()
