"""Stable CPU-only manifest checks; no evaluation outcome files are read."""
import copy
import json
import unittest
from report_bfcl_transfer_prediction_eight import validate_manifest
from supervise_bfcl_stage_fork import ROOT


class ReportResourceChecks(unittest.TestCase):
    def setUp(self):
        self.fit = json.loads((ROOT/'rl/bfcl_source_stage0_fit_jobs_20261013/manifest.json').read_text())
        self.held = json.loads((ROOT/'rl/bfcl_source_stage0_heldout_jobs_20261015/manifest.json').read_text())

    def test_original_fit_six_manifest_is_accepted(self):
        master = validate_manifest(ROOT, self.fit)
        self.assertEqual(len(master['jobs']), 51)

    def test_original_heldout_eight_manifest_is_accepted(self):
        master = validate_manifest(ROOT, self.held)
        self.assertEqual(len(master['jobs']), 51)

    def test_eight_card_resource_cannot_be_misreported_as_six(self):
        plan = copy.deepcopy(self.held)
        plan['max_simultaneous_gpus'] = 6
        with self.assertRaises(ValueError):
            validate_manifest(ROOT, plan)

    def test_registered_heldout_job_bytes_cannot_be_substituted(self):
        plan = copy.deepcopy(self.held)
        plan['jobs'][0]['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            validate_manifest(ROOT, plan)


if __name__ == '__main__':
    unittest.main(verbosity=2)
