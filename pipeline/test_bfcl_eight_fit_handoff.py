"""CPU checks for preserving original jobs while transferring only control ownership."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from supervise_bfcl_source_evaluation_eight import ROOT, validate_manifest, validate_handoff
from bfcl_ssh_evaluation_control_eight import check_control, resource_allocation


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class EightFitHandoffTests(unittest.TestCase):
    def setUp(self):
        self.plan = read(ROOT/'rl/BFCL_SOURCE_STAGE0_FIT_EIGHT_EXECUTION_20261014.json')
        self.authority = read(ROOT/'rl/BFCL_SOURCE_STAGE0_FIT_EIGHT_SSH_AUTHORITY_20261014.json')
        self.handoff = read(ROOT/self.plan['handoff']['path'])
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'experiments/.runtime')
        self.addCleanup(self.temp.cleanup)

    def mutated_handoff(self, handoff):
        path = Path(self.temp.name)/'handoff.json'
        path.write_text(json.dumps(handoff))
        plan = copy.deepcopy(self.plan)
        plan['handoff'] = ref(path)
        return plan

    def test_original_full_matrix_and_current_authority_are_valid(self):
        validate_manifest(ROOT, self.plan)
        validate_handoff(ROOT, self.plan)
        owner = check_control(ROOT, self.plan, self.authority)
        self.assertEqual(owner['complete_training_review_event'], 'E145')
        self.assertEqual(len(self.plan['jobs']), 80)

    def test_original_job_bytes_cannot_be_substituted(self):
        plan = copy.deepcopy(self.plan)
        plan['jobs'][0]['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            validate_handoff(ROOT, plan)

    def test_carried_job_cannot_move_to_another_gpu(self):
        h = copy.deepcopy(self.handoff)
        h['carried'][0]['gpu'] = 2
        with self.assertRaises(ValueError):
            validate_handoff(ROOT, self.mutated_handoff(h))

    def test_carried_original_hard_budget_cannot_be_extended(self):
        h = copy.deepcopy(self.handoff)
        process = read(ROOT/h['carried'][0]['process_receipt']['path'])
        process['hard_seconds'] += 1
        path = Path(self.temp.name)/'process.json'
        path.write_text(json.dumps(process))
        h['carried'][0]['process_receipt'] = ref(path)
        with self.assertRaises(ValueError):
            validate_handoff(ROOT, self.mutated_handoff(h))

    def test_completed_job_cannot_be_restarted_as_pending(self):
        h = copy.deepcopy(self.handoff)
        name = h['complete_at_latch'].pop(0)
        h['pending_at_latch'].append(name)
        with self.assertRaises(ValueError):
            validate_handoff(ROOT, self.mutated_handoff(h))

    def test_original_control_history_cannot_be_overwritten(self):
        plan = copy.deepcopy(self.plan)
        plan['control_root'] = plan['output_root']
        with self.assertRaises(ValueError):
            validate_handoff(ROOT, plan)

    def test_prior_six_card_cap_does_not_authorize_eight(self):
        with self.assertRaises(ValueError):
            resource_allocation(ROOT, ref(ROOT/'experiments/SSH_FUTURE_GPU_CAP_V2_20261013.json'))

    def test_current_stop_admission_is_obeyed(self):
        with tempfile.NamedTemporaryFile(dir=ROOT/'experiments/.runtime', prefix='eight_stop_test_', suffix='.tmp') as f:
            authority = copy.deepcopy(self.authority)
            authority['stop_file'] = str(Path(f.name).relative_to(ROOT))
            with self.assertRaises(RuntimeError):
                check_control(ROOT, self.plan, authority)


if __name__ == '__main__':
    unittest.main(verbosity=2)
