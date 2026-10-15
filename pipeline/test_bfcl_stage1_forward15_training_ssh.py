"""Actual15 identities with isolated CPU routing/certificate fixtures; no GPU science."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import bfcl_stage1_forward15_training_ssh as adapter

PROTOCOL = {'path': 'rl/BFCL_SOURCE_STAGE1_FORWARD15_PROTOCOL_20261015.json',
            'sha256': '0f661ed24ac04533be5d12529cf219942f8dfe3dd1af58de641c2eb352d611cf'}


class Forward15Training(unittest.TestCase):
    def test_actual15_layout_and_reused_eight_lane_worker(self):
        master, jobs = adapter.load_plan(adapter.ROOT, PROTOCOL)
        self.assertEqual(len(jobs), 15)
        self.assertEqual(sum(j['token_budget'] for j in jobs), 983040)
        self.assertEqual(sum(j['token_budget']+32768 for j in jobs), 1474560)
        self.assertEqual({j['source_group'] for j in jobs if j['role'] != 'uniform_control'},
                         {'MathAPI+TradingBot', 'MathAPI+TravelAPI', 'MessageAPI+VehicleControlAPI', 'TravelAPI'})
        barrier, lock = threading.Barrier(8), threading.Lock()
        active = peak = 0
        names, audits, first = [], [], set()
        def launch(gpu, case):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active); names.append(case['name'])
                wait = gpu not in first; first.add(gpu)
            if wait: barrier.wait(timeout=10)
            with lock: active -= 1
            return {'returncode': 0, 'hard_timeout': False, 'status': 'runner_complete_pending_replay_audit'}
        def audit(name):
            with lock: audits.append(name)
        schedule = {'cases': master['jobs'], 'gpu_lanes': master['gpu_lanes'],
                    'allowed_gpus': master['allowed_gpus'], 'max_simultaneous_gpus': 8}
        with tempfile.TemporaryDirectory() as tmp:
            result = adapter.core.execute_lanes(Path(tmp), schedule, lambda gpu: {'fixture_only': True}, launch, audit)
        self.assertEqual(peak, 8)
        self.assertEqual(len(names), 15); self.assertEqual(len(set(names)), 15)
        self.assertEqual(set(names), {entry['name'] for entry in master['jobs']})
        self.assertEqual(len(result['complete']), 15); self.assertEqual(result['pending'], []); self.assertEqual(result['failures'], [])
        self.assertEqual(len([x for x in audits if x is not None]), 15); self.assertEqual(audits.count(None), 1)

    def test_phase_scope_and_metadata_bindings_restore(self):
        before = {name: getattr(adapter.core, name) for name in ('load_plan', 'audit_batch', 'exclusive', 'SCOPE', 'CONTROL_CODE')}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'launch.json'
            with adapter.integration() as core:
                self.assertIs(core.load_plan, adapter.load_plan)
                self.assertEqual(core.SCOPE, adapter.SCOPE)
                self.assertTrue({adapter.SELF, adapter.CONTRACT} <= core.CONTROL_CODE)
                core.exclusive(path, {'scientific_jobs': 51, 'fixture_only': True})
            self.assertEqual(json.loads(path.read_text())['scientific_jobs'], 15)
        for name, value in before.items(): self.assertEqual(getattr(adapter.core, name), value)

    def test_old51_scope_and_missing_actual_forecasts_cannot_admit(self):
        master, jobs = adapter.load_plan(adapter.ROOT, PROTOCOL)
        with self.assertRaisesRegex(ValueError, 'only the registered forward15'):
            adapter.load_plan(adapter.ROOT, master['prior_full_stage1_design'])
        with adapter.integration() as core:
            with self.assertRaisesRegex(ValueError, 'Wrong current stage1 training ownership'):
                core.check_scope(adapter.ROOT, PROTOCOL, master, {}, {'scope': 'stage1_fixed_parent_source_training_51'})
            with patch.object(core, 'check_scope', return_value={'fixture_only': True}):
                with self.assertRaisesRegex(ValueError, 'requires actual original-five and H1'):
                    core.prerequisites(adapter.ROOT, PROTOCOL, master, {}, {'forward_bundle': None})
        self.assertFalse((adapter.ROOT/master['output_root']).exists())

    def test_fixture_certificate15_budget_guard_and_no_overwrite(self):
        master, jobs = adapter.load_plan(adapter.ROOT, PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); control = root/'fixture_control'; control.mkdir()
            for entry, job in zip(master['jobs'], jobs):
                p = root/entry['path']; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((adapter.ROOT/entry['path']).read_bytes())
                out = root/job['output']; out.mkdir(parents=True)
                (out/'audit.json').write_text(json.dumps({'status': 'passed', 'job_sha256': entry['sha256'],
                    'cost': {'new_model_tokens': 65536}, 'fixture_only_not_raw_scientific_audit': True}))
            first = root/jobs[0]['output']/'audit.json'
            bad = json.loads(first.read_text()); bad['cost']['new_model_tokens'] = 98305; first.write_text(json.dumps(bad))
            authority = {'forward_bundle': {'path': 'fixture_only', 'sha256': 'not_an_actual_receipt'}}
            with self.assertRaisesRegex(ValueError, 'Incomplete original15'):
                adapter.audit_batch(root, PROTOCOL, master, authority, control)
            certificate = root/master['complete_training_audit_path']; self.assertFalse(certificate.exists())
            bad['cost']['new_model_tokens'] = 65536; first.write_text(json.dumps(bad))
            adapter.audit_batch(root, PROTOCOL, master, authority, control)
            value = json.loads(certificate.read_text())
            self.assertEqual(value['complete_jobs'], 15); self.assertEqual(value['new_sampled_tokens'], 983040)
            self.assertFalse(value['original_full51_complete'])
            self.assertFalse((root/master['output_root']/'audit_status.json').exists())
            with self.assertRaises(FileExistsError): adapter.audit_batch(root, PROTOCOL, master, authority, control)


if __name__ == '__main__': unittest.main(verbosity=2)
