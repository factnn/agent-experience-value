"""Targeted tests of actual forward15 scope and new32-panel integration,no GPU samples."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import bfcl_stage1_forward32_contract as contract
import bfcl_stage1_forward32_ssh as control
import prepare_bfcl_stage1_forward32 as prepare
from bfcl_source_eval_contract import digest, pinned, policy_role
from supervise_bfcl_stage_fork import ROOT, read

PROTOCOL = {'path': 'rl/BFCL_SOURCE_STAGE1_FORWARD15_PROTOCOL_20261015.json',
            'sha256': '0f661ed24ac04533be5d12529cf219942f8dfe3dd1af58de641c2eb352d611cf'}


class Forward32Integration(unittest.TestCase):
    def test_runtime_pins_cover_reused_controllers_and_job_manifest_agree(self):
        required = {'pipeline/bfcl_stage1_training_ssh.py', 'pipeline/bfcl_stage1_probe_ssh.py',
                    'pipeline/supervise_bfcl_stage1_probe.py', 'pipeline/supervise_bfcl_stage_fork.py'}
        self.assertTrue(required <= set(contract.RUNTIME))
        hashes = {path: digest(ROOT/path) for path in contract.RUNTIME}
        manifest = {'runtime_sha256': hashes}; job = {'input_sha256': dict(hashes)}
        control.validate_runtime(ROOT, manifest, [job])
        job['input_sha256'][contract.RUNTIME[0]] = 'fixture_changed_sha'
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            control.validate_runtime(ROOT, manifest, [job])

    def test_actual32_cell_roles_and_reused_eight_lane_worker(self):
        master = pinned(ROOT, PROTOCOL)
        analysis = pinned(ROOT, master['analysis_contract'])
        training = {entry['name']: pinned(ROOT, entry) for entry in master['jobs']}
        cells = prepare.cells(master)
        self.assertEqual(len(cells), 32); self.assertEqual(len(set(cells)), 32)
        roles = collections.Counter(policy_role(case, training.get(case), analysis) for case, repeat in cells)
        self.assertEqual(roles, {'baseline': 2, 'control': 6, 'heldout': 24})
        entries = [{'name': prepare.name(case, repeat)} for case, repeat in cells]
        schedule = {'cases': entries, 'gpu_lanes': {str(g): [e['name'] for i, e in enumerate(entries) if i % 8 == g] for g in range(8)},
                    'allowed_gpus': list(range(8)), 'max_simultaneous_gpus': 8}
        barrier, lock = threading.Barrier(8), threading.Lock()
        seen, audits, first = [], [], set(); active = peak = 0
        def launch(gpu, case):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active); seen.append(case['name']); wait = gpu not in first; first.add(gpu)
            if wait: barrier.wait(timeout=10)
            with lock: active -= 1
            return {'returncode': 0, 'hard_timeout': False, 'status': 'runner_complete_pending_replay_audit'}
        def audit(case):
            with lock: audits.append(case)
        with tempfile.TemporaryDirectory() as tmp:
            result = control.execute_lanes(Path(tmp), schedule, lambda gpu: {'fixture_only': True}, launch, audit)
        self.assertEqual(peak, 8); self.assertEqual(len(seen), 32); self.assertEqual(len(set(seen)), 32)
        self.assertEqual(len(result['complete']), 32); self.assertEqual(result['pending'], []); self.assertEqual(result['failures'], [])
        self.assertEqual(len([case for case in audits if case is not None]), 32); self.assertEqual(audits.count(None), 1)

    def test_actual_common_is_step7_parent_and_deferred_fit_policy_refused(self):
        master = pinned(ROOT, PROTOCOL); analysis = pinned(ROOT, master['analysis_contract'])
        policy = ROOT/master['stage_start']['path']
        required = ['summary.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
        job = {'policy_case': 'common', 'role': 'baseline', 'policy_fingerprint': master['stage_start']['policy_fingerprint'],
               'policy_artifacts': {name: digest(policy/name) for name in required}}
        self.assertEqual(contract.source_policy(ROOT, job, master, analysis, {}), policy)
        self.assertEqual(master['stage_start']['source_step'], 7)
        old = pinned(ROOT, master['stage0_execution_protocol'])
        job['policy_fingerprint'] = old['source_stage0']['policy_fingerprint']
        with self.assertRaisesRegex(ValueError, 'stage1 policy fingerprint'):
            contract.source_policy(ROOT, job, master, analysis, {})
        job['policy_case'] = master['deferred_jobs'][0]['name']
        with self.assertRaisesRegex(ValueError, 'outside the active15'):
            contract.source_policy(ROOT, job, master, analysis, {})

    def test_all_roles_missing_forecasts_and_actual_incomplete15_refused(self):
        master = pinned(ROOT, PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp:
            for role in ['baseline', 'control', 'heldout', 'fit']:
                path = Path(tmp)/(role+'.json')
                path.write_text(json.dumps({'execution_protocol': PROTOCOL, 'stage': 1, 'batch_mode': contract.MODE,
                                             'role': role, 'forward_bundle': None}))
                with self.assertRaisesRegex(ValueError, 'Every forward32 target role'):
                    contract.load_job(ROOT, path, digest(path), admit=True, authority_ref=None)
        if (ROOT/master['complete_training_audit_path']).exists():
            self.skipTest('Actual15 is now complete;use real reviewed target registration rather than this negative check.')
        with self.assertRaisesRegex(ValueError, 'All15 reviewed training endpoints'):
            prepare.prepare(ROOT, PROTOCOL, {'path': 'fixture_only_missing_binding', 'sha256': 'not_a_receipt'})
        self.assertFalse((ROOT/prepare.REGISTRATION).exists()); self.assertFalse((ROOT/prepare.OUTPUT).exists())

    def test_training_owner_cannot_admit_targets_and_runner_exits_before_inference(self):
        master = pinned(ROOT, PROTOCOL)
        wrong = {'mode': 'human_authorized_ssh_takeover', 'status': 'authorized',
                 'scope': 'stage1_fixed_parent_forward_training_15', 'execution_protocol': PROTOCOL,
                 'resource_authorization': master['resource_authorization']}
        with patch.object(contract, 'pinned', side_effect=[wrong, master]):
            with self.assertRaisesRegex(ValueError, 'evaluation authority/scope'):
                contract.check_owner(ROOT, {'path': 'fixture_only', 'sha256': 'fixture'}, PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'missing_forecast_fixture.json'; output = Path(tmp)/'must_not_exist'
            path.write_text(json.dumps({'execution_protocol': PROTOCOL, 'stage': 1, 'batch_mode': contract.MODE,
                                       'role': 'baseline', 'forward_bundle': None, 'output': str(output)}))
            result = subprocess.run([sys.executable, str(ROOT/'pipeline/run_bfcl_stage1_forward32.py'),
                                     '--job', str(path), '--job-sha256', digest(path),
                                     '--authority', 'fixture_only_missing_authority', '--authority-sha256', 'not_authority'],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
            self.assertEqual(result.returncode, 3, result.stderr)
            failure = json.loads(result.stderr.strip().splitlines()[-1])
            self.assertFalse(failure['scientific_inference_started']); self.assertEqual(failure['new_scientific_tokens'], 0)
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main(verbosity=2)
