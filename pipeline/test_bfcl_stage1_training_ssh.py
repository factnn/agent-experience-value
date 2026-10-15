"""CPU tests of new stage1 control integration; fixture callbacks never launch science."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import bfcl_stage1_training_ssh as controller


PROTOCOL = {'path': 'rl/BFCL_SOURCE_STAGE1_EXECUTION_PROTOCOL_20261015.json',
            'sha256': '88750665635e88e18731d267079e431febfba1f22d72713b9a5c2f17d454682d'}


class Stage1TrainingControl(unittest.TestCase):
    def test_failure_categories_distinguish_queries_control_and_real_child(self):
        self.assertEqual(controller.admission_failure_category('publication', RuntimeError('Publication dependency unavailable after3 bounded attempts')), 'publication_dependency_query')
        self.assertEqual(controller.admission_failure_category('publication', RuntimeError('Current SSH owner stopped new stage1 training admission')), 'current_control_stop')
        self.assertEqual(controller.admission_failure_category('frozen_input_identity', FileNotFoundError('fixture missing input')), 'execution_input_or_query_dependency')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failure = {'phase': 'runner', 'case': 'fixture_case', 'type': 'RuntimeError'}
            self.assertEqual(controller.worker_failure_category(root, failure), 'control_or_execution_dependency_before_scientific_child')
            (root/'fixture_case.console.process.json').write_text('{"fixture_only":true}')
            self.assertEqual(controller.worker_failure_category(root, failure), 'scientific_process_failure')
            self.assertEqual(controller.worker_failure_category(root, {'phase': 'preflight', 'type': 'TimeoutExpired'}), 'execution_evidence_or_gpu_query_dependency')

    def test_actual_frozen51_layout_on_eight_reused_worker_lanes(self):
        master, jobs = controller.load_plan(controller.ROOT, PROTOCOL)
        schedule = {'cases': master['jobs'], 'gpu_lanes': master['gpu_lanes'],
                    'allowed_gpus': master['allowed_gpus'], 'max_simultaneous_gpus': 8}
        lock, barrier = threading.Lock(), threading.Barrier(8)
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
        with tempfile.TemporaryDirectory() as tmp:
            state = controller.execute_lanes(Path(tmp), schedule, lambda gpu: {'routing_fixture_only': True}, launch, audit)
        expected = [entry['name'] for entry in master['jobs']]
        self.assertEqual(peak, 8)
        self.assertEqual(len(names), 51); self.assertEqual(set(names), set(expected))
        self.assertEqual(len(set(names)), 51)
        self.assertEqual(len([x for x in audits if x is not None]), 51)
        self.assertEqual(audits.count(None), 1)
        self.assertEqual(len(state['complete']), 51)
        self.assertEqual(state['pending'], []); self.assertEqual(state['failures'], [])
        self.assertEqual(sum(job['token_budget'] for job in jobs), 3342336)
        self.assertEqual(sum(job['token_budget']+32768 for job in jobs), 5013504)

    def test_wrong_scope_and_missing_actual_forward_refuse_before_any_child(self):
        master, jobs = controller.load_plan(controller.ROOT, PROTOCOL)
        with self.assertRaisesRegex(ValueError, 'Wrong current stage1 training ownership'):
            controller.check_scope(controller.ROOT, PROTOCOL, master, {}, {'scope': 'old_stage0_owner'})
        with patch.object(controller, 'check_scope', return_value={'fixture_only': True}):
            with self.assertRaisesRegex(ValueError, 'requires actual original-five and H1'):
                controller.prerequisites(controller.ROOT, PROTOCOL, master, {}, {'forward_bundle': None})
        self.assertFalse((controller.ROOT/master['output_root']).exists())

    def test_current_training_stop_rejects_before_owner_or_model_loading(self):
        master, jobs = controller.load_plan(controller.ROOT, PROTOCOL)
        # Deliberately incomplete authority fixture in a temporary root. It
        # cannot authorize execution: no owner,forward receipt or controller pins.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runtime = root/'experiments/.runtime'; runtime.mkdir(parents=True)
            authority = {'mode': 'human_authorized_ssh_takeover', 'status': 'authorized', 'scope': controller.SCOPE,
                         'execution_protocol': PROTOCOL, 'resource_authorization': master['resource_authorization'],
                         'stop_file': 'experiments/.runtime/train', 'global_stop_file': 'experiments/.runtime/global',
                         'fixture_not_execution_authority': True}
            path = root/'incomplete_authority_fixture.json'; path.write_text(json.dumps(authority))
            (runtime/'train').touch()
            with self.assertRaisesRegex(RuntimeError, 'stopped new stage1 training admission'):
                controller.check_scope(root, PROTOCOL, master, controller.ref(root, path), authority)


if __name__ == '__main__': unittest.main(verbosity=2)
