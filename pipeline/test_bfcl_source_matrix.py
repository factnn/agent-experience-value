import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import report_bfcl_source_matrix as matrix


def put(root, relative, value):
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class MatrixFinalizationTests(unittest.TestCase):
    def fixture(self, root):
        jobs, audits, events = [], {}, []
        for i in range(51):
            name = 'case_'+str(i)
            output = 'original/'+name
            job = {'case_id': name, 'output': output}
            entry = put(root, 'jobs/'+name+'.json', job) | {'name': name}
            jobs.append(entry)
            audit = {'status': 'passed', 'case_id': name, 'cost': {'new_model_tokens': 65536}}
            audits[name] = audit
            put(root, output+'/audit.json', audit)
            put(root, output+'/summary.json', {'status': 'complete', 'synthetic_fixture': True})
            files = {n: {'sha256': hashlib.sha256((root/output/n).read_bytes()).hexdigest(),
                         'bytes': (root/output/n).stat().st_size}
                     for n in ['summary.json', 'audit.json']}
            put(root, output+'/evidence_manifest.json', {'case_id': name, 'files': files,
                'fresh_full_raw_audit_matches_saved': True})
            eid = 'Efixture'+str(i)
            events.append({'source': output+'/summary.json', 'id': eid})
            put(root, 'experiments/reviews/'+eid+'.json', {'status': 'reviewed'})
        put(root, 'experiments/index.json', events)
        names = [e['name'] for e in jobs]
        master = {'jobs': jobs, 'output_root': 'original', 'input_sha256': {}, 'allowed_gpus': [4, 5, 6, 7],
                  'gpu_lanes': {str(g): names[g-4::4] for g in range(4, 8)}}
        master_ref = put(root, 'master.json', master)
        kept, pending = names[:24], names[24:]
        original = {'complete': kept, 'pending': pending, 'active': {}, 'failures': [
            {'case': pending[0], 'gpu': 4, 'type': 'RuntimeError',
             'message': 'Goal explicitly unavailable for work: blocked'}]}
        resumed = {'complete': pending, 'pending': [], 'active': {}, 'failures': []}
        old_terminal = {'status': 'incomplete_requires_review', 'complete_jobs': 24}
        new_terminal = {'status': 'audited_complete', 'complete_jobs': 27}
        admission_failure = [{'phase': 'preflight', 'gpu_children_launched': 0}]
        failed_ref = put(root, 'failed_plan.json', {'output_root': 'failed'})
        continuation = {'jobs': jobs[24:], 'output_root': 'resumed', 'input_sha256': {},
            'allowed_gpus': [4, 5, 6, 7], 'max_simultaneous_gpus': 4, 'automatic_retry': False,
            'gpu_lanes': {g: [n for n in lane if n in pending] for g, lane in master['gpu_lanes'].items()},
            'execution_continuation': {'original_scientific_protocol': master_ref,
                'completed_cases_retained': kept, 'pending_cases_only': pending,
                'failed_initial_admission': {'protocol': failed_ref, 'failures_path': 'failed/failures.json'}}}
        continuation_ref = put(root, 'continuation.json', continuation)
        for folder, queue, terminal in [('original', original, old_terminal), ('resumed', resumed, new_terminal)]:
            put(root, folder+'/queue.json', queue)
            put(root, folder+'/supervisor.json', terminal)
            put(root, folder+'/launch.json', {'supervisor_pid': 0})
        put(root, 'original/failures.json', original['failures'])
        put(root, 'failed/failures.json', admission_failure)
        put(root, 'resumed/audit_status.json', {'status': 'passed', 'complete_jobs': 27,
            'jobs': [audits[n] for n in pending], 'new_sampled_tokens': 27*65536})
        return master_ref, master, continuation_ref, continuation, original, resumed, old_terminal, new_terminal, admission_failure, audits

    def test_complete_51_replays_once_and_preserves_failure_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, _, continuation_ref, _, _, _, _, _, _, audits = self.fixture(root)
            failures = {n: (root/n).read_bytes() for n in ['original/failures.json', 'failed/failures.json']}
            with patch.object(matrix, 'audit_source', side_effect=lambda r, j, s: audits[j['case_id']]) as replay:
                result = matrix.finalize(root, ref, continuation_ref)
            self.assertEqual(replay.call_count, 51)
            self.assertEqual(result['complete_jobs'], 51)
            self.assertEqual(result['new_sampled_tokens'], 51*65536)
            self.assertEqual(len({a['case_id'] for a in result['jobs']}), 51)
            self.assertEqual(result['execution_recovery']['retained_jobs'], 24)
            self.assertEqual(result['execution_recovery']['continued_jobs'], 27)
            self.assertFalse(result['evaluation_performed'])
            self.assertFalse(result['goal_complete'])
            for n, before in failures.items():
                self.assertEqual((root/n).read_bytes(), before)
            before = (root/'original/audit_status.json').read_bytes()
            with self.assertRaises(FileExistsError):
                matrix.finalize(root, ref, continuation_ref)
            self.assertEqual((root/'original/audit_status.json').read_bytes(), before)

    def test_partial_queue_has_no_raw_replay_or_complete_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, _, continuation_ref, _, _, resumed, _, _, _, _ = self.fixture(root)
            resumed['active'] = {'4': resumed['complete'].pop()}
            put(root, 'resumed/queue.json', resumed)
            with patch.object(matrix, 'audit_source') as replay, self.assertRaises(ValueError):
                matrix.finalize(root, ref, continuation_ref)
            replay.assert_not_called()
            self.assertFalse((root/'original/audit_status.json').exists())

    def test_changed_job_lanes_duplicate_completion_and_training_failure_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.fixture(Path(tmp))
            ref, master, _, continuation, original, resumed, old, new, failure, _ = data
            args = [ref, master, continuation, original, resumed, old, new, failure]
            mutations = [
                lambda x: x[2]['jobs'][0].update(sha256='changed-seed-or-recipe'),
                lambda x: x[2]['gpu_lanes']['4'].reverse(),
                lambda x: x[4]['complete'].__setitem__(-1, x[4]['complete'][0]),
                lambda x: x[3]['pending'].append(x[3]['complete'][0]),
                lambda x: x[3]['failures'][0].update(message='Training failed'),
                lambda x: x[7][0].update(gpu_children_launched=1),
            ]
            for mutate in mutations:
                # Production snapshots are independently decoded JSON files.
                # Copy each argument separately so fixture aliases cannot also
                # mutate the supposedly frozen master when changing a continuation.
                bad = [copy.deepcopy(arg) for arg in args]
                mutate(bad)
                with self.assertRaises(ValueError):
                    matrix.validate_partition(*bad)

    def test_fresh_audit_or_continuation_substitution_leaves_no_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, _, continuation_ref, _, _, _, _, _, _, audits = self.fixture(root)
            bad = copy.deepcopy(audits)
            bad['case_25']['cost']['new_model_tokens'] += 1
            with patch.object(matrix, 'audit_source', side_effect=lambda r, j, s: bad[j['case_id']]), self.assertRaises(ValueError):
                matrix.finalize(root, ref, continuation_ref)
            self.assertFalse((root/'original/audit_status.json').exists())
            batch = matrix.read(root/'resumed/audit_status.json')
            batch['jobs'][0] = audits['case_0']
            put(root, 'resumed/audit_status.json', batch)
            with patch.object(matrix, 'audit_source') as replay, self.assertRaises(ValueError):
                matrix.finalize(root, ref, continuation_ref)
            replay.assert_not_called()
            self.assertFalse((root/'original/audit_status.json').exists())


if __name__ == '__main__':
    unittest.main()
