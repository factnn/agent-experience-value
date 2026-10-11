import json
from pathlib import Path
import sys
import tempfile
import unittest

from supervise_bfcl_stage_fork import supervise, process_has_not_exited
from watch_research_events import discover


class SupervisorTests(unittest.TestCase):
    def test_exited_zombie_scheduler_is_not_running(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(process_has_not_exited(123, root))
            (root / '123').mkdir()
            for state in ['R', 'S', 'T', 'D', 'Z', 'X']:
                (root / '123/stat').write_text(f'123 (scheduler (child)) {state} 1 0')
                self.assertEqual(process_has_not_exited(123, root), state not in {'Z', 'X'})

    def run_child(self, body, hard=5):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        out = root / 'batch/replay'
        out.mkdir(parents=True)
        prefix = f'import pathlib,json,time; p=pathlib.Path({str(out)!r}); '
        result = supervise([sys.executable, '-c', prefix + body], out,
                           root / 'run.log', hard, grace_seconds=0.1)
        return result, out, list(discover(root, ['batch']))

    def test_hard_timeout_keeps_partial_cost_and_notifies(self):
        result, out, events = self.run_child(
            r'(p/"generation_calls.jsonl").write_text(json.dumps({"generated_ids":[1,2,3]})+"\n"+"{"); time.sleep(30)', 0.5)
        self.assertTrue(result['hard_timeout'])
        self.assertEqual(result['recorded_model_tokens_lower_bound'], 3)
        self.assertEqual(result['malformed_journal_rows'], 1)
        self.assertEqual([x['kind'] for x in events], ['scheduler_failure'])
        self.assertTrue((out / 'generation_calls.jsonl').exists())

    def test_zero_exit_incomplete_summary_still_notifies(self):
        result, _, events = self.run_child('(p/"summary.json").write_text(json.dumps({"status":"incomplete"}))')
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['status'], 'failed_or_incomplete')
        self.assertEqual([x['kind'] for x in events], ['scheduler_failure'])

    def test_complete_requires_further_replay_audit(self):
        result, _, events = self.run_child('(p/"summary.json").write_text(json.dumps({"status":"complete"}))')
        self.assertEqual(result['status'], 'runner_complete_pending_replay_audit')
        self.assertEqual([x['kind'] for x in events], ['complete'])

    def test_nonzero_exit_notifies_without_summary(self):
        result, _, events = self.run_child('raise SystemExit(7)')
        self.assertEqual(result['returncode'], 7)
        self.assertFalse(result['hard_timeout'])
        self.assertEqual([x['kind'] for x in events], ['scheduler_failure'])


if __name__ == '__main__':
    unittest.main()
