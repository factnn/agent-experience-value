import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import watch_research_events as w


class ResearchEventsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.job = self.root / 'batch/job'
        self.job.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_partial_completion_restart_and_preserve_review(self):
        source = self.job / 'summary.json'
        source.write_text('{"status":')
        self.assertEqual(w.record(self.root, ['batch'])[0], [])
        w.write(source, {'status': 'running', 'completed_tasks': 1})
        self.assertEqual(w.record(self.root, ['batch'])[0], [])
        w.write(source, {'status': 'complete', 'completed_tasks': 22})
        rows, pending, _ = w.record(self.root, ['batch'])
        self.assertEqual(pending, ['E001'])
        doc = self.root / 'experiments/E001.md'
        doc.write_text('Human analysis stays intact.\n')
        w.write(self.root / 'experiments/reviews/E001.json', {'status': 'reviewed'})
        self.assertEqual(w.record(self.root, ['batch'])[1], [])
        self.assertEqual(doc.read_text(), 'Human analysis stays intact.\n')
        self.assertEqual(len(w.record(self.root, ['batch'])[0]), 1)
        # A corrected terminal fact creates a new record, not silent replacement.
        w.write(source, {'status': 'complete', 'completed_tasks': 21})
        rows, pending, _ = w.record(self.root, ['batch'])
        self.assertEqual(len(rows), 2)
        self.assertEqual(pending, ['E002'])

    def test_failure_and_empty_failure_list(self):
        w.write(self.root / 'batch/failures.json', [])
        w.write(self.job / 'failure.json', {'type': 'RuntimeError'})
        self.assertEqual([r['kind'] for r in w.record(self.root, ['batch'])[0]], ['failure'])

    def test_corrupt_registry_never_resets_numbering(self):
        index = self.root / 'experiments/index.json'
        index.parent.mkdir()
        index.write_text('[')
        with self.assertRaises(ValueError):
            w.record(self.root, ['batch'])
        self.assertEqual(index.read_text(), '[')

    def test_readonly_goal_and_fail_closed(self):
        self.assertEqual(w.goal_status(self.root, 'own'), 'unknown')
        with sqlite3.connect(self.root / 'goals_1.sqlite') as db:
            db.execute('CREATE TABLE thread_goals(thread_id TEXT, status TEXT)')
            db.execute('INSERT INTO thread_goals VALUES (?, ?)', ('own', 'paused'))
        self.assertEqual(w.goal_status(self.root, 'own'), 'paused')
        self.assertEqual(w.goal_status(self.root, 'other'), 'missing')

    def test_inactive_loop_never_records_or_notifies(self):
        for status in ('paused', 'complete', 'blocked', 'budget_limited', 'usage_limited', 'unknown', 'missing'):
            with patch.object(w, 'ROOT', self.root), patch.dict(w.os.environ, {'CODEX_HOME': str(self.root), 'CODEX_THREAD_ID': 'own'}), patch('sys.argv', ['watch', '--once']), patch.object(w, 'goal_status', return_value=status), patch.object(w, 'record') as record, patch.object(w, 'notify') as notify:
                w.main()
                record.assert_not_called()
                notify.assert_not_called()

    def test_notify_only_own_thread_without_shell(self):
        with patch.object(w.subprocess, 'run') as run:
            run.return_value.returncode = 0
            self.assertTrue(w.notify('own', ['E001']))
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ['codex', 'queue', '--thread', 'own'])
            self.assertIn('非用户新的指令', command[-1])


if __name__ == '__main__':
    unittest.main()
