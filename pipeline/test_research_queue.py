import unittest
from unittest.mock import patch
from types import SimpleNamespace
from research_queue import enqueue
from recover_research_queue_when_idle import eligible


class QueueTests(unittest.TestCase):
    def test_storage_failure_is_recorded_and_not_success(self):
        with patch('research_queue.subprocess.run', return_value=SimpleNamespace(
                returncode=1, stdout=b'', stderr=b'queue storage failed: (code: 522) secret')):
            receipt = enqueue('fixture-thread', 'fixture-message')
        self.assertFalse(receipt['queued'])
        self.assertEqual(receipt['error'], 'queue_storage_sqlite_522')
        self.assertNotIn('secret', str(receipt))

    def test_restart_requires_active_goal_and_two_all_idle_observations_once(self):
        state = {'goal_query_ok': True, 'goal_status': 'active', 'all_idle': True}
        delivery = {'error': 'queue_storage_sqlite_522'}
        self.assertTrue(eligible(state, delivery, 2, False))
        for changed, d, n, tried in [(state | {'all_idle': False}, delivery, 2, False),
                                   (state | {'goal_status': 'paused'}, delivery, 2, False),
                                   (state | {'goal_query_ok': False}, delivery, 2, False),
                                   (state, {'error': 'queue_command_failed'}, 2, False),
                                   (state, delivery, 1, False), (state, delivery, 2, True)]:
            self.assertFalse(eligible(changed, d, n, tried))


if __name__ == '__main__':
    unittest.main()
