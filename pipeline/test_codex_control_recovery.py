import unittest
from recover_codex_goal_when_idle import action, incident_paths


class IdleRecoveryTests(unittest.TestCase):
    def test_distinct_incidents_preserve_old_records(self):
        self.assertEqual(incident_paths(None), ('control_recovery.json', 'experiments/control_recovery_receipt.json'))
        self.assertNotEqual(incident_paths(None), incident_paths('20261011_1626'))
        for invalid in ['../old', '', 'space name']:
            with self.assertRaises(ValueError):incident_paths(invalid)

    def test_never_interrupt_active_or_unknown_sessions(self):
        for s in [{'sqlite_522': True, 'all_idle': False},
                  {'sqlite_522': False, 'all_idle': True},
                  {'sqlite_522': True}]:
            self.assertEqual(action(s, 10, False), 'wait')

    def test_two_idle_observations_and_one_attempt_only(self):
        state = {'sqlite_522': True, 'all_idle': True}
        self.assertEqual(action(state, 1, False), 'wait')
        self.assertEqual(action(state, 2, False), 'restart_once')
        self.assertEqual(action(state, 2, True), 'wait_after_single_attempt')
        self.assertEqual(action({'goal_query_ok': True, 'goal_status': 'paused'}, 2, True), 'healthy')


if __name__ == '__main__':
    unittest.main()
