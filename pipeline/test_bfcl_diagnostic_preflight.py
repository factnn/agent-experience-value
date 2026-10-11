import unittest
from bfcl_diagnostic_preflight import require_active


class GoalReadTests(unittest.TestCase):
    def test_transient_unknown_rechecks_without_assuming_authority(self):
        values=iter(['unknown','active']);sleeps=[]
        self.assertEqual(require_active(lambda:next(values),sleeps.append),['unknown','active'])
        self.assertEqual(sleeps,[5])

    def test_explicit_pause_never_retried(self):
        sleeps=[]
        with self.assertRaisesRegex(RuntimeError,'paused'):require_active(lambda:'paused',sleeps.append)
        self.assertEqual(sleeps,[])

    def test_persistent_unknown_fails_closed(self):
        sleeps=[]
        with self.assertRaisesRegex(RuntimeError,'three read attempts'):require_active(lambda:'unknown',sleeps.append)
        self.assertEqual(sleeps,[5,5])


if __name__=='__main__':unittest.main()
