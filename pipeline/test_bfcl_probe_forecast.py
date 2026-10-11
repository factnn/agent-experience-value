import copy
import unittest
from bfcl_probe_forecast import beta_mixed,forecast,score

class ForecastTests(unittest.TestCase):
    def test_beta_predictive_known_cases(self):
        self.assertAlmostEqual(beta_mixed(0,0),.6)
        self.assertAlmostEqual(beta_mixed(0,4),beta_mixed(4,4))
        self.assertLess(beta_mixed(0,4),beta_mixed(2,4))
    def test_heterogeneous_tasks_and_disjoint_validation(self):
        a=[dict(task_id=t,source_group='g',stage=0,policy_fingerprint='f',draw_id='A:'+t,rewards=[v]*4) for t,v in [('easy',1),('hard',0)]]
        f=forecast(a,{'successes_in_window':0,'observed_episodes':0})
        b=copy.deepcopy(a)
        for row in b:row['draw_id']='B:'+row['task_id']
        result=score(f,b)
        self.assertLess(result['primary_difference'],0)
        self.assertEqual(f['task_level_statistics']['within_uncertainty_unbiased'],0)
        with self.assertRaises(ValueError):score(f,a)
        with self.assertRaises(ValueError):score(f,b[:1])

if __name__=='__main__':unittest.main()
