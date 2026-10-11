import unittest
from bfcl_source_group_contract import NORMALIZATION,validate_source,validate_intervention,learner_progress

class SourceGroupContractTests(unittest.TestCase):
    def test_legacy_transition_only_at_declared_empty_optimizer_start(self):
        old={'num_generations':4,'num_iterations':1,'learning_rate':1e-5};new=old|NORMALIZATION
        p={'global_step':0,'micro_step':0,'at_optimizer_boundary':True}
        self.assertEqual(validate_source(old,new,p,{'state':{}},True),0)
        for args in [(old,new,p,{'state':{}},False),(old,new,p,{'state':{0:{}}},True),(old,new|{'learning_rate':2e-5},p,{'state':{}},True)]:
            with self.assertRaises(ValueError):validate_source(*args)
        p=learner_progress(0,6,100,200)
        with self.assertRaises(ValueError):validate_source(old,new,p,{'state':{0:{}}},True)
    def test_nested_corrected_sources_and_counter_mismatch(self):
        a={'num_generations':4,'num_iterations':1}|NORMALIZATION;p=learner_progress(6,3,100,200)
        self.assertEqual(validate_source(a,a,p,{'state':{0:{}}},False),9)
        for change in [{'global_step':9},{'micro_step':35},{'fresh_rollout_on_next_group':False}]:
            with self.assertRaises(ValueError):validate_source(a,a,p|change,{'state':{}},False)
    def test_whole_source_group_and_uniform_control(self):
        registry={'a':'g1','b':'g1','c':'g2'}
        job={'task_pool':['a','b'],'role':'source_group','source_group':'g1','token_budget':65536,'allocator_seed':1,'rollout_seed_base':2}
        validate_intervention(job,registry)
        for changes in [{'task_pool':['a']},{'task_pool':['a','c']},{'role':'uniform_control'},{'token_budget':0}]:
            with self.assertRaises(ValueError):validate_intervention(job|changes,registry)
        validate_intervention(job|{'task_pool':list(registry),'role':'uniform_control'},registry)

if __name__=='__main__':unittest.main()
