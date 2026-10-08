import unittest
from rl_allocation import TaskAllocator


class AllocationTest(unittest.TestCase):
    def test_common_warmup_and_all_failure_costs(self):
        left=TaskAllocator(['a','b','c'],'uniform',seed=9)
        right=TaskAllocator(['a','b','c'],'frontier',seed=9)
        chosen=[]
        for i in range(3):
            a,b=left.choose(),right.choose()
            self.assertEqual(a,b);chosen.append(a['task_id'])
            for policy in [left,right]:
                policy.observe(a['task_id'],[0,0,0,0],100+i,str(i))
        self.assertEqual(set(chosen),{'a','b','c'})
        self.assertEqual(left.total_sampled_tokens,303)
        self.assertEqual(right.total_episodes,12)

    def test_frontier_prefers_mixed_and_keeps_exploration(self):
        policy=TaskAllocator(['easy','mixed','hard'],'frontier',exploration=.2)
        outcomes={'easy':[1]*16,'mixed':[0,1]*8,'hard':[0]*16}
        for i in range(3):
            choice=policy.choose();key=choice['task_id']
            policy.observe(key,outcomes[key],500,str(i))
        q=policy.probabilities()
        self.assertAlmostEqual(sum(q.values()),1)
        self.assertGreater(q['mixed'],q['easy'])
        self.assertAlmostEqual(q['easy'],q['hard'])
        self.assertTrue(all(x >= .2/3 for x in q.values()))

    def test_no_unrequested_tasks_and_no_future_observations(self):
        policy=TaskAllocator(['train'])
        with self.assertRaises(ValueError):policy.observe('test',[1],100,'x')
        policy.choose()
        with self.assertRaises(RuntimeError):policy.choose()
        with self.assertRaises(ValueError):policy.observe('train',[float('nan')],100,'x')
        policy.observe('train',[0],100,'x')
        policy.choose()
        with self.assertRaises(ValueError):policy.observe('train',[1],100,'x')
        self.assertEqual(policy.total_sampled_tokens,100)

    def test_old_policy_history_expires(self):
        policy=TaskAllocator(['train'],'frontier',window=4)
        for i,reward in enumerate([0,1]):
            policy.choose();policy.observe('train',[reward]*4,100,str(i))
        self.assertEqual(list(policy.history['train']),[1]*4)


if __name__=='__main__':unittest.main()
