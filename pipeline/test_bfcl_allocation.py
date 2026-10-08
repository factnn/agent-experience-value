"""Counterfactual forks and cost-bearing visible-combination allocation checks."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from bfcl_allocation import CombinationAllocator
from rl_learning_state import save_learning_state,restore_learning_state,tensor_hash

class AllocationChecks(unittest.TestCase):
    registry={'a':'X','b':'X','c':'Y','d':'Z'}
    def history(self):
        a=CombinationAllocator(self.registry,seed=7)
        for index,(key,rewards) in enumerate([('a',[0]*4),('c',[0,1,0,1]),('d',[1]*4)]):
            a.choose(forced=key);a.observe(key,rewards,100+index,f'probe:{index}')
        return a
    def test_distinct_rules_same_paid_history(self):
        a=self.history();state=json.loads(json.dumps(a.state_dict()))
        forks={r:CombinationAllocator.from_state_dict(state,rule=r,pool=['a','b','c'])
            for r in ['uniform','frontier','coverage']}
        for r,b in forks.items():
            self.assertEqual(b.history,a.history);self.assertEqual(b.counts,a.counts)
            self.assertEqual(b.total_sampled_tokens,303)
            self.assertAlmostEqual(sum(b.probabilities().values()),1)
            self.assertEqual(set(b.probabilities()),{'a','b','c'})
        self.assertGreater(forks['frontier'].probabilities()['c'],forks['uniform'].probabilities()['c'])
        self.assertGreater(forks['coverage'].probabilities()['c'],forks['uniform'].probabilities()['c'])
        # No forced 87-task warmup or free knowledge; first selection is allocation.
        self.assertEqual(forks['frontier'].choose()['phase'],'allocation')
    def test_rng_roundtrip_and_group_boundary(self):
        a=self.history();b=CombinationAllocator.from_state_dict(json.loads(json.dumps(a.state_dict())))
        self.assertEqual(a.choose(),b.choose())
        with self.assertRaises(RuntimeError):a.state_dict()
        with self.assertRaises(ValueError):a.observe('bad',[1],1,'bad')
    def test_complete_state_restores_new_allocator_and_adam(self):
        model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        model(torch.ones(1,2)).sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        a=self.history();expected=tensor_hash(model.named_parameters())
        provenance={'base_revision':'local','task_manifest_sha256':'test','algorithm_config_sha256':'test'}
        with tempfile.TemporaryDirectory() as root:
            save_learning_state(Path(root)/'saved',model,optimizer,scheduler,a,{'global_step':1,'at_optimizer_boundary':True},provenance)
            with torch.no_grad():
                for p in model.parameters():p.add_(3)
            optimizer.state.clear()
            b,progress=restore_learning_state(Path(root)/'saved',model,optimizer,scheduler,provenance,rule='coverage')
            self.assertEqual(tensor_hash(model.named_parameters()),expected)
            self.assertTrue(optimizer.state);self.assertEqual(progress['global_step'],1)
            self.assertEqual(b.rule,'coverage');self.assertEqual(b.history,a.history)
            self.assertEqual(b.total_sampled_tokens,303)
            self.assertEqual(a.rng.getstate(),b.rng.getstate())

    def test_prepared_optimizer_restores_same_backend(self):
        from accelerate import Accelerator
        from accelerate.optimizer import AcceleratedOptimizer
        model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        provenance={'base_revision':'local','task_manifest_sha256':'test','algorithm_config_sha256':'test'}
        with tempfile.TemporaryDirectory() as root:
            saved=Path(root)/'saved'
            save_learning_state(saved,model,optimizer,scheduler,self.history(),
                {'global_step':0,'at_optimizer_boundary':True},provenance)
            wrapper=Accelerator(cpu=True).prepare(optimizer)
            self.assertIsInstance(wrapper,AcceleratedOptimizer)
            self.assertIs(wrapper.optimizer,optimizer)
            # Exactly the HF Trainer order: prepare optimizer, create scheduler,
            # on_train_begin restores the underlying state then training starts.
            new_scheduler=torch.optim.lr_scheduler.LambdaLR(wrapper,lambda _:1.)
            restored,progress=restore_learning_state(saved,model,wrapper.optimizer,new_scheduler,provenance)
            self.assertEqual(progress['global_step'],0);self.assertEqual(restored.total_sampled_tokens,303)
            model(torch.ones(1,2)).sum().backward();wrapper.step();new_scheduler.step();wrapper.zero_grad(set_to_none=True)
            self.assertTrue(wrapper.state)

if __name__=='__main__':unittest.main()
