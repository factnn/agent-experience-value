"""Stochastic AdamW continuation and allocation-fork acceptance on CPU."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from rl_allocation import TaskAllocator
from rl_learning_state import save_learning_state,restore_learning_state,tensor_hash

PROVENANCE={'base_revision':'cpu-acceptance-v1','task_manifest_sha256':'a'*64,
            'algorithm_config_sha256':'b'*64}


class Learner(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Linear(3,3,bias=False)
        self.base.weight.requires_grad_(False)
        self.head=torch.nn.Linear(3,1);self.dropout=torch.nn.Dropout(.25)
        self.register_buffer('scale',torch.tensor(1.25))
    def forward(self,x):return self.head(self.dropout(self.base(x)))*self.scale


def components():
    torch.manual_seed(42);np.random.seed(43);random.seed(44)
    model=Learner();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.02)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.8)
    return model,optimizer,scheduler,TaskAllocator(['a','b'],seed=5)


def update(model,optimizer,scheduler,allocator):
    choice=allocator.choose()
    x=torch.randn(4,3)+float(np.random.rand())+random.random()
    loss=model(x).square().mean();loss.backward();optimizer.step();scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    rewards=[0,1] if choice['task_id']=='a' else [0,0]
    allocator.observe(choice['task_id'],rewards,17+int(choice['task_id']=='b'),allocator.selection_count)
    model.scale.add_(.01)
    return {'choice':choice,'loss':float(loss),'x':x.clone(),
            'parameters':[p.detach().clone() for p in model.parameters()]}


class LearningStateTests(unittest.TestCase):
    def checkpoint(self,path):
        parts=components()
        for _ in range(2):update(*parts)
        metadata=save_learning_state(path,*parts,
            progress={'global_step':2,'trainer_micro_step':8,'at_optimizer_boundary':True},
            provenance=PROVENANCE)
        return parts,metadata

    def test_exact_stochastic_adam_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint';parts,metadata=self.checkpoint(path)
            expected=update(*parts)
            model,opt,scheduler,_=components()
            model.eval()  # Restore must recover dropout/training mode too.
            allocator,progress=restore_learning_state(path,model,opt,scheduler,PROVENANCE)
            actual=update(model,opt,scheduler,allocator)
            self.assertEqual(progress['global_step'],2)
            self.assertEqual(expected['choice'],actual['choice'])
            self.assertEqual(expected['loss'],actual['loss'])
            self.assertTrue(torch.equal(expected['x'],actual['x']))
            for a,b in zip(expected['parameters'],actual['parameters']):self.assertTrue(torch.equal(a,b))
            self.assertEqual(parts[2].state_dict(),scheduler.state_dict())
            self.assertEqual(parts[3].state_dict(),allocator.state_dict())
            self.assertTrue(metadata['frozen_base_sha256'])

    def test_weights_only_does_not_reproduce_adam_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint';_,_=self.checkpoint(path)
            model,opt,scheduler,_=components()
            allocator,_=restore_learning_state(path,model,opt,scheduler,PROVENANCE)
            expected=update(model,opt,scheduler,allocator)
            allocator,_=restore_learning_state(path,model,opt,scheduler,PROVENANCE)
            # Remove only optimizer moments, preserving restored LR/RNG/model.
            opt.state.clear()
            actual=update(model,opt,scheduler,allocator)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(expected['parameters'],actual['parameters'])))

    def test_rules_fork_from_identical_state_and_paid_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint';_,_=self.checkpoint(path);forks=[]
            for rule in ['uniform','frontier']:
                model,opt,scheduler,_=components()
                allocator,progress=restore_learning_state(path,model,opt,scheduler,PROVENANCE,rule=rule)
                forks.append((tensor_hash(model.named_parameters()),allocator,copy.deepcopy(opt.state_dict())))
            self.assertEqual(forks[0][0],forks[1][0])
            self.assertEqual(forks[0][1].history,forks[1][1].history)
            self.assertEqual(forks[0][1].total_sampled_tokens,35)
            self.assertEqual(forks[0][1].total_sampled_tokens,forks[1][1].total_sampled_tokens)
            self.assertNotEqual(forks[0][1].probabilities(),forks[1][1].probabilities())
            for a,b in zip(forks[0][2]['state'].values(),forks[1][2]['state'].values()):
                for k in a:self.assertTrue(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k])
            choice=forks[0][1].choose();forks[0][1].observe(choice['task_id'],[1],99,'branch-a')
            self.assertEqual(forks[1][1].total_sampled_tokens,35)

    def test_registry_provenance_and_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint';_,_=self.checkpoint(path)
            model,opt,scheduler,_=components()
            with self.assertRaisesRegex(ValueError,'provenance'):
                restore_learning_state(path,model,opt,scheduler,PROVENANCE|{'base_revision':'other'})
            reverse=torch.optim.AdamW(list(reversed([p for p in model.parameters() if p.requires_grad])),lr=.02)
            with self.assertRaisesRegex(ValueError,'parameter-group'):
                restore_learning_state(path,model,reverse,scheduler,PROVENANCE)
            with torch.no_grad():model.base.weight.add_(1)
            with self.assertRaisesRegex(ValueError,'frozen base'):
                restore_learning_state(path,model,opt,scheduler,PROVENANCE)
            payload=path/'learning_state.pt';payload.write_bytes(payload.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'checksum'):
                restore_learning_state(path,model,opt,scheduler,PROVENANCE)

    def test_boundary_and_allocator_json_roundtrip(self):
        model,opt,scheduler,allocator=components();allocator.choose()
        with self.assertRaises(RuntimeError):allocator.state_dict()
        unfinished=TaskAllocator(['a','b'])
        with self.assertRaises(ValueError):TaskAllocator.from_state_dict(unfinished.state_dict(),rule='frontier')
        model,opt,scheduler,allocator=components()
        for _ in range(2):update(model,opt,scheduler,allocator)
        recovered=TaskAllocator.from_state_dict(json.loads(json.dumps(allocator.state_dict())))
        self.assertEqual(allocator.choose(),recovered.choose())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'optimizer update'):
                save_learning_state(Path(directory)/'bad',model,opt,scheduler,recovered,
                    {'at_optimizer_boundary':False},PROVENANCE)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LearningStateTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'scope':'CPU stochastic learner component acceptance, not full Qwen3/TRL resume',
        'tests':result.testsRun,'passed':result.wasSuccessful(),
        'verified':['AdamW moments and scheduler continuation','Python/NumPy/Torch RNG continuation',
            'frozen base and mutable buffer restoration','same-state uniform/frontier forks',
            'training history and cumulative cost preservation','branch isolation',
            'payload/provenance and boundary rejection'],
        'remaining':['Qwen3 LoRA/Accelerate resume integration','trainer microbatch boundary and cached input reset',
                     'common-checkpoint online sampling/update audit']}
    (Path(__file__).resolve().parents[1]/'rl/learning_state_cpu_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
