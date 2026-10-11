"""Exercise the new intervention callback with actual CPU AdamW/state payloads."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from train_bfcl_source_group import SourceState,r
from bfcl_source_group_contract import NORMALIZATION,learner_progress
from bfcl_dose_checkpoint import assert_equal

class SourceRestoreTests(unittest.TestCase):
    def test_actual_restore_preserves_history_and_moments_with_declared_pool_and_seed(self):
        torch.set_num_threads(2)
        for step in [0,1]:
            with self.subTest(source_step=step),tempfile.TemporaryDirectory() as d:
                root=Path(d);source=root/'source';out=root/'branch';out.mkdir()
                registry={'a':'g1','b':'g2','c':'g1'}
                old={'num_generations':4,'num_iterations':1,'learning_rate':1e-5}
                source_algorithm=old if step==0 else old|NORMALIZATION
                source_protocol={'base_revision':'unit-test','manifest_sha256':'fixture','algorithm':source_algorithm,'training_registry':registry}
                target=source_protocol|{'algorithm':old|NORMALIZATION}
                protocol_path=root/'source_protocol.json';protocol_path.write_text(json.dumps(source_protocol))
                model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
                allocator=r.CombinationAllocator(registry,seed=17);allocator.choose(forced='a');allocator.observe('a',[1,0,1,0],10,'ancestor-probe')
                if step:
                    model(torch.ones(1,3)).sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                progress=learner_progress(0,step,10,0)
                manifest=r.save_learning_state(source/'components',model,optimizer,scheduler,allocator,progress,r.provenance(source_protocol,protocol_path))
                before_opt=copy.deepcopy(optimizer.state_dict());before_weights=copy.deepcopy(model.state_dict());before_history=copy.deepcopy(allocator.state_dict())
                new_model=torch.nn.Linear(3,2);raw=torch.optim.AdamW(new_model.parameters(),lr=1e-5);new_scheduler=torch.optim.lr_scheduler.LambdaLR(raw,lambda _:1.)
                job={'source_checkpoint':str(source),'source_payload_sha256':manifest['payload_sha256'],'source_protocol':{'path':str(protocol_path),'sha256':r.digest(protocol_path)},'source_step':step,'allow_legacy_zero_optimizer_source':step==0,'task_pool':['b'],'allocator_seed':123}
                t=SimpleNamespace(job=job,protocol=target,model=new_model,optimizer=SimpleNamespace(optimizer=raw,scaler=None),lr_scheduler=new_scheduler,_step=0,_buffered_inputs=None,accelerator=SimpleNamespace(gradient_accumulation_steps=1),evidence_dir=out)
                SourceState(t).on_train_begin(SimpleNamespace(gradient_accumulation_steps=4),SimpleNamespace(global_step=0),None)
                assert_equal(before_weights,new_model.state_dict());assert_equal(before_opt,raw.state_dict())
                for k,v in before_history.items():
                    if k not in ['pool','rule','rng_state']:assert_equal(v,t.allocator.state_dict()[k])
                self.assertEqual(t.allocator.pool,['b']);self.assertEqual(t.source_step,step);self.assertEqual(t._step,step*4)
                self.assertEqual(t.allocator.probabilities(),{'b':1.})
                self.assertEqual(t.inherited_tokens,10)

if __name__=='__main__':unittest.main()
