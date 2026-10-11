import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from bfcl_paired_fork_contract import validate_pair_boundary


class PairedForkTests(unittest.TestCase):
    def test_explicit_nested_counter_mapping_and_rejections(self):
        p={'global_step':2,'branch_local_optimizer_steps':2,'source_branch_start_step':6,
           'cumulative_learner_optimizer_steps':8,'micro_step':32,'at_optimizer_boundary':True,
           'fresh_rollout_on_next_group':True,'inherited_acquisition_tokens':165799,'branch_sampled_tokens':26337}
        algorithm={'num_generations':4,'num_iterations':1}
        self.assertEqual(validate_pair_boundary(p,algorithm),8)
        for mutation in [{'global_step':8},{'source_branch_start_step':0},{'micro_step':31},
                         {'fresh_rollout_on_next_group':False},{'branch_sampled_tokens':-1}]:
            with self.assertRaises(ValueError):validate_pair_boundary(p|mutation,algorithm)

    def test_save_does_not_reload_continuous_model_or_change_state(self):
        import torch
        from accept_bfcl_deterministic_pair import save_without_restore,r
        from bfcl_dose_checkpoint import assert_equal,rng_state
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        model(torch.ones(1,3)).sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        allocation={'fixed':'unit test'}
        allocator=SimpleNamespace(state_dict=lambda:copy.deepcopy(allocation),rule='uniform',total_sampled_tokens=165809)
        protocol=json.loads((r.ROOT/'rl/BFCL_DOSE_PROTOCOL_20261010.json').read_text())
        trainer=SimpleNamespace(model=model,optimizer=SimpleNamespace(optimizer=optimizer),lr_scheduler=scheduler,
            allocator=allocator,state=SimpleNamespace(global_step=2),source_step=6,_step=32,
            inherited_tokens=165799,sampled_tokens=10,protocol=protocol)
        weights=copy.deepcopy(model.state_dict());opt=copy.deepcopy(optimizer.state_dict());rng=rng_state()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);plan=path/'plan.json';plan.write_text(json.dumps({'source_protocol':'rl/BFCL_DOSE_PROTOCOL_20261010.json'}))
            with patch.object(r,'restore_learning_state',side_effect=AssertionError('Reference must never restore')):
                manifest=save_without_restore(trainer,path/'components',plan)
            self.assertEqual(manifest['cumulative_acquisition_tokens'],165809)
            self.assertFalse(json.loads((path/'components/save_verification.json').read_text())['continuous_model_reloaded'])
        assert_equal(weights,model.state_dict());assert_equal(opt,optimizer.state_dict());assert_equal(rng,rng_state())

    def test_pair_audit_rejects_microgradient_or_final_optimizer_corruption(self):
        import report_bfcl_deterministic_pair as r
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=root/'continuous';f=root/'restored'
            plan={'output_root':temp,'save_after_local_updates':2}
            write(root/'reference_receipt.json',{'paired_protocol_sha256':'sha','sha256':{}})
            write(c/'boundary/components/manifest.json',{'payload_sha256':'boundary','cumulative_acquisition_tokens':120})
            write(f/'restored.json',{'source_payload_sha256':'boundary'})
            from report_bfcl_stage_fork import compare_rows
            # Empty paired sections are isolated here: full case validation is separately required by check_case.
            for name in ['allocation','groups','generation_calls','rollouts','microbatches']:
                for folder in [c,f]:(folder/(name+'.jsonl')).write_text('')
            for folder,step in [(c,3),(f,1)]:
                (folder/'updates.jsonl').write_text(json.dumps({'step':step,'before_sha256':'a','after_sha256':'b','parameters_changed':True})+'\n')
                (folder/'metrics.jsonl').write_text(json.dumps({'step':step,'grad_norm':1.,'loss':0.})+'\n')
                (folder/'microgradients.jsonl').write_text(json.dumps({'local_optimizer_step':step-1,'cumulative_optimizer_step':8,'micro_step_after':33,'loss':0.,'gradient_tensor_hash':'good','gradient_tensor_count':1})+'\n')
            components={k:{} for k in ['trainable','optimizer','scheduler','allocator','buffers','module_training','rng']}
            def check(plan,case,sha):
                return ({'inherited_tokens':100 if case=='continuous' else 120,'new_sampled_tokens':30 if case=='continuous' else 10,'final_fingerprint':'b'},
                        {'new_model_tokens':30 if case=='continuous' else 10})
            with patch.object(r,'check_case',side_effect=check),patch.object(r,'payload',return_value=components):
                self.assertEqual(r.audit_pair(plan,'sha')['status'],'passed')
                p=f/'microgradients.jsonl';p.write_text(p.read_text().replace('good','bad'))
                result=r.audit_pair(plan,'sha')
                self.assertEqual(result['status'],'mismatch_requires_review')
                self.assertTrue(any(x['section']=='microgradients' for x in result['differences']))
                p.write_text(p.read_text().replace('bad','good'))
            corrupted=copy.deepcopy(components);corrupted['optimizer']={'bad':True}
            with patch.object(r,'check_case',side_effect=check),patch.object(r,'payload',side_effect=[components,corrupted]):
                result=r.audit_pair(plan,'sha')
                self.assertEqual(result['status'],'mismatch_requires_review')
                self.assertIn({'section':'final_state','fields':['optimizer']},result['differences'])


if __name__=='__main__':unittest.main()
