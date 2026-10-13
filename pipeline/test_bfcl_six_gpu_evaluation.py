"""CPU-only administrative resource checks; no model evaluation is performed."""
import copy
import json
import prepare_bfcl_source_evaluation as prepare
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bfcl_ssh_evaluation_control as control
import test_bfcl_ssh_evaluation_control as old
from supervise_bfcl_source_evaluation import verify_publication


class SixGpuEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        plan, authority = old.EvaluationOwnershipTests().fixture(root)
        ref=old.put(root,'experiments/SSH_FUTURE_GPU_CAP_V2_20261013.json',{'max_simultaneous_gpus_for_subsequent_batches':6})
        plan.update(resource_authorization=ref, allowed_gpus=[0,1,4,5,6,7],max_simultaneous_gpus=6,automatic_retry=False)
        authority.update(resource_authorization=ref,allowed_gpus=plan['allowed_gpus'],max_simultaneous_gpus=6,
                         evaluation_manifest=old.put(root,'manifest.json',plan))
        old.put(root,'rl/STATUS.json',{'bfcl_intervention':{'phase':'audited_complete','active':{},'pending':[],
                'failures':[],'scheduler_pid':0,'path':'dose'}})
        old.put(root,'dose/audit_status.json',{'status':'passed'})
        old.put(root,'experiments/reviews/Edose.json',{'status':'reviewed'})
        old.put(root,'experiments/index.json',[{'id':'Efixture','source':'training/audit_status.json'},
                                              {'id':'Edose','source':'dose/audit_status.json'}])
        return plan,authority

    def test_all_six_actual_devices_keep_old_guard_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,authority=self.fixture(root)
            with patch.object(control,'ROOT',root),patch.object(control,'resource_preflight',side_effect=AssertionError('Do not spoof old four-card guard')):
                for gpu in [0,1,4,5,6,7]:
                    with patch.object(control.subprocess,'check_output',side_effect=['GPU-test, 3','']) as smi:
                        result=control.preflight(gpu,plan,authority)
                        self.assertEqual(result['gpu'],gpu);self.assertTrue(result['no_compute_process'])
                        self.assertEqual(smi.call_args_list[0].args[0][2],str(gpu))

    def test_explicit_authority_and_cap_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,authority=self.fixture(root)
            for change in [dict(resource_authorization=None),dict(allowed_gpus=[4,5,6,7]),dict(max_simultaneous_gpus=7)]:
                with self.assertRaises(ValueError):control.check_control(root,plan,authority|change)
            wrong=copy.deepcopy(plan);wrong['automatic_retry']=True
            wrong_authority=authority|{'evaluation_manifest':old.put(root,'manifest.json',wrong)}
            with self.assertRaises(ValueError):control.check_control(root,wrong,wrong_authority)
            old.put(root,'experiments/SSH_FUTURE_GPU_CAP_V2_20261013.json',{'max_simultaneous_gpus_for_subsequent_batches':4})
            with self.assertRaises(ValueError):control.check_control(root,plan,authority)

    def test_live_scheduler_unreviewed_dose_and_occupancy_fail_closed(self):
        for condition in ['scheduler','audit','review','memory','compute']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,authority=self.fixture(root)
                if condition=='audit':old.put(root,'dose/audit_status.json',{'status':'failed'})
                if condition=='review':old.put(root,'experiments/reviews/Edose.json',{'status':'pending'})
                with patch.object(control,'ROOT',root),patch.object(control,'process_has_not_exited',return_value=condition=='scheduler'),patch.object(control.subprocess,'check_output',side_effect=['GPU-test, '+('65' if condition=='memory' else '3'),'GPU-test' if condition=='compute' else '']):
                    with self.assertRaises(RuntimeError,msg=condition):control.preflight(0,plan,authority)

    def test_prepare_preserves_all_80_panels_seeds_and_budget_under_six_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            entries=[];registry={'common':{'role':'baseline'}}
            for i in range(51):
                case='case_'+str(i);registry[case]={'role':'fit' if i<39 else 'heldout'}
                entries.append(old.put(root,'jobs/'+case+'.json',{'case_id':case,'output':'policies/'+case,'input_sha256':{}})|{'name':case})
            for case in registry:
                folder='policies/'+case
                old.put(root,folder+'/summary.json',{'status':'complete','final_fingerprint':case})
                old.put(root,folder+'/audit.json',{'status':'passed'})
                old.put(root,folder+'/adapter/adapter_config.json',{})
                old.put(root,folder+'/adapter/adapter_model.safetensors',{})
            for name in prepare.EVAL_CODE:old.put(root,'pipeline/'+name,{'synthetic_code':True})
            old.put(root,'training/audit_status.json',{'status':'passed','complete_jobs':51})
            old.put(root,'experiments/index.json',[{'id':'Eall','source':'training/audit_status.json'}])
            old.put(root,'experiments/reviews/Eall.json',{'status':'reviewed'})
            master=old.put(root,'master.json',{'jobs':entries,'output_root':'training','allowed_gpus':[4,5,6,7],
                'required_review_events':[],'input_sha256':{},'analysis_contract':old.put(root,'analysis.json',{}),
                'source_stage0':{'path':'policies/common','policy_fingerprint':'common'},
                'evaluation':{'task_ids':['a','b'],'task_seed_vectors':[[1,2],[3,4]],'hard_seconds_per_panel':77400}})
            resource=old.put(root,'experiments/SSH_FUTURE_GPU_CAP_V2_20261013.json',{'max_simultaneous_gpus_for_subsequent_batches':6})
            with patch.object(prepare,'source_registry',return_value=registry),patch.object(prepare,'load_job'):
                four=prepare.prepare(root,master,root/'reg4',root/'out4','fit')
                six=prepare.prepare(root,master,root/'reg6',root/'out6','fit',resource_authorization=resource)
            self.assertEqual(len(four['jobs']),80);self.assertEqual(len(six['jobs']),80)
            self.assertEqual(six['max_simultaneous_gpus'],6)
            self.assertEqual(six['hard_seconds_per_panel'],four['hard_seconds_per_panel'])
            for first,second in zip(four['jobs'],six['jobs']):
                a=json.loads((root/first['path']).read_text());b=json.loads((root/second['path']).read_text())
                a.pop('output');b.pop('output');self.assertEqual(a,b)
            allocated=[n for lane in six['gpu_lanes'].values() for n in lane]
            self.assertEqual(len(allocated),80);self.assertEqual(len(set(allocated)),80)

    def test_resource_record_must_be_in_verified_remote_publication(self):
        with tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryDirectory() as remote:
            root=Path(tmp);plan,authority=self.fixture(root)
            # Publication test uses no child jobs; it only exercises signed file inclusion.
            plan=plan|{'jobs':[]};ref=old.put(root,'publication_manifest.json',plan)
            authority_ref=old.put(root,'authority.json',authority)
            subprocess.run(['git','init','-q'],cwd=root,check=True)
            subprocess.run(['git','init','--bare','-q',remote],check=True)
            subprocess.run(['git','remote','add','origin',remote],cwd=root,check=True)
            def commit(paths):
                subprocess.run(['git','add','--',*paths],cwd=root,check=True)
                subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','Synthetic resource publication'],cwd=root,check=True)
                return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            first=commit(['publication_manifest.json','authority.json','goal.json','training/audit_status.json','experiments/reviews/Efixture.json','pipeline/control.py'])
            with self.assertRaises(ValueError):verify_publication(root,root/ref['path'],ref['sha256'],plan,first,authority_ref)
            published=commit([authority['resource_authorization']['path']])
            subprocess.run(['git','push','-q','origin','HEAD:refs/heads/main'],cwd=root,check=True)
            self.assertEqual(verify_publication(root,root/ref['path'],ref['sha256'],plan,published,authority_ref)['remote_main'],published)

if __name__=='__main__':unittest.main()
