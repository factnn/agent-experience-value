"""CPU checks for stop ownership, handoff exclusivity and real resource limits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bfcl_ssh_resource_control as control

class ResourceControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.failure = {'case':'a','gpu':4,'type':'RuntimeError','message':control.STOP_MESSAGE}
        self.queue = {'complete':[],'active':{'4':'old'},'pending':['a','b'],'failures':[self.failure]}
        self.jobs = [self.put('jobs/'+n+'.json',{'output':'train/'+n}) | {'name':n} for n in ['a','b']]
        self.master = {'jobs':self.jobs,'input_sha256':{},'output_root':'master'}
        self.master_ref = self.put('master.json',self.master)
        prior_ref = self.put('prior.json',{'output_root':'prior'})
        self.put('prior/queue.json',self.queue); self.put('prior/failures.json',[self.failure])
        snapq = self.put('snapshot/queue.json',self.queue); snapf = self.put('snapshot/failures.json',[self.failure])
        handoff = {'status':'authorized_transfer','original_scientific_protocol':self.master_ref,
            'source_protocol':prior_ref,'source_output_root':'prior','source_stop_file':'experiments/.runtime/global',
            'transferred_pending':['a','b'],'source_snapshot':{'queue':snapq,'failures':snapf},
            'admission_stop_failure':self.failure}
        self.plan = {'allowed_gpus':[0,1],'max_simultaneous_gpus':2,'automatic_retry':False,
            'jobs':self.jobs,'input_sha256':{},'gpu_lanes':{'0':['a'],'1':['b']},
            'execution_continuation':{'original_scientific_protocol':self.master_ref,
                'previous_control_segments':[{'protocol':self.master_ref},{'protocol':self.master_ref},{'protocol':prior_ref}],
                'handoff_receipt':self.put('handoff.json',handoff)}}
        self.authority = {'mode':'human_authorized_ssh_takeover','scope':'source_stage0_resource_continuation',
            'status':'authorized','allowed_gpus':[0,1],'max_simultaneous_gpus':2,'aggregate_max_simultaneous_gpus':6,
            'stop_file':'experiments/.runtime/phase','global_stop_file':'experiments/.runtime/global',
            'training_protocol':self.put('plan.json',self.plan),'control_input_sha256':{},
            'owner_goal':{'thread_id':'current','record':self.put('goal.json',{'thread_id':'current','status_at_registration':'active'})},
            'resource_authorization':self.put('resource.json',{'goal_thread_id':'current','goal_status_at_registration':'active','new_gpus':[0,1],'max_total_parallel_jobs':6})}
        self.put('rl/STATUS.json',{'bfcl_intervention':{'phase':'audited_complete','active':{},'pending':[],'failures':[], 'scheduler_pid':123,'path':'dose'}})
        self.put('dose/audit_status.json',{'status':'passed'})
        self.put('experiments/index.json',[{'source':'dose/audit_status.json','id':'E001'}])
        self.put('experiments/reviews/E001.json',{'status':'reviewed'})
    def put(self,path,data):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
        return {'path':path,'sha256':control.digest(p)}
    def test_valid_and_prior_active_can_finish(self):
        control.check_control(self.root,self.plan,self.authority,require_unstarted=True)
        q=copy.deepcopy(self.queue);q['active']={};q['complete']=['old'];self.put('prior/queue.json',q)
        control.check_control(self.root,self.plan,self.authority)
    def test_current_global_stop_is_never_ignored(self):
        self.put(self.authority['global_stop_file'],{'reason':'user stop'})
        with self.assertRaisesRegex(RuntimeError,'stopped'):
            control.check_control(self.root,self.plan,self.authority)
    def test_transferred_child_evidence_rejected(self):
        self.put('prior/a.console.process.json',{'pid':123})
        with self.assertRaisesRegex(ValueError,'already launched'):
            control.check_control(self.root,self.plan,self.authority)
    def test_snapshot_without_real_latch_rejected(self):
        q=copy.deepcopy(self.queue);q['failures']=[];self.put('prior/queue.json',q);self.put('prior/failures.json',[])
        with self.assertRaisesRegex(ValueError,'changed ownership'):
            control.check_control(self.root,self.plan,self.authority)
    def test_prior_cannot_launch_transferred_case(self):
        q=copy.deepcopy(self.queue);q['active']['5']='a';self.put('prior/queue.json',q)
        with self.assertRaisesRegex(ValueError,'changed ownership'):
            control.check_control(self.root,self.plan,self.authority)
    def test_exclusive_initial_outputs_but_later_own_outputs_allowed(self):
        self.put('train/a/run.json',{'pid':123})
        with self.assertRaisesRegex(ValueError,'output already exists'):
            control.check_control(self.root,self.plan,self.authority,require_unstarted=True)
        control.check_control(self.root,self.plan,self.authority)
    def test_released_original_lane_is_allowed(self):
        self.plan['allowed_gpus']=[0,1,4];self.plan['max_simultaneous_gpus']=3
        self.plan['gpu_lanes']={'0':['a'],'1':[],'4':['b']}
        self.authority['allowed_gpus']=[0,1,4];self.authority['max_simultaneous_gpus']=3
        self.authority['training_protocol']=self.put('plan.json',self.plan)
        control.check_control(self.root,self.plan,self.authority)
    def test_unapproved_original_gpu_rejected(self):
        self.plan['allowed_gpus']=[0,1,2];self.plan['max_simultaneous_gpus']=3
        self.authority['allowed_gpus']=[0,1,2];self.authority['max_simultaneous_gpus']=3
        self.authority['training_protocol']=self.put('plan.json',self.plan)
        with self.assertRaisesRegex(ValueError,'newly authorized'):
            control.check_control(self.root,self.plan,self.authority)
    def test_wrong_owner_rejected(self):
        self.authority['owner_goal']['thread_id']='old'
        with self.assertRaisesRegex(ValueError,'Goal'):
            control.check_control(self.root,self.plan,self.authority)
    def flight(self,outputs,live=None):
        with patch.object(control,'ROOT',self.root), patch.object(control,'process_has_not_exited',return_value=False), patch.object(control,'owned_training_pids',return_value=live or []), patch.object(control.subprocess,'check_output',side_effect=outputs):
            return control.preflight(0,self.plan,self.authority)
    def test_zero_utilization_is_not_an_empty_gpu(self):
        with self.assertRaisesRegex(RuntimeError,'GPU occupied'):
            self.flight(['gpu0, 13106',''])
        with self.assertRaisesRegex(RuntimeError,'GPU occupied'):
            self.flight(['gpu0, 3','gpu0\n'])
        self.assertTrue(self.flight(['gpu0, 3',''])['no_compute_process'])
    def test_aggregate_cap_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'aggregate'):
            self.flight([],live=[{'case':str(i),'pid':i,'gpu':str(i)} for i in range(6)])

if __name__ == '__main__':
    unittest.main()
