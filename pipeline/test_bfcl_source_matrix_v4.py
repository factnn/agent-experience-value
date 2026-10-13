"""Synthetic lineage tests; mocked raw auditor is not a scientific 51-job audit."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from bfcl_source_matrix_v4 import build_report_v4, checked_handoff
import test_bfcl_source_matrix_v3 as fixtures
from test_bfcl_source_matrix import put
from supervise_bfcl_stage_fork import read


class FourSegmentTests(unittest.TestCase):
    def fixture(self, root, transfer_count=3, released_lane=False):
        ref, v3_ref, audits = fixtures.ThreeSegmentTests().fixture(root)
        master, v3 = read(root/ref['path']), read(root/v3_ref['path'])
        names = [e['name'] for e in v3['jobs']]
        transferred = names[-transfer_count:]
        complete = names[:-transfer_count]
        failures = [{'case': transferred[0], 'gpu': 4, 'type': 'RuntimeError', 'message': 'SSH owner stopped further job admission'}]
        snapshot = {'complete': complete[:-2], 'active': {'4':complete[-2], '5':complete[-1]}, 'pending': transferred, 'failures': failures}
        receipt = {'status':'authorized_transfer', 'original_scientific_protocol':ref, 'source_protocol':v3_ref, 'source_output_root':'final',
            'transferred_pending':transferred, 'source_snapshot':{'queue':put(root,'handoff/queue.json',snapshot), 'failures':put(root,'handoff/failures.json',failures)},
            'admission_stop_failure':failures[0], 'no_child_evidence':{c:{'original_output_absent':True,'v3_process_receipt_absent':True,'v3_run_receipt_absent':True,'v3_console_absent':True,'original_job':next(e for e in master['jobs'] if e['name']==c)} for c in transferred}}
        handoff_ref = put(root,'handoff/receipt.json',receipt)
        put(root,'final/queue.json',{'complete':complete,'active':{},'pending':transferred,'failures':failures})
        put(root,'final/failures.json',failures)
        put(root,'final/supervisor.json',{'status':'incomplete_requires_review','complete_jobs':len(complete),'pending_jobs':transferred})
        (root/'final/audit_status.json').unlink()
        v3_auth_ref = put(root,'rl/BFCL_SOURCE_STAGE0_SSH_AUTHORITY_20261013.json',{'training_protocol':v3_ref,'mode':'human_authorized_ssh_takeover','control_input_sha256':{}})
        put(root,'final/launch.json',{'supervisor_pid':0,'control_authority_sha256':v3_auth_ref['sha256']})
        v4 = copy.deepcopy(v3)
        v4.update(output_root='fourth',jobs=[e for e in master['jobs'] if e['name'] in transferred],allowed_gpus=[0,1],max_simultaneous_gpus=2,
            gpu_lanes={'0':transferred[::2],'1':transferred[1::2]})
        v4['execution_continuation'].update(handoff_receipt=handoff_ref,pending_cases_only=transferred,
            completed_cases_retained=v3['execution_continuation']['completed_cases_retained']+snapshot['complete'],
            previous_control_segments=v3['execution_continuation']['previous_control_segments']+[{'protocol':v3_ref,'output_root':'final'}])
        if released_lane:
            v4.update(allowed_gpus=[0,1,4],max_simultaneous_gpus=3,gpu_lanes={'0':[transferred[0]],'1':[transferred[1]],'4':[transferred[2]]})
        v4_ref=put(root,'v4.json',v4)
        auth_ref=put(root,'v4_authority.json',{'training_protocol':v4_ref,'mode':'human_authorized_ssh_takeover','control_input_sha256':{},'allowed_gpus':v4['allowed_gpus'],'max_simultaneous_gpus':v4['max_simultaneous_gpus'],'aggregate_max_simultaneous_gpus':6})
        put(root,'fourth/launch.json',{'supervisor_pid':0,'control_authority_path':auth_ref['path'],'control_authority_sha256':auth_ref['sha256']})
        put(root,'fourth/queue.json',{'complete':transferred,'pending':[],'active':{},'failures':[]})
        put(root,'fourth/supervisor.json',{'status':'audited_complete','complete_jobs':len(transferred),'pending_jobs':[]})
        put(root,'fourth/audit_status.json',{'status':'passed','complete_jobs':len(transferred),'jobs':[audits[n] for n in transferred],'new_sampled_tokens':len(transferred)*65536})
        for plan, cases in [(v3,complete),(v4,transferred)]:
            entries={e['name']:e for e in plan['jobs']}
            for gpu,lane in plan['gpu_lanes'].items():
                for idx,case in enumerate(lane):
                    if case not in cases:continue
                    out=read(root/entries[case]['path'])['output']
                    put(root,plan['output_root']+'/'+case+'.console.process.json',{'pid':0,'started_unix':400 if released_lane and plan is v4 and gpu=='4' else idx*100})
                    put(root,out+'/run.json',{'case_id':case,'job_sha256':entries[case]['sha256'],'gpu':gpu,'pid':0})
                    put(root,out+'/supervisor.json',{'returncode':0,'hard_timeout':False,'wall_seconds':50})
        return ref,v4_ref,audits

    def test_variable_v3_completion_and_exact_51_single_count(self):
        for transferred,released in [(2,False),(3,False),(3,True)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);ref,v4,audits=self.fixture(root,transferred,released)
                before=(root/'final/failures.json').read_bytes()
                replay=Mock(side_effect=lambda r,j,s:audits[j['case_id']])
                result=build_report_v4(root,ref,v4,replay)
                self.assertEqual(replay.call_count,51)
                self.assertEqual(result['new_sampled_tokens'],51*65536)
                self.assertEqual(result['execution_recovery']['segment_complete_counts'],[24,12,15-transferred,transferred])
                self.assertEqual((root/'final/failures.json').read_bytes(),before)
                self.assertFalse((root/'final/audit_status.json').exists())

    def test_started_transfer_or_fake_v3_batch_or_overlap_refused(self):
        for mutation in ['child','batch','overlap','science_failure','wrong_gpu','missing_live_completion','resource_overlap']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);ref,v4,_=self.fixture(root)
                plan=read(root/v4['path']);case=plan['jobs'][0]['name']
                if mutation=='child':put(root,'final/'+case+'.console.process.json',{'pid':0})
                if mutation=='batch':put(root,'final/audit_status.json',{'status':'passed'})
                if mutation=='overlap':
                    q=read(root/'fourth/queue.json');q['complete'][0]=read(root/'final/queue.json')['complete'][0];put(root,'fourth/queue.json',q)
                if mutation=='science_failure':
                    q=read(root/'final/queue.json');q['failures'][0]['message']='Training failed';put(root,'final/queue.json',q)
                if mutation=='wrong_gpu':
                    out=read(root/plan['jobs'][0]['path'])['output'];run=read(root/out/'run.json');run['gpu']='2';put(root,out+'/run.json',run)
                if mutation=='missing_live_completion':
                    q=read(root/'final/queue.json');q['active']={'4':q['complete'].pop()};put(root,'final/queue.json',q)
                if mutation=='resource_overlap':
                    other=plan['jobs'][2]['name'];proc=read(root/'fourth'/(other+'.console.process.json'));proc['started_unix']=0;put(root,'fourth/'+other+'.console.process.json',proc)
                replay=Mock()
                with self.assertRaises(ValueError,msg=mutation):build_report_v4(root,ref,v4,replay)
                replay.assert_not_called()

    def test_sealed_no_child_fields_and_original_job_identity_required(self):
        for field in ['original_output_absent', 'v3_process_receipt_absent',
                      'v3_run_receipt_absent', 'v3_console_absent', 'original_job']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);ref,v4_ref,_=self.fixture(root)
                v4=read(root/v4_ref['path']); recovery=v4['execution_continuation']
                receipt=read(root/recovery['handoff_receipt']['path'])
                record=receipt['no_child_evidence'][receipt['transferred_pending'][0]]
                record[field] = {'name':'substituted'} if field=='original_job' else False
                recovery['handoff_receipt']=put(root,'handoff/receipt.json',receipt)
                v3_ref=recovery['previous_control_segments'][-1]['protocol']
                with self.assertRaises(ValueError,msg=field):
                    checked_handoff(root,ref,v3_ref,read(root/v3_ref['path']),v4)

    def test_fresh_saved_disagreement_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref,v4,audits=self.fixture(root)
            bad=copy.deepcopy(audits);bad['case_0']['cost']['new_model_tokens']+=1
            with self.assertRaises(ValueError):build_report_v4(root,ref,v4,lambda r,j,s:bad[j['case_id']])

if __name__=='__main__':unittest.main()
