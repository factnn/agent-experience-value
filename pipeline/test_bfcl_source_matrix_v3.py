import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import report_bfcl_source_matrix as matrix
from test_bfcl_source_matrix import MatrixFinalizationTests, put


class ThreeSegmentTests(unittest.TestCase):
    def fixture(self, root):
        data=MatrixFinalizationTests().fixture(root)
        ref,master,v2_ref,v2,original,_,_,_,_,audits=data
        put(root,'original/supervisor.json',{'status':'incomplete_requires_review','complete_jobs':24,'pending_jobs':original['pending']})
        pending=original['pending'][12:]
        second={'complete':original['pending'][:12],'pending':pending,'active':{},'failures':[
            {'case':pending[0],'gpu':6,'type':'RuntimeError','message':'Goal explicitly unavailable for work: blocked'}]}
        put(root,'resumed/queue.json',second)
        put(root,'resumed/supervisor.json',{'status':'incomplete_requires_review','complete_jobs':12,'pending_jobs':pending})
        put(root,'resumed/failures.json',second['failures'])
        (root/'resumed/audit_status.json').unlink()
        v3=copy.deepcopy(v2)
        v3.update(output_root='final',jobs=master['jobs'][36:],gpu_lanes={g:[n for n in lane if n in pending] for g,lane in master['gpu_lanes'].items()})
        v3['execution_continuation'].update(completed_cases_retained=original['complete']+second['complete'],pending_cases_only=pending,
            previous_control_segments=[{'protocol':ref,'output_root':'original','complete_jobs':24},
                                       {'protocol':v2_ref,'output_root':'resumed','complete_jobs':12}])
        v3_ref=put(root,'v3.json',v3)
        put(root,'final/queue.json',{'complete':pending,'pending':[],'active':{},'failures':[]})
        put(root,'final/supervisor.json',{'status':'audited_complete','complete_jobs':15,'pending_jobs':[]})
        put(root,'final/launch.json',{'supervisor_pid':0})
        put(root,'final/audit_status.json',{'status':'passed','complete_jobs':15,'jobs':[audits[n] for n in pending],'new_sampled_tokens':15*65536})
        return ref,v3_ref,audits

    def test_all_51_replayed_once_and_all_failures_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref,v3,audits=self.fixture(root)
            paths=['original/failures.json','resumed/failures.json','failed/failures.json']
            before={p:(root/p).read_bytes() for p in paths}
            with patch.object(matrix,'audit_source',side_effect=lambda r,j,s:audits[j['case_id']]) as replay:
                result=matrix.finalize(root,ref,v3)
            self.assertEqual(replay.call_count,51)
            self.assertEqual(result['complete_jobs'],51)
            self.assertEqual(result['new_sampled_tokens'],51*65536)
            self.assertEqual(result['execution_recovery']['retained_jobs'],36)
            self.assertEqual(result['execution_recovery']['continued_jobs'],15)
            for p in paths:self.assertEqual((root/p).read_bytes(),before[p])
            with self.assertRaises(FileExistsError):matrix.finalize(root,ref,v3)

    def test_partial_final_queue_prevents_replay_and_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref,v3,_=self.fixture(root)
            queue=matrix.read(root/'final/queue.json')
            queue['active']={'4':queue['complete'].pop()}
            put(root,'final/queue.json',queue)
            with patch.object(matrix,'audit_source') as replay,self.assertRaises(ValueError):
                matrix.finalize(root,ref,v3)
            replay.assert_not_called()
            self.assertFalse((root/'original/audit_status.json').exists())

    def test_prior_scientific_failure_or_overlap_cannot_be_relabelled_control(self):
        for overlap in [False,True]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);ref,v3,_=self.fixture(root)
                queue=matrix.read(root/'resumed/queue.json')
                if overlap:queue['complete'][0]=matrix.read(root/'original/queue.json')['complete'][0]
                else:queue['failures'][0]['message']='Training failed'
                put(root,'resumed/queue.json',queue)
                with patch.object(matrix,'audit_source') as replay,self.assertRaises(ValueError):matrix.finalize(root,ref,v3)
                replay.assert_not_called()

    def test_final_batch_audit_substitution_prevents_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref,v3,audits=self.fixture(root)
            batch=matrix.read(root/'final/audit_status.json')
            batch['jobs'][0]=audits['case_0']
            put(root,'final/audit_status.json',batch)
            with patch.object(matrix,'audit_source') as replay,self.assertRaises(ValueError):matrix.finalize(root,ref,v3)
            replay.assert_not_called()


if __name__=='__main__':unittest.main()
