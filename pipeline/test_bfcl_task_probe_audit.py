import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import report_bfcl_task_probe as audit
from bfcl_probe_forecast import forecast, score
from bfcl_task_probe_statistics import compare_independent_panels

class ProbeAuditTests(unittest.TestCase):
    def fixture(self,root):
        case={'name':'g0','source_group':'g','task_ids':['easy','hard'],'seeds':{'feature':[101,102],'validation':[201,202]},'historical_signal':{'successes_in_window':0,'observed_episodes':0},'token_ceiling':100}
        plan={'output_root':'probe','policy_fingerprint':'fp'};out=root/'probe/g0';out.mkdir(parents=True)
        def write(name,obj): (out/name).write_text(json.dumps(obj))
        def lines(name,rows):(out/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        panels={};calls=[];rollouts=[]
        for phase in ['feature','validation']:
            rows=[]
            for i,task in enumerate(case['task_ids']):
                group=i+(0 if phase=='feature' else 2);draw=phase+task;seed=case['seeds'][phase][i];reward=1-i
                rows.append(dict(task_id=task,source_group='g',policy_fingerprint='fp',stage=0,draw_id=draw,phase=phase,seed=seed,rewards=[reward]*4,sampled_tokens=4))
                for k in range(4):
                    meta=dict(group=group,task_id=task,draw_id=draw,seed=seed,phase=phase,rollout_index=k)
                    calls.append(dict(meta,input_ids=[1],generated_ids=[2],generation_allowance=4096,remaining_completion_budget=8192))
                    rollouts.append(dict(meta,prompt_ids=[1],completion_ids=[2],model_token_mask=[1],segments=[dict(input_ids=[1],generated_ids=[2],start=0)],bridges=[],reward=reward))
            panels[phase]=rows;lines(phase+'.jsonl',rows)
        f=forecast(panels['feature'],case['historical_signal']);write('predictions.json',f);sha=audit.digest(out/'predictions.json')
        write('prediction_receipt.json',dict(sealed_before_validation=True,validation_rows_at_seal=0,sealed_unix=1,sha256=sha))
        write('validation_start.json',dict(started_unix=2,prediction_sha256=sha));lines('generation_calls.jsonl',calls);lines('rollouts.jsonl',rollouts)
        write('run.json',dict(protocol_sha256='protocol',case=case,policy_fingerprint='fp'))
        write('summary.json',dict(status='complete',weights_unchanged=True,optimizer_updates=0,policy_fingerprint='fp',task_statistics=compare_independent_panels(panels['feature'],panels['validation']),scores=score(f,panels['validation']),prediction_sha256=sha,new_sampled_tokens=16,wall_seconds=1,scope='fixture'))
        return plan,case,out
    def test_audit_accepts_valid_and_rejects_prediction_or_mask_corruption(self):
        for corruption in [None,'predictions.json','rollouts.jsonl','validation_start.json']:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);plan,case,out=self.fixture(root)
                if corruption=='predictions.json':
                    p=out/corruption;obj=json.loads(p.read_text());obj['task_predictions']['easy']['task_beta']=0;p.write_text(json.dumps(obj))
                if corruption=='rollouts.jsonl':
                    p=out/corruption;rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['model_token_mask']=[0];p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
                if corruption=='validation_start.json':
                    p=out/corruption;obj=json.loads(p.read_text());obj['started_unix']=0;p.write_text(json.dumps(obj))
                with patch.object(audit,'ROOT',root):
                    if corruption:
                        with self.assertRaises(AssertionError):audit.audit_case(plan,case,'protocol')
                    else:self.assertEqual(audit.audit_case(plan,case,'protocol')['cost']['new_model_tokens'],16)

if __name__=='__main__':unittest.main()
