"""Late-stage provenance and raw-evidence rejection tests; no GPU sampling."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import test_bfcl_task_probe_audit as old_fixture
import report_bfcl_stage1_probe as audit
from bfcl_stage1_probe_contract import load_design, load_protocol, validate_history, validate_execution
from bfcl_source_eval_contract import digest, pinned
from bfcl_probe_forecast import forecast, score
from bfcl_task_probe_statistics import compare_independent_panels

ROOT = Path(__file__).resolve().parents[1]
DESIGN = 'rl/bfcl_late_stage_probe_inputs_20261012.json'


class Stage1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = {'path':DESIGN, 'sha256':digest(ROOT/DESIGN)}
        cls.design, cls.binding, cls.algorithm_ref = load_design(ROOT, cls.ref)
        cls.manifest = pinned(ROOT, cls.binding['manifest'])
        cls.payload = torch.load(ROOT/cls.binding['learning_state']['path'], map_location='cpu', weights_only=False)
        cls.algorithm = pinned(ROOT, cls.algorithm_ref)['algorithm']

    def test_actual_parent_and_reject_stale_history_or_missing_optimizer(self):
        self.assertEqual(self.binding['source_step'], 7)
        self.assertEqual(len(validate_history(self.design, self.binding, self.manifest, self.payload, self.algorithm)), 16)
        design = copy.deepcopy(self.design)
        changed = next(c for c in design['cases'] if c['historical_signal_changed'])
        changed['historical_signal'] = changed['stage0_historical_signal']
        with self.assertRaisesRegex(ValueError, 'history'):
            validate_history(design, self.binding, self.manifest, self.payload, self.algorithm)
        payload = self.payload | {'optimizer':{'state':{}}}
        with self.assertRaisesRegex(ValueError, 'AdamW'):
            validate_history(self.design, self.binding, self.manifest, payload, self.algorithm)
        design = copy.deepcopy(self.design); design['cases'][0]['seeds']['validation'][0] = design['cases'][0]['seeds']['feature'][0]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            validate_history(design, self.binding, self.manifest, self.payload, self.algorithm)

    def test_design_is_not_execution_and_matrix_review_is_required(self):
        with self.assertRaisesRegex(ValueError, 'not an execution'):
            load_protocol(ROOT, Path(DESIGN), self.ref['sha256'])
        plan = {k:self.design[k] for k in ['cases','policy_path','policy_fingerprint','inherited_acquisition_tokens','parent_binding']}
        plan.update(kind='stage1_fixed_policy_probe_execution',stage=1,source_protocol=self.algorithm_ref,
                    stage0_transfer_report={'path':'report.json'},automatic_retry=False,max_simultaneous_gpus=4,allowed_gpus=[4,5,6,7])
        report = {'status':'passed','stage':0,'labels':{'panel_count':104},'evidence':{'execution_protocol':self.binding['stage0_execution_protocol']}}
        review = {'status':'reviewed','evidence':['report.json']}
        validate_execution(plan, self.design, self.binding, self.algorithm_ref, report, review)
        for corruption in ['stage','history','report','review','ancestor']:
            p,r,v = copy.deepcopy((plan,report,review))
            if corruption == 'stage': p['stage'] = 0
            if corruption == 'history': p['cases'][0]['historical_signal']['frontier'] = -1
            if corruption == 'report': r['labels']['panel_count'] = 80
            if corruption == 'review': v['evidence'] = []
            if corruption == 'ancestor': p['parent_binding']['sha256'] = 'wrong'
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_execution(p, self.design, self.binding, self.algorithm_ref, r, v)


class Stage1RawAuditTests(unittest.TestCase):
    def fixture(self, root):
        plan,case,out = old_fixture.ProbeAuditTests().fixture(root)
        def read(name):return json.loads((out/name).read_text())
        def write(name,obj):(out/name).write_text(json.dumps(obj))
        def rows(name):return [json.loads(x) for x in (out/name).read_text().splitlines()]
        def lines(name,items):(out/name).write_text(''.join(json.dumps(x)+'\n' for x in items))
        binding = {'learning_state':{'sha256':'parent'},'source_step':7}
        (root/'binding.json').write_text(json.dumps(binding))
        plan.update(stage=1,cases=[case],parent_binding={'path':'binding.json','sha256':digest(root/'binding.json')})
        for name in ['feature.jsonl','validation.jsonl','rollouts.jsonl']:
            lines(name,[x | {'stage':1,'policy_fingerprint':'fp'} for x in rows(name)])
        f,v = rows('feature.jsonl'),rows('validation.jsonl')
        predictions = forecast(f,case['historical_signal']);write('predictions.json',predictions)
        sha = digest(out/'predictions.json')
        write('prediction_receipt.json',read('prediction_receipt.json') | {'sha256':sha,'feature_rows':2})
        write('validation_start.json',read('validation_start.json') | {'prediction_sha256':sha})
        identity = {'policy_stage':1,'parent_binding':plan['parent_binding'],'parent_payload_sha256':'parent'}
        write('run.json',read('run.json') | identity | {'source_step':7,'started_unix':0,'optimizer_updates':0,'deterministic_algorithms':True,'cublas_workspace_config':':4096:8'})
        write('summary.json',read('summary.json') | identity | {'parent_state_unchanged':True,'case':case['name'],'source_group':case['source_group'],
            'task_statistics':compare_independent_panels(f,v),'scores':score(predictions,v),'prediction_sha256':sha,'panels':{'feature':2,'validation':2}})
        return plan,case,out

    def test_raw_replay_and_reject_stage_parent_masks_and_late_sealing(self):
        for corruption in [None,'stage','parent','mask','timing','raw_policy','phase']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as d:
                root=Path(d);plan,case,out=self.fixture(root)
                if corruption in ['stage','mask','raw_policy','phase']:
                    name = 'feature.jsonl' if corruption in ['stage','phase'] else 'rollouts.jsonl'
                    p=out/name;rows=[json.loads(x) for x in p.read_text().splitlines()]
                    field,value={'stage':('stage',0),'mask':('model_token_mask',[0]),'raw_policy':('policy_fingerprint','wrong'),'phase':('phase','validation')}[corruption]
                    rows[0][field]=value;p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
                if corruption in ['parent','timing']:
                    p=out/('run.json' if corruption=='parent' else 'validation_start.json');obj=json.loads(p.read_text())
                    obj['parent_payload_sha256' if corruption=='parent' else 'started_unix']='wrong' if corruption=='parent' else -1
                    p.write_text(json.dumps(obj))
                with patch.object(audit,'ROOT',root):
                    if corruption:
                        with self.assertRaises(AssertionError):audit.audit_case(plan,case,'protocol')
                    else:
                        result=audit.audit_case(plan,case,'protocol')
                        self.assertEqual(result['stage'],1)
                        self.assertEqual(result['cost']['new_model_tokens'],16)


if __name__ == '__main__':
    unittest.main()
