"""CPU feature isolation, frozen extrapolation and nested Git seal checks."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from bfcl_transfer_predictor import SIGNALS, fit_models
from bfcl_probe_forecast import beta_mixed
from bfcl_source_eval_contract import digest
from seal_bfcl_forward_predictions import feature_row, forward_artifact, verify_forward_seal


def fixture():
    groups=['a','b','c','d']
    analysis={'source_split':{'all_groups':groups,'fit':groups[:3],'heldout':['d']},
              'fitting':{'lambda':1.0},'features':{'allowed_models':['fit_mean',*SIGNALS]}}
    early={'stage':0,'policy_fingerprint':'early','validation_labels_used':False,
           'rows':[{'source_group':g,**{s:i/10 for s in SIGNALS}} for i,g in enumerate(groups)]}
    for row in early['rows']:row['historical_coverage']=1
    models=fit_models(early,{'a':-.1,'b':0,'c':.1},analysis)
    late=copy.deepcopy(early);late.update(stage=1,policy_fingerprint='late')
    for row in late['rows']:row['primary_task_beta']+=.2;row['historical_coverage']=.5
    return analysis,models,late


class ForwardTests(unittest.TestCase):
    def test_task_features_keep_structure_and_reject_validation_or_stage0(self):
        case={'task_ids':['easy','hard'],'seeds':{'feature':[1,2]},'source_group':'g',
              'historical_signal':{'observed_episodes':0,'successes_in_window':0,'frontier':1,'coverage_deficit':1}}
        rows=[{'task_id':task,'seed':i+1,'source_group':'g','phase':'feature','stage':1,'policy_fingerprint':'late',
               'draw_id':task,'rewards':[1-i]*4} for i,task in enumerate(case['task_ids'])]
        result,_=feature_row(rows,case,'late')
        self.assertAlmostEqual(result['primary_task_beta'],beta_mixed(0,4))
        self.assertAlmostEqual(result['current_pooled_beta'],beta_mixed(4,8))
        self.assertLess(result['primary_task_beta'],result['current_pooled_beta'])
        for key,value in [('phase','validation'),('stage',0),('policy_fingerprint','early')]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):feature_row(bad,case,'late')

    def test_early_normalization_is_immutable_and_late_labels_are_not_an_input(self):
        analysis,models,late=fixture();before=copy.deepcopy(models)
        result=forward_artifact(models,late,analysis,'early')
        self.assertAlmostEqual(result['point_predictions']['d']['primary_task_beta'],.3)
        self.assertEqual(result['point_predictions']['d']['historical_coverage'],0)
        self.assertEqual(models,before)
        # Unused malicious label fields cannot change the extrapolation.
        contaminated=late | {'G_labels':{'d':999},'validation_rewards':[1]*100}
        self.assertEqual(forward_artifact(models,contaminated,analysis,'early'),result)
        for fault in ['stage','validation','missing_model','nonfinite','negative_scale']:
            m,l=copy.deepcopy((models,late))
            if fault=='stage':m['fit_stage']=1
            if fault=='validation':l['validation_labels_used']=True
            if fault=='missing_model':m['models'].pop('current_pooled_beta')
            if fault=='nonfinite':m['models']['primary_task_beta']['slope']=float('nan')
            if fault=='negative_scale':m['models']['primary_task_beta']['fit_population_std']=-1
            with self.subTest(fault=fault),self.assertRaises(ValueError):forward_artifact(m,l,analysis,'early')

    def git_fixture(self,root,remote):
        def git(*args):
            p=subprocess.run(['git',*args],cwd=root,capture_output=True,text=True)
            if p.returncode:raise RuntimeError('Fixture Git failed: '+args[0])
            return p.stdout.strip()
        def write(name,data):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2)+'\n')
            return {'path':name,'sha256':digest(p)}
        def publish():
            git('add','.');git('commit','-m','synthetic fixture');commit=git('rev-parse','HEAD');git('push','origin','main');return commit
        git('init','--bare','--initial-branch=main',str(remote));git('init','--initial-branch=main')
        git('config','user.name','CPU Test');git('config','user.email','cpu-test@example.invalid');git('remote','add','origin',str(remote))
        analysis,models,late=fixture();analysis_ref=write('analysis.json',analysis)
        predictor=Path(__file__).with_name('bfcl_transfer_predictor.py')
        (root/'pipeline').mkdir();(root/'pipeline/bfcl_transfer_predictor.py').write_bytes(predictor.read_bytes())
        master_ref=write('master.json',{'analysis_contract':analysis_ref,'source_stage0':{'policy_fingerprint':'early'}})
        parent_ref=write('parent.json',{'stage0_execution_protocol':master_ref,'policy_fingerprint':'late'})
        for name,data in [('models.json',models),('predictions.json',{}),('fit_labels.json',{})]:write('early/'+name,data)
        early_ref=write('early/receipt.json',{'stage':0,'sealed_unix':1,'execution_protocol_sha256':master_ref['sha256'],
            'analysis_contract_sha256':analysis_ref['sha256'],'heldout_groups':['d'],
            'implementation_sha256':{'pipeline/bfcl_transfer_predictor.py':digest(predictor)},
            'files':{n:digest(root/'early'/n) for n in ['models.json','predictions.json','fit_labels.json']}})
        early_commit=publish()
        write('early/publication.json',{'receipt_sha256':early_ref['sha256'],'commit':early_commit,'verified_remote_at_unix':2})
        report_ref=write('report.json',{'evidence':{'prediction_seal':early_ref}})
        probe_ref=write('probe.json',{'stage':1,'parent_binding':parent_ref,'stage0_transfer_report':report_ref})
        late.update(source_protocol=probe_ref,parent_binding=parent_ref,analysis_contract_sha256=analysis_ref['sha256'])
        write('forward/early_models.json',models);write('forward/late_features.json',late)
        write('forward/predictions.json',forward_artifact(models,late,analysis,'early'))
        receipt={'kind':'early_to_late_prediction_seal','stage':1,'late_target_labels_used':False,'sealed_unix':3,
                 'stage1_probe_protocol':probe_ref,'parent_binding':parent_ref,'analysis_contract':analysis_ref,
                 'stage0_execution_protocol':master_ref,'early_prediction_seal':early_ref,'heldout_groups':['d'],
                 'publication_required_before_any_stage1_target_evaluation':True,
                 'files':{n:digest(root/'forward'/n) for n in ['early_models.json','late_features.json','predictions.json']}}
        reference=write('forward/receipt.json',receipt);commit=publish()
        write('forward/publication.json',{'receipt_sha256':reference['sha256'],'commit':commit,'verified_remote_at_unix':4})
        return reference,probe_ref,parent_ref,analysis_ref,write

    def test_published_nested_seal_rejects_early_evaluation_and_resealed_false_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=parent/'work';root.mkdir()
            ref,probe,binding,analysis,write=self.git_fixture(root,parent/'remote.git')
            result=verify_forward_seal(root,ref,probe,binding,analysis,evaluation_started_unix=5)
            self.assertEqual(result['stage'],1)
            self.assertEqual(verify_forward_seal(root,ref,probe,binding,analysis,check_remote=False),result)
            with self.assertRaisesRegex(ValueError,'preceded'):
                verify_forward_seal(root,ref,probe,binding,analysis,evaluation_started_unix=3.5)
            with self.assertRaisesRegex(ValueError,'binding mismatch'):
                verify_forward_seal(root,ref,probe,binding | {'sha256':'wrong'},analysis)
            code=root/'pipeline/bfcl_transfer_predictor.py';original=code.read_bytes();code.write_bytes(original+b'\n# changed\n')
            with self.assertRaisesRegex(ValueError,'implementation differs'):
                verify_forward_seal(root,ref,probe,binding,analysis)
            code.write_bytes(original)
            prediction=json.loads((root/'forward/predictions.json').read_text())
            prediction['point_predictions']['d']['primary_task_beta']=.99
            write('forward/predictions.json',prediction)
            receipt=json.loads((root/ref['path']).read_text());receipt['files']['predictions.json']=digest(root/'forward/predictions.json')
            altered=write(ref['path'],receipt)
            with self.assertRaisesRegex(ValueError,'not the original fixed model'):
                verify_forward_seal(root,altered,probe,binding,analysis)

    def test_resealed_refitted_model_cannot_replace_original_published_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=parent/'work';root.mkdir()
            ref,probe,binding,analysis,write=self.git_fixture(root,parent/'remote.git')
            models=json.loads((root/'forward/early_models.json').read_text());models['models']['primary_task_beta']['fit_mean']=.5
            write('forward/early_models.json',models)
            receipt=json.loads((root/ref['path']).read_text());receipt['files']['early_models.json']=digest(root/'forward/early_models.json')
            altered=write(ref['path'],receipt)
            with self.assertRaisesRegex(ValueError,'differ from the published early model'):
                verify_forward_seal(root,altered,probe,binding,analysis)


if __name__=='__main__':unittest.main()
