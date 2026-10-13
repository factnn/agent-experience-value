import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import bfcl_ssh_evaluation_control as control
from supervise_bfcl_source_evaluation import verify_publication


def put(root, name, data):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))
    return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class EvaluationOwnershipTests(unittest.TestCase):
    def fixture(self,root,phase='fit'):
        (root/'experiments/.runtime').mkdir(parents=True)
        master_ref=put(root,'master.json',{'output_root':'training'})
        manifest={'phase':phase,'execution_protocol':master_ref,'jobs':[{'name':str(i)} for i in range(80 if phase=='fit' else 24)],'input_sha256':{}}
        manifest_ref=put(root,'manifest.json',manifest)
        goal_ref=put(root,'goal.json',{'thread_id':'current','status_at_registration':'active'})
        audit_ref=put(root,'training/audit_status.json',{'status':'passed','complete_jobs':51})
        review_ref=put(root,'experiments/reviews/Efixture.json',{'status':'reviewed'})
        put(root,'experiments/index.json',[{'id':'Efixture','source':audit_ref['path']}])
        code_ref=put(root,'pipeline/control.py',{'version':'original'})
        authority={'mode':'human_authorized_ssh_takeover','scope':'source_stage0_evaluation','status':'authorized',
            'phase':phase,'evaluation_manifest':manifest_ref,'execution_protocol':master_ref,
            'owner_goal':{'thread_id':'current','record':goal_ref},'current_goal_status_at_registration':'active',
            'complete_training_audit':audit_ref,'complete_training_review':review_ref,
            'stop_file':'experiments/.runtime/evaluation.stop','global_stop_file':'experiments/.runtime/all.stop',
            'control_input_sha256':{code_ref['path']:code_ref['sha256']}}
        return manifest,authority

    def test_current_owner_binds_each_exact_phase_without_old_goal_query(self):
        for phase in ['fit','heldout']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,authority=self.fixture(root,phase)
                with patch('codex_goal_status.goal_status',side_effect=AssertionError('Old Goal queried')):
                    result=control.check_control(root,plan,authority)
                self.assertEqual(result['phase'],phase)
                self.assertEqual(result['owner_goal_thread'],'current')

    def test_wrong_owner_phase_or_modified_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,authority=self.fixture(root)
            cases=[]
            wrong=copy.deepcopy(authority);wrong['owner_goal']['thread_id']='historical';cases.append(wrong)
            wrong=copy.deepcopy(authority);wrong['phase']='heldout';cases.append(wrong)
            wrong=copy.deepcopy(authority);wrong['evaluation_manifest']['sha256']='changed';cases.append(wrong)
            for wrong in cases:
                with self.assertRaises(ValueError):control.check_control(root,plan,wrong)

    def test_stops_controller_mutation_and_incomplete_training_remain_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,authority=self.fixture(root)
            for name in ['stop_file','global_stop_file']:
                p=root/authority[name];p.touch()
                with self.assertRaises(RuntimeError):control.check_control(root,plan,authority)
                p.unlink()
            original=(root/'pipeline/control.py').read_bytes()
            (root/'pipeline/control.py').write_text('changed')
            with self.assertRaises(ValueError):control.check_control(root,plan,authority)
            (root/'pipeline/control.py').write_bytes(original)
            authority['complete_training_audit']=put(root,'training/audit_status.json',{'status':'passed','complete_jobs':50})
            with self.assertRaises(ValueError):control.check_control(root,plan,authority)

    def test_unreviewed_or_wrong_matrix_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,authority=self.fixture(root)
            authority['complete_training_review']=put(root,'experiments/reviews/Efixture.json',{'status':'pending'})
            with self.assertRaises(ValueError):control.check_control(root,plan,authority)
            authority['complete_training_review']=put(root,'experiments/reviews/Efixture.json',{'status':'reviewed'})
            put(root,'experiments/index.json',[{'id':'Efixture','source':'another/audit_status.json'}])
            with self.assertRaises(ValueError):control.check_control(root,plan,authority)

    def test_resource_preflight_still_runs_after_owner_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,authority=self.fixture(root)
            with patch.object(control,'ROOT',root),patch.object(control,'resource_preflight',side_effect=RuntimeError('GPU occupied')) as resource:
                with self.assertRaisesRegex(RuntimeError,'GPU occupied'):control.preflight(4,plan,authority)
            resource.assert_called_once_with(4,plan,authority)

    def test_authority_and_its_evidence_must_be_in_the_published_commit(self):
        with tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryDirectory() as remote:
            root=Path(tmp);_,authority=self.fixture(root)
            plan={'jobs':[],'input_sha256':{}}
            manifest_ref=put(root,'published_manifest.json',plan)
            authority_ref=put(root,'authority.json',authority)
            subprocess.run(['git','init','-q'],cwd=root,check=True)
            subprocess.run(['git','init','--bare','-q',remote],check=True)
            subprocess.run(['git','remote','add','origin',remote],cwd=root,check=True)
            def commit(paths):
                subprocess.run(['git','add','--',*paths],cwd=root,check=True)
                subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','Synthetic control publication'],cwd=root,check=True)
                return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            first=commit(['published_manifest.json'])
            with self.assertRaises(ValueError):verify_publication(root,root/manifest_ref['path'],manifest_ref['sha256'],plan,first,authority_ref)
            published=commit(['authority.json','goal.json','training/audit_status.json','experiments/reviews/Efixture.json','pipeline/control.py'])
            subprocess.run(['git','push','-q','origin','HEAD:refs/heads/main'],cwd=root,check=True)
            result=verify_publication(root,root/manifest_ref['path'],manifest_ref['sha256'],plan,published,authority_ref)
            self.assertEqual(result['remote_main'],published)


if __name__=='__main__':unittest.main()
