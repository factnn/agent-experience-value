"""CPU scheduling/publication tests, never a scientific GPU execution."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import supervise_bfcl_stage1_probe as s


def plan():
    cases = [{'name':f'group_{i:02d}'} for i in range(16)]
    return {'kind':'stage1_fixed_policy_probe_execution', 'output_root':'rl/test', 'allowed_gpus':[4,5,6,7],
            'max_simultaneous_gpus':4, 'automatic_retry':False, 'cases':cases,
            'gpu_lanes':{str(g):[c['name'] for c in cases[g-4::4]] for g in range(4,8)},
            'input_sha256':{p:'sha' for p in s.NEW_RUNTIME_CODE}}


OK = {'returncode':0, 'hard_timeout':False, 'status':'runner_complete_pending_replay_audit'}


class SupervisorTests(unittest.TestCase):
    def test_no_missing_duplicate_or_unpinned_cases(self):
        p=plan();master={'input_sha256':{'pipeline/old.py':'old'}};p['input_sha256'].update(master['input_sha256'])
        s.validate_schedule(p,master)
        for fault in ['missing','duplicate','gpu','dependency','new_runtime']:
            q=copy.deepcopy(p)
            if fault=='missing':q['gpu_lanes']['4'].pop()
            if fault=='duplicate':q['gpu_lanes']['4'][-1]=q['gpu_lanes']['5'][0]
            if fault=='gpu':q['allowed_gpus'][0]=0
            if fault=='dependency':q['input_sha256']['pipeline/old.py']='changed'
            if fault=='new_runtime':q['input_sha256'].pop(s.NEW_RUNTIME_CODE[0])
            with self.subTest(fault=fault),self.assertRaises(ValueError):s.validate_schedule(q,master)

    def test_all_cases_once_four_concurrent_and_final_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);barrier=threading.Barrier(4);lock=threading.Lock();seen=[];audits=[];active=0;maximum=0
            def launch(gpu,case):
                nonlocal active,maximum
                with lock:active+=1;maximum=max(maximum,active);seen.append((gpu,case['name']))
                barrier.wait(timeout=5)
                with lock:active-=1
                return dict(OK)
            def audit(name):
                with lock:audits.append(name)
            result=s.execute_lanes(root,plan(),lambda g:{'gpu':g},launch,audit)
            self.assertEqual(maximum,4);self.assertEqual(len(seen),16);self.assertEqual(len(set(n for _,n in seen)),16)
            self.assertEqual(audits[-1],None);self.assertEqual(len(audits),17)
            self.assertFalse(result['failures']);self.assertFalse(result['pending'])
            self.assertEqual(s.read(root/'supervisor.json')['status'],'audited_complete')

    def test_failure_stops_future_admission_and_preserves_other_active_work(self):
        for fault in ['runner','timeout','audit']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                root=Path(d);barrier=threading.Barrier(4);failure_written=threading.Event();seen=[];lock=threading.Lock()
                original_write=s.write
                def write(path,data):
                    original_write(path,data)
                    if path.name=='failures.json':failure_written.set()
                def launch(gpu,case):
                    with lock:seen.append(case['name'])
                    barrier.wait(timeout=5)
                    if gpu==4:
                        if fault=='runner':return OK | {'returncode':1}
                        if fault=='timeout':return OK | {'hard_timeout':True}
                        return dict(OK)
                    self.assertTrue(failure_written.wait(timeout=5))
                    return dict(OK)
                def audit(name):
                    if name=='group_00' and fault=='audit':raise ValueError('synthetic bad evidence')
                    self.assertIsNotNone(name)
                with patch.object(s,'write',write):
                    result=s.execute_lanes(root,plan(),lambda g:{'gpu':g},launch,audit)
                self.assertEqual(len(seen),4);self.assertEqual(len(result['complete']),3)
                self.assertEqual(len(result['pending']),12);self.assertEqual(len(result['failures']),1)
                self.assertFalse(result['active']);self.assertEqual(s.read(root/'supervisor.json')['status'],'incomplete_requires_review')

    def test_preflight_and_batch_audit_failures_are_not_complete(self):
        for fault in ['preflight','batch']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                root=Path(d);launched=[]
                def check(gpu):
                    if fault=='preflight':raise RuntimeError('Goal unavailable or GPU occupied')
                    return {'gpu':gpu}
                def launch(gpu,case):launched.append(case['name']);return dict(OK)
                def audit(name):
                    if name is None:raise RuntimeError('synthetic final audit failure')
                result=s.execute_lanes(root,plan(),check,launch,audit)
                self.assertEqual(len(launched),0 if fault=='preflight' else 16)
                self.assertTrue(result['failures']);self.assertTrue((root/'failures.json').exists())
                self.assertEqual(s.read(root/'supervisor.json')['status'],'incomplete_requires_review')

    def test_initial_failure_receipt_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'protocol.json';path.write_text(json.dumps(plan()));sha=s.digest(path)
            with patch.object(s,'load_protocol',side_effect=ValueError('stage0 report incomplete')),patch.object(s,'supervise') as launch:
                with self.assertRaisesRegex(ValueError,'stage0 report incomplete'):s.run(root,path,sha,'unused')
                launch.assert_not_called()
                receipt=(root/'rl/test/failures.json').read_bytes()
                self.assertEqual(json.loads(receipt)[0]['gpu_children_launched'],0)
                with self.assertRaises(FileExistsError):s.run(root,path,sha,'unused')
                self.assertEqual((root/'rl/test/failures.json').read_bytes(),receipt)

    def test_real_git_remote_requires_pushed_matching_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=parent/'work';root.mkdir();remote=parent/'remote.git'
            def git(*args):
                r=subprocess.run(['git',*args],cwd=root,capture_output=True,text=True)
                if r.returncode:raise RuntimeError('Fixture Git command failed: '+args[0])
                return r.stdout.strip()
            git('init','--bare','--initial-branch=main',str(remote));git('init','--initial-branch=main')
            git('config','user.name','CPU Test');git('config','user.email','cpu-test@example.invalid')
            git('remote','add','origin',str(remote))
            (root/'protocol.json').write_text('{"stage":1}\n');git('add','protocol.json');git('commit','-m','fixture')
            commit=git('rev-parse','HEAD');files={'protocol.json':s.digest(root/'protocol.json')}
            with self.assertRaises(RuntimeError):s.verify_publication(root,files,commit)
            git('push','origin','main')
            self.assertEqual(s.verify_publication(root,files,commit)['remote_main'],commit)
            with self.assertRaises(ValueError):s.verify_publication(root,{'protocol.json':'wrong'},commit)
            (root/'protocol.json').write_text('{"stage":2}\n');git('add','protocol.json');git('commit','-m','not pushed')
            with self.assertRaisesRegex(ValueError,'not on known remote'):
                s.verify_publication(root,{'protocol.json':s.digest(root/'protocol.json')},git('rev-parse','HEAD'))


if __name__=='__main__':unittest.main()
