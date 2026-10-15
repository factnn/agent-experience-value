"""Only the newly changed pool capacity and bounded dependency recovery."""
import subprocess,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
import bfcl_stage1_probe_ssh as adapter

class AdapterTests(unittest.TestCase):
 def test_original_workers_actually_use_eight_and_run_each_case_once(self):
  names=['case_'+str(i) for i in range(16)]
  plan={'cases':[{'name':n} for n in names],'allowed_gpus':list(range(8)),'max_simultaneous_gpus':8,'gpu_lanes':{str(g):[names[g],names[g+8]] for g in range(8)}}
  lock=threading.Lock();barrier=threading.Barrier(8);running=0;peak=0;seen=[]
  def launch(g,c):
   nonlocal running,peak
   with lock:running+=1;peak=max(peak,running);seen.append(c['name'])
   if int(c['name'].split('_')[-1])<8:barrier.wait(timeout=5)
   time.sleep(.01)
   with lock:running-=1
   return {'returncode':0,'hard_timeout':False,'status':'runner_complete_pending_replay_audit'}
  with tempfile.TemporaryDirectory() as d:result=adapter.execute_lanes(Path(d),plan,lambda g:{'gpu':g},launch,lambda case:None)
  self.assertEqual(peak,8);self.assertCountEqual(seen,names);self.assertEqual(len(seen),16)
  self.assertFalse(result['pending']);self.assertFalse(result['failures'])

 def test_finite_dependency_recovery_and_stop_checks(self):
  calls=[];delays=[];checks=[]
  def query():
   calls.append(1)
   if len(calls)<3:raise subprocess.TimeoutExpired(['git','ls-remote'],30)
   return SimpleNamespace(returncode=0,stdout=b'abc refs/heads/main\n')
  result,observations=adapter.bounded_remote(query,lambda:checks.append(1),delays.append)
  self.assertEqual(len(calls),3);self.assertEqual(delays,[5,15]);self.assertEqual(len(observations),2)
  calls.clear();delays.clear()
  with self.assertRaisesRegex(RuntimeError,'after3'):
   adapter.bounded_remote(lambda:(_ for _ in ()).throw(subprocess.TimeoutExpired(['git'],30)),lambda:None,delays.append)
  self.assertEqual(delays,[5,15])
  stopped=[False];calls=[]
  def stop_check():
   if stopped[0]:raise RuntimeError('current stop')
  def stop_query():
   calls.append(1);stopped[0]=True;raise subprocess.TimeoutExpired(['git'],30)
  with self.assertRaisesRegex(RuntimeError,'current stop'):adapter.bounded_remote(stop_query,stop_check,lambda _:None)
  self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
