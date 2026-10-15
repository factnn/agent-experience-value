"""Current SSH/eight-card adapter for the existing fixed stage1 probe, no scientific redesign."""
import argparse,hashlib,importlib,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import bfcl_stage1_probe_contract as contract
import supervise_bfcl_stage1_probe as original
from bfcl_source_eval_contract import digest,pinned
from supervise_bfcl_stage_fork import ROOT,read,write,supervise,process_has_not_exited

CAP='experiments/SSH_FUTURE_GPU_CAP_V3_20261014.json'
SELF='pipeline/bfcl_stage1_probe_ssh.py'

def validate_resources(root,plan):
 cap=pinned(root,plan['resource_authorization'])
 if (plan['resource_authorization']['path']!=CAP or cap['max_simultaneous_gpus_for_subsequent_batches']!=8
     or plan['allowed_gpus']!=list(range(8)) or plan['max_simultaneous_gpus']!=8 or plan['automatic_retry'] is not False):
  raise ValueError('Require the current exact eight-card resource amendment and single attempts')

def load_protocol(root,path,expected_sha):
 plan=pinned(root,{'path':str(path),'sha256':expected_sha})
 if plan.get('kind')!='stage1_fixed_policy_probe_execution':raise ValueError('Design inputs are not execution')
 design,binding,algorithm=contract.load_design(root,plan['design_inputs'])
 report=pinned(root,plan['stage0_transfer_report']);review=pinned(root,plan['stage0_transfer_review'])
 # Preserve every scientific condition of the original validator; only the
 # separately authorized resource fields are projected to its old interface.
 contract.validate_execution(plan|{'allowed_gpus':[4,5,6,7],'max_simultaneous_gpus':4},design,binding,algorithm,report,review)
 validate_resources(root,plan)
 for name,h in plan['input_sha256'].items():
  if digest(root/name)!=h:raise ValueError('Execution dependency changed: '+name)
 return plan,binding

def validate_schedule(plan,master):
 names=[c['name'] for c in plan['cases']]
 if len(names)!=16 or len(set(names))!=16 or any(Path(n).name!=n or n in ('.','..') for n in names):raise ValueError('Invalid16 original cases')
 if set(plan['gpu_lanes'])!={str(g) for g in range(8)}:raise ValueError('Invalid eight lanes')
 scheduled=[n for lane in plan['gpu_lanes'].values() for n in lane]
 if len(scheduled)!=16 or set(scheduled)!=set(names):raise ValueError('Missing/repeated/substituted case')
 for name,h in master['input_sha256'].items():
  if plan['input_sha256'].get(name)!=h:raise ValueError('Inherited dependency changed: '+name)
 if not set(original.NEW_RUNTIME_CODE+[SELF])<=plan['input_sha256'].keys():raise ValueError('Missing stage1 adapter dependencies')

def check_control(root,plan,authority):
 if (authority.get('mode')!='human_authorized_ssh_takeover' or authority.get('scope')!='stage1_fixed_policy_probe'
     or authority.get('status')!='authorized' or pinned(root,authority['probe_protocol'])!=plan):
  raise ValueError('Wrong current SSH probe authority')
 validate_resources(root,plan)
 if authority['resource_authorization']!=plan['resource_authorization']:raise ValueError('Wrong resource authority')
 for name in ('stop_file','global_stop_file'):
  p=(root/authority[name]).resolve()
  if p.parent!=(root/'experiments/.runtime').resolve():raise ValueError('Stop marker outside project runtime')
  if p.exists():raise RuntimeError('Current SSH owner stopped further stage1 probe admission')
 owner=pinned(root,authority['owner_goal']['record'])
 if owner['thread_id']!=authority['owner_goal']['thread_id'] or owner['status_at_registration']!='active':raise ValueError('Wrong current registered owner')
 if authority['stage0_transfer_report']!=plan['stage0_transfer_report'] or authority['stage0_transfer_review']!=plan['stage0_transfer_review']:raise ValueError('Wrong complete stage0 evidence')
 audit=pinned(root,authority['complete_training_audit']);review=pinned(root,authority['complete_training_review'])
 if audit['status']!='passed' or audit['complete_jobs']!=51 or review['status']!='reviewed':raise ValueError('Missing complete51 audit/review')
 for name,h in authority['control_input_sha256'].items():
  if digest(root/name)!=h:raise ValueError('Controller changed: '+name)
 return {'owner_goal_thread':owner['thread_id'],'control_mode':authority['mode'],'scope':authority['scope']}

def preflight(root,gpu,plan,authority,publication):
 owner=check_control(root,plan,authority)
 # The immutable receipt was created from actual GitHub verification before
 # admission. Keep checking its exact bytes and all published local inputs;
 # a later transient network read is not needed to re-prove that past event.
 proof=pinned(root,publication)
 for name,h in proof['files'].items():
  if digest(root/name)!=h:raise ValueError('Published local bytes changed: '+name)
 if gpu not in plan['allowed_gpus']:raise ValueError('GPU outside registered lanes')
 prior=read(root/'rl/STATUS.json')['bfcl_intervention']
 if prior['phase']!='audited_complete' or prior['active'] or prior['pending'] or prior['failures'] or process_has_not_exited(prior['scheduler_pid']):raise RuntimeError('Prior dose queue is not complete/exited')
 prior_path=prior['path']+'/audit_status.json';events=[e for e in read(root/'experiments/index.json') if e['source']==prior_path]
 if read(root/prior_path)['status']!='passed' or not events or read(root/'experiments/reviews'/(events[-1]['id']+'.json'))['status']!='reviewed':raise RuntimeError('Prior dose audit/review missing')
 info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
 if int(info[1].strip())>64 or info[0].strip() in apps:raise RuntimeError('GPU occupied')
 return owner|{'gpu':gpu,'memory_mib':int(info[1].strip()),'no_compute_process':True,'resource_authorization':plan['resource_authorization'],'publication':publication}

def bounded_remote(query,check,sleep=time.sleep):
 observations=[]
 for attempt in range(3):
  check()
  try:
   result=query()
   if result.returncode or not result.stdout.strip():raise RuntimeError('GitHub publication query unavailable')
   return result,observations
  except (subprocess.TimeoutExpired,RuntimeError) as error:
   observations.append({'attempt':attempt+1,'classification':'publication_dependency_query','type':type(error).__name__,'detail':'Timeout or unavailable remote main; no scientific child started'})
   if attempt==2:raise RuntimeError('Publication dependency unavailable after3 bounded attempts') from error
   check();sleep((5,15)[attempt])

def verify_publication(root,files,commit,check):
 for name,h in files.items():
  check();result=subprocess.run(['git','show',commit+':'+name],cwd=root,capture_output=True,timeout=30)
  if result.returncode or hashlib.sha256(result.stdout).hexdigest()!=h:raise ValueError('Commit bytes differ: '+name)
 result,attempts=bounded_remote(lambda:subprocess.run(['git','ls-remote','origin','refs/heads/main'],cwd=root,capture_output=True,timeout=30),check)
 remote=result.stdout.decode().split()[0]
 result=subprocess.run(['git','merge-base','--is-ancestor',commit,remote],cwd=root,capture_output=True,timeout=30)
 if result.returncode:raise ValueError('Publication commit not on known remote main')
 return {'commit':commit,'remote_main':remote,'verified_unix':time.time(),'files':files,'dependency_query_failures_before_success':attempts,'query_policy':{'attempts_max':3,'timeout_seconds':30,'backoff_seconds':[5,15]},'scientific_child_started':False}

def execute_lanes(out,plan,check,launch,audit):
 # The original callbacks, single-attempt worker, stop-on-failure semantics,
 # active-job bounds and audit behavior remain. Adapt only pool capacity.
 previous=original.ThreadPoolExecutor
 def pool(*args,**kwargs):
  kwargs['max_workers']=plan['max_simultaneous_gpus'];return ThreadPoolExecutor(*args,**kwargs)
 original.ThreadPoolExecutor=pool
 try:return original.execute_lanes(out,plan,check,launch,audit)
 finally:original.ThreadPoolExecutor=previous

def run(root,path,expected_sha,authority_ref,commit):
 plan,binding=load_protocol(root,path,expected_sha);authority=pinned(root,authority_ref);check=lambda:check_control(root,plan,authority)
 check();validate_schedule(plan,pinned(root,binding['stage0_execution_protocol']))
 rel=Path(plan['output_root'])
 if rel.is_absolute() or '..' in rel.parts or rel.parts[:1]!=('rl',):raise ValueError('Invalid output root')
 out=root/rel;out.mkdir(exist_ok=False)
 try:
  files=original.publication_files(path.relative_to(root),expected_sha,plan,binding)
  files[authority_ref['path']]=authority_ref['sha256']
  for ref in (plan['resource_authorization'],authority['owner_goal']['record'],authority['complete_training_audit'],authority['complete_training_review']):files[ref['path']]=ref['sha256']
  for eid in plan['required_review_events']:
   name='experiments/reviews/'+eid+'.json';files[name]=digest(root/name)
  pub=verify_publication(root,files,commit,check)
  with (out/'publication_verified.json').open('x') as f:json.dump(pub,f,indent=2);f.write('\n')
  proof={'path':str((out/'publication_verified.json').relative_to(root)),'sha256':digest(out/'publication_verified.json')}
  if plan['output_root'] not in read(root/'experiments/watch_roots.json'):raise ValueError('Output not registered')
  for eid in plan['required_review_events']:
   if read(root/'experiments/reviews'/(eid+'.json'))['status']!='reviewed':raise ValueError('Missing required review')
  initial={g:preflight(root,g,plan,authority,proof) for g in plan['allowed_gpus']}
 except Exception as error:
  write(out/'failures.json',[{'classification':'control_or_publication_dependency_admission','type':type(error).__name__,'message':str(error),'scientific_children_launched':0}]);write(out/'supervisor.json',{'status':'incomplete_requires_review','automatic_retry':False});raise
 write(out/'launch.json',{'supervisor_pid':os.getpid(),'started_unix':time.time(),'protocol_sha256':expected_sha,'authority':authority_ref,'publication':proof,'preflights':initial,'max_simultaneous_gpus':8})
 def launch(gpu,case):
  return supervise([str(root/'.venv-rl/bin/python'),'-u',str(root/SELF),'run','--protocol',str(path),'--protocol-sha256',expected_sha,'--case',case['name']],out/case['name'],out/(case['name']+'.console.log'),case['hard_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8'))
 def audit(case):
  cmd=[str(root/'.venv-rl/bin/python'),str(root/SELF),'audit','--protocol',str(path),'--protocol-sha256',expected_sha]
  if case is not None:cmd+=['--case',case]
  with (out/((case or 'batch')+'.audit.console.log')).open('x') as log:
   result=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=300,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  receipt=read(out/case/'audit.json' if case is not None else out/'audit_status.json')
  if result.returncode or receipt['status']!='passed' or receipt['stage']!=1 or receipt['complete_cases']!=(1 if case is not None else 16):raise ValueError('Original raw audit failed: '+str(case))
 result=execute_lanes(out,plan,lambda g:preflight(root,g,plan,authority,proof),launch,audit)
 if result['failures'] or result['pending']:raise RuntimeError('Incomplete batch; preserve evidence, no scientific retry')
 return result

def main():
 operation=sys.argv.pop(1)
 if operation in ('run','audit'):
  module=importlib.import_module('run_bfcl_stage1_probe' if operation=='run' else 'report_bfcl_stage1_probe')
  module.load_protocol=load_protocol;return module.main()
 if operation!='supervise':raise ValueError('Unknown operation')
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True);ap.add_argument('--authority',required=True);ap.add_argument('--authority-sha256',required=True);ap.add_argument('--publication-commit',required=True);a=ap.parse_args()
 run(ROOT,a.protocol.resolve(),a.protocol_sha256,{'path':a.authority,'sha256':a.authority_sha256},a.publication_commit)

if __name__=='__main__':main()
