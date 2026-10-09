"""Run the prospectively frozen full-pool follow-up on at most three idle GPUs."""
import json
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
import watch_bfcl_intervention as w

w.OUT=w.ROOT/'rl/bfcl_intervention_followup_20261009'
w.PROTOCOL=w.ROOT/'rl/BFCL_FOLLOWUP_PROTOCOL_20261009.json'
REPORT='24_bfcl_followup_design.md'
HARD_SECONDS=14400
original_status=w.status

def status(active,pending,phase):
    original_status(active,pending,phase)
    current=w.read(w.ROOT/'rl/STATUS.json')
    current['status']='bfcl_followup_'+phase
    current['bfcl_first_wave']={'phase':'audited_complete',
        'comparison':'rl/bfcl_intervention_20261008/comparison.json',
        'report':'23_bfcl_intervention_complete_result.md'}
    current['bfcl_intervention']['report']=REPORT
    w.write(w.ROOT/'rl/STATUS.json',current)
w.status=status

def publish(label,active):
    # Another authorized writer may briefly hold Git's index; never remove its lock.
    for attempt in range(3):
        try:return w.publish(label,active)
        except subprocess.CalledProcessError:
            print(json.dumps({'publication':'local_git_retry','attempt':attempt+1}),flush=True)
            time.sleep(5)
    return False
def check(job):
    if time.monotonic()-job['start']>HARD_SECONDS and job['process'].poll() is None:
        os.killpg(job['process'].pid,signal.SIGTERM)
        w.write(job['out']/'scheduler_failure.json',{'reason':'hard_14400_second_limit','partial_data_retained':True})
    return job['process'].poll()

def launch(mode,name,gpu,extra):
    job=w.launch(mode,name,gpu,[*extra,'--protocol',str(w.PROTOCOL)])
    path=w.OUT/(name+'.launch.json');record=w.read(path)
    record['hard_seconds']=HARD_SECONDS;w.write(path,record)
    return job

class AdoptedProcess:
    """Observe an existing orphan; its original OS exit status is unavailable."""
    def __init__(self,record,out):
        self.pid=record['pid'];self.command=record['command'];self.out=out;self.returncode=None
    def poll(self):
        proc=Path('/proc')/str(self.pid)
        try:
            state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
            command=(proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
            if state!='Z' and command==[s.encode() for s in self.command]:return None
        except FileNotFoundError:pass
        summary=self.out/'summary.json'
        self.returncode=0 if summary.exists() and w.read(summary)['status']=='complete' else 1
        return self.returncode

def recover(p):
    active=[];eval_queue=[];branch_queue=[];failed=[]
    original=w.read(w.OUT/'scheduler.json')
    specs=[('eval_common','eval',{'kind':'baseline'})]
    for index,condition in enumerate(p['conditions']):
        for rule in p['rules']:
            specs.extend([
                (f'train_{index}_{rule}','train',{'kind':'branch','index':index,'condition':condition,'rule':rule}),
                (f'eval_{index}_{rule}','eval',{'kind':'eval','index':index,'rule':rule,'policy':str(w.OUT/f'train_{index}_{rule}')})])
    for name,mode,spec in specs:
        out=w.OUT/name;record_path=w.OUT/(name+'.launch.json')
        if record_path.exists():
            record=w.read(record_path);process=AdoptedProcess(record,out)
            if process.poll() is None:
                elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(record['started_at_utc'])).total_seconds()
                job={'name':name,'mode':mode,'gpu':record['gpu'],'process':process,
                    'log':(w.OUT/(name+'.console.log')).open('a'),'start':time.monotonic()-elapsed,
                    'command':record['command'],'out':out,'last_signature':None}
                if mode=='train':job.update({'index':spec['index'],'rule':spec['rule']})
                active.append(job)
            elif process.returncode:
                failed.append({'name':name,'reason':'previous_process_exited_without_complete_summary','partial_data_retained':True})
            continue
        if out.exists():raise RuntimeError(f'{name}: output without launch ledger; preserve and inspect')
        if mode=='train':branch_queue.append(spec)
        elif spec['kind']=='baseline':eval_queue.append(spec)
        else:
            training=w.OUT/f"train_{spec['index']}_{spec['rule']}"/'summary.json'
            if training.exists() and w.read(training)['status']=='complete':eval_queue.append(spec)
    assert len(active)<=3 and len({j['gpu'] for j in active})==len(active)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    w.write(w.OUT/f'scheduler_recovery_{stamp}.json',{'previous_scheduler_pid':original['pid'],
        'replacement_scheduler_pid':os.getpid(),'protocol_sha256':hashlib.sha256(w.PROTOCOL.read_bytes()).hexdigest(),
        'adopted':[{'name':j['name'],'pid':j['process'].pid,'gpu':j['gpu']} for j in active],
        'reason':'Previous CPU scheduler exited; existing GPU jobs continued.',
        'budgets_reset':False,'completed_or_partial_jobs_resampled':False,
        'original_exit_status_unavailable':'Adopted jobs use PID lifecycle and complete output summaries; final data audit remains required.'})
    original.update({'pid':os.getpid(),'previous_scheduler_pid':original['pid'],'resumed_at_utc':datetime.now(timezone.utc).isoformat()})
    w.write(w.OUT/'scheduler.json',original)
    return active,eval_queue+branch_queue,failed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    lock=open('/tmp/agent-experience-bfcl-followup-20261009.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=w.read(w.PROTOCOL)
    common=w.ROOT/p['source_common_path']
    manifest=w.read(common/'components/manifest.json')
    assert manifest['payload_sha256']==p['source_common_payload_sha256']
    assert w.read(common/'summary.json')['status']=='complete'
    if args.resume:
        active,queue,failed=recover(p)
    else:
        w.OUT.mkdir(parents=True,exist_ok=False)
        active=[];failed=[]
        queue=[{'kind':'baseline'}]+[{'kind':'branch','index':i,'condition':c,'rule':r}
            for i,c in enumerate(p['conditions']) for r in p['rules']]
    # Local link only; published source manifest references the archived first wave.
    if not args.resume:(w.OUT/'common').symlink_to(common,target_is_directory=True)
    w.write(w.OUT/'source_common.json',{'path':p['source_common_path'],
        'payload_sha256':manifest['payload_sha256'],'inherited_probe_tokens':90122,
        'new_probe_tokens':0,'conditional_repeats':2,'source_protocol':p['source_common_protocol']})
    if not args.resume:w.write(w.OUT/'scheduler.json',{'pid':os.getpid(),'gpus':w.GPUS,'max_simultaneous_gpus':3,
        'protocol':str(w.PROTOCOL.relative_to(w.ROOT)),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'hard_seconds':HARD_SECONDS})
    status(active,queue,'running')
    publish('Resume existing full-pool BFCL study without resampling' if args.resume else 'Launch frozen full-pool conditional-repeat BFCL study',active)
    while queue or active:
        completed=[]
        for job in active:
            code=check(job)
            if code is None:continue
            job['log'].close();completed.append(job)
            if code:
                failed.append({'name':job['name'],'returncode':code});continue
            summary=w.read(job['out']/'summary.json')
            if summary['status']!='complete':
                failed.append({'name':job['name'],'status':summary['status']});continue
            if job['mode']=='train':
                queue.insert(0,{'kind':'eval','index':job['index'],'rule':job['rule'],'policy':str(job['out'])})
        for job in completed:active.remove(job)
        if completed:
            w.write(w.OUT/'failures.json',failed);status(active,queue,'running')
            publish('Publish completed full-pool BFCL follow-up stage',active)
        free=w.idle()-{j['gpu'] for j in active}
        for gpu in w.GPUS:
            if len(active)>=3 or not queue:break
            if gpu not in free:continue
            spec=queue.pop(0)
            if spec['kind']=='baseline':job=launch('eval','eval_common',gpu,['--policy',str(common)])
            elif spec['kind']=='branch':
                job=launch('train',f"train_{spec['index']}_{spec['rule']}",gpu,
                    ['--common',str(common),'--condition',spec['condition'],'--rule',spec['rule']])
                job.update({'index':spec['index'],'rule':spec['rule']})
            else:job=launch('eval',f"eval_{spec['index']}_{spec['rule']}",gpu,['--policy',spec['policy']])
            active.append(job)
        dirty=False
        for job in active:
            path=job['out']/'progress.json'
            if path.exists():
                try:signature=path.read_text();json.loads(signature)
                except (ValueError,OSError):continue
                if signature!=job['last_signature']:job['last_signature']=signature;dirty=True
        status(active,queue,'running')
        if dirty:publish('Publish full-pool BFCL follow-up milestone',active)
        time.sleep(20)
    w.write(w.OUT/'failures.json',failed)
    if failed:
        status([],[],'incomplete');publish('Retain bounded incomplete BFCL follow-up',[]);return
    audit=subprocess.run([str(w.PYTHON),str(w.ROOT/'pipeline/report_bfcl_intervention.py'),
        '--root',str(w.OUT),'--protocol',str(w.PROTOCOL)],cwd=w.ROOT,capture_output=True,text=True)
    (w.OUT/'audit.console.log').write_text(audit.stdout+audit.stderr)
    w.write(w.OUT/'audit_status.json',{'returncode':audit.returncode,'status':'passed' if audit.returncode==0 else 'failed'})
    status([],[],'audited_complete' if audit.returncode==0 else 'audit_failed')
    publish('Publish audited full-pool BFCL conditional-repeat G and V',[])

if __name__=='__main__':main()
