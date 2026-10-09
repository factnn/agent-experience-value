"""Run the prospectively frozen full-pool follow-up on at most three idle GPUs."""
import json
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

def main():
    p=w.read(w.PROTOCOL)
    w.OUT.mkdir(parents=True,exist_ok=False)
    common=w.ROOT/p['source_common_path']
    manifest=w.read(common/'components/manifest.json')
    assert manifest['payload_sha256']==p['source_common_payload_sha256']
    assert w.read(common/'summary.json')['status']=='complete'
    # Local link only; published source manifest references the archived first wave.
    (w.OUT/'common').symlink_to(common,target_is_directory=True)
    w.write(w.OUT/'source_common.json',{'path':p['source_common_path'],
        'payload_sha256':manifest['payload_sha256'],'inherited_probe_tokens':90122,
        'new_probe_tokens':0,'conditional_repeats':2,'source_protocol':p['source_common_protocol']})
    w.write(w.OUT/'scheduler.json',{'pid':os.getpid(),'gpus':w.GPUS,'max_simultaneous_gpus':3,
        'protocol':str(w.PROTOCOL.relative_to(w.ROOT)),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'hard_seconds':HARD_SECONDS})
    queue=[{'kind':'baseline'}]+[{'kind':'branch','index':i,'condition':c,'rule':r}
        for i,c in enumerate(p['conditions']) for r in p['rules']]
    active=[];failed=[];status(active,queue,'running')
    publish('Launch frozen full-pool conditional-repeat BFCL study',active)
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
