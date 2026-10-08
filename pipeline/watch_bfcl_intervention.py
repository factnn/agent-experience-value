"""Bounded CPU scheduler: one common state, <=3 idle GPUs, publish audit evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv-rl/bin/python'
RUNNER=ROOT/'pipeline/run_bfcl_intervention.py'
OUT=ROOT/'rl/bfcl_intervention_20261008'
PROTOCOL=ROOT/'rl/BFCL_INTERVENTION_PROTOCOL_20261008.json'
GPUS=['4','5','6']

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
def read(path):return json.loads(path.read_text())
def idle():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return {index.strip() for line in result.splitlines() for index,memory in [line.split(',')] if int(memory.strip())<=64}
def publish(label,active):
    paused=[]
    try:
        for job in active:
            if job['process'].poll() is None:
                os.kill(job['process'].pid,signal.SIGSTOP);paused.append(job['process'].pid)
        files=[]
        for path in OUT.rglob('*'):
            if path.suffix not in ['.json','.jsonl'] or 'adapter' in path.parts:continue
            raw=path.read_bytes()
            if raw and not raw.endswith(b'\n'):return False
            if path.suffix=='.json':json.loads(raw)
            files.append(str(path.relative_to(ROOT)))
        files+=['rl/STATUS.json']
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True,capture_output=True)
        changed=subprocess.run(['git','diff','--cached','--quiet','--',*files],cwd=ROOT).returncode
        if changed:subprocess.run(['git','commit','--only','-m',label,'--',*files],cwd=ROOT,check=True,capture_output=True)
    finally:
        for pid in paused:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
    pushed=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True).returncode==0
    print(json.dumps({'publication':'pushed' if pushed else 'push_failed',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}),flush=True)
    return pushed

def launch(mode,name,gpu,extra):
    assert gpu in idle(),'GPU is occupied; do not displace another user'
    command=[str(PYTHON),str(RUNNER),'--mode',mode,'--out',str((OUT/name).relative_to(ROOT)),*extra]
    log=(OUT/(name+'.console.log')).open('w')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;env['PYTHONUNBUFFERED']='1';env['TOKENIZERS_PARALLELISM']='false'
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job={'name':name,'mode':mode,'gpu':gpu,'process':process,'log':log,'start':time.monotonic(),
        'command':command,'out':OUT/name,'last_signature':None}
    write(OUT/(name+'.launch.json'),{'pid':process.pid,'gpu':gpu,'command':command,
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'hard_seconds':5400})
    return job

def status(active,pending,phase):
    current=read(ROOT/'rl/STATUS.json')
    current.update({'status':'bfcl_intervention_'+phase,'active_gpu_count':len(active),
        'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    current['bfcl_intervention']={'path':str(OUT.relative_to(ROOT)),'protocol':str(PROTOCOL.relative_to(ROOT)),
        'scheduler_pid':os.getpid(),'active':[{'name':j['name'],'gpu':j['gpu'],'pid':j['process'].pid} for j in active],
        'pending':pending,'phase':phase,'report':'22_bfcl_common_start_intervention.md'}
    write(ROOT/'rl/STATUS.json',current)

def check(job):
    if time.monotonic()-job['start']>5400 and job['process'].poll() is None:
        os.killpg(job['process'].pid,signal.SIGTERM)
        write(job['out']/'scheduler_failure.json',{'reason':'hard_5400_second_limit','partial_data_retained':True})
    return job['process'].poll()

def milestone(active,pending,phase):
    dirty=False
    for job in active:
        path=job['out']/'progress.json'
        if path.exists():
            try:signature=path.read_text();json.loads(signature)
            except (ValueError,OSError):continue
            if signature!=job['last_signature']:job['last_signature']=signature;dirty=True
    status(active,pending,phase)
    if dirty:publish('Publish common-start BFCL intervention milestone',active)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    p=read(PROTOCOL);write(OUT/'scheduler.json',{'pid':os.getpid(),'gpus':GPUS,'max_simultaneous_gpus':3,
        'protocol':str(PROTOCOL.relative_to(ROOT)),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Brief SIGSTOP only our child processes while staging closed evidence files; main wall times include publication pauses.'})
    common=launch('common','common',GPUS[0],[]);active=[common];status(active,[], 'building_common')
    publish('Launch frozen BFCL common-start probe state',active)
    while check(common) is None:
        milestone(active,[],'building_common');time.sleep(20)
    common['log'].close()
    if common['process'].returncode!=0:
        status([],[],'common_failed');publish('Retain failed common-state intervention attempt',[]);return
    assert read(OUT/'common/summary.json')['status']=='complete'
    status([],[],'common_ready');publish('Publish complete shared BFCL base and charged history',[])
    # Fixed queue order, independent of observed reward/evaluation outcomes.
    queue=[{'kind':'baseline'}]+[{'kind':'branch','index':i,'condition':c,'rule':r}
        for i,c in enumerate(p['conditions']) for r in p['rules']]
    active=[];failed=[]
    while queue or active:
        completed=[]
        for job in active:
            code=check(job)
            if code is None:continue
            job['log'].close();completed.append(job)
            if code:
                failed.append({'name':job['name'],'returncode':code});continue
            summary=read(job['out']/'summary.json')
            if summary['status']!='complete':failed.append({'name':job['name'],'status':summary['status']});continue
            if job['mode']=='train':
                queue.insert(0,{'kind':'eval','index':job['index'],'rule':job['rule'],'policy':str(job['out'])})
        for job in completed:active.remove(job)
        if completed:
            write(OUT/'failures.json',failed);status(active,queue,'running');publish('Publish finished BFCL intervention stage',active)
        free=idle()-{j['gpu'] for j in active}
        for gpu in GPUS:
            if len(active)>=3 or not queue:break
            if gpu not in free:continue
            spec=queue.pop(0)
            if spec['kind']=='baseline':job=launch('eval','eval_common',gpu,['--policy',str(OUT/'common')])
            elif spec['kind']=='branch':
                job=launch('train',f"train_{spec['index']}_{spec['rule']}",gpu,
                    ['--common',str(OUT/'common'),'--condition',spec['condition'],'--rule',spec['rule']])
                job.update({'index':spec['index'],'rule':spec['rule']})
            else:job=launch('eval',f"eval_{spec['index']}_{spec['rule']}",gpu,['--policy',spec['policy']])
            active.append(job)
        milestone(active,queue,'running');time.sleep(20)
    write(OUT/'failures.json',failed)
    if failed:
        status([],[],'incomplete');publish('Publish bounded incomplete BFCL intervention matrix',[]);return
    audit=subprocess.run([str(PYTHON),str(ROOT/'pipeline/report_bfcl_intervention.py'),'--root',str(OUT)],cwd=ROOT,capture_output=True,text=True)
    (OUT/'audit.console.log').write_text(audit.stdout+audit.stderr)
    write(OUT/'audit_status.json',{'returncode':audit.returncode,'status':'passed' if audit.returncode==0 else 'failed'})
    status([],[],'audited_complete' if audit.returncode==0 else 'audit_failed')
    publish('Publish audited common-start BFCL G and V matrix',[])
if __name__=='__main__':main()
