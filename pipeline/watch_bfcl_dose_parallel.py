"""Resource-only dose scheduling amendment: adopt training, evaluate saved states."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
import watch_bfcl_dose as d

w=d.w
AMENDMENT=w.ROOT/'rl/BFCL_DOSE_RESOURCE_AMENDMENT_20261010.json'


def choose_gpu(spec, active, idle, done, allowed):
    """Never assign the training GPU to evaluation while training is pending/live."""
    busy={j['gpu'] for j in active}
    candidates=['4'] if spec['mode']=='train' else ['5','6','7']+(['4'] if 'train' in done else [])
    return next((g for g in candidates if g in allowed and g in idle and g not in busy), None)


def recover(p):
    active=[];done=set();failures=[]
    for spec in d.plan(p):
        path=w.OUT/(spec['name']+'.launch.json')
        if not path.exists():
            assert not (w.OUT/spec['name']).exists(), 'Unrecorded output must be inspected, never resampled'
            continue
        job=d.adopt(spec);code=job['process'].poll()
        if code is None: active.append(job)
        else:
            job['log'].close();done.add(job['name'])
            if code:failures.append({'name':job['name'],'reason':'exited_without_complete_summary',
                                    'original_exit_status_unavailable':True,'partial_data_retained':True})
    return active,done,failures


def main():
    lock=open('/tmp/agent-experience-bfcl-dose-20261010.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=w.read(w.PROTOCOL);amendment=w.read(AMENDMENT)
    protocol_hash=hashlib.sha256(w.PROTOCOL.read_bytes()).hexdigest()
    assert protocol_hash==amendment['unchanged_scientific_protocol_sha256']
    hard=p['schedule']['hard_seconds_per_process'];allowed=amendment['allowed_gpus']
    cap=amendment['max_simultaneous_gpu_processes'];assert cap<=4
    previous=w.read(w.OUT/'scheduler.json')
    archive=w.OUT/f"scheduler_previous_{previous['pid']}.json"
    if not archive.exists(): w.write(archive,previous)
    active,done,failures=recover(p)
    assert len(active)<=cap and len({j['gpu'] for j in active})==len(active)
    assert all(j['gpu'] in allowed for j in active)
    w.write(w.OUT/'scheduler.json',{'pid':os.getpid(),'previous_scheduler_pid':previous['pid'],
        'gpus':allowed,'max_simultaneous_gpus':cap,'protocol':str(w.PROTOCOL.relative_to(w.ROOT)),
        'protocol_sha256':protocol_hash,'resource_amendment':str(AMENDMENT.relative_to(w.ROOT)),
        'started_at_utc':previous['started_at_utc'],'replacement_started_at_utc':datetime.now(timezone.utc).isoformat(),
        'hard_seconds':hard,'resume':True,'resampling_permitted':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'note':'Training adopted without interruption. Per-job deadlines inherit original launch UTC; original exit codes of adopted jobs unavailable.'})
    w.write(w.OUT/f"scheduler_recovery_{os.getpid()}.json",{'previous_scheduler_pid':previous['pid'],
        'replacement_scheduler_pid':os.getpid(),'adopted':[{'name':j['name'],'pid':j['process'].pid,
            'gpu':j['gpu']} for j in active],'budgets_reset':False,'resampled_jobs':0,
        'reason':'User authorized parallel use of currently idle GPUs, within original four-GPU ceiling.'})
    publication_pending=True
    while True:
        dirty=False
        for job in list(active):
            code=job['process'].poll();elapsed=time.monotonic()-job['start']
            if code is None and elapsed>hard:
                try:os.killpg(job['process'].pid,signal.SIGKILL if elapsed>hard+30 else signal.SIGTERM)
                except ProcessLookupError:pass
                w.write(job['out']/'scheduler_failure.json',{'reason':'hard_wall_limit',
                    'hard_seconds':hard,'partial_data_retained':True})
            if code is not None:
                job['log'].close();done.add(job['name']);active.remove(job);dirty=True
                path=job['out']/'summary.json'
                if code or not path.exists() or w.read(path)['status']!='complete':
                    failures.append({'name':job['name'],'returncode':code,'partial_data_retained':True})
            else:
                path=job['out']/'progress.json'
                if path.exists():
                    try:signature=path.read_text();json.loads(signature)
                    except (OSError,ValueError):signature=None
                    if signature and signature!=job['last_signature']:
                        job['last_signature']=signature;dirty=True
        pending=[s for s in d.plan(p) if s['name'] not in done and s['name'] not in {j['name'] for j in active}]
        idle=w.idle()
        for spec in list(pending):
            if len(active)>=cap:break
            gpu=choose_gpu(spec,active,idle,done,allowed)
            if gpu is None:continue
            assert not (w.OUT/(spec['name']+'.launch.json')).exists(), 'Never relaunch an observed job'
            job=w.launch(spec['mode'],spec['name'],gpu,[*spec['extra'],'--protocol',str(w.PROTOCOL)])
            record=w.read(w.OUT/(spec['name']+'.launch.json'));record['hard_seconds']=hard
            record['resource_amendment']=str(AMENDMENT.relative_to(w.ROOT))
            w.write(w.OUT/(spec['name']+'.launch.json'),record)
            active.append(job);pending.remove(spec);dirty=True
        w.write(w.OUT/'failures.json',failures)
        d.status(active,[s['name'] for s in pending],'running' if active else 'waiting_for_idle_gpu',failures)
        publication_pending=publication_pending or dirty
        if publication_pending:publication_pending=not d.publish('Publish parallel BFCL dose development progress',active)
        if not active and not pending:break
        time.sleep(20)
    audit=subprocess.run([str(w.PYTHON),str(w.ROOT/'pipeline/report_bfcl_dose.py')],
        cwd=w.ROOT,capture_output=True,text=True)
    (w.OUT/'audit.console.log').write_text(audit.stdout+audit.stderr)
    w.write(w.OUT/'audit_status.json',{'returncode':audit.returncode,
        'status':'passed' if audit.returncode==0 else 'failed_or_incomplete'})
    phase='audited_complete' if audit.returncode==0 and not failures else 'incomplete'
    d.status([],[],phase,failures)
    d.publish('Publish audited BFCL development dose curve' if phase=='audited_complete'
        else 'Retain bounded incomplete BFCL dose experiment',[])


if __name__=='__main__':main()
