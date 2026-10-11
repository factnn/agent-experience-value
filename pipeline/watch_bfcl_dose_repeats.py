"""Four-GPU bounded queue: three new dose trajectories and two eval seeds."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime,timezone
import watch_bfcl_dose as d

w=d.w
MASTER=w.ROOT/'rl/BFCL_DOSE_REPEATS_PROTOCOL_20261011.json'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_path(spec):
    path=w.ROOT/spec['protocol']['path']
    assert digest(path)==spec['protocol']['sha256']
    return path


def plan(p):
    """Task eligibility depends on complete snapshots, never observed scores."""
    jobs=[]
    for repeat in p['training_repeats']:
        jobs.append({'name':'train_'+repeat['name'],'mode':'train','protocol':repeat['protocol'],
            'extra':['--common',str(w.ROOT/p['source_common_path'])]})
    e1=p['evaluation_repeats'][1]
    jobs.append({'name':'eval_common_e1','mode':'eval','protocol':e1['protocol'],
        'extra':['--policy',str(w.ROOT/p['source_common_path'])]})
    old=w.ROOT/p['previous_wave']['path']
    old_labels=w.read(old/'train/dose_checkpoints.json');seen=set()
    for label in old_labels:
        if label['policy'] in seen:continue
        seen.add(label['policy'])
        jobs.append({'name':f"eval_discovery_step_{label['optimizer_steps']:03d}_e1",'mode':'eval',
            'protocol':e1['protocol'],'extra':['--policy',str(w.ROOT/label['policy'])]})
    for repeat in p['training_repeats']:
        ledger=w.OUT/('train_'+repeat['name'])/'dose_checkpoints.json'
        seen=set()
        for label in w.read(ledger) if ledger.exists() else []:
            policy=w.ROOT/label['policy']
            if label['policy'] in seen:continue
            seen.add(label['policy'])
            assert w.read(policy/'summary.json')['status']=='complete'
            for evaluation in p['evaluation_repeats']:
                jobs.append({'name':f"eval_{repeat['name']}_step_{label['optimizer_steps']:03d}_{evaluation['name']}",
                    'mode':'eval','protocol':evaluation['protocol'],'extra':['--policy',str(policy)]})
    return jobs


def available_jobs(p, done, active):
    running={j['name'] for j in active}
    return [s for s in plan(p) if s['name'] not in done and s['name'] not in running]


def status(p,active,pending,phase,failures):
    s=w.read(w.ROOT/'rl/STATUS.json')
    s.update({'status':'bfcl_dose_repeats_'+phase,'active_gpu_count':len(active),
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),'latest_report':'31_dose_repeats_design.md'})
    s['bfcl_intervention']={'path':p['output'],'protocol':str(MASTER.relative_to(w.ROOT)),
        'scheduler_pid':os.getpid(),'phase':phase,'report':'31_dose_repeats_design.md',
        'active':[{'name':j['name'],'gpu':j['gpu'],'pid':j['process'].pid} for j in active],
        'pending':[j['name'] for j in pending],
        'future_evaluations':'All unique dose snapshots from the three new training trajectories, at both predeclared eval seeds.',
        'failures':failures,'training_repeats':3,'evaluation_repeats':2,'max_simultaneous_gpus':4}
    w.write(w.ROOT/'rl/STATUS.json',s)


def launch(spec,gpu,hard):
    job=w.launch(spec['mode'],spec['name'],gpu,[*spec['extra'],'--protocol',str(protocol_path(spec))])
    path=w.OUT/(spec['name']+'.launch.json');row=w.read(path)
    row['hard_seconds']=hard;row['protocol']=spec['protocol'];w.write(path,row)
    return job


def recover(p):
    active=[];done=set();failures=[]
    for spec in plan(p):
        path=w.OUT/(spec['name']+'.launch.json')
        if not path.exists():
            assert not (w.OUT/spec['name']).exists(),'Unrecorded output requires inspection, not resampling'
            continue
        record=w.read(path);assert record['protocol']==spec['protocol']
        job=d.adopt(spec);code=job['process'].poll()
        if code is None:active.append(job)
        else:
            job['log'].close();done.add(job['name'])
            if code:failures.append({'name':job['name'],'reason':'exited_without_complete_summary',
                                    'original_exit_status_unavailable':True})
    return active,done,failures


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    lock=open('/tmp/agent-experience-bfcl-dose-repeats-20261011.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=w.read(MASTER);w.OUT=w.ROOT/p['output'];w.RUNNER=w.ROOT/'pipeline/run_bfcl_dose.py'
    hard=p['hard_seconds_per_process'];cap=p['max_simultaneous_gpu_processes'];assert cap==4
    assert digest(w.ROOT/p['previous_protocol']['path'])==p['previous_protocol']['sha256']
    assert digest(w.ROOT/p['previous_wave']['path']/'comparison.json')==p['previous_wave']['comparison_sha256']
    for spec in p['training_repeats']+p['evaluation_repeats']:protocol_path(spec)
    manifest=w.read(w.ROOT/p['source_common_path']/'components/manifest.json')
    assert manifest['payload_sha256']==p['source_common_payload_sha256']
    if args.resume:active,done,failures=recover(p)
    else:w.OUT.mkdir(parents=True,exist_ok=False);active=[];done=set();failures=[]
    assert len(active)<=cap and len({j['gpu'] for j in active})==len(active)
    w.write(w.OUT/'scheduler.json',{'pid':os.getpid(),'gpus':p['allowed_gpus'],
        'max_simultaneous_gpus':cap,'protocol':str(MASTER.relative_to(w.ROOT)),
        'protocol_sha256':digest(MASTER),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'hard_seconds':hard,'resume':args.resume,'resampling_permitted':False,
        'source_sha256':digest(Path(__file__)),
        'note':'Queue fills idle authorized cards; adopted jobs keep original UTC deadlines. Publication pauses only own workers.'})
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
        pending=available_jobs(p,done,active);free=w.idle()-{j['gpu'] for j in active}
        for gpu in p['allowed_gpus']:
            if len(active)>=cap or not pending:break
            if gpu not in free:continue
            spec=pending.pop(0)
            assert not (w.OUT/(spec['name']+'.launch.json')).exists(),'Never rerun an observed job'
            active.append(launch(spec,gpu,hard));dirty=True
        w.write(w.OUT/'failures.json',failures)
        status(p,active,pending,'running' if active else 'waiting_for_idle_gpu',failures)
        publication_pending=publication_pending or dirty
        if publication_pending:publication_pending=not d.publish('Publish four-GPU BFCL dose repeats progress',active)
        if not active and not pending:break
        time.sleep(20)
    audit=subprocess.run([str(w.PYTHON),str(w.ROOT/'pipeline/report_bfcl_dose_repeats.py'),
        '--protocol',str(MASTER)],cwd=w.ROOT,capture_output=True,text=True)
    (w.OUT/'audit.console.log').write_text(audit.stdout+audit.stderr)
    w.write(w.OUT/'audit_status.json',{'returncode':audit.returncode,
        'status':'passed' if audit.returncode==0 else 'failed_or_incomplete'})
    phase='audited_complete' if audit.returncode==0 and not failures else 'incomplete'
    status(p,[],[],phase,failures)
    d.publish('Publish audited BFCL training and evaluation dose repeats' if phase=='audited_complete'
        else 'Retain bounded incomplete BFCL dose repeats',[])


if __name__=='__main__':main()
