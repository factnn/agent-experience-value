"""Durable one-GPU, fixed-order dose scheduler with bounded jobs and publication."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
import watch_bfcl_intervention as w

w.OUT = w.ROOT/'rl/bfcl_intervention_dose_20261010'
w.PROTOCOL = w.ROOT/'rl/BFCL_DOSE_PROTOCOL_20261010.json'
w.RUNNER = w.ROOT/'pipeline/run_bfcl_dose.py'
w.GPUS = ['4']


class AdoptedProcess:
    def __init__(self, record, out):
        self.pid=record['pid']; self.command=record['command']; self.out=out
        self.returncode=None
    def poll(self):
        proc=Path('/proc')/str(self.pid)
        try:
            state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
            command=(proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
            if state!='Z' and command==[s.encode() for s in self.command]: return None
        except FileNotFoundError: pass
        path=self.out/'summary.json'
        self.returncode=0 if path.exists() and w.read(path)['status']=='complete' else 1
        return self.returncode


def plan(p):
    specs=[{'name':'train','mode':'train','extra':['--common',str(w.ROOT/p['source_common_path'])]},
           {'name':'eval_common','mode':'eval','extra':['--policy',str(w.ROOT/p['source_common_path'])]}]
    ledger=w.OUT/'train/dose_checkpoints.json'
    seen=set()
    for row in w.read(ledger) if ledger.exists() else []:
        if row['policy'] in seen: continue
        seen.add(row['policy'])
        specs.append({'name':f"eval_step_{row['optimizer_steps']:03d}",'mode':'eval',
            'extra':['--policy',str(w.ROOT/row['policy'])]})
    return specs


def status(active, pending, phase, failures):
    w.status(active, pending, phase)
    current=w.read(w.ROOT/'rl/STATUS.json')
    current['status']='bfcl_dose_'+phase
    current['bfcl_intervention']['report']='29_dose_execution_protocol.md'
    current['bfcl_followup_complete']={'status':'audited_complete',
        'comparison':'rl/bfcl_intervention_followup_20261009/comparison.json',
        'report':'28_full_pool_followup_complete_result.md'}
    current['bfcl_intervention']['failures']=failures
    w.write(w.ROOT/'rl/STATUS.json',current)


def publish(label, active):
    for attempt in range(3):
        try:
            if w.publish(label, active): return True
        except subprocess.CalledProcessError:
            print(json.dumps({'publication':'git_retry','attempt':attempt+1}),flush=True)
        time.sleep(3)
    return False


def adopt(spec):
    record=w.read(w.OUT/(spec['name']+'.launch.json'))
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(record['started_at_utc'])).total_seconds()
    return {'name':spec['name'],'mode':spec['mode'],'gpu':record['gpu'],
        'process':AdoptedProcess(record,w.OUT/spec['name']),
        'log':(w.OUT/(spec['name']+'.console.log')).open('a'),
        'start':time.monotonic()-elapsed,'out':w.OUT/spec['name'],
        'command':record['command'],'last_signature':None}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    lock=open('/tmp/agent-experience-bfcl-dose-20261010.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=w.read(w.PROTOCOL); hard=p['schedule']['hard_seconds_per_process']
    manifest=w.read(w.ROOT/p['source_common_path']/'components/manifest.json')
    assert manifest['payload_sha256']==p['source_common_payload_sha256']
    assert w.read(w.ROOT/p['source_common_path']/'summary.json')['status']=='complete'
    if not args.resume: w.OUT.mkdir(parents=True,exist_ok=False)
    else: assert w.OUT.is_dir()
    w.write(w.OUT/'scheduler.json',{'pid':os.getpid(),'gpus':w.GPUS,'max_simultaneous_gpus':1,
        'protocol':str(w.PROTOCOL.relative_to(w.ROOT)),
        'protocol_sha256':hashlib.sha256(w.PROTOCOL.read_bytes()).hexdigest(),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'hard_seconds':hard,
        'resume':args.resume,'resampling_permitted':False,
        'note':'Brief publication pauses only our child process; wall time includes pauses.'})
    w.write(w.OUT/'source_common.json',{'path':p['source_common_path'],
        'payload_sha256':manifest['payload_sha256'],'inherited_probe_tokens':90122,'new_probe_tokens':0})
    done=set(); active=[]; failures=[]; publication_pending=True
    if args.resume:
        for spec in plan(p):
            if (w.OUT/(spec['name']+'.launch.json')).exists():
                job=adopt(spec); code=job['process'].poll()
                if code is None: active.append(job)
                else:
                    job['log'].close(); done.add(job['name'])
                    if code: failures.append({'name':job['name'],'reason':'exited_without_complete_summary',
                                              'original_exit_status_unavailable':True})
        assert len(active)<=1
    while True:
        dirty=False
        if active:
            job=active[0]; code=job['process'].poll()
            elapsed=time.monotonic()-job['start']
            if code is None and elapsed>hard:
                sig=signal.SIGKILL if elapsed>hard+30 else signal.SIGTERM
                try: os.killpg(job['process'].pid,sig)
                except ProcessLookupError: pass
                w.write(job['out']/'scheduler_failure.json',{'reason':'hard_wall_limit',
                    'hard_seconds':hard,'partial_data_retained':True})
            if code is not None:
                job['log'].close();done.add(job['name']);active=[];dirty=True
                summary=job['out']/'summary.json'
                if code or not summary.exists() or w.read(summary)['status']!='complete':
                    failures.append({'name':job['name'],'returncode':code,'partial_data_retained':True})
            else:
                path=job['out']/'progress.json'
                if path.exists():
                    try:
                        signature=path.read_text();json.loads(signature)
                    except (ValueError,OSError): signature=None
                    if signature and signature!=job['last_signature']:
                        job['last_signature']=signature;dirty=True
        pending=[s for s in plan(p) if s['name'] not in done and s['name'] not in {j['name'] for j in active}]
        if not active and pending and '4' in w.idle():
            spec=pending.pop(0)
            assert not (w.OUT/(spec['name']+'.launch.json')).exists(), 'never resample a launched job'
            job=w.launch(spec['mode'],spec['name'],'4',[*spec['extra'],'--protocol',str(w.PROTOCOL)])
            record=w.read(w.OUT/(spec['name']+'.launch.json'));record['hard_seconds']=hard
            w.write(w.OUT/(spec['name']+'.launch.json'),record)
            active=[job];dirty=True
        w.write(w.OUT/'failures.json',failures)
        status(active,[s['name'] for s in pending],'running' if active else 'waiting_for_idle_gpu',failures)
        publication_pending=publication_pending or dirty
        if publication_pending:
            publication_pending=not publish('Publish frozen BFCL dose experiment progress',active)
        if not active and not pending: break
        time.sleep(20)
    audit=subprocess.run([str(w.PYTHON),str(w.ROOT/'pipeline/report_bfcl_dose.py')],
        cwd=w.ROOT,capture_output=True,text=True)
    (w.OUT/'audit.console.log').write_text(audit.stdout+audit.stderr)
    w.write(w.OUT/'audit_status.json',{'returncode':audit.returncode,
        'status':'passed' if audit.returncode==0 else 'failed_or_incomplete'})
    phase='audited_complete' if audit.returncode==0 and not failures else 'incomplete'
    status([],[],phase,failures)
    publish('Publish audited BFCL development dose curve' if phase=='audited_complete'
        else 'Retain bounded incomplete BFCL dose experiment',[])


if __name__=='__main__': main()
