"""Persist/push authorized pilot milestones and run final audits after exit.

Run as a detached CPU process. Git scope is limited to these pilot artifacts and
STATUS.json; it never stages docs/, adapters, or unrelated workspace changes.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
NAMES=['allocation_pilot_20261008_uniform','allocation_pilot_20261008_frontier']


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def alive(pid):
    if not pid:return False
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def call(args):return subprocess.run(args,cwd=ROOT,text=True,capture_output=True)


def main():
    previous=None;audited=set();pair_audited=False;started=time.monotonic();pending_push=False
    while True:
        runs=[];stage=['rl/STATUS.json']
        for name in NAMES:
            path=ROOT/'rl'/name;run=read(path/'run.json',{})
            if not run:continue
            summary=read(path/'summary.json');progress=read(path/'progress.json',{})
            evaluating=read(path/'evaluation.json',[]);running=alive(run.get('pid'))
            status='complete' if summary else ('evaluating' if evaluating else 'training') if running else 'failed_or_timed_out'
            if summary and name not in audited:
                result=call(['.venv-rl/bin/python','pipeline/report_rl_allocation.py',str(path)])
                print(json.dumps({'audit_run':name,'exit_code':result.returncode,'output':result.stdout,'error':result.stderr}),flush=True)
                if result.returncode==0:audited.add(name)
                else:status='audit_failed'
            entry={'rule':run['rule'],'path':f'rl/{name}','status':status,
                'pid':run.get('pid') if running else None,'gpu':run['gpu'],
                'progress':progress,'completed_eval_tasks':len(evaluating)}
            if summary:entry['summary']=f'rl/{name}/summary.json'
            runs.append(entry)
            for file in ['run.json','tasks.json','trainer_config.json','progress.json','allocation.jsonl',
                         'updates.jsonl','metrics.jsonl','rollouts.jsonl','groups.jsonl','evaluation.json']:
                if (path/file).exists():stage.append(f'rl/{name}/{file}')
            if summary or not running:
                for file in ['summary.json','audit_report.json','generation_calls.jsonl']:
                    if (path/file).exists():stage.append(f'rl/{name}/{file}')
                stage.append(f'rl/{name}.console.log')
        if len(audited)==2 and not pair_audited:
            result=call(['.venv-rl/bin/python','pipeline/report_rl_allocation.py',*[str(ROOT/'rl'/n) for n in NAMES]])
            print(json.dumps({'pair_audit_exit_code':result.returncode,'output':result.stdout,'error':result.stderr}),flush=True)
            pair_audited=result.returncode==0
        if pair_audited:stage.append('rl/allocation_pilot_20261008_comparison.json')
        active=sum(alive(r.get('pid')) for r in runs)
        done=len(runs)==2 and active==0 and all(r['status'] in ['complete','failed_or_timed_out','audit_failed'] for r in runs)
        status={'status':'allocation_pilot_complete' if pair_audited else 'allocation_pilot_finished_with_errors' if done else 'allocation_pilot_running',
                'updated_at_utc':datetime.now(timezone.utc).isoformat(),'active_gpu_count':active,
                'runs':runs,'protocol':'rl/ALLOCATION_PILOT_PROTOCOL.md',
                'note':'Single-seed engineering pilot; final comparison requires both audits. All generated costs charged; no confirmed efficacy claim.'}
        signature=[(r['status'],r['progress'].get('optimizer_steps'),r['completed_eval_tasks']) for r in runs]+[pair_audited,active]
        if len(runs)==2 and signature!=previous:
            (ROOT/'rl/STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
            added=call(['git','add','--',*stage])
            if added.returncode==0:
                changed=call(['git','diff','--cached','--quiet','--',*stage]).returncode
                if changed==1:
                    committed=call(['git','commit','-m','Record online allocation pilot milestone','--',*stage])
                    print(json.dumps({'commit_exit_code':committed.returncode,'output':committed.stdout,'error':committed.stderr}),flush=True)
                    pending_push=committed.returncode==0 or pending_push
                previous=signature
        if pending_push:
            result=call(['git','push','origin','main'])
            print(json.dumps({'push_exit_code':result.returncode,'output':result.stdout,'error':result.stderr}),flush=True)
            pending_push=result.returncode!=0
        if done and not pending_push:break
        if time.monotonic()-started>7500:break
        time.sleep(30)


if __name__=='__main__':main()
