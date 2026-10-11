"""Fail closed on explicit Goal stops; briefly retry only unavailable reads."""
import os
from pathlib import Path
import subprocess
import time
from codex_goal_status import goal_status
from supervise_bfcl_stage_fork import ROOT, read, process_has_not_exited


def require_active(query, sleep=time.sleep):
    observations=[]
    for attempt in range(3):
        status=query();observations.append(status)
        if status=='active':return observations
        if status!='unknown':
            raise RuntimeError('Goal explicitly unavailable for work: '+str(status))
        if attempt<2:sleep(5)
    raise RuntimeError('Goal API unavailable after three read attempts: '+','.join(observations))


def preflight(gpu, plan):
    observations=require_active(lambda:goal_status(os.environ['CODEX_HOME'],os.environ['CODEX_THREAD_ID']))
    state=read(ROOT/'rl/STATUS.json')['bfcl_intervention']
    if state['phase']!='audited_complete' or state['active'] or state['pending'] or state['failures']:
        raise RuntimeError('Dose queue is not audited complete')
    if process_has_not_exited(state['scheduler_pid']):raise RuntimeError('Old dose scheduler still alive')
    source=state['path']+'/audit_status.json'
    assert read(ROOT/source)['status']=='passed'
    events=[e for e in read(ROOT/'experiments/index.json') if e['source']==source]
    assert events and read(ROOT/'experiments/reviews'/(events[-1]['id']+'.json'))['status']=='reviewed'
    assert gpu in plan['allowed_gpus']
    info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
                                  '--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if int(info[1].strip())>64 or info[0].strip() in apps:raise RuntimeError('GPU occupied')
    return {'goal_observations':observations,'gpu':gpu,'memory_mib':int(info[1].strip()),'no_compute_process':True}
