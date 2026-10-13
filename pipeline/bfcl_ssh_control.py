"""Resource and explicit SSH-owner checks for the 2026-10-13 handoff.

The human transferred execution to the current SSH chat and asked it to continue.
The old chat's Goal remains untouched; it no longer owns this registered batch.
Frozen scientific dependencies, job identities and resource checks still apply.
"""
from pathlib import Path
import subprocess
from supervise_bfcl_stage_fork import ROOT, read, process_has_not_exited


def preflight(gpu, plan, authority):
    if authority['mode'] != 'human_authorized_ssh_takeover':
        raise RuntimeError('Unknown control authority')
    if authority['status'] != 'authorized' or (ROOT/authority['stop_file']).exists():
        raise RuntimeError('SSH owner stopped further job admission')
    if plan['allowed_gpus'] != [4, 5, 6, 7] or gpu not in plan['allowed_gpus']:
        raise RuntimeError('GPU outside the authorized four-card lanes')
    if plan['max_simultaneous_gpus'] != 4 or plan['automatic_retry'] is not False:
        raise RuntimeError('Resource/retry authorization differs')
    state=read(ROOT/'rl/STATUS.json')['bfcl_intervention']
    if state['phase'] != 'audited_complete' or state['active'] or state['pending'] or state['failures']:
        raise RuntimeError('Dose queue is not audited complete')
    if process_has_not_exited(state['scheduler_pid']):
        raise RuntimeError('Old dose scheduler still alive')
    source=state['path']+'/audit_status.json'
    if read(ROOT/source)['status'] != 'passed':
        raise RuntimeError('Previous scientific prerequisite audit did not pass')
    events=[e for e in read(ROOT/'experiments/index.json') if e['source']==source]
    if not events or read(ROOT/'experiments/reviews'/(events[-1]['id']+'.json'))['status'] != 'reviewed':
        raise RuntimeError('Previous batch has not been reviewed')
    info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True).splitlines()
    if int(info[1].strip())>64 or info[0].strip() in apps:
        raise RuntimeError('GPU occupied')
    return {'control_mode':authority['mode'],'authority_registered_at':authority['registered_at'],
            'gpu':gpu,'memory_mib':int(info[1].strip()),'no_compute_process':True}
