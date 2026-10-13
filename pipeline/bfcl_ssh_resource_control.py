"""Narrow resource continuation after the recorded V3 admission handoff."""
from pathlib import Path
import subprocess
from bfcl_source_eval_contract import pinned
from supervise_bfcl_source_training import digest
from supervise_bfcl_stage_fork import ROOT, read, process_has_not_exited

STOP_MESSAGE = 'SSH owner stopped further job admission'

def check_control(root, plan, authority, *, require_unstarted=False):
    if (authority.get('mode') != 'human_authorized_ssh_takeover'
            or authority.get('scope') != 'source_stage0_resource_continuation'
            or authority.get('status') != 'authorized'):
        raise ValueError('Current explicit resource-continuation authority required')
    allowed = authority['allowed_gpus']
    if (plan['allowed_gpus'] != allowed or len(allowed) != len(set(allowed))
            or not {0, 1}.issubset(allowed) or not set(allowed).issubset({0, 1, 4, 5, 6, 7})
            or not 2 <= len(allowed) <= 3
            or plan['max_simultaneous_gpus'] != len(allowed) or authority['max_simultaneous_gpus'] != len(allowed)
            or authority['aggregate_max_simultaneous_gpus'] != 6
            or plan['automatic_retry'] is not False):
        raise ValueError('Require newly authorized GPUs plus at most one released old lane; six aggregate GPUs, no retries')
    for name in ('stop_file', 'global_stop_file'):
        path = (root/authority[name]).resolve()
        if path.parent != (root/'experiments/.runtime').resolve():
            raise ValueError('Stop marker must be in project runtime')
        if path.exists():
            raise RuntimeError('Current SSH owner stopped resource-continuation admission')
    if authority['stop_file'] == authority['global_stop_file']:
        raise ValueError('Separate phase and global stop markers required')
    if pinned(root, authority['training_protocol']) != plan:
        raise ValueError('Authority does not bind this exact continuation')
    for path, sha in authority['control_input_sha256'].items():
        if digest(root/path) != sha:
            raise ValueError('Registered resource controller changed: '+path)
    goal = pinned(root, authority['owner_goal']['record'])
    resource = pinned(root, authority['resource_authorization'])
    if (goal['thread_id'] != authority['owner_goal']['thread_id']
            or goal['status_at_registration'] != 'active'
            or resource['goal_thread_id'] != goal['thread_id']
            or resource['goal_status_at_registration'] != 'active'
            or resource['new_gpus'] != [0, 1] or resource['max_total_parallel_jobs'] != 6):
        raise ValueError('Current Goal or explicit resource authorization differs')
    recovery = plan['execution_continuation']
    master_ref = recovery['original_scientific_protocol']
    master = pinned(root, master_ref)
    segments = recovery['previous_control_segments']
    if len(segments) != 3 or segments[0]['protocol'] != master_ref:
        raise ValueError('Require original, V2, V3 control lineage')
    prior_ref = segments[-1]['protocol']; prior = pinned(root, prior_ref)
    handoff = pinned(root, recovery['handoff_receipt'])
    if (handoff['status'] != 'authorized_transfer'
            or handoff['original_scientific_protocol'] != master_ref
            or handoff['source_protocol'] != prior_ref
            or handoff['source_output_root'] != prior['output_root']
            or handoff['source_stop_file'] != authority['global_stop_file']):
        raise ValueError('Handoff belongs to another authority or control segment')
    pending = handoff['transferred_pending']
    if not pending or len(set(pending)) != len(pending):
        raise ValueError('Handoff requires unique unstarted jobs')
    expected = [e for e in master['jobs'] if e['name'] in pending]
    if (len(expected) != len(pending) or plan['jobs'] != expected
            or plan['input_sha256'] != master['input_sha256']):
        raise ValueError('Scientific job identities or dependencies changed')
    lanes = plan['gpu_lanes']
    names = [n for lane in lanes.values() for n in lane]
    if set(lanes) != {str(g) for g in allowed} or len(names) != len(set(names)) or set(names) != set(pending):
        raise ValueError('Transferred jobs must occur exactly once across new lanes')
    snapshot = pinned(root, handoff['source_snapshot']['queue'])
    failures = pinned(root, handoff['source_snapshot']['failures'])
    failure = handoff['admission_stop_failure']
    if (failure.get('type') != 'RuntimeError' or failure.get('message') != STOP_MESSAGE
            or failure not in failures or failures != snapshot['failures']
            or failure['case'] not in snapshot['pending']
            or not set(pending).issubset(snapshot['pending'])
            or set(pending) & (set(snapshot['complete']) | set(snapshot['active'].values()))):
        raise ValueError('No proven V3 admission-stop latch for transferred jobs')
    prior_root = root/prior['output_root']
    actual = read(prior_root/'queue.json')
    actual_failures = read(prior_root/'failures.json')
    if (failure not in actual_failures or actual['failures'] != actual_failures
            or not set(pending).issubset(actual['pending'])
            or set(pending) & (set(actual['complete']) | set(actual['active'].values()))):
        raise ValueError('V3 changed ownership after the handoff snapshot')
    for entry in expected:
        case = entry['name']; job = pinned(root, entry)
        if (prior_root/(case+'.console.process.json')).exists():
            raise ValueError('Transferred job was already launched by V3: '+case)
        if require_unstarted and (root/job['output']).exists():
            raise ValueError('Transferred output already exists: '+case)
    return {'transferred_pending': pending, 'prior_control_root': prior['output_root'],
            'handoff_sha256': recovery['handoff_receipt']['sha256']}

def owned_training_pids(root, master):
    live = []
    for entry in master['jobs']:
        job = pinned(root, entry); path = root/job['output']/'run.json'
        if path.exists():
            record = read(path)
            if process_has_not_exited(record['pid']):
                live.append({'case': entry['name'], 'pid': record['pid'], 'gpu': record['gpu']})
    return live

def preflight(gpu, plan, authority):
    checked = check_control(ROOT, plan, authority)
    if gpu not in plan['allowed_gpus']:
        raise ValueError('GPU is outside this registered resource continuation')
    state = read(ROOT/'rl/STATUS.json')['bfcl_intervention']
    if (state['phase'] != 'audited_complete' or state['active'] or state['pending']
            or state['failures'] or process_has_not_exited(state['scheduler_pid'])):
        raise RuntimeError('Prior dose queue is not audited complete and exited')
    source = state['path']+'/audit_status.json'
    if read(ROOT/source)['status'] != 'passed':
        raise RuntimeError('Prior dose audit did not pass')
    events = [e for e in read(ROOT/'experiments/index.json') if e['source'] == source]
    if not events or read(ROOT/'experiments/reviews'/(events[-1]['id']+'.json'))['status'] != 'reviewed':
        raise RuntimeError('Prior dose batch has not been reviewed')
    master = pinned(ROOT, plan['execution_continuation']['original_scientific_protocol'])
    live = owned_training_pids(ROOT, master)
    if len(live) >= authority['aggregate_max_simultaneous_gpus']:
        raise RuntimeError('Six-GPU aggregate admission limit reached')
    info = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                   '--format=csv,noheader'],text=True).splitlines()
    if int(info[1].strip()) > 64 or info[0].strip() in apps:
        raise RuntimeError('GPU occupied')
    return checked | {'control_mode': authority['mode'], 'gpu':gpu, 'memory_mib':int(info[1]),
                      'no_compute_process':True, 'live_owned_training_pids':live,
                      'observed_project_live_pids':live, 'aggregate_live_before_admission':len(live), 'aggregate_max_simultaneous_gpus':6}
