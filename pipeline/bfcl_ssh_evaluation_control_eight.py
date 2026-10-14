"""Bind each frozen evaluation phase to the current human-authorized SSH owner.

The training audit, research review, phase publication and heldout seal gates
remain mandatory. This module never queries or changes the historical Goal.
"""
import subprocess
from bfcl_source_eval_contract import pinned
from bfcl_ssh_control import preflight as resource_preflight
from supervise_bfcl_stage_fork import ROOT, read, process_has_not_exited
from supervise_bfcl_source_training import digest


def resource_allocation(root, reference):
    """The current explicit eight-card administrative amendment, not a science change."""
    if reference['path'] != 'experiments/SSH_FUTURE_GPU_CAP_V3_20261014.json':
        raise ValueError('Unknown evaluation resource amendment')
    cap = pinned(root, reference)
    if cap['max_simultaneous_gpus_for_subsequent_batches'] != 8:
        raise ValueError('The explicit eight-card resource amendment is required')
    return list(range(8)), 8


def check_control(root, plan, authority):
    if (authority.get('mode') != 'human_authorized_ssh_takeover'
            or authority.get('scope') != 'source_stage0_evaluation'
            or authority.get('status') != 'authorized'):
        raise ValueError('Evaluation must have explicit current SSH ownership')
    resource_ref = plan.get('resource_authorization')
    if resource_ref is not None:
        gpus, maximum = resource_allocation(root, resource_ref)
        if (authority.get('resource_authorization') != resource_ref
                or plan['allowed_gpus'] != gpus or authority.get('allowed_gpus') != gpus
                or plan['max_simultaneous_gpus'] != maximum
                or authority.get('max_simultaneous_gpus') != maximum
                or plan['automatic_retry'] is not False):
            raise ValueError('Eight-card evaluation lacks exact current resource authority')
    elif authority.get('resource_authorization') is not None:
        raise ValueError('Resource authority cannot substitute an unamended manifest')
    phase = plan['phase']
    if (phase not in ('fit', 'heldout') or authority['phase'] != phase
            or len(plan['jobs']) != {'fit': 80, 'heldout': 24}[phase]):
        raise ValueError('Authority or panel count belongs to a different phase')
    if pinned(root, authority['evaluation_manifest']) != plan:
        raise ValueError('Authority is not bound to these exact evaluation bytes')
    if authority['execution_protocol'] != plan['execution_protocol']:
        raise ValueError('Authority is bound to a different scientific protocol')
    for name in ('stop_file', 'global_stop_file'):
        relative = authority[name]
        path = (root/relative).resolve()
        if path.parent != (root/'experiments/.runtime').resolve():
            raise ValueError('Stop marker must belong to this project runtime')
        if path.exists():
            raise RuntimeError('Current SSH owner stopped further evaluation admission')
    goal = pinned(root, authority['owner_goal']['record'])
    if (goal['thread_id'] != authority['owner_goal']['thread_id']
            or goal['status_at_registration'] != 'active'
            or authority['current_goal_status_at_registration'] != 'active'):
        raise ValueError('Evaluation owner does not match the registered current Goal')
    master = pinned(root, plan['execution_protocol'])
    audit_ref = authority['complete_training_audit']
    if audit_ref['path'] != master['output_root']+'/audit_status.json':
        raise ValueError('Training certificate belongs to another matrix')
    batch = pinned(root, audit_ref)
    if batch['status'] != 'passed' or batch['complete_jobs'] != 51:
        raise ValueError('The complete original 51-policy audit is mandatory')
    review_ref = authority['complete_training_review']
    checked = pinned(root, review_ref)
    event_id = review_ref['path'].rsplit('/', 1)[-1].removesuffix('.json')
    if review_ref['path'] != 'experiments/reviews/'+event_id+'.json' or checked['status'] != 'reviewed':
        raise ValueError('Completed training matrix review is mandatory')
    events = [e for e in read(root/'experiments/index.json')
              if e['source'] == audit_ref['path'] and e['id'] == event_id]
    if len(events) != 1:
        raise ValueError('Review is not for the exact complete training matrix event')
    for path, sha in authority['control_input_sha256'].items():
        if digest(root/path) != sha:
            raise ValueError('Registered evaluation controller changed: '+path)
    return {'phase': phase, 'owner_goal_thread': goal['thread_id'],
            'complete_training_audit_sha256': audit_ref['sha256'],
            'complete_training_review_event': event_id}


def preflight(gpu, plan, authority):
    owner = check_control(ROOT, plan, authority)
    if plan.get('resource_authorization') is None:
        resource = resource_preflight(gpu, plan, authority)
    else:
        if gpu not in plan['allowed_gpus']:
            raise RuntimeError('GPU outside the authorized evaluation lanes')
        # Preserve every prior-dose gate from the immutable four-card guard.
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
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip().split(',')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True).splitlines()
        if int(info[1].strip()) > 64 or info[0].strip() in apps:
            raise RuntimeError('GPU occupied')
        resource = {'control_mode': authority['mode'], 'gpu': gpu,
                    'memory_mib': int(info[1].strip()), 'no_compute_process': True,
                    'resource_authorization': plan['resource_authorization']}
    return resource | owner
