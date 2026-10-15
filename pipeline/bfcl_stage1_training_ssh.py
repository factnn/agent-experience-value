"""Current SSH adapter for the frozen51 stage1 RL jobs; no automatic science retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from bfcl_source_eval_contract import digest, pinned
from bfcl_stage1_matrix import validate_jobs
from bfcl_stage1_probe_ssh import execute_lanes, verify_publication
from bfcl_stage1_forward_bundle import verify_bundle
from supervise_bfcl_stage_fork import ROOT, read, write, supervise, process_has_not_exited

SELF = 'pipeline/bfcl_stage1_training_ssh.py'
SCOPE = 'stage1_fixed_parent_source_training_51'
CAP = 'experiments/SSH_FUTURE_GPU_CAP_V3_20261014.json'
CONTROL_CODE = {SELF, 'pipeline/bfcl_stage1_matrix.py', 'pipeline/bfcl_stage1_forward_bundle.py',
                'pipeline/bfcl_stage1_probe_ssh.py', 'pipeline/supervise_bfcl_stage1_probe.py',
                'pipeline/supervise_bfcl_stage_fork.py', 'pipeline/train_bfcl_source_group.py',
                'pipeline/report_bfcl_source_training.py'}


def ref(root, path):
    return {'path': str(path.resolve().relative_to(root.resolve())), 'sha256': digest(path)}


def exclusive(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def input_digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def admission_failure_category(phase, error):
    if 'Current SSH owner stopped' in str(error):
        return 'current_control_stop'
    if phase == 'publication' and (isinstance(error, subprocess.TimeoutExpired) or 'Publication dependency unavailable' in str(error)):
        return 'publication_dependency_query'
    if isinstance(error, OSError):
        return 'execution_input_or_query_dependency'
    return 'control_identity_or_resource_admission'


def worker_failure_category(control, failure):
    phase = failure['phase']
    if phase == 'runner':
        if not (control/(failure['case']+'.console.process.json')).exists():
            return 'control_or_execution_dependency_before_scientific_child'
        return 'scientific_process_failure'
    if phase == 'preflight':
        if failure['type'] in ('OSError', 'FileNotFoundError', 'TimeoutExpired'):
            return 'execution_evidence_or_gpu_query_dependency'
        return 'control_admission'
    return {'audit': 'raw_training_audit_failure', 'batch_audit': 'full_matrix_audit_failure'}.get(phase, 'execution_failure')


def load_plan(root, protocol_ref):
    master = pinned(root, protocol_ref)
    if (master['kind'] != 'source_stage1_training_and_transfer_execution' or master['stage'] != 1
            or master['allowed_gpus'] != list(range(8)) or master['max_simultaneous_gpus'] != 8
            or master['automatic_retry'] is not False or master['training_jobs'] != 51):
        raise ValueError('Require the original registered51 stage1/eight-free-card design')
    cap = pinned(root, master['resource_authorization'])
    if master['resource_authorization']['path'] != CAP or cap['max_simultaneous_gpus_for_subsequent_batches'] != 8:
        raise ValueError('Wrong current GPU resource amendment')
    early = pinned(root, master['stage0_execution_protocol'])
    probe = pinned(root, master['stage1_probe_protocol'])
    parent = pinned(root, master['parent_binding'])
    jobs = [pinned(root, entry) for entry in master['jobs']]
    validate_jobs(root, early, probe, parent, jobs)
    names = [entry['name'] for entry in master['jobs']]
    scheduled = [name for lane in master['gpu_lanes'].values() for name in lane]
    if (set(master['gpu_lanes']) != {str(g) for g in range(8)} or len(scheduled) != 51
            or set(scheduled) != set(names) or any(Path(n).name != n or n in ('.', '..') for n in names)
            or names != [job['case_id'] for job in jobs]):
        raise ValueError('Missing,duplicated,substituted or unsafe training job identity')
    return master, jobs


def check_scope(root, protocol_ref, master, authority_ref, authority):
    if (authority.get('mode') != 'human_authorized_ssh_takeover' or authority.get('status') != 'authorized'
            or authority.get('scope') != SCOPE or authority.get('execution_protocol') != protocol_ref
            or authority.get('resource_authorization') != master['resource_authorization']):
        raise ValueError('Wrong current stage1 training ownership/scope')
    if digest(root/authority_ref['path']) != authority_ref['sha256']:
        raise ValueError('Registered current training authority changed')
    for name in ('stop_file', 'global_stop_file'):
        path = (root/authority[name]).resolve()
        if path.parent != (root/'experiments/.runtime').resolve():
            raise ValueError('Stop marker outside current project runtime')
        if path.exists():
            raise RuntimeError('Current SSH owner stopped new stage1 training admission')
    owner = pinned(root, authority['owner_goal']['record'])
    if (owner['thread_id'] != authority['owner_goal']['thread_id'] or owner['status_at_registration'] != 'active'
            or owner['control_scope'] != SCOPE):
        raise ValueError('Wrong current stage1 training owner record')
    if not CONTROL_CODE <= authority['control_input_sha256'].keys():
        raise ValueError('Unpinned stage1 training controller')
    for name, value in authority['control_input_sha256'].items():
        if digest(root/name) != value:
            raise ValueError('Current training controller changed: '+name)
    return {'control_mode': authority['mode'], 'scope': SCOPE, 'current_owner_thread': owner['thread_id']}


def prerequisites(root, protocol_ref, master, authority_ref, authority):
    scope = check_scope(root, protocol_ref, master, authority_ref, authority)
    if not authority.get('forward_bundle') or not authority.get('forward_publication'):
        raise ValueError('Stage1 training requires actual original-five and H1 late forecasts/publication')
    forward = verify_bundle(root, authority['forward_bundle'], protocol_ref, time.time())
    receipt = pinned(root, authority['forward_bundle'])
    definition = pinned(root, receipt['definition'])
    publication_path = (root/authority['forward_bundle']['path']).parent/'publication.json'
    if (authority['forward_publication'] != ref(root, publication_path)
            or authority['stop_file'] != definition['training_stop_file']
            or authority['global_stop_file'] != definition['global_stop_file']
            or authority['complete_probe_audit'] != receipt['complete_probe_audit']
            or authority['complete_probe_review'] != receipt['complete_probe_review']):
        raise ValueError('Wrong actual forward/probe/stopping bindings')
    probe = pinned(root, master['stage1_probe_protocol'])
    directory = root/probe['output_root']
    final = read(directory/'supervisor.json'); queue = read(directory/'queue.json'); launch = read(directory/'launch.json')
    if (final['status'] != 'audited_complete' or final['complete_cases'] != 16
            or len(queue['complete']) != 16 or queue['active'] or queue['pending'] or queue['failures']
            or process_has_not_exited(launch['supervisor_pid'])):
        raise RuntimeError('Original late probe queue is not complete and exited')
    required = set(master['required_review_events']) | set(authority['required_review_events'])
    for eid in required:
        if read(root/'experiments/reviews'/(eid+'.json'))['status'] != 'reviewed':
            raise ValueError('Missing required execution review: '+eid)
    return scope | {'forward_verified': forward, 'probe_complete_and_exited': True}


def preflight(root, gpu, protocol_ref, master, authority_ref, authority, publication_ref):
    checked = prerequisites(root, protocol_ref, master, authority_ref, authority)
    publication = pinned(root, publication_ref)
    for name, value in publication['files'].items():
        if digest(root/name) != value:
            raise ValueError('Published execution bytes changed: '+name)
    if gpu not in master['allowed_gpus']:
        raise ValueError('GPU outside registered lanes')
    info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines()
    if int(info[1].strip()) > 64 or info[0].strip() in apps:
        raise RuntimeError('GPU occupied; no unrelated process may be stopped')
    return checked | {'gpu': gpu, 'memory_mib': int(info[1].strip()), 'no_compute_process': True,
                      'publication': publication_ref, 'resource_authorization': master['resource_authorization']}


def execution_files(root, protocol_ref, master, authority_ref, authority):
    files = {protocol_ref['path']: protocol_ref['sha256'], authority_ref['path']: authority_ref['sha256']}
    refs = [authority['owner_goal']['record'], authority['forward_bundle'], authority['forward_publication'],
            authority['complete_probe_audit'], authority['complete_probe_review'], master['resource_authorization'],
            master['stage0_execution_protocol'], master['stage1_probe_protocol'], master['parent_binding'], *master['jobs']]
    files.update({item['path']: item['sha256'] for item in refs})
    files.update({name: value for name, value in master['input_sha256'].items() if name.startswith('pipeline/')})
    files.update(authority['control_input_sha256'])
    for eid in set(master['required_review_events']) | set(authority['required_review_events']):
        name = 'experiments/reviews/'+eid+'.json'; files[name] = digest(root/name)
    return files


def audit_case(root, master, entry, control_root):
    job = pinned(root, entry)
    output = root/job['output']
    if (output/'audit.json').exists():
        raise FileExistsError('Preserve existing original audit; no duplicate audit execution')
    with (control_root/(entry['name']+'.audit.console.log')).open('x') as log:
        result = subprocess.run([str(root/'.venv-rl/bin/python'), str(root/'pipeline/report_bfcl_source_training.py'),
                                 '--job', str(root/entry['path'])], cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                stdout=log, stderr=subprocess.STDOUT, timeout=300)
    audit = read(output/'audit.json')
    if result.returncode or audit['status'] != 'passed' or audit['job_sha256'] != entry['sha256']:
        raise ValueError('Original full training raw audit failed: '+entry['name'])
    if audit['cost']['new_model_tokens'] > job['token_budget']+32768:
        raise ValueError('Actual complete-group token ceiling exceeded')
    return audit


def audit_batch(root, protocol_ref, master, authority, control_root):
    audits = []
    for entry in master['jobs']:
        job = pinned(root, entry); audit = read(root/job['output']/'audit.json')
        if audit['status'] != 'passed' or audit['job_sha256'] != entry['sha256'] or audit['cost']['new_model_tokens'] > job['token_budget']+32768:
            raise ValueError('Incomplete51 original raw training audits')
        audits.append(audit)
    actual = sum(audit['cost']['new_model_tokens'] for audit in audits)
    if actual > master['new_training_token_ceiling']:
        raise ValueError('Total actual training token ceiling exceeded')
    certificate = {'status': 'passed', 'stage': 1, 'complete_jobs': 51, 'execution_protocol': protocol_ref,
                   'forward_bundle': authority['forward_bundle'], 'new_sampled_tokens': actual, 'jobs': audits,
                   'ancestor_tokens_recharged': 0, 'evaluation_performed': False,
                   'scope': 'Full51 late training raw audits. Execution-agent review and transfer evaluation remain separate.'}
    exclusive(root/master['output_root']/'audit_status.json', certificate)
    exclusive(control_root/'training_audit_binding.json', {'complete_audit': ref(root, root/master['output_root']/'audit_status.json')})


def run(root, protocol_ref, authority_ref, commit):
    master, jobs = load_plan(root, protocol_ref)
    authority = pinned(root, authority_ref)
    # No output, GPU check or scientific child before current actual prerequisites.
    prerequisites(root, protocol_ref, master, authority_ref, authority)
    relative = Path(authority['control_root'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('rl',) or len(relative.parts) < 2:
        raise ValueError('Invalid current control output path')
    control = root/relative
    if control == root/master['output_root'] or control.is_relative_to(root/master['output_root']):
        raise ValueError('Separate control receipts from scientific branch outputs')
    if any((root/job['output']).exists() for job in jobs):
        raise FileExistsError('An original training branch already has evidence; use separately registered continuation, never rerun')
    control.mkdir(exist_ok=False)
    check = lambda: check_scope(root, protocol_ref, master, authority_ref, authority)
    phase = 'frozen_input_identity'
    try:
        # One full frozen dependency verification. Actual trainer restores full
        # parent weights/optimizer/history/RNG and checks its own pinned inputs.
        for name, value in master['input_sha256'].items():
            check()
            if input_digest(root/name) != value: raise ValueError('Frozen training input changed: '+name)
        files = execution_files(root, protocol_ref, master, authority_ref, authority)
        phase = 'publication'
        publication = verify_publication(root, files, commit, check)
        exclusive(control/'publication_verified.json', publication)
        publication_ref = ref(root, control/'publication_verified.json')
        phase = 'watch_registration'
        watched = read(root/'experiments/watch_roots.json')
        if master['output_root'] not in watched or str(relative) not in watched:
            raise ValueError('Actual training and control outputs not registered')
        phase = 'gpu_admission'
        initial = {gpu: preflight(root, gpu, protocol_ref, master, authority_ref, authority, publication_ref) for gpu in master['allowed_gpus']}
    except Exception as error:
        exclusive(control/'failures.json', [{'classification': admission_failure_category(phase, error), 'phase': phase,
            'type': type(error).__name__, 'message': str(error), 'scientific_children_launched': 0}])
        exclusive(control/'supervisor.json', {'status': 'incomplete_requires_review', 'complete_jobs': 0, 'automatic_retry': False})
        raise
    exclusive(control/'launch.json', {'supervisor_pid': os.getpid(), 'started_unix': time.time(),
        'execution_protocol': protocol_ref, 'authority': authority_ref, 'publication': publication_ref,
        'preflights': initial, 'max_simultaneous_gpus': 8, 'scientific_jobs': 51})
    entries = {entry['name']: entry for entry in master['jobs']}
    schedule = {'cases': [entry | {'hard_seconds': job['hard_seconds']} for entry, job in zip(master['jobs'], jobs)],
                'gpu_lanes': master['gpu_lanes'], 'allowed_gpus': master['allowed_gpus'], 'max_simultaneous_gpus': 8}
    def launch(gpu, case):
        entry = entries[case['name']]; job = pinned(root, entry)
        if (root/job['output']).exists(): raise FileExistsError('Never overwrite prior scientific evidence: '+case['name'])
        return supervise([str(root/'.venv-rl/bin/python'), '-u', str(root/'pipeline/train_bfcl_source_group.py'),
                          '--job', str(root/entry['path']), '--job-sha256', entry['sha256']], root/job['output'],
                         control/(case['name']+'.console.log'), job['hard_seconds'],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    def audit(name):
        if name is None: return audit_batch(root, protocol_ref, master, authority, control)
        return audit_case(root, master, entries[name], control)
    result = execute_lanes(control, schedule,
        lambda gpu: preflight(root, gpu, protocol_ref, master, authority_ref, authority, publication_ref), launch, audit)
    # Preserve the original worker records. Classify failures without relabeling
    # control/query failures as scientific negative outcomes.
    classified = [failure | {'classification': worker_failure_category(control, failure)} for failure in result['failures']]
    exclusive(control/'training_control_result.json', {'complete_jobs': len(result['complete']),
        'pending_jobs': result['pending'], 'classified_failures': classified,
        'automatic_scientific_retry': False, 'evaluation_started': False})
    if result['failures'] or result['pending']:
        raise RuntimeError('Incomplete original51 batch; preserve all evidence and register any continuation separately')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'authority'):
        parser.add_argument('--'+name, required=True); parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--publication-commit', required=True)
    args = parser.parse_args()
    run(ROOT, {'path': args.protocol, 'sha256': args.protocol_sha256},
        {'path': args.authority, 'sha256': args.authority_sha256}, args.publication_commit)


if __name__ == '__main__': main()
