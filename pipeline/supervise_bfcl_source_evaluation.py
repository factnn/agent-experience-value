"""Four-card, fail-closed execution of one published fit/heldout evaluation phase."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time
from bfcl_source_eval_contract import pinned, load_job
from bfcl_transfer_labels import source_registry
from prepare_bfcl_source_evaluation import phase_cases
from supervise_bfcl_source_training import digest
from supervise_bfcl_stage_fork import ROOT, read, write, supervise
from bfcl_ssh_evaluation_control import preflight as owner_preflight, check_control


def validate_manifest(root, plan):
    master = pinned(root, plan['execution_protocol'])
    analysis = pinned(root, master['analysis_contract'])
    registry = source_registry(master, [pinned(root, e) for e in master['jobs']], analysis)
    selected = phase_cases(registry, plan['phase'])
    expected = {(c, e) for c in selected for e in range(2)}
    seen, names = set(), set()
    if plan['automatic_retry'] is not False or plan['max_simultaneous_gpus'] != 4:
        raise ValueError('Retry or concurrency contract changed')
    if plan['allowed_gpus'] != master['allowed_gpus'] or len(set(plan['allowed_gpus'])) != 4:
        raise ValueError('GPU allocation changed')
    if plan['hard_seconds_per_panel'] != master['evaluation']['hard_seconds_per_panel']:
        raise ValueError('Panel runtime limit changed')
    for entry in plan['jobs']:
        job = pinned(root, entry)
        key = job['policy_case'], job['evaluation_repeat']
        if type(key[1]) is not int or key not in expected or key in seen or entry['name'] in names:
            raise ValueError('Duplicate or unexpected panel')
        if job['execution_protocol'] != plan['execution_protocol'] or job.get('prediction_seal') != plan['prediction_seal']:
            raise ValueError('Execution or prediction seal substitution')
        if job['role'] != registry[key[0]]['role'] or job['output'] != str(Path(plan['output_root'])/entry['name']):
            raise ValueError('Panel role or output substitution')
        seen.add(key)
        names.add(entry['name'])
    if seen != expected:
        raise ValueError('Missing scheduled panels')
    lanes = plan['gpu_lanes']
    if set(lanes) != {str(g) for g in plan['allowed_gpus']}:
        raise ValueError('Missing GPU lane')
    allocated = [name for lane in lanes.values() for name in lane]
    if len(allocated) != len(names) or set(allocated) != names:
        raise ValueError('Lane omission or duplication')
    return master


def verify_publication(root, path, expected_sha, plan, commit, control_authority_ref=None):
    files = {str(path.relative_to(root)): expected_sha}
    files.update({entry['path']: entry['sha256'] for entry in plan['jobs']})
    files.update({name: sha for name, sha in plan['input_sha256'].items() if name.startswith('pipeline/')})
    if control_authority_ref is not None:
        authority = pinned(root, control_authority_ref)
        files[control_authority_ref['path']] = control_authority_ref['sha256']
        files.update(authority['control_input_sha256'])
        for reference in [authority['owner_goal']['record'], authority['complete_training_audit'], authority['complete_training_review']]:
            files[reference['path']] = reference['sha256']
    for name, sha in files.items():
        result = subprocess.run(['git', 'show', commit+':'+name], cwd=root, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
            raise ValueError('Publication does not contain registered execution bytes: '+name)
    result = subprocess.run(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root, capture_output=True, timeout=30)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('Cannot verify remote execution publication')
    remote = result.stdout.decode().split()[0]
    if subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=root, capture_output=True).returncode:
        raise ValueError('Execution commit is not verified on remote main')
    return {'commit': commit, 'remote_main': remote, 'manifest_sha256': expected_sha, 'verified_unix': time.time()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--publication-commit', required=True)
    parser.add_argument('--control-authority', type=Path, required=True)
    parser.add_argument('--control-authority-sha256', required=True)
    args = parser.parse_args()
    path = args.manifest.resolve()
    if digest(path) != args.manifest_sha256:
        raise ValueError('Evaluation manifest changed')
    plan = read(path)
    authority_ref = {'path': str(args.control_authority.resolve().relative_to(ROOT)), 'sha256': args.control_authority_sha256}
    authority = pinned(ROOT, authority_ref)
    if authority['evaluation_manifest'] != {'path': str(path.relative_to(ROOT)), 'sha256': args.manifest_sha256}:
        raise ValueError('Evaluation authority is not bound to this phase manifest')
    check_control(ROOT, plan, authority)
    def preflight(gpu, plan):
        return owner_preflight(gpu, plan, authority)

    root = ROOT/plan['output_root']
    root.mkdir(parents=True, exist_ok=False)
    try:
        validate_manifest(ROOT, plan)
        publication = verify_publication(ROOT, path, args.manifest_sha256, plan, args.publication_commit, authority_ref)
        for name, sha in plan['input_sha256'].items():
            if digest(ROOT/name) != sha:
                raise ValueError('Evaluation dependency changed: '+name)
        for eid in plan['required_review_events']:
            if read(ROOT/'experiments/reviews'/(eid+'.json'))['status'] != 'reviewed':
                raise ValueError('Missing required research review')
        if plan['output_root'] not in read(ROOT/'experiments/watch_roots.json'):
            raise ValueError('Evaluation output must be registered with event monitor')
        initial = {g: preflight(g, plan) for g in plan['allowed_gpus']}
    except Exception as error:
        write(root/'failures.json', [{'phase': 'preflight', 'type': type(error).__name__, 'message': str(error), 'gpu_children_launched': 0}])
        raise
    write(root/'launch.json', {'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'publication': publication,
                              'control_authority': authority_ref, 'owner_goal_thread': authority['owner_goal']['thread_id'],
                              'preflights': initial, 'phase': plan['phase'], 'max_simultaneous_gpus': 4})
    lock, stop = threading.Lock(), threading.Event()
    state = {'active': {}, 'complete': [], 'pending': [e['name'] for e in plan['jobs']], 'failures': []}
    entries = {e['name']: e for e in plan['jobs']}
    def publish():
        write(root/'queue.json', state)
    def worker(gpu):
        for name in plan['gpu_lanes'][str(gpu)]:
            if stop.is_set():
                return
            entry = entries[name]
            try:
                job_path = ROOT/entry['path']
                # Recheck policy bytes and the actual remote seal before each launch.
                job, _, _, _, _ = load_job(ROOT, job_path, entry['sha256'])
                checked = preflight(gpu, plan)
                with lock:
                    if stop.is_set():
                        return
                    state['pending'].remove(name)
                    state['active'][str(gpu)] = name
                    publish()
                write(root/(name+'.preflight.json'), checked)
                result = supervise([str(ROOT/'.venv-rl/bin/python'), '-u', str(ROOT/'pipeline/eval_bfcl_source_transfer.py'),
                    '--job', str(job_path), '--job-sha256', entry['sha256']], ROOT/job['output'],
                    root/(name+'.console.log'), plan['hard_seconds_per_panel'],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                if result['returncode'] != 0 or result['hard_timeout']:
                    raise RuntimeError('Evaluation failed: '+name)
                with (root/(name+'.audit.console.log')).open('x') as log:
                    audit = subprocess.run([str(ROOT/'.venv-rl/bin/python'), str(ROOT/'pipeline/report_bfcl_source_eval.py'),
                        '--job', str(job_path)], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                        stdout=log, stderr=subprocess.STDOUT, timeout=300)
                if audit.returncode:
                    raise RuntimeError('Evaluation audit failed: '+name)
                with lock:
                    state['active'].pop(str(gpu))
                    state['complete'].append(name)
                    publish()
            except Exception as error:
                with lock:
                    state['active'].pop(str(gpu), None)
                    state['failures'].append({'case': name, 'gpu': gpu, 'type': type(error).__name__, 'message': str(error)})
                    stop.set()
                    publish()
                    write(root/'failures.json', state['failures'])
                return
    publish()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, plan['allowed_gpus']))
    complete = not state['failures'] and not state['pending']
    if complete:
        audits = [read(ROOT/read(ROOT/e['path'])['output']/'audit.json') for e in plan['jobs']]
        if not all(a['status'] == 'passed' for a in audits):
            raise RuntimeError('Final panel audit check failed')
        write(root/'audit_status.json', {'status': 'passed', 'phase': plan['phase'], 'complete_panels': len(audits),
            'new_sampled_tokens': sum(a['cost']['new_model_tokens'] for a in audits), 'jobs': audits,
            'scope': 'One complete evaluation phase; prediction sealing or heldout scoring remains.'})
    write(root/'supervisor.json', {'status': 'audited_complete' if complete else 'incomplete_requires_review',
        'complete_panels': len(state['complete']), 'pending_panels': state['pending'], 'automatic_retry': False})


if __name__ == '__main__':
    main()
