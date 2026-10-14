"""Registered-resource, fail-closed execution of one published fit/heldout evaluation phase."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import signal
from pathlib import Path
import subprocess
import threading
import time
from bfcl_source_eval_contract import pinned, load_job
from bfcl_transfer_labels import source_registry
from prepare_bfcl_source_evaluation import phase_cases
from supervise_bfcl_source_training import digest
from supervise_bfcl_stage_fork import ROOT, read, write, supervise
from bfcl_ssh_evaluation_control_eight import preflight as owner_preflight, check_control, resource_allocation


def validate_manifest(root, plan):
    master = pinned(root, plan['execution_protocol'])
    analysis = pinned(root, master['analysis_contract'])
    registry = source_registry(master, [pinned(root, e) for e in master['jobs']], analysis)
    selected = phase_cases(registry, plan['phase'])
    expected = {(c, e) for c in selected for e in range(2)}
    seen, names = set(), set()
    gpus, maximum = (master['allowed_gpus'], 4) if plan.get('resource_authorization') is None else resource_allocation(root, plan['resource_authorization'])
    if plan['automatic_retry'] is not False or plan['max_simultaneous_gpus'] != maximum:
        raise ValueError('Retry or concurrency contract changed')
    if plan['allowed_gpus'] != gpus or len(set(plan['allowed_gpus'])) != maximum:
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
        references = [authority['owner_goal']['record'], authority['complete_training_audit'], authority['complete_training_review']]
        if authority.get('resource_authorization') is not None:
            references.append(authority['resource_authorization'])
        for reference in references:
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



def process_state(pid, start_ticks=None):
    path = Path('/proc')/str(pid)/'stat'
    if not path.exists():
        return 'missing'
    fields = path.read_text().rsplit(')', 1)[1].split()
    if start_ticks is not None and int(fields[19]) != start_ticks:
        return 'reused'
    return fields[0]


def completed_panel(root, plan, name):
    entry = next(e for e in plan['jobs'] if e['name'] == name)
    job = pinned(root, entry)
    out = root/job['output']
    if not all((out/p).exists() for p in ('run.json', 'summary.json', 'audit.json', 'supervisor.json')):
        return False
    run, summary, audit, terminal = [read(out/p) for p in ('run.json', 'summary.json', 'audit.json', 'supervisor.json')]
    if summary.get('status') != 'complete' or audit.get('status') != 'passed':
        return False
    if terminal['returncode'] != 0 or terminal['hard_timeout']:
        raise RuntimeError('Existing scientific terminal failed; do not rerun: '+name)
    if (run['job_sha256'] != entry['sha256'] or run['role'] != job['role']
            or run['policy_fingerprint'] != job['policy_fingerprint']
            or run['optimizer_updates'] != 0 or not summary['weights_unchanged']
            or summary['completed_tasks'] != 32):
        raise ValueError('Existing complete panel is not the original frozen job: '+name)
    return True


def validate_handoff(root, plan):
    if plan['phase'] != 'fit' or plan['allowed_gpus'] != list(range(8)) or plan['max_simultaneous_gpus'] != 8:
        raise ValueError('This adapter is only for the original fit80 eight-card continuation')
    original = pinned(root, plan['original_manifest'])
    for key in ('phase', 'execution_protocol', 'output_root', 'hard_seconds_per_panel',
                'prediction_seal', 'required_review_events', 'jobs'):
        if plan[key] != original[key]:
            raise ValueError('Eight-card handoff changed scientific field: '+key)
    if any(plan['input_sha256'].get(p) != sha for p, sha in original['input_sha256'].items()):
        raise ValueError('Original frozen dependency was changed')
    if plan['control_root'] == plan['output_root']:
        raise ValueError('New control records must not overwrite original history')
    receipt = pinned(root, plan['handoff'])
    if receipt['original_manifest'] != plan['original_manifest'] or receipt['resource_authorization'] != plan['resource_authorization']:
        raise ValueError('Handoff binding differs')
    stop = root/receipt['old_stop_marker']['path']
    if digest(stop) != receipt['old_stop_marker']['sha256']:
        raise ValueError('Old-controller admission latch disappeared or changed')
    names = {e['name'] for e in plan['jobs']}
    done = receipt['complete_at_latch']
    carry = receipt['carried']
    remaining = receipt['pending_at_latch']
    allocated = done + [x['panel'] for x in carry] + remaining
    if len(allocated) != len(names) or set(allocated) != names or len(set(allocated)) != len(allocated):
        raise ValueError('Original job identities omitted or duplicated at handoff')
    if len({x['gpu'] for x in carry}) != len(carry) or len({x['pid'] for x in carry}) != len(carry):
        raise ValueError('Carried process/GPU duplicated')
    for name in done:
        if not completed_panel(root, plan, name):
            raise ValueError('Previously completed evidence is not complete: '+name)
    for info in carry:
        if info['panel'] not in plan['gpu_lanes'][str(info['gpu'])]:
            raise ValueError('Carried job assigned to a different GPU')
        entry = next(e for e in plan['jobs'] if e['name'] == info['panel'])
        process = pinned(root, info['process_receipt'])
        if process['pid'] != info['pid'] or process['pgid'] != info['pid'] or process['hard_seconds'] != plan['hard_seconds_per_panel']:
            raise ValueError('Carried process lifetime or hard budget differs')
        if '--job' not in process['command'] or process['command'][process['command'].index('--job')+1] != str(root/entry['path']):
            raise ValueError('Carried process is for another job')
        if '--job-sha256' not in process['command'] or process['command'][process['command'].index('--job-sha256')+1] != entry['sha256']:
            raise ValueError('Carried process is for different job bytes')
        if process_state(info['pid'], info['start_ticks']) not in ('missing', 'reused', 'Z', 'X'):
            actual = (Path('/proc')/str(info['pid'])/'cmdline').read_bytes().split(b'\0')
            actual = [x.decode() for x in actual if x]
            if actual != process['command']:
                raise ValueError('Live carried command differs')
            env = (Path('/proc')/str(info['pid'])/'environ').read_bytes().split(b'\0')
            cuda = next((x.split(b'=', 1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
            if cuda != str(info['gpu']):
                raise ValueError('Live carried process uses a different GPU')
    for name in remaining:
        job = pinned(root, next(e for e in plan['jobs'] if e['name'] == name))
        if (root/job['output']).exists():
            raise ValueError('Pending job already has execution evidence; do not rerun: '+name)
    return receipt


def wait_carried(root, plan, info):
    process = pinned(root, info['process_receipt'])
    deadline = time.monotonic() + max(0, process['started_unix'] + process['hard_seconds'] - time.time())
    while process_state(info['pid'], info['start_ticks']) not in ('missing', 'reused', 'Z', 'X'):
        if time.monotonic() >= deadline:
            # Existing registered hard timeout, applied only to this lifetime and PGID.
            os.killpg(process['pgid'], signal.SIGTERM)
            raise RuntimeError('Carried original job reached its original hard timeout: '+info['panel'])
        time.sleep(5)
    audit_deadline = time.monotonic()+330
    while not completed_panel(root, plan, info['panel']):
        if time.monotonic() >= audit_deadline:
            raise RuntimeError('Carried job terminal/audit evidence unavailable; preserve, do not rerun: '+info['panel'])
        time.sleep(2)


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

    root = ROOT/plan['control_root']
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
        handoff = validate_handoff(ROOT, plan)
        carried = {str(x['gpu']): x for x in handoff['carried']}
        initial = {g: preflight(g, plan) for g in plan['allowed_gpus'] if str(g) not in carried}
    except Exception as error:
        write(root/'failures.json', [{'phase': 'preflight', 'type': type(error).__name__, 'message': str(error), 'gpu_children_launched': 0}])
        raise
    write(root/'launch.json', {'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'publication': publication,
                              'control_authority': authority_ref, 'owner_goal_thread': authority['owner_goal']['thread_id'],
                              'preflights': initial, 'phase': plan['phase'], 'max_simultaneous_gpus': plan['max_simultaneous_gpus'],
                              'original_manifest': plan['original_manifest'], 'handoff': plan['handoff'], 'carried_processes': handoff['carried']})
    lock, stop = threading.Lock(), threading.Event()
    state = {'active': {g: x['panel'] for g, x in carried.items()},
             'complete': list(handoff['complete_at_latch']), 'pending': list(handoff['pending_at_latch']), 'failures': []}
    for info in handoff['carried']:
        source = ROOT/info['process_receipt']['path']
        (root/(info['panel']+'.console.process.json')).write_bytes(source.read_bytes())
        write(root/(info['panel']+'.legacy_adoption.json'), info)
    entries = {e['name']: e for e in plan['jobs']}
    def publish():
        write(root/'queue.json', state)
    def worker(gpu):
        for name in plan['gpu_lanes'][str(gpu)]:
            if name in state['complete']:
                continue
            if stop.is_set():
                return
            entry = entries[name]
            phase = 'control_admission'
            try:
                if str(gpu) in carried and carried[str(gpu)]['panel'] == name:
                    phase = 'legacy_execution_or_audit'
                    wait_carried(ROOT, plan, carried[str(gpu)])
                    with lock:
                        state['active'].pop(str(gpu))
                        state['complete'].append(name)
                        publish()
                    continue
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
                phase = 'scientific_execution'
                result = supervise([str(ROOT/'.venv-rl/bin/python'), '-u', str(ROOT/'pipeline/eval_bfcl_source_transfer.py'),
                    '--job', str(job_path), '--job-sha256', entry['sha256']], ROOT/job['output'],
                    root/(name+'.console.log'), plan['hard_seconds_per_panel'],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                if result['returncode'] != 0 or result['hard_timeout']:
                    raise RuntimeError('Evaluation failed: '+name)
                phase = 'dependency_audit'
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
                    state['failures'].append({'case': name, 'gpu': gpu, 'type': type(error).__name__, 'message': str(error), 'classification': phase})
                    stop.set()
                    publish()
                    write(root/'failures.json', state['failures'])
                return
    publish()
    with ThreadPoolExecutor(max_workers=plan['max_simultaneous_gpus']) as pool:
        list(pool.map(worker, plan['allowed_gpus']))
    complete = not state['failures'] and not state['pending']
    if complete:
        audits = [read(ROOT/read(ROOT/e['path'])['output']/'audit.json') for e in plan['jobs']]
        if not all(a['status'] == 'passed' for a in audits):
            raise RuntimeError('Final panel audit check failed')
        write(root/'audit_status.json', {'status': 'passed', 'phase': plan['phase'], 'complete_panels': len(audits),
            'new_sampled_tokens': sum(a['cost']['new_model_tokens'] for a in audits), 'jobs': audits,
            'original_manifest': plan['original_manifest'], 'handoff': plan['handoff'],
            'scope': 'All80 original fit job identities, completed once across original six-card and eight-card controls. No scientific retries. Actual prediction sealing and heldout remain.'})
    write(root/'supervisor.json', {'status': 'audited_complete' if complete else 'incomplete_requires_review',
        'complete_panels': len(state['complete']), 'pending_panels': state['pending'], 'automatic_retry': False})


if __name__ == '__main__':
    main()
