"""Published stage1 probes, four fixed lanes, bounded attempts and no retry.

This entry point cannot run the stage1 design-input inventory directly. The
complete stage0 transfer report/review is required by load_protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time
from bfcl_source_eval_contract import digest, pinned
from bfcl_stage1_probe_contract import load_protocol
from bfcl_diagnostic_preflight import preflight
from supervise_bfcl_stage_fork import ROOT, read, write, supervise


NEW_RUNTIME_CODE = [
    'pipeline/bfcl_stage1_probe_contract.py',
    'pipeline/run_bfcl_stage1_probe.py',
    'pipeline/report_bfcl_stage1_probe.py',
    'pipeline/supervise_bfcl_stage1_probe.py',
    'pipeline/bfcl_source_eval_contract.py',
]


def validate_schedule(plan, master):
    if plan['allowed_gpus'] != [4, 5, 6, 7] or plan['max_simultaneous_gpus'] != 4 or plan['automatic_retry'] is not False:
        raise ValueError('Four-card single-attempt contract changed')
    names = [c['name'] for c in plan['cases']]
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError('Require sixteen distinct probe cases')
    if any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Invalid case path')
    if set(plan['gpu_lanes']) != {'4', '5', '6', '7'}:
        raise ValueError('Missing or extra GPU lane')
    scheduled = [n for lane in plan['gpu_lanes'].values() for n in lane]
    if len(scheduled) != 16 or set(scheduled) != set(names):
        raise ValueError('Omitted, repeated or substituted source case')
    for name, sha in master['input_sha256'].items():
        if plan['input_sha256'].get(name) != sha:
            raise ValueError('Inherited runtime dependency omitted or changed: '+name)
    if not set(NEW_RUNTIME_CODE) <= plan['input_sha256'].keys():
        raise ValueError('Missing stage1 runtime dependencies')


def publication_files(path, expected_sha, plan, binding):
    files = {str(path): expected_sha}
    for ref in [plan[k] for k in ('design_inputs', 'parent_binding', 'stage0_transfer_report', 'stage0_transfer_review', 'source_protocol')]:
        files[ref['path']] = ref['sha256']
    for key in ('stage0_execution_protocol', 'parent_training_job', 'summary', 'audit', 'manifest', 'adapter_config'):
        ref = binding[key]; files[ref['path']] = ref['sha256']
    files.update({p:h for p,h in plan['input_sha256'].items() if p.startswith('pipeline/')})
    # Large local model/payload files are hash-checked by load_protocol, never
    # pretended to be available in Git merely because their manifests are there.
    return files


def verify_publication(root, files, commit):
    for name, sha in files.items():
        result = subprocess.run(['git', 'show', commit+':'+name], cwd=root, capture_output=True, timeout=30)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
            raise ValueError('Published bytes do not match: '+name)
    result = subprocess.run(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root, capture_output=True, timeout=30)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('Cannot verify remote publication')
    remote = result.stdout.decode().split()[0]
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=root, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError('Execution commit is not on known remote main')
    return {'commit':commit, 'remote_main':remote, 'verified_unix':time.time(), 'files':files}


def execute_lanes(root, plan, check, launch, audit):
    """Admit each case once; a failed lane closes admission for all lanes.

    Already admitted independent work finishes its existing bounded attempt.
    Callbacks let CPU tests exercise concurrency/failure paths without GPU jobs.
    """
    lock, stop = threading.Lock(), threading.Event()
    state = {'active':{}, 'complete':[], 'pending':[c['name'] for c in plan['cases']], 'failures':[]}
    cases = {c['name']:c for c in plan['cases']}
    def publish():
        write(root/'queue.json', state)
    def worker(gpu):
        for name in plan['gpu_lanes'][str(gpu)]:
            if stop.is_set():
                return
            phase = 'preflight'
            try:
                checked = check(gpu)
                with lock:
                    if stop.is_set():
                        return
                    state['pending'].remove(name)
                    state['active'][str(gpu)] = name
                    publish()
                write(root/(name+'.preflight.json'), checked)
                phase = 'runner'
                result = launch(gpu, cases[name])
                if result['returncode'] != 0 or result['hard_timeout'] or result['status'] != 'runner_complete_pending_replay_audit':
                    raise RuntimeError('Probe runner failed or incomplete: '+name)
                phase = 'audit'
                audit(name)
                with lock:
                    state['active'].pop(str(gpu))
                    state['complete'].append(name)
                    publish()
            except Exception as error:
                with lock:
                    state['active'].pop(str(gpu), None)
                    state['failures'].append({'case':name, 'gpu':gpu, 'phase':phase, 'type':type(error).__name__, 'message':str(error)})
                    stop.set(); publish(); write(root/'failures.json', state['failures'])
                return
    publish()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, plan['allowed_gpus']))
    if not state['failures'] and not state['pending']:
        try:
            audit(None)
        except Exception as error:
            state['failures'].append({'phase':'batch_audit', 'type':type(error).__name__, 'message':str(error)})
            publish(); write(root/'failures.json', state['failures'])
    complete = not state['failures'] and not state['pending'] and len(state['complete']) == len(cases)
    write(root/'supervisor.json', {'status':'audited_complete' if complete else 'incomplete_requires_review',
        'complete_cases':len(state['complete']), 'pending_cases':state['pending'], 'automatic_retry':False})
    return state


def run(root, path, expected_sha, commit):
    if digest(path) != expected_sha:
        raise ValueError('Protocol hash changed')
    plan = read(path)
    if plan.get('kind') != 'stage1_fixed_policy_probe_execution':
        raise ValueError('Design inputs are not executable')
    relative = Path(plan['output_root'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('rl',) or len(relative.parts) < 2:
        raise ValueError('Output must be a new relative RL directory')
    out = root/relative
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan, binding = load_protocol(root, path, expected_sha)
        master = pinned(root, binding['stage0_execution_protocol'])
        validate_schedule(plan, master)
        files = publication_files(path.relative_to(root), expected_sha, plan, binding)
        publication = verify_publication(root, files, commit)
        if plan['output_root'] not in read(root/'experiments/watch_roots.json'):
            raise ValueError('Output not registered with event monitor')
        for eid in plan['required_review_events']:
            if read(root/'experiments/reviews'/(eid+'.json'))['status'] != 'reviewed':
                raise ValueError('Missing required research review')
        initial = {gpu:preflight(gpu, plan) for gpu in plan['allowed_gpus']}
    except Exception as error:
        write(out/'failures.json', [{'phase':'preflight', 'type':type(error).__name__, 'message':str(error), 'gpu_children_launched':0}])
        write(out/'supervisor.json', {'status':'incomplete_requires_review', 'complete_cases':0, 'automatic_retry':False})
        raise
    write(out/'launch.json', {'supervisor_pid':os.getpid(), 'started_unix':time.time(), 'protocol_sha256':expected_sha,
        'publication':publication, 'preflights':initial, 'max_simultaneous_gpus':4})
    def launch(gpu, case):
        return supervise([str(root/'.venv-rl/bin/python'), '-u', str(root/'pipeline/run_bfcl_stage1_probe.py'),
            '--protocol', str(path), '--protocol-sha256', expected_sha, '--case', case['name']], out/case['name'],
            out/(case['name']+'.console.log'), case['hard_seconds'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    def audit(case):
        cmd = [str(root/'.venv-rl/bin/python'), str(root/'pipeline/report_bfcl_stage1_probe.py'),
               '--protocol', str(path), '--protocol-sha256', expected_sha]
        if case is not None:
            cmd += ['--case', case]
        with (out/((case or 'batch')+'.audit.console.log')).open('x') as log:
            result = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if result.returncode:
            raise RuntimeError('Probe raw audit failed: '+str(case))
        receipt = read(out/case/'audit.json' if case is not None else out/'audit_status.json')
        if receipt['status'] != 'passed' or receipt['stage'] != 1 or receipt['complete_cases'] != (1 if case is not None else 16):
            raise ValueError('Audit output is incomplete or from a different stage')
    state = execute_lanes(out, plan, lambda gpu:preflight(gpu, plan), launch, audit)
    if state['failures'] or state['pending']:
        raise RuntimeError('Probe batch incomplete; preserve evidence and review, do not retry')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--publication-commit', required=True)
    args = parser.parse_args()
    run(ROOT, args.protocol.resolve(), args.protocol_sha256, args.publication_commit)


if __name__ == '__main__':
    main()
