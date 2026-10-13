"""Run only the original unstarted jobs under the registered additional resource authorization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import threading
import time
from supervise_bfcl_stage_fork import ROOT, read, write, supervise
from supervise_bfcl_source_training import digest
from bfcl_ssh_resource_control import check_control, preflight


def main():
    parser = argparse.ArgumentParser()
    for name in ['protocol', 'control-authority']:
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    args = parser.parse_args()
    assert digest(args.protocol) == args.protocol_sha256
    assert digest(args.control_authority) == args.control_authority_sha256
    plan, authority = read(args.protocol), read(args.control_authority)
    assert authority['training_protocol'] == {'path': str(args.protocol.resolve().relative_to(ROOT)), 'sha256': args.protocol_sha256}
    root = ROOT/plan['output_root']
    root.mkdir(parents=True, exist_ok=False)
    try:
        check_control(ROOT, plan, authority, require_unstarted=True)
        for path, sha in plan['input_sha256'].items():
            assert digest(ROOT/path) == sha, path
        for eid in plan['required_review_events']:
            assert read(ROOT/'experiments/reviews'/(eid+'.json'))['status'] == 'reviewed'
        initial = {gpu:preflight(gpu,plan,authority) for gpu in plan['allowed_gpus']}
    except Exception as error:
        write(root/'failures.json', [{'phase':'preflight','type':type(error).__name__,
                                     'message':str(error),'gpu_children_launched':0}])
        raise
    write(root/'launch.json', {'supervisor_pid':os.getpid(),'started_unix':time.time(),
        'protocol_sha256':args.protocol_sha256,'control_authority_sha256':args.control_authority_sha256,
        'control_authority_path':str(args.control_authority.resolve().relative_to(ROOT)),
        'preflights':initial,'max_simultaneous_gpus':plan['max_simultaneous_gpus'],'aggregate_max_simultaneous_gpus':6})
    lock, stop = threading.Lock(), threading.Event()
    state = {'active':{},'complete':[],'pending':[e['name'] for e in plan['jobs']],'failures':[]}
    jobs = {e['name']:e for e in plan['jobs']}
    def publish():
        write(root/'queue.json',state)
    def worker(gpu):
        for name in plan['gpu_lanes'][str(gpu)]:
            if stop.is_set():
                return
            entry = jobs[name]
            try:
                job_path = ROOT/entry['path']
                assert digest(job_path) == entry['sha256']
                job = read(job_path)
                with lock:
                    if stop.is_set():
                        return
                    checked = preflight(gpu,plan,authority)
                    reserved = set(state['active'].values())
                    observed = {x['case'] for x in checked['observed_project_live_pids']}
                    aggregate = len(reserved | observed | {name})
                    if aggregate > 6:
                        raise RuntimeError('Six-GPU aggregate admission limit reached')
                    if (ROOT/job['output']).exists():
                        raise FileExistsError('Original job output already exists: '+name)
                    checked.update({'reserved_project_cases':sorted(reserved),
                                    'aggregate_after_admission':aggregate})
                    state['pending'].remove(name)
                    state['active'][str(gpu)] = name
                    publish()
                    write(root/(name+'.preflight.json'),checked)
                result = supervise([str(ROOT/'.venv-rl/bin/python'),'-u',str(ROOT/'pipeline/train_bfcl_source_group.py'),
                    '--job',str(job_path),'--job-sha256',entry['sha256']], ROOT/job['output'],
                    root/(name+'.console.log'),job['hard_seconds'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                assert result['returncode'] == 0 and not result['hard_timeout'], 'Training failed: '+name
                with (root/(name+'.audit.console.log')).open('x') as log:
                    audit = subprocess.run([str(ROOT/'.venv-rl/bin/python'),str(ROOT/'pipeline/report_bfcl_source_training.py'),
                        '--job',str(job_path)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                        stdout=log,stderr=subprocess.STDOUT,timeout=300)
                assert audit.returncode == 0, 'Training audit failed: '+name
                with lock:
                    state['active'].pop(str(gpu)); state['complete'].append(name); publish()
            except Exception as error:
                with lock:
                    state['active'].pop(str(gpu),None)
                    state['failures'].append({'case':name,'gpu':gpu,'type':type(error).__name__,'message':str(error)})
                    stop.set(); publish(); write(root/'failures.json',state['failures'])
                return
    publish()
    with ThreadPoolExecutor(max_workers=plan['max_simultaneous_gpus']) as pool:
        list(pool.map(worker,plan['allowed_gpus']))
    complete = not state['failures'] and not state['pending']
    if complete:
        audits = [read(ROOT/read(ROOT/e['path'])['output']/'audit.json') for e in plan['jobs']]
        assert all(a['status'] == 'passed' for a in audits)
        write(root/'audit_status.json',{'status':'passed','complete_jobs':len(audits),
            'new_sampled_tokens':sum(a['cost']['new_model_tokens'] for a in audits),'jobs':audits,
            'evaluation_performed':False,'scope':'Original unstarted training only; transfer and late stage remain.'})
    write(root/'supervisor.json',{'status':'audited_complete' if complete else 'incomplete_requires_review',
        'complete_jobs':len(state['complete']),'pending_jobs':state['pending'],'automatic_retry':False})

if __name__ == '__main__':
    main()
