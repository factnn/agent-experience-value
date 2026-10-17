"""Continue only the unexecuted16 probes after a recorded common-audit repair."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from bfcl_qwen8_initial_probe import (audit_probe, bind_probe, check_binding,
    complete_probe, exclusive, load_master, pinned, ref)
from bfcl_stage1_probe_ssh import execute_lanes
from supervise_bfcl_stage_fork import ROOT, read, supervise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--continuation', type=Path, required=True)
    ap.add_argument('--continuation-sha256', required=True)
    ap.add_argument('--publication', type=Path, required=True)
    args = ap.parse_args()
    continuation_ref = {'path': str(args.continuation.relative_to(ROOT)),
                        'sha256': args.continuation_sha256}
    c = pinned(continuation_ref)
    assert c['kind'] == 'qwen8_unexecuted_probe_continuation_after_json_audit_fix'
    master_ref = c['original_master']
    p, _ = load_master(ROOT / master_ref['path'], master_ref['sha256'])
    assert c['scientific_jobs_to_repeat'] == [] and c['automatic_retry'] is False
    assert c['unexecuted_cases'] == [x['name'] for x in p['cases']]
    assert c['new_model_token_ceiling'] == p['probe_new_model_token_ceiling']
    assert c['remaining_maximum_sum_worker_seconds'] == sum(x['hard_seconds'] for x in p['cases'])
    assert c['current_ssh_owner'] == p['current_ssh_owner']
    for name, sha in c['input_sha256'].items():
        assert ref(ROOT / name)['sha256'] == sha, name
    publication = read(args.publication)
    assert publication['scientific_child_started'] is False
    assert publication['commit'] == publication['remote_main']
    assert publication['files'][continuation_ref['path']] == continuation_ref['sha256']
    for name, sha in publication['files'].items():
        assert ref(ROOT / name)['sha256'] == sha, name
    failure = pinned(c['original_control_failure'])
    assert failure['phase'] == 'common_audit' and failure['type'] == 'AssertionError'
    old = ROOT / p['control_root']
    launch = read(old / 'launch.json')
    for pid in [launch['supervisor_pid'], read(old / 'common.console.process.json')['pid']]:
        proc = Path('/proc') / str(pid) / 'stat'
        assert not proc.exists() or proc.read_text().split(') ', 1)[1].split()[0] == 'Z', 'Old process remains live'
    assert not (ROOT / p['probe_output']).exists()
    assert not (ROOT / p['binding_root']).exists()
    control = ROOT / c['control_root']
    assert not (control / 'launch.json').exists()

    def admitted(gpu):
        for name in [p['stop_file'], p['global_stop_file'], c['stop_file']]:
            if (ROOT / name).exists():
                raise RuntimeError('Current SSH owner stopped new admission: ' + name)
        assert gpu in p['allowed_gpus']
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True, timeout=15).splitlines()
        if int(info[1].strip()) > 64 or info[0].strip() in apps:
            raise RuntimeError('GPU occupied; no new scientific child launched')
        return {'gpu': gpu, 'memory_mib': int(info[1].strip()), 'no_compute_process': True,
                'continuation': continuation_ref, 'current_owner': p['current_ssh_owner']}

    phase = 'continuation_admission'
    try:
        # Check current stop markers before even the CPU postprocessing segment.
        free = [admitted(gpu) for gpu in p['allowed_gpus']]
        exclusive(control / 'launch.json', {'supervisor_pid': os.getpid(), 'started_unix': time.time(),
                  'master': master_ref, 'continuation': continuation_ref,
                  'publication': ref(args.publication), 'initial_free_gpu_checks': free,
                  'common_generation_repeated': False, 'maximum_simultaneous_gpus': 8})
        phase = 'common_audit_json_fix'
        from bfcl_qwen8_common_audit_json_fix import audit_common
        audit, signals = audit_common(p)
        plan, plan_ref = bind_probe(p, master_ref, audit, signals)
        exclusive(control / 'bound_probe_plan.json', plan_ref)
        (ROOT / p['probe_output']).mkdir(exist_ok=False)
        phase = 'source_probe_execution'

        def check(gpu):
            check_binding(plan)
            for name, sha in c['input_sha256'].items():
                assert ref(ROOT / name)['sha256'] == sha, name
            return admitted(gpu)

        def run(gpu, case):
            command = [str(ROOT / '.venv-rl/bin/python'), '-u',
                       str(ROOT / 'pipeline/run_bfcl_qwen8_initial_probe.py'),
                       '--master', str(ROOT / master_ref['path']), '--master-sha256', master_ref['sha256'],
                       '--mode', 'probe', '--case', case['name'], '--probe-plan', str(ROOT / plan_ref['path']),
                       '--probe-plan-sha256', plan_ref['sha256']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1')
            return supervise(command, ROOT / p['probe_output'] / case['name'],
                             control / (case['name'] + '.console.log'), case['hard_seconds'], env)

        def audit_case(name):
            if name is None:
                complete_probe(plan, plan_ref)
                return
            case = next(x for x in plan['cases'] if x['name'] == name)
            exclusive(ROOT / p['probe_output'] / name / 'audit.json', audit_probe(plan, case, plan_ref['sha256']))

        result = execute_lanes(control, plan, check, run, audit_case)
        exclusive(control / 'result.json', {'phase': phase, 'queue': result, 'automatic_retry': False,
                  'common_generation_repeated': False, 'goal_complete': False})
    except Exception as error:
        exclusive(control / 'control_failure.json', {'phase': phase, 'type': type(error).__name__,
                  'message': str(error), 'scientific_retry': False, 'partial_evidence_retained': True})
        raise


if __name__ == '__main__':
    main()
