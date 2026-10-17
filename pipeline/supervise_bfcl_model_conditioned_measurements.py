"""Current SSH eight-lane adapter for the finite E368 forward measurement."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from transformers import AutoTokenizer

from audit_bfcl_model_conditioned_measurement import audit_case
from bfcl_model_conditioned_contract import (SIGNALS, check_bytes, digest, exclusive, load_execution,
    pinned, read, ref)
from bfcl_stage1_probe_ssh import execute_lanes
from seal_bfcl_qwen8_cross_size_predictions import native_target_record_check
from supervise_bfcl_stage_fork import supervise


def complete(root, execution_ref, execution, design):
    output = root / design['output_root']
    rows = [read(output / case['name'] / 'audit.json') for case in design['cases']]
    if any(row['status'] != 'passed' or row['case'] != case['name'] for row, case in zip(rows, design['cases'])):
        raise ValueError('Require all original32 source/model raw audits')
    cost = {key: sum(row['cost'][key] for row in rows) for key in rows[0]['cost']}
    if (cost['forward_calls_started'] > design['maximum_forward_calls']
            or cost['forward_input_positions_started'] > design['maximum_forward_input_positions']
            or cost['actual_supervised_worker_wall_seconds'] > design['maximum_worker_seconds']):
        raise ValueError('Finite measurement resource ceiling exceeded; preserve evidence and review')
    groups = {}
    for profile in design['profiles']:
        selected = [row for row in rows if row['profile'] == profile]
        if len(selected) != 16 or len({row['source_group'] for row in selected}) != 16:
            raise ValueError('Missing/repeated original source group')
        groups[profile] = {row['source_group']: row['source_features'] for row in selected}
    features = {'kind': 'complete_E368_model_conditioned_source_features', 'execution': execution_ref,
                'design': execution['design'], 'profiles': design['profiles'], 'signals': list(SIGNALS),
                'groups': groups, 'feature_phase_only': True, 'reward_or_grade_filtering': False,
                'target_labels_used': False, 'model_updates': 0,
                'scope': 'Two fixed proxies, not evidence of transfer value or independent mechanism replications.'}
    exclusive(output / 'features.json', features)
    result = {'status': 'passed', 'complete_cases': 32, 'execution': execution_ref,
              'features': ref(root, output / 'features.json'), 'cost': cost, 'rows': rows,
              'target_evaluation': False, 'reserved21_accessed': False, 'goal_complete': False,
              'scope': 'Complete stored-likelihood integrity review only; old4B fitting and actual8B forecasts still require their separate seal.'}
    exclusive(output / 'audit_status.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execution', type=Path, required=True)
    ap.add_argument('--execution-sha256', required=True)
    ap.add_argument('--publication', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = {'path': str(args.execution.resolve().relative_to(root)), 'sha256': args.execution_sha256}
    execution, design = load_execution(root, reference)
    control = root / design['control_root']
    output = root / design['output_root']
    publication = read(args.publication)
    if (publication['scientific_child_started'] is not False or publication['commit'] != publication['remote_main']
            or publication['files'].get(reference['path']) != reference['sha256']):
        raise ValueError('Actual remote publication was not verified before scoring admission')
    for name, sha in execution['runtime_code_sha256'].items():
        if publication['files'].get(name) != sha:
            raise ValueError('Runtime code absent from verified publication: ' + name)
    for name, sha in publication['files'].items():
        if digest(root / name) != sha:
            raise ValueError('Published bytes changed: ' + name)
    if output.exists() or (control / 'launch.json').exists():
        raise ValueError('Scoring was previously admitted; no automatic repeat or overwrite')

    def admitted(gpu):
        for name in [design['stop_file'], design['global_stop_file']]:
            if (root / name).exists():
                raise RuntimeError('Current SSH owner stopped new admission: ' + name)
        if gpu not in execution['allowed_gpus']:
            raise ValueError('GPU outside current authorized eight lanes')
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True, timeout=15).splitlines()
        if int(info[1].strip()) > 64 or info[0].strip() in apps:
            raise RuntimeError('GPU occupied; no new scoring child launched')
        return {'gpu': gpu, 'memory_mib': int(info[1].strip()), 'no_compute_process': True,
                'current_ssh_owner': execution['current_ssh_owner'], 'execution': reference}

    phase = 'current_scoring_admission'
    try:
        preflight = [admitted(gpu) for gpu in execution['allowed_gpus']]
        access = native_target_record_check(root)
        tokenizer = AutoTokenizer.from_pretrained(design['profiles']['qwen4b']['base_model_path'],
                                                  local_files_only=True, use_fast=True)
        exclusive(control / 'launch.json', {'supervisor_pid': os.getpid(), 'started_unix': time.time(),
                  'execution': reference, 'publication': ref(root, args.publication),
                  'initial_free_gpu_checks': preflight, 'native_target_record_check': access,
                  'new_model_token_ceiling': 0, 'maximum_worker_seconds': design['maximum_worker_seconds']})
        output.mkdir(exist_ok=False)
        phase = 'forward_measurement_execution'

        def check(gpu):
            for name, sha in execution['runtime_code_sha256'].items():
                if digest(root / name) != sha:
                    raise ValueError('Execution code changed: ' + name)
            return admitted(gpu)

        def run(gpu, case):
            command = [str(root / '.venv-rl/bin/python'), '-u',
                       str(root / 'pipeline/run_bfcl_model_conditioned_measurement.py'),
                       '--execution', str(args.execution.resolve()), '--execution-sha256', reference['sha256'],
                       '--case', case['name']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4', PYTHONHASHSEED='0')
            return supervise(command, output / case['name'], control / (case['name'] + '.console.log'),
                             case['hard_seconds'], env)

        def audit(name):
            if name is None:
                complete(root, reference, execution, design)
                return
            exclusive(output / name / 'audit.json', audit_case(root, reference, name, tokenizer))

        result = execute_lanes(control, execution, check, run, audit)
        exclusive(control / 'result.json', {'phase': phase, 'queue': result,
                  'automatic_retry': False, 'new_model_tokens': 0, 'goal_complete': False})
    except Exception as error:
        exclusive(control / 'control_failure.json', {'phase': phase, 'type': type(error).__name__,
                  'message': str(error), 'scientific_retry': False, 'partial_evidence_retained': True,
                  'classification': 'measurement_admission_or_execution_failure_not_transfer_negative'})
        raise


if __name__ == '__main__':
    main()
