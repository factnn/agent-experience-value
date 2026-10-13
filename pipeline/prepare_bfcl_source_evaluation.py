"""Bind completed policies to the predeclared evaluation panels; no GPU launch."""
import argparse
from pathlib import Path
import time
from bfcl_source_eval_contract import digest, pinned, load_job, verify_seal
from bfcl_transfer_labels import source_registry
from bfcl_ssh_evaluation_control import resource_allocation
from supervise_bfcl_stage_fork import ROOT, read, write

EVAL_CODE = ['bfcl_source_eval_contract.py', 'eval_bfcl_source_transfer.py', 'report_bfcl_source_eval.py',
             'bfcl_transfer_labels.py', 'bfcl_transfer_predictor.py', 'seal_bfcl_transfer_predictions.py',
             'prepare_bfcl_source_evaluation.py', 'supervise_bfcl_source_evaluation.py',
             'bfcl_ssh_evaluation_control.py', 'bfcl_ssh_control.py']


def phase_cases(registry, phase):
    if phase not in ('fit', 'heldout'):
        raise ValueError('Require fit or heldout phase')
    return [case for case, row in registry.items() if (row['role'] == 'heldout') == (phase == 'heldout')]


def prepare(root, master_ref, registration, output, phase, prediction_seal=None, resource_authorization=None):
    master = pinned(root, master_ref)
    gpus, maximum = (master['allowed_gpus'], 4) if resource_authorization is None else resource_allocation(root, resource_authorization)
    batch = read(root/master['output_root']/'audit_status.json')
    if batch['status'] != 'passed' or batch['complete_jobs'] != 51:
        raise ValueError('Complete training audit is required before binding endpoint policies')
    batch_source = master['output_root'] + '/audit_status.json'
    events = [e for e in read(root/'experiments/index.json') if e['source'] == batch_source]
    if not events or read(root/'experiments/reviews'/(events[-1]['id']+'.json'))['status'] != 'reviewed':
        raise ValueError('The completed training matrix must be reviewed before evaluation registration')
    required_reviews = list(dict.fromkeys([*master['required_review_events'], events[-1]['id']]))
    analysis = pinned(root, master['analysis_contract'])
    training = [pinned(root, entry) for entry in master['jobs']]
    registry = source_registry(master, training, analysis)
    selected = phase_cases(registry, phase)
    if phase == 'heldout':
        if prediction_seal is None:
            raise ValueError('Heldout registration requires published actual predictions')
        verify_seal(root, prediction_seal, master_ref['sha256'], master['analysis_contract']['sha256'],
                    analysis['source_split']['heldout'])
    elif prediction_seal is not None:
        raise ValueError('Fit phase must precede the prediction seal')
    if output.exists() or registration.exists():
        raise FileExistsError('Do not overwrite a registration or an observed evaluation attempt')
    input_sha = dict(training[0]['input_sha256'])
    input_sha.update({str(Path('pipeline')/n): digest(root/'pipeline'/n) for n in EVAL_CODE})
    # Verify the full parent runtime, including base weights, once here; jobs retain
    # the smaller runtime set plus their policy artifacts, as in the training design.
    from supervise_bfcl_source_training import digest as streaming_digest
    for path, sha in master['input_sha256'].items():
        if streaming_digest(root/path) != sha:
            raise ValueError('Frozen parent dependency changed: '+path)
    for path, sha in input_sha.items():
        if streaming_digest(root/path) != sha:
            raise ValueError('Evaluation dependency changed: '+path)
    training_by_case = {j['case_id']: j for j in training}
    jobs = []
    for case in selected:
        if case == 'common':
            policy = root/master['source_stage0']['path']
            fingerprint = master['source_stage0']['policy_fingerprint']
            files = ['summary.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
        else:
            policy = root/training_by_case[case]['output']
            summary, audit = read(policy/'summary.json'), read(policy/'audit.json')
            if summary['status'] != 'complete' or audit['status'] != 'passed':
                raise ValueError('Incomplete policy: '+case)
            fingerprint = summary['final_fingerprint']
            files = ['summary.json', 'audit.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
        artifacts = {n: digest(policy/n) for n in files}
        for e in range(2):
            name = case+'_e'+str(e)
            jobs.append((name, {'execution_protocol': master_ref, 'policy_case': case, 'role': registry[case]['role'],
                'evaluation_repeat': e, 'policy_fingerprint': fingerprint, 'policy_artifacts': artifacts,
                'task_ids': master['evaluation']['task_ids'], 'task_seeds': master['evaluation']['task_seed_vectors'][e],
                'input_sha256': input_sha, 'output': str((output/name).relative_to(root)),
                'prediction_seal': prediction_seal}))
    registration.mkdir(parents=True, exist_ok=False)
    entries = []
    try:
        for name, job in jobs:
            path = registration/(name+'.json')
            write(path, job)
            entry = {'name': name, 'path': str(path.relative_to(root)), 'sha256': digest(path)}
            load_job(root, path, entry['sha256'], check_remote=False)
            entries.append(entry)
        manifest = {'kind': 'source_stage0_evaluation_execution', 'phase': phase, 'registered_unix': time.time(),
            'execution_protocol': master_ref, 'output_root': str(output.relative_to(root)), 'jobs': entries,
            'allowed_gpus': gpus, 'max_simultaneous_gpus': maximum, 'automatic_retry': False,
            'gpu_lanes': {str(g): [e['name'] for i, e in enumerate(entries) if i % len(gpus) == n] for n, g in enumerate(gpus)},
            'hard_seconds_per_panel': master['evaluation']['hard_seconds_per_panel'],
            'prediction_seal': prediction_seal, 'required_review_events': required_reviews,
            'input_sha256': master['input_sha256'] | input_sha,
            'scope': 'Registered panels only. Fit phase precedes actual prediction publication; 21 reserved targets stay closed.'}
        if resource_authorization is not None:
            manifest['resource_authorization'] = resource_authorization
        write(registration/'manifest.json', manifest)
        return manifest
    except Exception as error:
        write(registration/'registration_failure.json', {'type': type(error).__name__, 'message': str(error),
                                                       'gpu_jobs_launched': 0, 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['fit', 'heldout'], required=True)
    parser.add_argument('--prediction-receipt', type=Path)
    parser.add_argument('--prediction-receipt-sha256')
    parser.add_argument('--resource-authorization', type=Path)
    parser.add_argument('--resource-authorization-sha256')
    args = parser.parse_args()
    ref = {'path': str(args.protocol.resolve().relative_to(ROOT)), 'sha256': args.protocol_sha256}
    prediction_seal = None
    if args.prediction_receipt:
        prediction_seal = {'path': str(args.prediction_receipt.resolve().relative_to(ROOT)), 'sha256': args.prediction_receipt_sha256}
    resource_ref = None
    if args.resource_authorization:
        resource_ref = {'path': str(args.resource_authorization.resolve().relative_to(ROOT)), 'sha256': args.resource_authorization_sha256}
    result = prepare(ROOT, ref, args.registration.resolve(), args.output.resolve(), args.phase, prediction_seal, resource_ref)
    print(__import__('json').dumps({'phase': result['phase'], 'panels': len(result['jobs']), 'gpu_jobs_launched': 0}))


if __name__ == '__main__':
    main()
