"""Register the exact32 prospective target jobs after reviewed15 training endpoints."""
import argparse
import json
from pathlib import Path
import time

from bfcl_source_eval_contract import digest, pinned, policy_role
from bfcl_stage1_forward15_contract import MODE, validate
from bfcl_stage1_forward32_contract import RUNTIME, admission_binding, source_policy
from bfcl_stage1_forward_bundle import verify_bundle
from supervise_bfcl_stage_fork import ROOT, read

OUTPUT = 'rl/bfcl_source_stage1_forward_eval_20261015'
REGISTRATION = 'rl/bfcl_source_stage1_forward_jobs_20261015'


def cells(master):
    return [(case, repeat) for case in ['common', *[entry['name'] for entry in master['jobs']]] for repeat in range(2)]


def name(case, repeat):
    return ('s1_common' if case == 'common' else case)+'_e'+str(repeat)


def ref(root, path):
    return {'path': str(path.resolve().relative_to(root.resolve())), 'sha256': digest(path)}


def exclusive(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def prepare(root, protocol_ref, binding_ref):
    master = pinned(root, protocol_ref); validate(root, master)
    if not (root/master['complete_training_audit_path']).exists():
        raise ValueError('All15 reviewed training endpoints are required before target registration')
    binding, batch = admission_binding(root, binding_ref, protocol_ref, master)
    verify_bundle(root, binding['forward_bundle'], protocol_ref, time.time())
    registration, output = root/REGISTRATION, root/OUTPUT
    if registration.exists() or output.exists():
        raise FileExistsError('Preserve existing original target registrations/results')
    analysis = pinned(root, master['analysis_contract'])
    training = {entry['name']: pinned(root, entry) for entry in master['jobs']}
    inputs = master['input_sha256'] | {path: digest(root/path) for path in RUNTIME}
    jobs = []
    for case, repeat in cells(master):
        source = training.get(case)
        if case == 'common':
            policy = root/master['stage_start']['path']
            fingerprint = master['stage_start']['policy_fingerprint']
            required = ['summary.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
        else:
            policy = root/source['output']; fingerprint = read(policy/'summary.json')['final_fingerprint']
            required = ['summary.json', 'audit.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
        job = {'kind': 'stage1_forward_only_target_panel', 'stage': 1, 'batch_mode': MODE,
               'execution_protocol': protocol_ref, 'admission_binding': binding_ref,
               'forward_bundle': binding['forward_bundle'], 'prediction_seal': None,
               'policy_case': case, 'role': policy_role(case, source, analysis), 'evaluation_repeat': repeat,
               'task_ids': master['evaluation']['task_ids'], 'task_seeds': master['evaluation']['task_seed_vectors'][repeat],
               'policy_fingerprint': fingerprint, 'policy_artifacts': {path: digest(policy/path) for path in required},
               'input_sha256': inputs, 'output': OUTPUT+'/'+name(case, repeat)}
        source_policy(root, job, master, analysis, batch)
        jobs.append(job)
    if len(jobs) != 32 or len({job['output'] for job in jobs}) != 32:
        raise ValueError('Require exactly the original32 distinct target panels')
    registration.mkdir(exist_ok=False)
    entries = []
    for job in jobs:
        identifier = name(job['policy_case'], job['evaluation_repeat'])
        path = registration/(identifier+'.json'); exclusive(path, job)
        entries.append({'name': identifier, **ref(root, path)})
    manifest = {'kind': 'stage1_forward32_evaluation_manifest', 'stage': 1, 'batch_mode': MODE,
                'phase': 'forward_validation', 'registered_unix': time.time(), 'execution_protocol': protocol_ref,
                'admission_binding': binding_ref, 'forward_bundle': binding['forward_bundle'], 'output_root': OUTPUT,
                'jobs': entries, 'allowed_gpus': master['allowed_gpus'], 'max_simultaneous_gpus': 8,
                'resource_authorization': master['resource_authorization'], 'automatic_retry': False,
                'hard_seconds_per_panel': master['evaluation']['hard_seconds_per_panel'],
                'new_token_ceiling': master['evaluation']['new_token_ceiling'],
                'gpu_lanes': {str(g): [entry['name'] for index, entry in enumerate(entries) if index % 8 == g] for g in range(8)},
                'required_review_events': list(dict.fromkeys(master['required_review_events']+binding['required_review_events'])),
                'runtime_sha256': {path: digest(root/path) for path in RUNTIME},
                'within_late_fitting_performed': False, 'reserved21_allowed': False,
                'scope': 'Current32 forward panels only. Old12-fit-source interventions/72panels remain deferred;this is not full104.'}
    manifest_path = registration/'manifest.json'; exclusive(manifest_path, manifest)
    return ref(root, manifest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('protocol', 'binding'):
        parser.add_argument('--'+field, required=True); parser.add_argument('--'+field+'-sha256', required=True)
    args = parser.parse_args()
    result = prepare(ROOT, {'path': args.protocol, 'sha256': args.protocol_sha256},
                     {'path': args.binding, 'sha256': args.binding_sha256})
    print(json.dumps(result, indent=2))
