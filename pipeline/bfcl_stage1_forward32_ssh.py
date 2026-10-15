"""Run the user-approved32 target panels once with separate current evaluation control."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from bfcl_source_eval_contract import digest, pinned, validate_panel
from bfcl_stage1_forward15_contract import MODE, validate
from bfcl_stage1_forward32_contract import SCOPE, RUNTIME, check_owner, admission_binding, source_policy
from bfcl_stage1_forward_bundle import verify_bundle
from bfcl_stage1_probe_ssh import execute_lanes, verify_publication
from bfcl_stage1_training_ssh import admission_failure_category
from prepare_bfcl_stage1_forward32 import cells, name, exclusive, ref
from supervise_bfcl_stage_fork import ROOT, read, supervise, process_has_not_exited

SELF = 'pipeline/bfcl_stage1_forward32_ssh.py'


def validate_runtime(root, manifest, jobs):
    expected = {path: digest(root/path) for path in RUNTIME}
    if manifest['runtime_sha256'] != expected:
        raise ValueError('Frozen target manifest runtime bytes changed')
    if any(any(job['input_sha256'].get(path) != value for path, value in expected.items()) for job in jobs):
        raise ValueError('Target job runtime differs from the original32 manifest')


def load_manifest(root, manifest_ref):
    manifest = pinned(root, manifest_ref)
    if (manifest['kind'] != 'stage1_forward32_evaluation_manifest' or manifest['stage'] != 1
            or manifest['batch_mode'] != MODE or manifest['phase'] != 'forward_validation'
            or manifest['automatic_retry'] is not False or manifest['max_simultaneous_gpus'] != 8
            or manifest['within_late_fitting_performed'] is not False or manifest['reserved21_allowed'] is not False):
        raise ValueError('Wrong current32 forward evaluation design')
    master = pinned(root, manifest['execution_protocol']); validate(root, master)
    binding, batch = admission_binding(root, manifest['admission_binding'], manifest['execution_protocol'], master)
    if (manifest['forward_bundle'] != binding['forward_bundle'] or len(manifest['jobs']) != 32
            or manifest['allowed_gpus'] != master['allowed_gpus'] or manifest['resource_authorization'] != master['resource_authorization']
            or manifest['new_token_ceiling'] != 8388608 or manifest['hard_seconds_per_panel'] != 77400):
        raise ValueError('Wrong active target count,resource,budget or actual forecast binding')
    verify_bundle(root, binding['forward_bundle'], manifest['execution_protocol'], time.time())
    jobs = [pinned(root, entry) for entry in manifest['jobs']]
    validate_runtime(root, manifest, jobs)
    expected = cells(master)
    if ([(job['policy_case'], job['evaluation_repeat']) for job in jobs] != expected
            or [entry['name'] for entry in manifest['jobs']] != [name(case, repeat) for case, repeat in expected]):
        raise ValueError('Target identities omitted,repeated,substituted or reordered')
    expected_lanes = {str(g): [entry['name'] for index, entry in enumerate(manifest['jobs']) if index % 8 == g] for g in range(8)}
    if manifest['gpu_lanes'] != expected_lanes: raise ValueError('Fixed32 target lane schedule changed')
    analysis = pinned(root, master['analysis_contract'])
    for job in jobs:
        if (job['execution_protocol'] != manifest['execution_protocol'] or job['admission_binding'] != manifest['admission_binding']
                or job['forward_bundle'] != manifest['forward_bundle'] or job.get('prediction_seal') is not None
                or job['stage'] != 1 or job['batch_mode'] != MODE):
            raise ValueError('Target job belongs to another batch/seal')
        validate_panel(job, master['evaluation']); source_policy(root, job, master, analysis, batch)
    return manifest, master, binding, jobs


def current(root, authority_ref, manifest_ref, manifest):
    authority = check_owner(root, authority_ref, manifest['execution_protocol'], live=True, started_unix=time.time())
    if authority['manifest'] != manifest_ref or authority['admission_binding'] != manifest['admission_binding']:
        raise ValueError('Current evaluation owner belongs to another original manifest')
    binding = pinned(root, manifest['admission_binding'])
    for control_root in binding['training_control_roots']:
        launch = read(root/control_root/'launch.json')
        if process_has_not_exited(launch['supervisor_pid']):
            raise RuntimeError('Prior training supervisor still owns an active execution segment')
    for eid in manifest['required_review_events']+authority['required_review_events']:
        if read(root/'experiments/reviews'/(eid+'.json'))['status'] != 'reviewed':
            raise ValueError('Missing required target execution review: '+eid)
    return authority


def preflight(root, gpu, authority_ref, manifest_ref, manifest, publication_ref):
    authority = current(root, authority_ref, manifest_ref, manifest)
    publication = pinned(root, publication_ref)
    for path, value in publication['files'].items():
        if digest(root/path) != value: raise ValueError('Published target execution bytes changed: '+path)
    if gpu not in manifest['allowed_gpus']: raise ValueError('GPU outside approved lanes')
    info = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines()
    if int(info[1].strip()) > 64 or info[0].strip() in apps: raise RuntimeError('GPU occupied;never stop unrelated work')
    return {'scope': SCOPE, 'authority': authority_ref, 'gpu': gpu, 'memory_mib': int(info[1].strip()),
            'no_compute_process': True, 'publication': publication_ref, 'admission_binding': manifest['admission_binding']}


def run(root, manifest_ref, authority_ref, commit):
    manifest, master, binding, jobs = load_manifest(root, manifest_ref)
    authority = current(root, authority_ref, manifest_ref, manifest)
    rel = Path(authority['control_root'])
    if rel.is_absolute() or '..' in rel.parts or rel.parts[:1] != ('rl',) or len(rel.parts) < 2:
        raise ValueError('Invalid current target control output')
    control = root/rel
    if control == root/manifest['output_root'] or control.is_relative_to(root/manifest['output_root']):
        raise ValueError('Keep execution control separate from scientific target panels')
    if any((root/job['output']).exists() for job in jobs):
        raise FileExistsError('An original target panel has evidence;register continuation instead of rerunning')
    control.mkdir(exist_ok=False)
    phase = 'publication'
    try:
        files = {manifest_ref['path']: manifest_ref['sha256'], authority_ref['path']: authority_ref['sha256'],
                 manifest['admission_binding']['path']: manifest['admission_binding']['sha256'],
                 manifest['execution_protocol']['path']: manifest['execution_protocol']['sha256'],
                 **{entry['path']: entry['sha256'] for entry in manifest['jobs']}, **authority['control_input_sha256']}
        for reference in (authority['owner_goal']['record'], binding['complete_training_audit'], binding['complete_training_review'], binding['forward_bundle'], binding['forward_publication']):
            files[reference['path']] = reference['sha256']
        for eid in set(manifest['required_review_events']+authority['required_review_events']):
            path = 'experiments/reviews/'+eid+'.json'; files[path] = digest(root/path)
        publication = verify_publication(root, files, commit, lambda: current(root, authority_ref, manifest_ref, manifest))
        exclusive(control/'publication_verified.json', publication); publication_ref = ref(root, control/'publication_verified.json')
        phase = 'watch_registration'
        watched = read(root/'experiments/watch_roots.json')
        if str(rel) not in watched or manifest['output_root'] not in watched: raise ValueError('Target/control outputs not registered')
        phase = 'gpu_admission'
        initial = {gpu: preflight(root, gpu, authority_ref, manifest_ref, manifest, publication_ref) for gpu in manifest['allowed_gpus']}
    except Exception as error:
        exclusive(control/'failures.json', [{'phase': phase, 'classification': admission_failure_category(phase, error),
                  'type': type(error).__name__, 'message': str(error), 'scientific_children_launched': 0}])
        exclusive(control/'supervisor.json', {'status': 'incomplete_requires_review', 'complete_panels': 0, 'automatic_retry': False})
        raise
    exclusive(control/'launch.json', {'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'manifest': manifest_ref,
              'authority': authority_ref, 'publication': publication_ref, 'preflights': initial,
              'scientific_panels': 32, 'max_simultaneous_gpus': 8})
    entries = {entry['name']: entry for entry in manifest['jobs']}
    def launch(gpu, case):
        entry = entries[case['name']]; job = pinned(root, entry)
        if (root/job['output']).exists(): raise FileExistsError('Never overwrite an original target panel')
        return supervise([str(root/'.venv-rl/bin/python'), '-u', str(root/'pipeline/run_bfcl_stage1_forward32.py'),
                          '--job', str(root/entry['path']), '--job-sha256', entry['sha256'],
                          '--authority', authority_ref['path'], '--authority-sha256', authority_ref['sha256']],
                         root/job['output'], control/(case['name']+'.console.log'), manifest['hard_seconds_per_panel'],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    def audit(case):
        if case is None:
            audits = [read(root/job['output']/'audit.json') for job in jobs]
            if any(audit['status'] != 'passed' or audit['job_sha256'] != entry['sha256'] for audit, entry in zip(audits, manifest['jobs'])):
                raise ValueError('Incomplete32 original raw target audits')
            tokens = sum(audit['cost']['new_model_tokens'] for audit in audits)
            if tokens > manifest['new_token_ceiling']: raise ValueError('Forward32 evaluation token ceiling exceeded')
            certificate = {'status': 'passed', 'stage': 1, 'batch_mode': MODE, 'complete_panels': 32,
                           'manifest': manifest_ref, 'execution_protocol': manifest['execution_protocol'],
                           'forward_bundle': manifest['forward_bundle'], 'admission_binding': manifest['admission_binding'],
                           'new_sampled_tokens': tokens, 'rows': audits, 'original_full104_complete': False}
            path = root/manifest['output_root']/'audit_status.json'; exclusive(path, certificate)
            exclusive(control/'target_audit_binding.json', {'complete_audit': ref(root, path)})
            return
        entry = entries[case]; job = pinned(root, entry)
        with (control/(case+'.audit.console.log')).open('x') as log:
            result = subprocess.run([str(root/'.venv-rl/bin/python'), str(root/'pipeline/report_bfcl_stage1_forward32.py'),
                                     '--job', str(root/entry['path'])], cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
        saved = read(root/job['output']/'audit.json')
        if (result.returncode or saved['status'] != 'passed' or saved['stage'] != 1 or saved['batch_mode'] != MODE
                or saved['job_sha256'] != entry['sha256'] or saved['cost']['new_model_tokens'] > 262144):
            raise ValueError('Original target raw audit failed: '+case)
    schedule = {'cases': manifest['jobs'], 'gpu_lanes': manifest['gpu_lanes'],
                'allowed_gpus': manifest['allowed_gpus'], 'max_simultaneous_gpus': 8}
    result = execute_lanes(control, schedule,
                           lambda gpu: preflight(root, gpu, authority_ref, manifest_ref, manifest, publication_ref), launch, audit)
    failures = []
    for failure in result['failures']:
        category = {'preflight': 'control_or_gpu_query_admission', 'audit': 'raw_target_audit_failure', 'batch_audit': 'full32_audit_failure'}.get(failure['phase'], 'scientific_process_failure')
        if failure['phase'] == 'runner':
            job = pinned(root, entries[failure['case']]); receipt = root/job['output']/'supervisor.json'
            if receipt.exists() and read(receipt).get('returncode') == 3: category = 'target_admission_before_inference'
            elif not (control/(failure['case']+'.console.process.json')).exists(): category = 'execution_dependency_before_scientific_child'
        failures.append(failure | {'classification': category})
    exclusive(control/'target_control_result.json', {'complete_panels': len(result['complete']), 'pending_panels': result['pending'],
              'classified_failures': failures, 'automatic_scientific_retry': False})
    if result['failures'] or result['pending']: raise RuntimeError('Forward32 incomplete;preserve original evidence and register any continuation separately')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'authority'):
        parser.add_argument('--'+field, required=True); parser.add_argument('--'+field+'-sha256', required=True)
    parser.add_argument('--publication-commit', required=True); args = parser.parse_args()
    run(ROOT, {'path': args.manifest, 'sha256': args.manifest_sha256},
        {'path': args.authority, 'sha256': args.authority_sha256}, args.publication_commit)
