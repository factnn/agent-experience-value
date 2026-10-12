"""Bind transfer evaluations to registered sources, panels, and published predictions."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(root, reference):
    path = root / reference['path']
    if digest(path) != reference['sha256']:
        raise ValueError('Pinned artifact changed: ' + reference['path'])
    return json.loads(path.read_text())


def policy_role(policy_case, training_job, analysis):
    if policy_case == 'common':
        if training_job is not None:
            raise ValueError('Common policy cannot alias a training job')
        return 'baseline'
    if training_job is None or training_job['case_id'] != policy_case:
        raise ValueError('Policy does not match its registered training job')
    if training_job['role'] == 'uniform_control':
        return 'control'
    if training_job['role'] != 'source_group':
        raise ValueError('Unsupported evaluation source')
    group = training_job['source_group']
    if group in analysis['source_split']['fit']:
        return 'fit'
    if group in analysis['source_split']['heldout']:
        return 'heldout'
    raise ValueError('Unknown source group')


def validate_panel(job, evaluation):
    repeat = job['evaluation_repeat']
    if type(repeat) is not int or repeat not in (0, 1):
        raise ValueError('Exactly the two registered evaluation repeats are allowed')
    if job['task_ids'] != evaluation['task_ids'] or job['task_seeds'] != evaluation['task_seed_vectors'][repeat]:
        raise ValueError('Cannot change target tasks, order, or seeds')
    if evaluation['rollouts_per_task'] != 1 or evaluation['reserved21_allowed']:
        raise ValueError('Different evaluation design needs a separate registration')


def verify_seal(root, reference, master_sha, analysis_sha, heldout_groups, check_remote=True):
    receipt = pinned(root, reference)
    directory = (root / reference['path']).parent
    if (receipt['execution_protocol_sha256'] != master_sha or receipt['analysis_contract_sha256'] != analysis_sha
            or receipt['heldout_groups'] != heldout_groups or receipt['stage'] != 0):
        raise ValueError('Prediction seal belongs to another analysis or stage')
    required = {'models.json', 'predictions.json', 'fit_labels.json'}
    if set(receipt['files']) != required:
        raise ValueError('Incomplete prediction seal')
    for name, sha in receipt['files'].items():
        if digest(directory / name) != sha:
            raise ValueError('Sealed prediction artifact changed')
    publication = json.loads((directory / 'publication.json').read_text())
    if publication['receipt_sha256'] != reference['sha256'] or publication['verified_remote_at_unix'] < receipt['sealed_unix']:
        raise ValueError('Prediction publication does not follow sealing')
    commit = publication['commit']
    expected = receipt['files'] | {Path(reference['path']).name: reference['sha256']}
    for name, sha in expected.items():
        relative = str((directory / name).relative_to(root))
        result = subprocess.run(['git', 'show', commit + ':' + relative], cwd=root, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
            raise ValueError('Git publication does not contain the sealed bytes')
    if check_remote:
        result = subprocess.run(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root, capture_output=True, timeout=30)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError('Cannot verify remote prediction publication')
        remote = result.stdout.decode().split()[0]
        result = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=root, capture_output=True)
        if result.returncode:
            raise RuntimeError('Prediction commit is not verified on current remote history')
    return receipt


def load_job(root, path, expected_sha, check_remote=True):
    if digest(path) != expected_sha:
        raise ValueError('Evaluation job changed')
    job = json.loads(path.read_text())
    master = pinned(root, job['execution_protocol'])
    analysis = pinned(root, master['analysis_contract'])
    validate_panel(job, master['evaluation'])
    batch = json.loads((root / master['output_root'] / 'audit_status.json').read_text())
    if batch['status'] != 'passed' or batch['complete_jobs'] != 51:
        raise ValueError('The complete training matrix must pass before evaluation')
    case = job['policy_case']
    training_job = None
    if case == 'common':
        policy = root / master['source_stage0']['path']
        expected_fingerprint = master['source_stage0']['policy_fingerprint']
        required = ['summary.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
    else:
        entry = next((e for e in master['jobs'] if e['name'] == case), None)
        if entry is None:
            raise ValueError('Unregistered policy case')
        training_job = pinned(root, entry)
        policy = root / training_job['output']
        summary = json.loads((policy / 'summary.json').read_text())
        audit = json.loads((policy / 'audit.json').read_text())
        if summary['status'] != 'complete' or audit['status'] != 'passed' or audit['job_sha256'] != entry['sha256']:
            raise ValueError('Source training is incomplete or unaudited')
        if audit['final_payload_sha256'] != summary['final_manifest']['payload_sha256']:
            raise ValueError('Source learning-state lineage changed')
        expected_fingerprint = summary['final_fingerprint']
        required = ['summary.json', 'audit.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']
    role = policy_role(case, training_job, analysis)
    if job['role'] != role or job['policy_fingerprint'] != expected_fingerprint:
        raise ValueError('Policy role or fingerprint changed')
    if set(job['policy_artifacts']) != set(required):
        raise ValueError('Missing source policy artifacts')
    for name, sha in job['policy_artifacts'].items():
        if digest(policy / name) != sha:
            raise ValueError('Source policy artifact changed')
    for file, sha in job['input_sha256'].items():
        if digest(root / file) != sha:
            raise ValueError('Frozen evaluation dependency changed: ' + file)
    if role == 'heldout':
        if not job.get('prediction_seal'):
            raise ValueError('Heldout target evaluation requires published point predictions')
        seal = verify_seal(root, job['prediction_seal'], job['execution_protocol']['sha256'],
                           master['analysis_contract']['sha256'], analysis['source_split']['heldout'], check_remote=check_remote)
    else:
        seal = None
    first_training_job = pinned(root, master['jobs'][0])
    algorithm = pinned(root, first_training_job['algorithm_protocol'])
    return job, master, algorithm, policy, seal
