"""Native8B target identity, full-cohort audit and actual pre-label forecasts."""
import datetime
import hashlib
from pathlib import Path
import subprocess
import time

from bfcl_source_eval_contract import digest, pinned, policy_role, validate_panel
from bfcl_qwen8_source_training import REVISION, identity, load_job as load_training_job
from seal_bfcl_qwen8_cross_size_predictions import verify_cross_size_seal, verify_chronology
from seal_bfcl_model_conditioned_predictions import verify_auxiliary_seal

SELF = 'pipeline/bfcl_qwen8_transfer_contract.py'
SCOPE = 'qwen3_8b_stage0_forward_target_evaluation'
RUNTIME = {SELF, 'pipeline/run_bfcl_qwen8_source_transfer.py', 'pipeline/report_bfcl_qwen8_source_transfer.py',
           'pipeline/bfcl_qwen8_source_training.py', 'pipeline/eval_bfcl_source_transfer.py',
           'pipeline/report_bfcl_source_eval.py', 'pipeline/report_bfcl_intervention.py',
           'pipeline/bfcl_source_eval_contract.py', 'pipeline/run_bfcl_intervention.py',
           'pipeline/rl_learning_state.py', 'pipeline/seal_bfcl_qwen8_cross_size_predictions.py',
           'pipeline/seal_bfcl_model_conditioned_predictions.py'}


def read(path):
    import json
    return json.loads(path.read_text())


def complete_training(root, execution_ref, master, binding):
    if (binding['kind'] != 'qwen3_8b_stage0_forward_target_admission'
            or identity(binding['execution_protocol']) != identity(execution_ref)
            or identity(binding['native_common_binding']) != identity(master['native_common_binding'])):
        raise ValueError('Target binding belongs to another native8B learner/cohort')
    batch = pinned(root, binding['complete_training_audit'])
    expected = {e['sha256'] for e in master['jobs']}
    if (batch['status'] != 'passed' or batch['complete_jobs'] != len(master['jobs'])
            or identity(batch['execution_protocol']) != identity(execution_ref)
            or len(batch['jobs']) != len(master['jobs'])
            or {row['job_sha256'] for row in batch['jobs']} != expected
            or any(row['status'] != 'passed' or row['native_model_revision'] != REVISION for row in batch['jobs'])):
        raise ValueError('Every registered native8B RL branch must pass before any target role')
    actual = sum(row['cost']['new_model_tokens'] for row in batch['jobs'])
    if actual != batch['new_sampled_tokens'] or actual > master['new_training_token_ceiling']:
        raise ValueError('Actual native8B training costs differ or exceed their finite ceiling')
    review = pinned(root, binding['complete_training_review'])
    if (review['kind'] != 'complete_native8b_source_training_execution_review'
            or review['status'] != 'reviewed'
            or identity(review['execution_protocol']) != identity(execution_ref)
            or identity(review['complete_audit']) != identity(binding['complete_training_audit'])
            or review['fresh_raw_audits_match_saved'] is not True
            or review['no_failed_or_omitted_active_branches'] is not True):
        raise ValueError('Complete native8B raw training execution review is required')
    terminal = pinned(root, binding['prior_training_terminal_proof'])
    if (terminal['all_owned_scientific_children_terminal'] is not True
            or identity(terminal['execution_protocol']) != identity(execution_ref)
            or sorted(terminal['scientific_job_sha256s']) != sorted(expected)):
        raise ValueError('Original training processes must be terminal before target admission')
    return batch


def verify_unavailable_auxiliary(root, disposition, master, started):
    """A bounded auxiliary failure must be published, rather than silently dropped."""
    receipt_ref = disposition['receipt']
    receipt = pinned(root, receipt_ref)
    if (receipt['kind'] != 'e368_bounded_measurement_failure_disposition'
            or receipt['target_model_revision'] != REVISION
            or identity(receipt['native_common_binding']) != identity(master['native_common_binding'])
            or receipt['native8b_target_labels_used'] is not False
            or receipt['all_owned_scientific_children_terminal'] is not True
            or receipt['automatic_retry'] is not False or receipt['actual_auxiliary_points_available'] is not False
            or receipt['post_label_repair_can_be_called_prospective'] is not False):
        raise ValueError('Auxiliary measurement cannot be silently skipped or repeatedly retried')
    design = pinned(root, receipt['bounded_design'])
    if (identity(receipt['bounded_design']) != identity(master['auxiliary_candidate_design'])
            or design['kind'] != 'distinct_model_conditioned_value_candidates_design'
            or identity(design['profiles']['qwen8b']['native_common_binding']) != identity(master['native_common_binding'])
            or design['maximum_worker_seconds'] != 115200):
        raise ValueError('Wrong bounded auxiliary measurement disposition')
    directory = (root/receipt_ref['path']).parent
    if not receipt['files'] or any(Path(name).name != name for name in receipt['files']):
        raise ValueError('Exact partial-cost and terminal evidence files are required')
    for name, sha in receipt['files'].items():
        if digest(directory/name) != sha:
            raise ValueError('Auxiliary failure evidence changed: '+name)
    evidence = pinned(root, receipt['partial_cost_and_terminal_evidence'])
    evidence_path = root/receipt['partial_cost_and_terminal_evidence']['path']
    if (evidence_path.parent != directory
            or receipt['files'].get(evidence_path.name) != receipt['partial_cost_and_terminal_evidence']['sha256']):
        raise ValueError('The actual incomplete-work ledger must be included in published disposition files')
    if (not evidence['failures'] or not evidence['raw_evidence_refs']
            or evidence['all_owned_scientific_children_terminal'] is not True
            or type(evidence['completed_cases']) is not int or not 0 <= evidence['completed_cases'] <= 32
            or evidence['complete_measurement_audit_passed'] is not False or evidence['scientific_retries'] != 0
            or evidence['native8b_target_labels_used'] is not False):
        raise ValueError('A failed bounded measurement needs its real incomplete-work ledger')
    for item in evidence['raw_evidence_refs']:
        if digest(root/item['path']) != item['sha256']:
            raise ValueError('Actual bounded-measurement failure evidence changed')
    publication = pinned(root, disposition['publication'])
    if ((root/disposition['publication']['path']).parent != directory
            or publication['receipt_sha256'] != receipt_ref['sha256']):
        raise ValueError('Wrong publication for auxiliary unavailability')
    verify_chronology(receipt, publication, started)
    files = {str((directory/name).relative_to(root)): sha for name, sha in receipt['files'].items()}
    files[receipt_ref['path']] = receipt_ref['sha256']
    for name, sha in files.items():
        result = subprocess.run(['git', 'show', publication['commit']+':'+name], cwd=root,
                                capture_output=True, timeout=30)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
            raise ValueError('Failure disposition was not published with its actual evidence')
    return receipt


def forecasts(root, binding, master, started):
    core = verify_cross_size_seal(root, binding['cross_size_seal'], master['native_common_binding'],
        master['analysis_contract'], publication_ref=binding['cross_size_publication'],
        check_remote=False, evaluation_started_unix=started)
    auxiliary = binding['auxiliary_forecast_disposition']
    if auxiliary['status'] == 'sealed':
        extra = verify_auxiliary_seal(root, auxiliary['receipt'], auxiliary['publication'],
            master['native_common_binding'], master['analysis_contract'], check_remote=False,
            evaluation_started_unix=started)
        if identity(extra['design']) != identity(master['auxiliary_candidate_design']):
            raise ValueError('Auxiliary candidates differ from their predeclared bounded design')
    elif auxiliary['status'] == 'unavailable_after_bounded_measurement':
        extra = verify_unavailable_auxiliary(root, auxiliary, master, started)
    else:
        raise ValueError('Auxiliary candidate status must be fixed and published before native8B targets')
    return core, extra


def owner(root, authority_ref, job, master, *, live, started):
    authority = pinned(root, authority_ref)
    if (authority['mode'] != 'human_authorized_ssh_takeover' or authority['status'] != 'authorized'
            or authority['scope'] != SCOPE or authority['current_ssh_owner'] != master['current_ssh_owner']
            or identity(authority['execution_protocol']) != identity(job['execution_protocol'])
            or identity(authority['admission_binding']) != identity(job['admission_binding'])
            or identity(authority['resource_authorization']) != identity(master['resource_authorization'])
            or authority['registered_unix'] > started):
        raise ValueError('Wrong separately registered current native8B target owner')
    for key in ['stop_file', 'global_stop_file']:
        path = (root/authority[key]).resolve()
        if path.parent != (root/'experiments/.runtime').resolve():
            raise ValueError('Stop marker outside current project runtime')
        if live and path.exists():
            raise RuntimeError('Current SSH owner stopped new native8B target admission')
    if not RUNTIME <= authority['control_input_sha256'].keys():
        raise ValueError('Unpinned native8B target runtime')
    for name, sha in authority['control_input_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Native8B target code changed: '+name)
    manifest = pinned(root, authority['manifest'])
    if (manifest['kind'] != 'native8b_forward_target_manifest'
            or identity(manifest['execution_protocol']) != identity(job['execution_protocol'])
            or identity(manifest['admission_binding']) != identity(job['admission_binding'])):
        raise ValueError('Current owner has another target manifest')
    return authority, manifest


def publication(root, publication_ref, job_ref, authority_ref, authority, binding, started):
    value = pinned(root, publication_ref)
    if (value['commit'] != value['remote_main'] or value['scientific_child_started'] is not False
            or datetime.datetime.fromisoformat(value['verified_at']).timestamp() > started):
        raise ValueError('Target code/forecasts were not verified on the actual remote before admission')
    required = [job_ref, authority_ref, authority['manifest'], pinned(root, job_ref)['execution_protocol'],
                authority['admission_binding'], binding['complete_training_audit'], binding['complete_training_review'],
                binding['prior_training_terminal_proof'],
                binding['cross_size_seal'], binding['cross_size_publication'],
                binding['auxiliary_forecast_disposition']['receipt'], binding['auxiliary_forecast_disposition']['publication']]
    expected = {item['path']: item['sha256'] for item in required} | authority['control_input_sha256']
    if any(value['files'].get(name) != sha for name, sha in expected.items()):
        raise ValueError('Verified remote admission receipt omits an actual target prerequisite')
    for name, sha in value['files'].items():
        if digest(root/name) != sha:
            raise ValueError('Published target admission bytes changed: '+name)
    return value


def load_job(root, path, expected_sha, *, authority_ref, publication_ref, admit=False, evaluation_started_unix=None):
    job_ref = {'path': str(path.resolve().relative_to(root.resolve())), 'sha256': expected_sha}
    job = pinned(root, job_ref)
    master = pinned(root, job['execution_protocol'])
    design = pinned(root, master['design'])
    if (master['kind'] != 'qwen3_8b_stage0_forward_training_and_transfer'
            or master['model_revision'] != REVISION or master['stage'] != 0
            or design['kind'] != 'qwen3_8b_stage0_source_forward_design'
            or job['kind'] != 'native8b_frozen_forward_target_panel' or job['stage'] != 0
            or master['evaluation'] != design['evaluation']
            or master['reserved21_allowed'] is not False or master['automatic_retry'] is not False):
        raise ValueError('Wrong registered native8B target design')
    validate_panel(job, master['evaluation'])
    if len(job['task_ids']) != 32 or master['evaluation']['reserved21_allowed'] is not False:
        raise ValueError('Use only the original32 target tasks; reserved21 remain closed')
    binding = pinned(root, job['admission_binding'])
    batch = complete_training(root, job['execution_protocol'], master, binding)
    started = time.time() if evaluation_started_unix is None else evaluation_started_unix
    current, manifest = owner(root, authority_ref, job, master, live=admit, started=started)
    entries = [e for e in manifest['jobs'] if identity(e) == identity(job_ref)]
    if len(entries) != 1:
        raise ValueError('Target job is not in the exact current target manifest')
    expected_cells = [(case, repeat) for case in ['common', *[e['name'] for e in master['jobs']]] for repeat in range(2)]
    if (len(manifest['jobs']) != len(expected_cells)
            or [(pinned(root,e)['policy_case'], pinned(root,e)['evaluation_repeat']) for e in manifest['jobs']] != expected_cells):
        raise ValueError('Target manifest omits, substitutes or repeats a frozen endpoint/seed')
    publication(root, publication_ref, job_ref, authority_ref, current, binding, started)
    core, extra = forecasts(root, binding, master, started)
    if identity(job['prediction_seal']) != identity(binding['cross_size_seal']):
        raise ValueError('Target points differ from the actual published cross-size predictions')
    common = pinned(root, master['native_common_binding'])
    source = None
    if job['policy_case'] == 'common':
        policy = root/common['common_policy']; fingerprint = common['policy_fingerprint']
        required = {'summary.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
    else:
        entry = next((e for e in master['jobs'] if e['name'] == job['policy_case']), None)
        if entry is None:
            raise ValueError('Target endpoint was not part of the complete native8B RL cohort')
        source, _, _, _ = load_training_job(root, root/entry['path'], entry['sha256'], job['execution_protocol'])
        policy = root/source['output']; summary = read(policy/'summary.json'); audit = read(policy/'audit.json')
        if (summary['status'] != 'complete' or audit != next((x for x in batch['jobs'] if x['job_sha256'] == entry['sha256']), None)
                or audit['final_payload_sha256'] != summary['final_manifest']['payload_sha256']):
            raise ValueError('Target endpoint differs from the completely audited native8B branch')
        fingerprint = summary['final_fingerprint']
        required = {'summary.json', 'audit.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
    if (job['role'] != policy_role(job['policy_case'], source, pinned(root, master['analysis_contract']))
            or job['policy_fingerprint'] != fingerprint or set(job['policy_artifacts']) != required):
        raise ValueError('Wrong native8B target role, weights or endpoint evidence')
    for name, sha in job['policy_artifacts'].items():
        if digest(policy/name) != sha:
            raise ValueError('Target endpoint bytes changed: '+name)
    if not RUNTIME <= job['input_sha256'].keys():
        raise ValueError('Missing native8B target runtime identity')
    for name, sha in job['input_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Frozen native8B target input changed: '+name)
    algorithm = pinned(root, master['algorithm_protocol'])
    if algorithm['base_revision'] != REVISION:
        raise ValueError('Target inference cannot use the4B base')
    proof = {'core_seal': binding['cross_size_seal'], 'core_publication': binding['cross_size_publication'],
             'auxiliary_disposition': binding['auxiliary_forecast_disposition'], 'execution_publication': publication_ref,
             'target_labels_used_in_forecasts': False, 'verified_for_every_target_role': True}
    return job | {'_runtime_forward_verification': proof}, master, algorithm, policy, core
