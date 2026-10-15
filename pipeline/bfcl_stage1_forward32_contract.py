"""Stage1 forward-only target admission: complete15,actual forecasts,every role.

Replay audits verify historical byte/timing evidence without applying today's
admission stop to already completed work. Live admission also checks current
phase-specific ownership and current stops; no old Goal control is queried.
"""
import hashlib
import time
from bfcl_source_eval_contract import digest, pinned, policy_role, validate_panel
from bfcl_stage1_forward15_contract import MODE, validate
from bfcl_stage1_forward_bundle import verify_bundle, verify_forward_target_gate
from supervise_bfcl_stage_fork import read

SELF = 'pipeline/bfcl_stage1_forward32_contract.py'
SCOPE = 'stage1_fixed_parent_forward_evaluation_32'
RUNTIME = [SELF, 'pipeline/run_bfcl_stage1_forward32.py', 'pipeline/report_bfcl_stage1_forward32.py',
           'pipeline/prepare_bfcl_stage1_forward32.py', 'pipeline/bfcl_stage1_forward32_ssh.py',
           'pipeline/bfcl_stage1_forward15_contract.py', 'pipeline/bfcl_stage1_forward_bundle.py',
           'pipeline/eval_bfcl_source_transfer.py', 'pipeline/report_bfcl_source_eval.py',
           'pipeline/bfcl_source_eval_contract.py', 'pipeline/report_bfcl_intervention.py',
           'pipeline/bfcl_stage1_training_ssh.py', 'pipeline/bfcl_stage1_probe_ssh.py',
           'pipeline/supervise_bfcl_stage1_probe.py', 'pipeline/supervise_bfcl_stage_fork.py']


def input_digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def check_owner(root, authority_ref, protocol_ref, *, live=True, started_unix=None):
    authority = pinned(root, authority_ref)
    master = pinned(root, protocol_ref)
    if (authority.get('mode') != 'human_authorized_ssh_takeover' or authority.get('status') != 'authorized'
            or authority.get('scope') != SCOPE or authority.get('execution_protocol') != protocol_ref
            or authority.get('resource_authorization') != master['resource_authorization']):
        raise ValueError('Wrong current forward32 evaluation authority/scope')
    owner = pinned(root, authority['owner_goal']['record'])
    if (owner['thread_id'] != authority['owner_goal']['thread_id'] or owner['status_at_registration'] != 'active'
            or owner['control_scope'] != SCOPE):
        raise ValueError('Wrong separately registered evaluation owner')
    if started_unix is not None and authority['registered_unix'] > started_unix:
        raise ValueError('Evaluation authority was registered after scientific use')
    binding = pinned(root, authority['admission_binding'])
    receipt = pinned(root, binding['forward_bundle']); definition = pinned(root, receipt['definition'])
    if (binding['execution_protocol'] != protocol_ref or authority['stop_file'] != definition['evaluation_stop_file']
            or authority['global_stop_file'] != definition['global_stop_file']):
        raise ValueError('Evaluation scope/stopping bindings changed')
    for name in ('stop_file', 'global_stop_file'):
        path = (root/authority[name]).resolve()
        if path.parent != (root/'experiments/.runtime').resolve():
            raise ValueError('Evaluation stop outside current project runtime')
        if live and path.exists():
            raise RuntimeError('Current SSH owner stopped new forward32 evaluation admission')
    if not set(RUNTIME) <= authority['control_input_sha256'].keys():
        raise ValueError('Missing required target execution code pins')
    for name, value in authority['control_input_sha256'].items():
        if digest(root/name) != value: raise ValueError('Pinned evaluation controller changed: '+name)
    return authority


def admission_binding(root, reference, protocol_ref, master):
    binding = pinned(root, reference)
    if (binding['kind'] != 'stage1_forward32_evaluation_admission_binding' or binding['stage'] != 1
            or binding['batch_mode'] != MODE or binding['execution_protocol'] != protocol_ref):
        raise ValueError('Wrong forward32 complete-training binding')
    if binding['complete_training_audit']['path'] != master['complete_training_audit_path']:
        raise ValueError('Expected the active15 certificate,not the deferred full51 matrix')
    batch = pinned(root, binding['complete_training_audit'])
    if (batch['status'] != 'passed' or batch['stage'] != 1 or batch['batch_mode'] != MODE
            or batch['complete_jobs'] != 15 or len(batch['jobs']) != 15 or batch['execution_protocol'] != protocol_ref
            or batch['forward_bundle'] != binding['forward_bundle'] or batch['original_full51_complete'] is not False
            or batch['evaluation_performed'] is not False):
        raise ValueError('All15 active branches must pass raw audit before any target panel')
    expected = {entry['sha256'] for entry in master['jobs']}
    if {row['job_sha256'] for row in batch['jobs']} != expected or any(row['status'] != 'passed' for row in batch['jobs']):
        raise ValueError('Complete15 certificate contains wrong or failed original job identities')
    actual = sum(row['cost']['new_model_tokens'] for row in batch['jobs'])
    if actual != batch['new_sampled_tokens'] or actual > master['new_training_token_ceiling']:
        raise ValueError('Complete15 actual training cost is inconsistent or beyond its finite ceiling')
    review = pinned(root, binding['complete_training_review'])
    if (review['kind'] != 'complete_stage1_forward15_training_execution_review' or review['status'] != 'reviewed'
            or review['execution_protocol'] != protocol_ref or review['complete_audit'] != binding['complete_training_audit']
            or review['fresh_raw_audits_match_saved'] is not True or review['no_failed_or_omitted_active_branches'] is not True):
        raise ValueError('Missing complete15 execution-agent raw review')
    if binding['within_late_fitting_allowed'] is not False:
        raise ValueError('This batch cannot fit a model on late target labels')
    publication_path = (root/binding['forward_bundle']['path']).parent/'publication.json'
    if binding['forward_publication']['path'] != str(publication_path.relative_to(root)):
        raise ValueError('Missing exact pre-acquisition forecast publication receipt')
    publication = pinned(root, binding['forward_publication'])
    if publication['receipt_sha256'] != binding['forward_bundle']['sha256']:
        raise ValueError('Forward publication receipt belongs to another sealed bundle')
    return binding, batch


def registered_job(root, path, expected_sha, authority):
    manifest = pinned(root, authority['manifest'])
    if (manifest['kind'] != 'stage1_forward32_evaluation_manifest' or len(manifest['jobs']) != 32
            or manifest['execution_protocol'] != authority['execution_protocol']
            or manifest['admission_binding'] != authority['admission_binding']):
        raise ValueError('Current evaluation authority does not bind the exact32 original jobs')
    relative = str(path.resolve().relative_to(root.resolve()))
    matches = [entry for entry in manifest['jobs'] if entry['path'] == relative and entry['sha256'] == expected_sha]
    if len(matches) != 1:
        raise ValueError('Target job is not an original panel of the current32 manifest')


def source_policy(root, job, master, analysis, batch):
    case = job['policy_case']; training = None
    if case == 'common':
        policy = root/master['stage_start']['path']
        expected = master['stage_start']['policy_fingerprint']
        required = {'summary.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
    else:
        entry = next((entry for entry in master['jobs'] if entry['name'] == case), None)
        if entry is None: raise ValueError('Policy is outside the active15 branches;deferred fit sources are not admitted')
        training = pinned(root, entry); policy = root/training['output']
        summary, audit = read(policy/'summary.json'), read(policy/'audit.json')
        if (summary['status'] != 'complete' or audit['status'] != 'passed' or audit['job_sha256'] != entry['sha256']
                or audit['final_payload_sha256'] != summary['final_manifest']['payload_sha256']
                or audit != next((row for row in batch['jobs'] if row['job_sha256'] == entry['sha256']), None)):
            raise ValueError('Source endpoint is incomplete,unaudited or changed from complete15 evidence')
        expected = summary['final_fingerprint']
        required = {'summary.json', 'audit.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
    role = policy_role(case, training, analysis)
    if role not in ('baseline', 'control', 'heldout') or job['role'] != role or job['policy_fingerprint'] != expected:
        raise ValueError('Wrong active role or stage1 policy fingerprint')
    if set(job['policy_artifacts']) != required:
        raise ValueError('Missing exact endpoint adapter/summary/audit artifacts')
    for name, value in job['policy_artifacts'].items():
        if digest(policy/name) != value: raise ValueError('Frozen source policy bytes changed: '+name)
    return policy


def load_job(root, path, expected_sha, *, admit=False, authority_ref=None, evaluation_started_unix=None):
    if digest(path) != expected_sha: raise ValueError('Target job changed')
    job = read(path)
    protocol_ref = job['execution_protocol']; master = pinned(root, protocol_ref)
    validate(root, master)
    if job['stage'] != 1 or job['batch_mode'] != MODE:
        raise ValueError('Different target stage or scope')
    if not job.get('forward_bundle'):
        raise ValueError('Every forward32 target role requires actual published early-five and H1 forecasts')
    if job.get('prediction_seal') is not None:
        raise ValueError('Within-late fitting/seals are deferred;never substitute an old stage0 seal')
    validate_panel(job, master['evaluation'])
    if admit:
        if not authority_ref: raise ValueError('Live target admission requires separate current evaluation ownership')
        authority = check_owner(root, authority_ref, protocol_ref, live=True, started_unix=time.time())
        if authority['admission_binding'] != job['admission_binding']:
            raise ValueError('Current owner belongs to another target admission binding')
        registered_job(root, path, expected_sha, authority)
        forward = verify_forward_target_gate(root, job, master, evaluation_started_unix)
    else:
        forward = verify_bundle(root, job['forward_bundle'], protocol_ref, evaluation_started_unix)
    binding, batch = admission_binding(root, job['admission_binding'], protocol_ref, master)
    if binding['forward_bundle'] != job['forward_bundle']:
        raise ValueError('Target job changed its actual pre-acquisition forecasts')
    analysis = pinned(root, master['analysis_contract'])
    policy = source_policy(root, job, master, analysis, batch)
    if not set(RUNTIME) <= job['input_sha256'].keys():
        raise ValueError('Target job omitted required versioned execution code')
    for name, value in job['input_sha256'].items():
        if input_digest(root/name) != value: raise ValueError('Frozen target dependency changed: '+name)
    algorithm = pinned(root, pinned(root, master['jobs'][0])['algorithm_protocol'])
    result = job | {'_runtime_forward_verification': forward}
    # The original inference engine only uses the seal's time and truth value.
    # For this batch it represents the forward seal,including baseline/Uniform.
    return result, master, algorithm, policy, pinned(root, job['forward_bundle'])
