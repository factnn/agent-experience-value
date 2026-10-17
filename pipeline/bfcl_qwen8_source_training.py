"""Native8B bindings for future registered source RL; no cohort admission here."""
from contextlib import contextmanager
from pathlib import Path

from bfcl_source_eval_contract import digest, pinned
from bfcl_source_group_contract import validate_intervention
from bfcl_qwen8_initial_probe import REVISION, verify_model

SELF = 'pipeline/bfcl_qwen8_source_training.py'
TRAINER = 'pipeline/run_bfcl_qwen8_source_training.py'
AUDITOR = 'pipeline/report_bfcl_qwen8_source_training.py'
REQUIRED = {SELF, TRAINER, AUDITOR, 'pipeline/train_bfcl_source_group.py',
            'pipeline/report_bfcl_source_training.py', 'pipeline/bfcl_source_group_contract.py',
            'pipeline/run_bfcl_intervention.py', 'pipeline/rl_learning_state.py',
            'pipeline/bfcl_qwen8_initial_probe.py', 'pipeline/bfcl_source_eval_contract.py'}


def identity(reference):
    return {'path': reference['path'], 'sha256': reference['sha256']}


def load_job(root, path, expected_sha, execution_ref):
    if digest(path) != expected_sha:
        raise ValueError('Native8B training job changed')
    relative = str(path.resolve().relative_to(root.resolve()))
    job = pinned(root, {'path': relative, 'sha256': expected_sha})
    master = pinned(root, execution_ref)
    if (master['kind'] != 'qwen3_8b_stage0_forward_training_and_transfer'
            or master['model_revision'] != REVISION or master['stage'] != 0
            or master['automatic_retry'] is not False or master['reserved21_allowed'] is not False):
        raise ValueError('A separately frozen native8B RL execution protocol is required')
    if identity(job['native_cohort_design']) != identity(master['design']):
        raise ValueError('Native8B branch belongs to another scientific design')
    design = pinned(root, master['design'])
    if (design['kind'] != 'qwen3_8b_stage0_source_forward_design'
            or design['model_revision'] != REVISION or design['stage'] != 0):
        raise ValueError('Wrong native8B scientific design')
    for field in ['source_groups', 'training_repeats', 'training']:
        if master[field] != design[field]:
            raise ValueError('Execution differs from frozen native8B design: '+field)
    matches = [e for e in master['jobs']
               if e['path'] == relative and e['sha256'] == expected_sha and e['name'] == job['case_id']]
    if len(matches) != 1:
        raise ValueError('Job is not a unique branch of the registered native8B cohort')
    if (job['stage'] != 0 or job['source_step'] != 0
            or job['allow_legacy_zero_optimizer_source'] is not False
            or identity(job['native_common_binding']) != identity(master['native_common_binding'])
            or identity(job['native_model_verification']) != identity(master['model_verification'])
            or job['native_model_revision'] != REVISION):
        raise ValueError('Wrong model-specific step0 branch identity')
    common = pinned(root, job['native_common_binding'])
    if (common['model_revision'] != REVISION or job['source_checkpoint'] != common['common_policy']
            or job['source_payload_sha256'] != common['learning_state']['sha256']):
        raise ValueError('A4B or other ancestor cannot be substituted for the native8B common state')
    common_audit = pinned(root, common['common_audit'])
    manifest = pinned(root, common['manifest'])
    if (common_audit['status'] != 'passed' or common_audit['common_optimizer_updates'] != 0
            or common_audit['policy_fingerprint'] != common['policy_fingerprint']
            or manifest['progress']['global_step'] != 0 or manifest['progress']['micro_step'] != 0
            or manifest['provenance']['base_revision'] != REVISION
            or manifest['payload_sha256'] != job['source_payload_sha256']):
        raise ValueError('Native8B common state is incomplete or has the wrong learner position')
    for name in ['learning_state', 'manifest', 'adapter_config', 'adapter_weights']:
        if digest(root/common[name]['path']) != common[name]['sha256']:
            raise ValueError('Native8B common artifact changed: '+name)
    source = pinned(root, job['source_protocol'])
    target = pinned(root, job['algorithm_protocol'])
    if identity(job['algorithm_protocol']) != identity(master['algorithm_protocol']):
        raise ValueError('Branch uses another target algorithm protocol')
    initial = pinned(root, common['master'])
    if identity(job['source_protocol']) != identity(initial['source_protocol']):
        raise ValueError('Wrong native8B source algorithm provenance')
    if source['base_revision'] != REVISION or target['base_revision'] != REVISION:
        raise ValueError('Only the declared native8B revision is permitted')
    if (source['algorithm'] != target['algorithm']
            or source['training_registry'] != target['training_registry']
            or source['manifest_sha256'] != target['manifest_sha256']
            or source['data_sha256'] != target['data_sha256']
            or source['seeds']['initialization'] != target['seeds']['initialization']):
        raise ValueError('Undeclared algorithm, task, or initialization transition')
    validate_intervention(job, target['training_registry'])
    if job['role'] not in ('source_group', 'uniform_control'):
        raise ValueError('Only declared source and full-poolUniform branches are allowed')
    if job['role'] == 'source_group' and job['source_group'] not in master['source_groups']:
        raise ValueError('Source is outside the separately registered native8B cohort')
    for field in ['token_budget', 'max_updates', 'soft_seconds', 'hard_seconds']:
        if job[field] != master['training'][field]:
            raise ValueError('Native8B registered training budget differs: '+field)
    if type(job['training_repeat']) is not int or not 0 <= job['training_repeat'] < master['training_repeats']:
        raise ValueError('Training repeat outside the registered native8B cohort')
    if not REQUIRED <= job['input_sha256'].keys():
        raise ValueError('Native8B job omitted its model-specific runner or original scientific dependencies')
    for name, sha in job['input_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Native8B training input changed: '+name)
    return job, master, common, manifest


@contextmanager
def native_factories(runtime, model_path, metadata):
    """Redirect exactly the legacy smoke/model factories in the current process."""
    tokenizer_factory, model_factory, original_write = runtime.AutoTokenizer, runtime.AutoModelForCausalLM, runtime.write
    expected = runtime.ROOT/'smoke/model'
    class NativeTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            if Path(path) != expected:
                raise ValueError('Unexpected tokenizer factory path')
            return tokenizer_factory.from_pretrained(model_path, **kwargs)
    class NativeModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            if Path(path) != expected:
                raise ValueError('Unexpected model factory path')
            return model_factory.from_pretrained(model_path, **kwargs)
    def annotated_write(path, value):
        if path.name == 'run.json':
            value = value | metadata | {'native_model_revision': REVISION,
                'native_model_path': str(model_path), 'target_evaluation': False,
                'reserved21_accessed': False,
                'model_factory_override': 'Process-local native8B factories; original scientific source trainer unchanged'}
        return original_write(path, value)
    runtime.AutoTokenizer, runtime.AutoModelForCausalLM, runtime.write = NativeTokenizer, NativeModel, annotated_write
    try:
        yield
    finally:
        runtime.AutoTokenizer, runtime.AutoModelForCausalLM, runtime.write = tokenizer_factory, model_factory, original_write


def model_path(master):
    return verify_model(master)
