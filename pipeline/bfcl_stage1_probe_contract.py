"""Bind late probes to the fixed, nonzero ancestor and its actual history.

Design inputs alone never constitute an executable protocol. Publication and
GPU/Goal checks additionally belong to the forthcoming stage1 supervisor.
"""
import json
from bfcl_source_eval_contract import digest, pinned
from bfcl_allocation import CombinationAllocator
from bfcl_source_group_contract import validate_source


def validate_history(design, binding, manifest, payload, algorithm):
    """Replay CPU state; never restore or advance its optimizer/RNG/history."""
    from rl_learning_state import tensor_hash
    if payload['progress'] != manifest['progress']:
        raise ValueError('Payload/manifest progress mismatch')
    step = validate_source(algorithm, algorithm, payload['progress'], payload['optimizer'], False)
    if step <= 0 or step != binding['source_step'] or binding['micro_step'] != 4*step:
        raise ValueError('Late ancestor must be the bound nonzero optimizer boundary')
    states = payload['optimizer']['state']
    if payload['optimizer_type'] != 'AdamW' or not states or any(int(v['step']) != step for v in states.values()):
        raise ValueError('Late ancestor AdamW state missing or inconsistent')
    if tensor_hash(payload['trainable'].items()) != manifest['trainable_sha256']:
        raise ValueError('Late ancestor parameter payload mismatch')
    if payload['provenance'] != manifest['provenance']:
        raise ValueError('Late ancestor provenance mismatch')
    allocator = CombinationAllocator.from_state_dict(payload['allocator'])
    if allocator.total_sampled_tokens != binding['cumulative_acquisition_tokens']:
        raise ValueError('Late acquisition lineage mismatch')
    if manifest['cumulative_acquisition_tokens'] != allocator.total_sampled_tokens:
        raise ValueError('Manifest acquisition lineage mismatch')
    signals = allocator.signals()
    cases = design['cases']
    if len(cases) != 16 or len({c['name'] for c in cases}) != 16 or {c['source_group'] for c in cases} != set(signals):
        raise ValueError('Require all sixteen distinct source groups')
    all_tasks, seeds = [], []
    for c in cases:
        tasks = c['task_ids']; group = c['source_group']
        if not tasks or len(set(tasks)) != len(tasks) or set(tasks) != {t for t,g in allocator.registry.items() if g == group}:
            raise ValueError('Require the entire original source group')
        if c['historical_signal'] != signals[group]:
            raise ValueError('Probe history is not the fixed late ancestor history')
        if c['token_ceiling'] != len(tasks)*8*8192 or c['hard_seconds'] != 600+4800*len(tasks):
            raise ValueError('Probe dose or runtime boundary changed')
        for phase in ('feature', 'validation'):
            if len(c['seeds'][phase]) != len(tasks):
                raise ValueError('Missing task seeds')
            seeds.extend(c['seeds'][phase])
        all_tasks.extend(tasks)
    if len(all_tasks) != 87 or len(set(all_tasks)) != 87 or set(all_tasks) != set(allocator.registry):
        raise ValueError('Probe task coverage changed')
    if len(set(seeds)) != 174 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Invalid or overlapping probe seeds')
    return signals


def load_design(root, reference):
    import torch
    design = pinned(root, reference)
    if design['kind'] != 'late_stage_probe_design_inputs_not_execution_protocol' or design['stage'] != 1:
        raise ValueError('Expected the separately frozen stage1 design inputs')
    binding = pinned(root, design['parent_binding'])
    master = pinned(root, binding['stage0_execution_protocol'])
    job = pinned(root, binding['parent_training_job'])
    if binding['parent_case'] != master['prespecified_late_stage_parent'] or job['case_id'] != binding['parent_case']:
        raise ValueError('Late ancestor was not prespecified')
    if job['role'] != 'uniform_control' or job['training_repeat'] != 0 or job['stage'] != 0:
        raise ValueError('Late ancestor must be the fixed stage0 Uniform repeat0')
    if binding['selection_uses_target_labels'] is not False:
        raise ValueError('Target-based ancestor selection forbidden')
    for key in ('summary', 'audit', 'learning_state', 'manifest', 'adapter_config', 'adapter_weights'):
        ref = binding[key]
        if digest(root/ref['path']) != ref['sha256']:
            raise ValueError('Changed ancestor artifact: '+key)
    expected_paths = {'summary':'summary.json', 'audit':'audit.json', 'learning_state':'components/learning_state.pt',
                      'manifest':'components/manifest.json', 'adapter_config':'adapter/adapter_config.json',
                      'adapter_weights':'adapter/adapter_model.safetensors'}
    if binding['policy_path'] != job['output'] or design['policy_path'] != job['output']:
        raise ValueError('Ancestor policy path mismatch')
    for key, suffix in expected_paths.items():
        if binding[key]['path'] != job['output']+'/'+suffix:
            raise ValueError('Ancestor artifact belongs to another job')
    manifest = pinned(root, binding['manifest']); summary = pinned(root, binding['summary']); audit = pinned(root, binding['audit'])
    if summary['status'] != 'complete' or audit['status'] != 'passed' or audit['job_sha256'] != binding['parent_training_job']['sha256']:
        raise ValueError('Ancestor is not completely audited')
    if manifest['payload_sha256'] != binding['learning_state']['sha256'] or audit['final_payload_sha256'] != manifest['payload_sha256']:
        raise ValueError('Ancestor payload binding mismatch')
    if design['policy_fingerprint'] != binding['policy_fingerprint'] or summary['final_fingerprint'] != binding['policy_fingerprint']:
        raise ValueError('Ancestor policy fingerprint mismatch')
    for key in ('source_step', 'micro_step'):
        if design[key] != binding[key]:
            raise ValueError('Stage position mismatch')
    if design['inherited_acquisition_tokens'] != binding['cumulative_acquisition_tokens']:
        raise ValueError('Design inherited cost mismatch')
    source = pinned(root, job['algorithm_protocol'])
    if manifest['provenance']['intervention_protocol_sha256'] != job['algorithm_protocol']['sha256']:
        raise ValueError('Ancestor algorithm mismatch')
    payload = torch.load(root/binding['learning_state']['path'], map_location='cpu', weights_only=False)
    validate_history(design, binding, manifest, payload, source['algorithm'])
    return design, binding, job['algorithm_protocol']


def validate_execution(plan, design, binding, algorithm_ref, report, review):
    if plan['kind'] != 'stage1_fixed_policy_probe_execution' or type(plan['stage']) is not int or plan['stage'] != 1:
        raise ValueError('A new stage1 execution protocol is required')
    for key in ('cases', 'policy_path', 'policy_fingerprint', 'inherited_acquisition_tokens'):
        if plan[key] != design[key]:
            raise ValueError('Execution differs from frozen late design: '+key)
    if plan['parent_binding'] != design['parent_binding'] or plan['source_protocol'] != algorithm_ref:
        raise ValueError('Late provenance or sampling recipe changed')
    if report['status'] != 'passed' or report['stage'] != 0 or report['labels']['panel_count'] != 104:
        raise ValueError('Complete stage0 transfer report required before late sampling')
    if report['evidence']['execution_protocol'] != binding['stage0_execution_protocol']:
        raise ValueError('Stage0 report belongs to a different matrix')
    if review['status'] != 'reviewed' or plan['stage0_transfer_report']['path'] not in review['evidence']:
        raise ValueError('Complete stage0 report must be reviewed first')
    if plan['automatic_retry'] is not False or plan['max_simultaneous_gpus'] != 4 or plan['allowed_gpus'] != [4,5,6,7]:
        raise ValueError('Execution resource/retry contract changed')


def load_protocol(root, path, expected_sha):
    plan = pinned(root, {'path':str(path), 'sha256':expected_sha})
    # Fail closed on the design-input file itself, before requiring other keys.
    if plan.get('kind') != 'stage1_fixed_policy_probe_execution':
        raise ValueError('Design inputs are not an execution protocol')
    design, binding, algorithm_ref = load_design(root, plan['design_inputs'])
    report = pinned(root, plan['stage0_transfer_report'])
    review = pinned(root, plan['stage0_transfer_review'])
    validate_execution(plan, design, binding, algorithm_ref, report, review)
    for relative, sha in plan['input_sha256'].items():
        if digest(root/relative) != sha:
            raise ValueError('Changed execution input: '+relative)
    return plan, binding
