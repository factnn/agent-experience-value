"""Add observed policy/ancestor dependence to the original full15 raw review.

The original case reviewer,raw auditor,certificate and repeat weighting stay
unchanged. This entry is used only after all original15 branches are complete.
"""
import json
from collections import defaultdict

import review_bfcl_stage1_forward15_training as original
from bfcl_source_eval_contract import pinned
from supervise_bfcl_stage_fork import ROOT, read

SELF = 'pipeline/review_bfcl_stage1_complete_policy_context.py'
OUTPUT = 'rl/bfcl_stage1_forward15_training_policy_dependencies_20261016.json'
ORIGINAL_REGISTER = original.register


def with_context(root, source, kind, facts, review, extra_paths, message):
    if review.get('kind') != 'complete_stage1_forward15_training_execution_review':
        raise ValueError('Only the complete original15 training review is adapted')
    master = pinned(root, original.PROTOCOL)
    rows, classes = [], defaultdict(list)
    for case in facts['cases']:
        entry = case['job'];job = pinned(root, entry)
        summary = read(root/job['output']/'summary.json')
        audit = read(root/job['output']/'audit.json')
        fingerprint = summary['final_fingerprint']
        row = {'case': entry['name'], 'source_group': job['source_group'],
               'training_repeat': job['training_repeat'], 'job': entry,
               'summary': original.ref(root, root/job['output']/'summary.json'),
               'audit': original.ref(root, root/job['output']/'audit.json'),
               'policy_fingerprint': fingerprint,
               'frozen_base_sha256': summary['final_manifest']['frozen_base_sha256'],
               'adapter_config_sha256': original.digest(root/job['output']/'adapter/adapter_config.json'),
               'full_learner_payload_sha256': audit['final_payload_sha256'],
               'optimizer_steps': audit['optimizer_steps'],
               'fresh_nonzero_gradient_steps': audit['fresh_nonzero_gradient_steps'],
               'changed_parameter_steps': audit['changed_parameter_steps'],
               'actual_new_model_tokens': audit['cost']['new_model_tokens']}
        rows.append(row);classes[fingerprint].append(entry['name'])
    if len(rows) != 15 or len({row['case'] for row in rows}) != 15:
        raise ValueError('Require all15 original cases for dependency description')
    bases = sorted({row['frozen_base_sha256'] for row in rows})
    context_path = root/'rl/bfcl_stage1_fixed_parent_optimizer_context_20261016.json'
    duplicate_classes = [{'policy_fingerprint': k, 'cases': v} for k, v in classes.items() if len(v)>1]
    description = {'kind': 'complete_forward15_policy_dependence_description',
        'execution_protocol': original.PROTOCOL, 'complete_training_audit': facts['complete_audit'],
        'prespecified_parent': master['parent_binding'], 'common_start': master['stage_start'],
        'fixed_parent_optimizer_context': original.ref(root, context_path),
        'complete_cases': 15, 'planned_training_repeats': 3, 'planned_evaluation_repeats': 2,
        'distinct_trainable_weight_fingerprints': len(classes), 'frozen_base_hashes': bases,
        'identical_weight_case_groups': duplicate_classes, 'rows': rows,
        'zero_fresh_gradient_cases': [row['case'] for row in rows if not row['fresh_nonzero_gradient_steps']],
        'primary_repeat_weights_unchanged': True, 'deduplicated_or_dropped_cases': [],
        'late_target_labels_used': False, 'new_scientific_tokens': 0, 'new_gpu_jobs': 0,
        'scope': 'Post-training descriptive dependence only,not a presampling predictor. Equal trainable weights under a common frozen base must not be counted as independent learned policies;full learner RNG/history/optimizer payloads may still differ. Distinct fingerprints do not prove independence. Keep every planned case and its original primary weight. Shared ancestor,target tasks and evaluation seeds remain dependent.'}
    path = root/OUTPUT
    if path.exists():
        if read(path) != description:raise ValueError('Preserve differing prior dependence evidence')
    else:
        original.exclusive(path, description)
    facts = facts | {'policy_dependency_description': original.ref(root, path),
                     'distinct_trainable_weight_fingerprints': len(classes),
                     'identical_weight_case_groups': duplicate_classes,
                     'fixed_parent_optimizer_context': original.ref(root, context_path),
                     'primary_repeat_weights_unchanged': True}
    review = review | {'evidence': [*review['evidence'], OUTPUT, str(context_path.relative_to(root))],
        'conclusion': review['conclusion']+' 同时逐案列出政策权重FP等价类、完整learner payload差别和0fresh梯度分支；相同权重不冒充独立学得政策，不因此删重复、补seed或改主权重。共同祖先的既有AdamW动量与新数据梯度/参数变化分开解释，仍无本批迁移结论。'}
    return ORIGINAL_REGISTER(root, source, kind, facts, review,
                             [*extra_paths, OUTPUT, SELF], message)


if __name__ == '__main__':
    before = original.register
    original.register = with_context
    try:
        result = original.review_complete(ROOT)
    finally:
        original.register = before
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
