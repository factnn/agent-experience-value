"""Prospective user-approved forward-only15/32 amendment; preserve original job bytes."""
import copy
from bfcl_source_eval_contract import digest, pinned

MODE = 'forward_only_15'
TRAINING_JOBS = 15
TARGET_PANELS = 32
TRAINING_CEILING = 1474560
EVALUATION_CEILING = 8388608
SELF = 'pipeline/bfcl_stage1_forward15_contract.py'
ORDER = 'Publish original early-five and separateH1 actual late forecasts before new stage1 RL acquisition and ALL32 target panels; complete/review15 training branches,then evaluate32 panels. No within-late fitting in this batch.'


def selected_entries(root, full, analysis):
    held = set(analysis['source_split']['heldout'])
    result = []
    for entry in full['jobs']:
        job = pinned(root, entry)
        if job['role'] == 'uniform_control' or job['source_group'] in held:
            result.append(entry)
    if len(result) != 15:
        raise ValueError('Require original four heldout groups x3 plus all3Uniform')
    return result


def evaluation_design(full):
    value = copy.deepcopy(full['evaluation'])
    value.update(policies_including_shared_start=16, total_rollouts=1024,
                 new_token_ceiling=EVALUATION_CEILING, required_order=ORDER,
                 batch_mode=MODE, target_panels=TARGET_PANELS,
                 within_late_prediction_seal_required=False,
                 within_late_fitting_allowed=False,
                 forward_original_five_and_H1_publication_required_for_every_role=True)
    return value


def validate(root, plan):
    if (plan['kind'] != 'source_stage1_training_and_transfer_execution' or plan['stage'] != 1
            or plan['batch_mode'] != MODE or plan['training_jobs'] != 15
            or plan['within_late_fit_deferred'] is not True):
        raise ValueError('Require the separately registered forward-only15/32 stage1 batch')
    full = pinned(root, plan['prior_full_stage1_design'])
    if full['training_jobs'] != 51 or full['stage'] != 1:
        raise ValueError('Wrong original51 prospective design')
    analysis = pinned(root, plan['analysis_contract'])
    expected = selected_entries(root, full, analysis)
    if plan['jobs'] != expected or plan['deferred_jobs'] != [e for e in full['jobs'] if e not in expected]:
        raise ValueError('Original source identities substituted,reranked or dropped without accounting')
    for key in ('stage0_execution_protocol', 'stage1_probe_protocol', 'parent_binding', 'stage_start',
                'analysis_contract', 'output_root', 'allowed_gpus', 'max_simultaneous_gpus', 'resource_authorization'):
        if plan[key] != full[key]:
            raise ValueError('Unchanged lineage/resource field differs: '+key)
    if (plan['automatic_retry'] is not False or plan['training_repeats'] != 3 or plan['source_groups'] != 4
            or plan['nominal_new_training_tokens'] != 983040 or plan['new_training_token_ceiling'] != TRAINING_CEILING
            or plan['evaluation'] != evaluation_design(full)):
        raise ValueError('Changed repeat,seed,target,budget,algorithm or access order')
    expected_lanes = {str(g): [e['name'] for index, e in enumerate(expected) if index % 8 == g] for g in range(8)}
    if plan['gpu_lanes'] != expected_lanes:
        raise ValueError('Prospective forward15 lane schedule changed')
    jobs = [pinned(root, entry) for entry in expected]
    cells = {(job['source_group'], job['training_repeat']) for job in jobs}
    if cells != {(g, repeat) for g in [None, *analysis['source_split']['heldout']] for repeat in range(3)}:
        raise ValueError('Missing original source/control repeat cell')
    for entry, job in zip(expected, jobs):
        if (job['stage'] != 1 or job['source_step'] != 7 or job['case_id'] != entry['name']
                or (job['token_budget'], job['max_updates'], job['soft_seconds'], job['hard_seconds']) != (65536, 128, 12600, 14400)):
            raise ValueError('Original frozen branch changed')
    if sum(job['token_budget']+32768 for job in jobs) != TRAINING_CEILING:
        raise ValueError('Wrong finite training ceiling')
    expected_budgets = {'late_probe_new_token_ceiling_separate': 5701632,
                        'training_new_token_ceiling': TRAINING_CEILING,
                        'forward32_evaluation_new_token_ceiling': EVALUATION_CEILING,
                        'post_probe_training_and_evaluation_ceiling': 9863168,
                        'stage1_including_original_probe_new_token_ceiling': 15564800,
                        'ancestor_tokens_recharged': 0, 'paid_api_calls': 0}
    if plan['budgets'] != expected_budgets:
        raise ValueError('Forward-only batch resource ceiling changed')
    if plan['input_sha256'] != full['input_sha256'] | {SELF: digest(root/SELF)}:
        raise ValueError('Original runtime inputs changed or forward15 contract unpinned')
    authorization = pinned(root, plan['scope_authorization'])
    if authorization['human_choice'] != 'forward15_32_now_full_matrix_deferred' or authorization['new_late_training_or_target_started_at_decision']:
        raise ValueError('Wrong prospective user scope decision')
    return jobs
