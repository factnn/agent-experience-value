"""Scientific identity of the finite native8B cross-size forward15/32 study."""
import copy

from bfcl_source_eval_contract import digest, pinned
from bfcl_qwen8_initial_probe import REVISION

SELF='pipeline/bfcl_qwen8_forward_design.py'
KIND='qwen3_8b_stage0_source_forward_design'
MODE='native8b_forward15_no_refit'
ORDER='Complete original16 native probes; publish actual128 cross-size forecasts and auxiliary32 forecasts or bounded-failure disposition; audit/review all15 native RL endpoints and terminal processes; only then admit all32 paired targets including common andUniform.'
TRAINING={'token_budget':65536,'max_updates':128,'soft_seconds':12600,'hard_seconds':14400}
BUDGETS={'nominal_training_new_tokens':983040,'training_new_token_ceiling':1474560,
    'evaluation_new_token_ceiling':8388608,'new_training_and_evaluation_token_ceiling':9863168,
    'maximum_training_worker_seconds':216000,'maximum_evaluation_worker_seconds':1382400,
    'maximum_post_probe_worker_seconds':1598400,'existing_native_common_and_probe_token_ceiling':5963776,
    'existing_native_common_and_probe_worker_seconds':432600,'existing_E368_measurement_worker_seconds':115200,
    'route_new_token_ceiling_excluding_separate_capacity_case':15826944,
    'route_worker_second_ceiling_excluding_separate_capacity_case':2146200,
    'ancestor_tokens_recharged':0,'paid_api_calls':0}


def zero_reference(groups):
    return {'kind':'unfitted_zero_gain_reference','intercept':0.0,'coefficient':0.0,
            'feature_scaling':None,'fit_labels_used':False,
            'actual_point_predictions':{g:0.0 for g in groups},
            'publication_required_before_every_target':True,
            'scope':'A fixed no-gain error reference, not a fitted mechanism candidate or a policy effect claim.'}


def evaluation_design(original):
    result=copy.deepcopy(original)
    result.update(policies_including_shared_start=16,total_rollouts=1024,new_token_ceiling=8388608,
        hard_seconds_per_panel=43200,target_panels=32,batch_mode=MODE,required_order=ORDER,
        within_native8b_fitting_allowed=False,
        cross_size_and_auxiliary_disposition_publication_required_for_every_role=True)
    return result


def selected(root,original,analysis):
    result=[]
    for entry in original['jobs']:
        job=pinned(root,entry)
        if job['role']=='uniform_control' or job['source_group'] in analysis['source_split']['heldout']:
            result.append((entry,job))
    if len(result)!=15:
        raise ValueError('Require the original four heldout sources x3 plus all3Uniform')
    return result


def blueprints(root,original,analysis):
    return [{'case_id':'q8_'+job['case_id'],'corresponding_early4b_job':entry,
             **{k:copy.deepcopy(job[k]) for k in ['role','source_group','training_repeat','prediction_split',
                  'task_pool','allocator_seed','rollout_seed_base']},
             'stage':0,'source_step':0,'allow_legacy_zero_optimizer_source':False,**TRAINING}
            for entry,job in selected(root,original,analysis)]


def validate(root,design):
    if (design['kind']!=KIND or design['stage']!=0 or design['model_revision']!=REVISION
            or design['batch_mode']!=MODE or design['training_jobs']!=15 or design['target_panels']!=32
            or design['source_group_count']!=4 or design['training_repeats']!=3
            or design['automatic_retry'] is not False or design['reserved21_allowed'] is not False
            or design['within_native8b_fitting_allowed'] is not False
            or design['allowed_gpus']!=list(range(8)) or design['max_simultaneous_gpus']!=8):
        raise ValueError('Wrong prospective native8B forward15/32 scientific design')
    analysis=pinned(root,design['analysis_contract'])
    original=pinned(root,design['original_early4b_execution_protocol'])
    initial=pinned(root,design['initial_native8b_execution_protocol'])
    common=pinned(root,design['native_common_binding'])
    if (design['source_groups']!=analysis['source_split']['heldout']
            or design['all_presampling_source_groups']!=analysis['source_split']['all_groups']
            or design['deferred_within8b_fit_source_groups']!=analysis['source_split']['fit']):
        raise ValueError('Do not select sources from partial8B results or silently restore all51')
    if design['training']!=TRAINING or design['budgets']!=BUDGETS:
        raise ValueError('Registered finite dose, timing or shared-cost accounting changed')
    if design['training_job_blueprints']!=blueprints(root,original,analysis):
        raise ValueError('Original source identities,pools,repeats or paired seeds changed')
    if design['evaluation']!=evaluation_design(original['evaluation']):
        raise ValueError('Original32 target identities/strata/seeds or frozen sampling recipe changed')
    if (design['zero_gain_reference']!=zero_reference(analysis['source_split']['all_groups'])
            or any(type(v) is not float for v in design['zero_gain_reference']['actual_point_predictions'].values())):
        raise ValueError('The prospective zero-gain reference must retain all16 exact constant points')
    if (design['model_verification']!=initial['model_verification']
            or design['resource_authorization']!=initial['resource_authorization']
            or design['current_ssh_owner']!=initial['current_ssh_owner']
            or common['master']!=design['initial_native8b_execution_protocol']
            or common['model_revision']!=REVISION):
        raise ValueError('Wrong own-model parent, cache, current owner or authorized resource cap')
    audit=pinned(root,common['common_audit'])
    if audit['status']!='passed' or audit['common_optimizer_updates']!=0:
        raise ValueError('Require the actual audited native8B step0 common state')
    cap=pinned(root,design['resource_authorization'])
    if cap['max_simultaneous_gpus_for_subsequent_batches']!=8:
        raise ValueError('Wrong current eight-free-card resource authorization')
    source=pinned(root,initial['source_protocol']);target=pinned(root,design['algorithm_protocol'])
    for name in ['algorithm','base_revision','training_registry','manifest_sha256','data_sha256','record_sha256']:
        if source[name]!=target[name]:
            raise ValueError('Scientific learner/task recipe changed: '+name)
    if (target['branch_raw_token_budget']!=65536 or target['branch_max_updates']!=128
            or target['branch_soft_wall_seconds']!=12600
            or target['seeds']['initialization']!=source['seeds']['initialization']):
        raise ValueError('Wrong declared native8B branch budget or own initialization seed')
    core=pinned(root,design['prediction_protocols']['core128'])
    auxiliary=pinned(root,design['prediction_protocols']['auxiliary32_design'])
    core_plan=pinned(root,core['probe_plan'])
    if (core_plan['common_binding']!=design['native_common_binding']
            or core['analysis_contract']!=design['analysis_contract']
            or auxiliary['profiles']['qwen8b']['native_common_binding']!=design['native_common_binding']
            or auxiliary['fit_rule']['analysis_contract']!=design['analysis_contract']):
        raise ValueError('Frozen forecast blocks belong to another native parent or source split')
    if design['implementation_sha256'].get(SELF)!=digest(root/SELF):
        raise ValueError('Scientific design validator was not frozen')
    for name,sha in design['implementation_sha256'].items():
        if digest(root/name)!=sha:
            raise ValueError('Frozen scientific implementation changed: '+name)
    return design
