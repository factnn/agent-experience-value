"""Audit completed common-start matrix and report paired G/V without imputations."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from bfcl_token_rollout import ROOT

def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def rate(rows,split):
    selected=[r for r in rows if r['split']==split]
    return sum(sum(r['rewards'])/len(r['rewards']) for r in selected)/len(selected)

def unchanged_policy_consistency(base_rows,branch_rows):
    """Compare already sampled trajectories; no new inference or reward changes."""
    baseline={(r['task_id'],r['rollout_index']):r for r in base_rows}
    fields=['prompt_ids','completion_ids','model_token_mask','segments','bridges',
        'reward','stop_reason','completed_user_turns','total_user_turns']
    differences=[]
    for row in branch_rows:
        key=(row['task_id'],row['rollout_index'])
        if key not in baseline:raise ValueError('evaluation task outside baseline')
        changed=[field for field in fields if row[field]!=baseline[key][field]]
        if changed:differences.append({'task_id':key[0],'rollout_index':key[1],'differing_fields':changed})
    return {'checked_rollouts':len(branch_rows),'baseline_rollouts':len(base_rows),
        'status':'exact_match' if not differences else 'runtime_or_sampling_difference_requires_review',
        'differences':differences,'new_model_calls':0,
        'scope':'Model inference control only; optimizer counters/history can differ despite unchanged parameters.'}

def signal_variation(rows):
    """Flag constants rather than fitting an unidentifiable predictor."""
    result={}
    for field in ['posterior_success','frontier','coverage_deficit','observed_episodes']:
        values=[r['expected_signals_before_sampling'][field] for r in rows]
        span=max(values)-min(values)
        result[field]={'values':values,'range':span,
            'status':'constant_no_predictive_contrast' if span<=1e-12 else 'varies_descriptive_only',
            'predictive_model_fitted':False}
    return result
def audit_sampling(out):
    calls=lines(out/'generation_calls.jsonl');rollouts=lines(out/'rollouts.jsonl')
    key='policy_step' if any('policy_step' in r for r in calls) else 'group'
    charged=sum(len(r['generated_ids']) for r in calls)
    inputs=sum(len(r['input_ids']) for r in calls)
    for row in rollouts:
        group=row[key];i=row['rollout_index']
        actual=[r for r in calls if r[key]==group and r['rollout_index']==i]
        assert len(actual)==len(row['segments'])
        mask=row['model_token_mask'];ids=row['completion_ids']
        assert len(ids)==len(mask) and set(mask)<={0,1}
        for call,segment in zip(actual,row['segments']):
            assert call['generated_ids']==segment['generated_ids'] and call['input_ids']==segment['input_ids']
            start=segment['start'];stop=start+len(segment['generated_ids'])
            assert ids[start:stop]==segment['generated_ids'] and mask[start:stop]==[1]*(stop-start)
            assert call['input_ids']==row['prompt_ids']+ids[:start]
            assert call['generation_allowance']==min(4096,call['remaining_completion_budget'])
        for b in row['bridges']:
            start=b['start'];stop=start+len(b['ids'])
            assert ids[start:stop]==b['ids'] and mask[start:stop]==[0]*(stop-start)
        assert sum(mask)==sum(len(s['generated_ids']) for s in row['segments'])
    return {'new_model_tokens':charged,'input_token_positions_excluding_padding':inputs,
        'external_tokens':sum(len(r['model_token_mask'])-sum(r['model_token_mask']) for r in rollouts),
        'rollouts':len(rollouts),'token_lineage_and_mask':'passed'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=ROOT/'rl/BFCL_INTERVENTION_PROTOCOL_20261008.json')
    args=ap.parse_args();root=args.root
    p=read(args.protocol);common=read(root/'common/summary.json')
    assert common['status']=='complete' and common['optimizer_updates']==0
    cost=audit_sampling(root/'common');assert cost['new_model_tokens']==common['sampled_tokens']
    baseline=read(root/'eval_common/summary.json');assert baseline['status']=='complete'
    base_rows=lines(root/'eval_common/outcomes.jsonl');assert len(base_rows)==len(p['eval_panel'])
    assert [{k:r[k] for k in p['eval_panel'][0]} for r in base_rows]==p['eval_panel']
    baseline_cost=audit_sampling(root/'eval_common');assert baseline_cost['new_model_tokens']==baseline['sampled_tokens']
    result={'status':'audited_complete','protocol':str(args.protocol.resolve().relative_to(ROOT)),
        'common_probe_cost':cost,'common_payload_sha256':common['common_manifest']['payload_sha256'],
        'common_optimizer_updates':0,'common_eval_cost':baseline_cost,
        'common_rates':{s:rate(base_rows,s) for s in ['id_eval','composition_transfer_eval']},
        'rows':[],'limitations':p['inference_limits'],'repeat_count':p.get('conditional_training_repeats',1)}
    result['common_probe_new_tokens_in_this_wave']=0 if 'source_common_path' in p else cost['new_model_tokens']
    result['common_probe_inherited_tokens']=cost['new_model_tokens'] if 'source_common_path' in p else 0
    for index,condition in enumerate(p['conditions']):
        condition_rows=[]
        uniform=lines(root/f'eval_{index}_uniform/outcomes.jsonl')
        assert read(root/f'eval_{index}_uniform/summary.json')['status']=='complete'
        for rule in p['rules']:
            train=root/f'train_{index}_{rule}';evaluation=root/f'eval_{index}_{rule}'
            summary=read(train/'summary.json');restored=read(train/'restored.json')
            assert summary['status']=='complete' and read(evaluation/'summary.json')['status']=='complete'
            assert restored['common_payload_sha256']==result['common_payload_sha256']
            assert restored['inherited_tokens']==common['sampled_tokens']
            assert restored['trainable_sha256']==common['common_manifest']['trainable_sha256']
            allocations=lines(train/'allocation.jsonl');groups=lines(train/'groups.jsonl');updates=lines(train/'updates.jsonl')
            assert len(allocations)==len(groups)==len(updates)==summary['optimizer_steps']
            for a,g in zip(allocations,groups):
                assert a['task_id'] in p['conditions'][condition]
                assert g['task_ids']==[a['task_id']]*4 and a['rewards']==g['rewards']
                assert abs(sum(a['probabilities'].values())-1)<1e-8
                assert set(a['probabilities'])==set(p['conditions'][condition])
                assert a['seed']==p['seeds']['branch']+10000*index+100*a['policy_step']
                assert (len(set(a['rewards']))>1)==any(abs(x)>1e-8 for x in g['advantages'])
            train_cost=audit_sampling(train);assert train_cost['new_model_tokens']==summary['branch_sampled_tokens']
            assert sum(a['sampled_tokens'] for a in allocations)==train_cost['new_model_tokens']
            assert summary['overshoot']<=4*p['algorithm']['whole_completion_budget']
            rows=lines(evaluation/'outcomes.jsonl');assert len(rows)==len(base_rows)
            assert [{k:r[k] for k in p['eval_panel'][0]} for r in rows]==p['eval_panel']
            assert [r['seed'] for r in rows]==[r['seed'] for r in base_rows]
            eval_cost=audit_sampling(evaluation);assert eval_cost['new_model_tokens']==read(evaluation/'summary.json')['sampled_tokens']
            metrics=lines(train/'metrics.jsonl');grad_steps=[r['step'] for r in metrics if r.get('grad_norm',0)>0]
            assert all(math.isfinite(r['grad_norm']) for r in metrics if 'grad_norm' in r)
            item={'condition':condition,'rule':rule,'initial_signals':restored['initial_signals'],
                'initial_probabilities':restored['initial_probabilities'],
                'actual_combo_exposure':dict(Counter(a['combination'] for a in allocations)),
                'actual_task_exposure':dict(Counter(a['task_id'] for a in allocations)),
                'task_sequence':[a['task_id'] for a in allocations],
                'optimizer_steps':summary['optimizer_steps'],'fresh_nonzero_gradient_steps':grad_steps,
                'changed_parameter_steps':[r['step'] for r in updates if r['parameters_changed']],
                'train_cost':train_cost,'eval_cost':eval_cost,'stop_reason':summary['stop_reason'],
                'budget_overshoot':summary['overshoot'],'generation_seconds':summary['generation_seconds'],
                'non_generation_training_seconds':summary['non_generation_training_seconds'],
                'paired_outcomes':[{ 'task_id':r['task_id'],'split':r['split'],
                    'common':b['rewards'],'uniform':u['rewards'],'branch':r['rewards']}
                    for r,b,u in zip(rows,base_rows,uniform)]}
            item['acquisition_decisions']=len(allocations)
            item['largest_group_fraction_of_acquisition_tokens']=max(a['sampled_tokens'] for a in allocations)/summary['branch_sampled_tokens']
            item['group_signal_and_update_diagnostics']=[{
                'task_id':a['task_id'],'combination':a['combination'],
                'prior_combination_signal':a['signals_before'][a['combination']],
                'observed_rewards':g['rewards'],'observed_advantages':g['advantages'],
                'sampled_tokens':a['sampled_tokens'],'stop_reasons':a['stop_reasons'],
                'parameters_changed':u['parameters_changed']}
                for a,g,u in zip(allocations,groups,updates)]
            item['final_parameters_equal_common']=summary['final_fingerprint']==common['policy_fingerprint']
            assert summary['final_manifest']['frozen_base_sha256']==common['common_manifest']['frozen_base_sha256']
            if 'source_common_path' in p:
                from audit_bfcl_parameter_displacement import audit as displacement_audit
                displacement=displacement_audit(root/'common',train,p['algorithm']['lora_alpha']/p['algorithm']['lora_r'])
                (train/'parameter_displacement.json').write_text(json.dumps(displacement,indent=2)+'\n')
                item['endpoint_parameter_displacement']={k:displacement[k] for k in [
                    'trainable_parameter_count','trainable_parameter_delta_l2','trainable_parameter_delta_rms',
                    'effective_weight_delta_frobenius','lora_scaling','interpretation']}
            if item['final_parameters_equal_common']:
                item['unchanged_model_evaluation_control']=unchanged_policy_consistency(
                    lines(root/'eval_common/rollouts.jsonl'),lines(evaluation/'rollouts.jsonl'))
            available_classes=sorted({c for task in p['conditions'][condition]
                for c in p['training_registry'][task].split('+')})
            item['classes_available_in_condition_training_pool']=available_classes
            exposed_classes=sorted({c for a in allocations for c in a['combination'].split('+')})
            item['classes_actually_exposed_in_branch']=exposed_classes
            item['class_exposure_basis']='Available-tool menus in acquired training tasks; not verified execution counts or per-tool gradient attribution.'
            item['heldout_constituent_classes_absent_from_branch_exposure']={stratum:sorted(set(stratum.split('+'))-set(exposed_classes))
                for stratum in sorted({r['stratum'] for r in rows if r['split']=='composition_transfer_eval'})}
            item['heldout_constituent_classes_absent_from_condition_pool']={stratum:sorted(set(stratum.split('+'))-set(available_classes))
                for stratum in sorted({r['stratum'] for r in rows if r['split']=='composition_transfer_eval'})}
            for s in ['id_eval','composition_transfer_eval']:
                item[s]={'rate':rate(rows,s),'G':rate(rows,s)-rate(base_rows,s),'V':rate(rows,s)-rate(uniform,s)}
            if 'evaluation_role' in p['eval_panel'][0]:
                item['evaluation_roles']={}
                for role in sorted({r['evaluation_role'] for r in rows}):
                    subset=[r for r in rows if r['evaluation_role']==role]
                    base_subset=[r for r in base_rows if r['evaluation_role']==role]
                    uniform_subset=[r for r in uniform if r['evaluation_role']==role]
                    item['evaluation_roles'][role]={s:{'tasks':sum(r['split']==s for r in subset),
                        'rate':rate(subset,s),'G':rate(subset,s)-rate(base_subset,s),
                        'V':rate(subset,s)-rate(uniform_subset,s)} for s in ['id_eval','composition_transfer_eval']
                        if any(r['split']==s for r in subset)}
            probabilities=restored['initial_probabilities'];signals=restored['initial_signals']
            combo_probabilities=Counter()
            for task,probability in probabilities.items():combo_probabilities[p['training_registry'][task]]+=probability
            item['initial_combo_probabilities']=dict(combo_probabilities)
            item['expected_signals_before_sampling']={feature:sum(probability*signals[p['training_registry'][task]][feature]
                for task,probability in probabilities.items())
                for feature in ['posterior_success','frontier','coverage_deficit','observed_episodes']}
            item['initial_task_distribution_entropy']= -sum(x*math.log(x) for x in probabilities.values() if x)
            # Matched strata are secondary, not chosen after seeing results.
            matched=[r for r in rows if condition in r['stratum']]
            item['matched_strata_outcomes']=[r['task_id'] for r in matched]
            for s in ['id_eval','composition_transfer_eval'] if matched else []:
                base_matched=[r for r in base_rows if condition in r['stratum']]
                uniform_matched=[r for r in uniform if condition in r['stratum']]
                item['matched_'+s]={'rate':rate(matched,s),
                    'G':rate(matched,s)-rate(base_matched,s),
                    'V':rate(matched,s)-rate(uniform_matched,s)}
            result['rows'].append(item);condition_rows.append(item)
        ref=condition_rows[0]['task_sequence']
        uniform_cost=condition_rows[0]['train_cost']['new_model_tokens']
        for item in condition_rows:
            item['same_task_sequence_as_uniform']=item['task_sequence']==ref
            item['actual_acquisition_cost_ratio_to_uniform']=item['train_cost']['new_model_tokens']/uniform_cost
            item['stopped_at_token_budget']=item['stop_reason']=='raw_token_budget'
    result['cost_interpretation']='Equal declared thresholds with whole-group overshoot are not exact equal realized compute; V is a descriptive paired contrast at reported actual costs, not proof of per-compute allocation superiority.'
    result['transfer_interpretation']='Combinations withheld from the global 87-task research train split; the directed condition pools need not contain every constituent class. This wave is not a strict test that each heldout constituent received parameter updates.'
    result['initial_signal_variation']={'all_conditions':signal_variation(result['rows']),
        'within_condition':{c:signal_variation([r for r in result['rows'] if r['condition']==c]) for c in p['conditions']}}
    result['analysis_addendum']='gpt8 feedback: retain original strategies/budgets; flag constant Coverage deficit, retain recorded task/combo probabilities and exposure, separate estimation granularity from update/transfer efficacy. No post-result replacement predictor or regression.'
    if p.get('conditional_training_repeats',1)>1:
        result['conditional_repeat_scope']='Distinct training/allocation seeds, conditional on one inherited common policy/probe history; not independent common histories or learner stages.'
        result['conditional_repeat_descriptive_summary']={}
        for rule in p['rules']:
            repeated=[r for r in result['rows'] if r['rule']==rule]
            result['conditional_repeat_descriptive_summary'][rule]={}
            for role in repeated[0]['evaluation_roles']:
                result['conditional_repeat_descriptive_summary'][rule][role]={}
                for split in repeated[0]['evaluation_roles'][role]:
                    result['conditional_repeat_descriptive_summary'][rule][role][split]={
                        metric:{'values':(values:=[r['evaluation_roles'][role][split][metric] for r in repeated]),
                            'mean':sum(values)/len(values),'range':max(values)-min(values)}
                        for metric in ['rate','G','V']}
    target=root/'comparison.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'audit':'passed','result':str(target),'common_rates':result['common_rates'],
        'rows':[{k:r[k] for k in ['condition','rule','id_eval','composition_transfer_eval','optimizer_steps','fresh_nonzero_gradient_steps','same_task_sequence_as_uniform']} for r in result['rows']]},ensure_ascii=False))
if __name__=='__main__':main()
