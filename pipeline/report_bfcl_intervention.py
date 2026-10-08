"""Audit completed common-start matrix and report paired G/V without imputations."""
import argparse
from collections import Counter
import json
from pathlib import Path
from bfcl_token_rollout import ROOT

def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def rate(rows,split):
    selected=[r for r in rows if r['split']==split]
    return sum(sum(r['rewards'])/len(r['rewards']) for r in selected)/len(selected)
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
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    p=read(ROOT/'rl/BFCL_INTERVENTION_PROTOCOL_20261008.json');common=read(root/'common/summary.json')
    assert common['status']=='complete' and common['optimizer_updates']==0
    cost=audit_sampling(root/'common');assert cost['new_model_tokens']==common['sampled_tokens']
    baseline=read(root/'eval_common/summary.json');assert baseline['status']=='complete'
    base_rows=lines(root/'eval_common/outcomes.jsonl');assert len(base_rows)==len(p['eval_panel'])
    assert [{k:r[k] for k in ['task_id','split','stratum']} for r in base_rows]==p['eval_panel']
    baseline_cost=audit_sampling(root/'eval_common');assert baseline_cost['new_model_tokens']==baseline['sampled_tokens']
    result={'status':'audited_complete','protocol':'rl/BFCL_INTERVENTION_PROTOCOL_20261008.json',
        'common_probe_cost':cost,'common_payload_sha256':common['common_manifest']['payload_sha256'],
        'common_optimizer_updates':0,'common_eval_cost':baseline_cost,
        'common_rates':{s:rate(base_rows,s) for s in ['id_eval','composition_transfer_eval']},
        'rows':[],'limitations':p['inference_limits'],'repeat_count':1}
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
            assert [{k:r[k] for k in ['task_id','split','stratum']} for r in rows]==p['eval_panel']
            assert [r['seed'] for r in rows]==[r['seed'] for r in base_rows]
            eval_cost=audit_sampling(evaluation);assert eval_cost['new_model_tokens']==read(evaluation/'summary.json')['sampled_tokens']
            metrics=lines(train/'metrics.jsonl');grad_steps=[r['step'] for r in metrics if r.get('grad_norm',0)>0]
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
            for s in ['id_eval','composition_transfer_eval']:
                item[s]={'rate':rate(rows,s),'G':rate(rows,s)-rate(base_rows,s),'V':rate(rows,s)-rate(uniform,s)}
            # Matched strata are secondary, not chosen after seeing results.
            matched=[r for r in rows if condition in r['stratum']]
            item['matched_strata_outcomes']=[r['task_id'] for r in matched]
            result['rows'].append(item);condition_rows.append(item)
        ref=condition_rows[0]['task_sequence']
        for item in condition_rows:item['same_task_sequence_as_uniform']=item['task_sequence']==ref
    target=root/'comparison.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'audit':'passed','result':str(target),'common_rates':result['common_rates'],
        'rows':[{k:r[k] for k in ['condition','rule','id_eval','composition_transfer_eval','optimizer_steps','fresh_nonzero_gradient_steps','same_task_sequence_as_uniform']} for r in result['rows']]},ensure_ascii=False))
if __name__=='__main__':main()
