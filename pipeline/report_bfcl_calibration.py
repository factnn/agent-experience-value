"""Audit frozen fixed-policy panels and report evidence without task selection."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from bfcl_token_rollout import ROOT,development_tasks


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def report(path):
    run=json.loads((path/'run.json').read_text());summary=json.loads((path/'summary.json').read_text())
    protocol=run['protocol'];tasks,_=development_tasks(protocol['task_ids'])
    groups=rows(path/'groups.jsonl');rollouts=rows(path/'rollouts.jsonl');calls=rows(path/'generation_calls.jsonl')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True)
    assert summary['optimizer_updates']==0 and summary['initial_weights_sha256']==summary['final_weights_sha256']==run['initial_weights_sha256']
    assert summary['groups']==len(groups)
    schedule=[(key,mode) for key in protocol['task_ids'] for mode in protocol['modes']]
    assert [(g['task_id'],g['mode']) for g in groups]==schedule[:len(groups)]
    assert [g['group'] for g in groups]==list(range(len(groups)))
    assert len(rollouts)==len(groups)*protocol['rollouts_per_group']
    assert sum(len(c['generated_ids']) for c in calls)==summary['sampled_tokens']
    by_sample=defaultdict(list)
    for c in calls:
        assert c['generation_allowance']==min(protocol['segment_cap'],c['remaining_completion_budget'])
        assert len(c['generated_ids'])<=c['generation_allowance']
        by_sample[(c['group'],c['rollout_index'])].append(c)
    for r in rollouts:
        g=groups[r['group']];assert r['task_id']==g['task_id'] and r['mode']==g['mode']
        assert r['seed']==g['seed']==protocol['seed']+100*protocol['task_ids'].index(r['task_id'])
        assert r['weights_sha256']==run['initial_weights_sha256']
        ids=r['completion_ids'];mask=r['model_token_mask'];full=r['prompt_ids']+ids
        assert len(ids)==len(mask)<=protocol['whole_completion_budget'] and set(mask)<={0,1}
        assert r['enable_thinking']==(r['mode']=='thinking')
        expected='<|im_start|>assistant\n'+('' if r['enable_thinking'] else '<think>\n\n</think>\n\n')
        assert tokenizer.decode(r['prompt_ids']).endswith(expected)
        partition=[];sample_calls=by_sample[(r['group'],r['rollout_index'])]
        assert len(sample_calls)==len(r['segments'])
        for segment,call in zip(r['segments'],sample_calls):
            start=segment['start'];n=len(segment['generated_ids'])
            assert call['input_ids']==segment['input_ids']==full[:len(r['prompt_ids'])+start]
            assert call['generated_ids']==segment['generated_ids']==ids[start:start+n]
            assert mask[start:start+n]==[1]*n;partition.extend(range(start,start+n))
        for bridge in r['bridges']:
            start=bridge['start'];n=len(bridge['ids'])
            assert ids[start:start+n]==bridge['ids'] and mask[start:start+n]==[0]*n
            assert tokenizer.decode(bridge['ids']).endswith(expected)
            partition.extend(range(start,start+n))
        assert sorted(partition)==list(range(len(ids)))
        assert len(r['grades'])==r['completed_user_turns']
        assert r['reward']==float(r['completed_user_turns']==r['total_user_turns'] and all(g['valid'] for g in r['grades']))
        following=[m for b in r['bridges'] for m in b['messages'] if m['role']=='user']
        expected_users=[m for turn in tasks[r['task_id']]['question'][1:] for m in turn]
        assert following==expected_users[:len(following)]
    for g in groups:
        rs=[r for r in rollouts if r['group']==g['group']]
        assert [r['reward'] for r in rs]==g['rewards']
        assert sum(sum(r['model_token_mask']) for r in rs)==g['sampled_tokens']
        assert g['mixed_rewards']==(len(set(g['rewards']))>1)
        assert g['censored']==any(r['stop_reason'].startswith('global_') for r in rs)
    assert sum(g['sampled_tokens'] for g in groups)==summary['sampled_tokens']
    assert summary['sampled_tokens']<=protocol['raw_token_budget']+protocol['rollouts_per_group']*protocol['segment_cap']
    paired=[key for key in protocol['task_ids'] if all(any(g['task_id']==key and g['mode']==mode and not g['censored'] for g in groups) for mode in protocol['modes'])]
    panels={}
    for mode in protocol['modes']:
        gs=[g for g in groups if g['mode']==mode];rs=[r for r in rollouts if r['mode']==mode]
        uncensored=[r for r in rs if not r['stop_reason'].startswith('global_')]
        paired_rs=[r for r in rs if r['task_id'] in paired]
        failure_counts=Counter(g['check'].get('error_type','unknown') for r in rs for g in r['grades'] if not g['valid'])
        panels[mode]={'attempted_groups':len(gs),'uncensored_groups':sum(not g['censored'] for g in gs),
            'attempted_rollouts':len(rs),'globally_censored_rollouts':len(rs)-len(uncensored),
            'successes_uncensored':sum(r['reward'] for r in uncensored),'uncensored_rollouts':len(uncensored),
            'paired_task_successes':sum(r['reward'] for r in paired_rs),'paired_task_rollouts':len(paired_rs),
            'mixed_reward_task_ids':[g['task_id'] for g in gs if not g['censored'] and g['mixed_rewards']],
            'complete_user_conversations':sum(r['completed_user_turns']==r['total_user_turns'] for r in rs),
            'valid_user_turns':sum(g['valid'] for r in rs for g in r['grades']),
            'completed_user_turns':sum(r['completed_user_turns'] for r in rs),
            'stop_reasons':dict(Counter(r['stop_reason'] for r in rs)),
            'failure_types':dict(failure_counts),
            'tool_error_responses':sum('error' in e['response'].lower() for r in rs for e in r['events']),
            'sampled_tokens':sum(g['sampled_tokens'] for g in gs),
            'external_tokens':sum(len(r['model_token_mask'])-sum(r['model_token_mask']) for r in rs),
            'generation_seconds':sum(g['generation_seconds'] for g in gs)}
    result={'audit':'passed','status':summary['status'],'stop_reason':summary['stop_reason'],
        'optimizer_updates':0,'weights_unchanged':True,'thinking_prefixes_verified':True,
        'planned_task_ids':protocol['task_ids'],'paired_uncensored_task_ids':paired,
        'observed_groups':groups,'panels':panels,'total_sampled_tokens':summary['sampled_tokens'],
        'wall_seconds':summary['wall_seconds'],'single_gpu_main_wall_hours':summary['wall_seconds']/3600,
        'source_files_still_match':all(hashlib.sha256((ROOT/'pipeline'/name).read_bytes()).hexdigest()==digest for name,digest in run['source_sha256'].items()),
        'interpretation':'Complete mixed binary-reward groups indicate a potential GRPO learning signal at this fixed base/config; no update or intervention gain was measured. Preserve full scientific task pool, not successful-case selection. Global censoring is reported separately.'}
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    r=report(args.run);print(json.dumps({k:v for k,v in r.items() if k!='observed_groups'}))
