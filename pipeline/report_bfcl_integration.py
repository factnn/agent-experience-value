"""Audit raw-token lineage, user boundaries, loss masks and learning signal."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from bfcl_token_rollout import development_tasks


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def audit(path):
    summary=json.loads((path/'summary.json').read_text());run=json.loads((path/'run.json').read_text())
    tasks,_=development_tasks(run['task_ids'])
    groups=rows(path/'groups.jsonl');rollouts=rows(path/'rollouts.jsonl')
    updates=rows(path/'updates.jsonl');calls=rows(path/'generation_calls.jsonl')
    assert len(groups)==len(updates)==summary['global_step']==2
    assert len(rollouts)==4*len(groups)
    assert sum(len(c['generated_ids']) for c in calls)==summary['sampled_tokens']
    if len(updates)>1:assert updates[1]['before_sha256']==updates[0]['after_sha256']
    assert summary['final_fingerprint']==updates[-1]['after_sha256']
    for r in rollouts:
        ids=r['completion_ids'];mask=r['model_token_mask'];full=r['prompt_ids']+ids
        assert len(ids)==len(mask)<=run['whole_completion_budget'] and set(mask)<={0,1}
        assert r['policy_fingerprint']==updates[r['policy_step']]['before_sha256']
        partition=[]
        for segment in r['segments']:
            start=segment['start'];generated=segment['generated_ids'];n=len(generated)
            assert segment['input_ids']==full[:len(r['prompt_ids'])+start]
            assert ids[start:start+n]==generated and mask[start:start+n]==[1]*n
            assert any(c['policy_step']==r['policy_step'] and c['input_ids']==segment['input_ids']
                and c['generated_ids']==generated for c in calls)
            partition.extend(range(start,start+n))
        for bridge in r['bridges']:
            start=bridge['start'];n=len(bridge['ids'])
            assert ids[start:start+n]==bridge['ids'] and mask[start:start+n]==[0]*n
            partition.extend(range(start,start+n))
        assert sorted(partition)==list(range(len(ids)))
        assert len(r['grades'])==r['completed_user_turns']
        assert r['reward']==float(r['completed_user_turns']==r['total_user_turns'] and all(g['valid'] for g in r['grades']))
        following=[m for b in r['bridges'] for m in b['messages'] if m['role']=='user']
        expected=[m for turn in tasks[r['task_id']]['question'][1:len(following)+1] for m in turn]
        # These two bounded cases have exactly one message per user turn.
        assert following==expected
    for g in groups:
        rs=[r for r in rollouts if r['policy_step']==g['policy_step']]
        assert [r['reward'] for r in rs]==g['rewards']
        assert [r['task_id'] for r in rs]==g['task_ids'] and len(set(g['task_ids']))==1
        assert sum(sum(r['model_token_mask']) for r in rs)==g['model_tokens_in_loss']
        assert sum(len(r['model_token_mask'])-sum(r['model_token_mask']) for r in rs)==g['external_tokens_excluded']
        mean=sum(g['rewards'])/4;std=math.sqrt(sum((r-mean)**2 for r in g['rewards'])/3)
        expected=[(r-mean)/(std+1e-4) for r in g['rewards']]
        assert all(abs(a-b)<1e-6 for a,b in zip(expected,g['advantages']))
    assert sum(sum(r['model_token_mask']) for r in rollouts)==summary['sampled_tokens']
    metrics=rows(path/'metrics.jsonl')
    result={'interface_lineage_boundary_and_loss_mask_audit':'passed',
        'model_rollouts':len(rollouts),'sampled_tokens':summary['sampled_tokens'],
        'successes':sum(r['reward'] for r in rollouts),
        'complete_user_conversations':sum(r['completed_user_turns']==r['total_user_turns'] for r in rollouts),
        'valid_user_turns':sum(g['valid'] for r in rollouts for g in r['grades']),
        'completed_user_turns':sum(r['completed_user_turns'] for r in rollouts),
        'stop_reasons':dict(Counter(r['stop_reason'] for r in rollouts)),
        'external_tokens_excluded':sum(g['external_tokens_excluded'] for g in groups),
        'groups_with_reward_variance':sum(len(set(g['rewards']))>1 for g in groups),
        'nonzero_gradient_steps':[m['step'] for m in metrics if float(m.get('grad_norm',0))>0],
        'parameter_change_steps':[u['step'] for u in updates if u['parameters_changed']],
        'wall_seconds':summary['wall_seconds'],'single_gpu_wall_hours':summary['wall_seconds']/3600,
        'limitation':'Two development cases; engineering updates only. No held-out generalization, selector effect, or frozen scientific recipe.'}
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();print(json.dumps(audit(args.run)))
