"""Audit fresh GRPO continuation, separating state restoration from determinism."""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(continuous, restored):
    a=json.loads((continuous/'summary.json').read_text())
    b=json.loads((restored/'summary.json').read_text())
    saved=json.loads((continuous/'checkpoint-1/components/manifest.json').read_text())
    initial=json.loads((restored/'restored.json').read_text())
    ua=rows(continuous/'updates.jsonl');ub=rows(restored/'updates.jsonl')
    aa=rows(continuous/'allocation.jsonl');ab=rows(restored/'allocation.jsonl')
    ga=rows(continuous/'groups.jsonl');gb=rows(restored/'groups.jsonl')
    assert a['global_step']==b['global_step']==2
    assert [u['step'] for u in ua]==[1,2] and [u['step'] for u in ub]==[2]
    assert initial['global_step']==1 and initial['fresh_buffer']
    assert initial['policy_fingerprint']==ua[0]['after_sha256']==ub[0]['before_sha256']
    assert initial['historical_cost']==saved['cumulative_acquisition_tokens']==aa[0]['sampled_tokens']
    assert [g['policy_step'] for g in gb]==[1] and len(ab)==1
    assert ab[0]['cost_before_choice']==initial['historical_cost']
    assert ab[0]['policy_fingerprint']==initial['policy_fingerprint']
    assert b['cumulative_cost']==initial['historical_cost']+ab[0]['sampled_tokens']
    for path, allocations in [(continuous,aa),(restored,ab)]:
        calls=rows(path/'generation_calls.jsonl');rollouts=rows(path/'rollouts.jsonl')
        for allocation in allocations:
            step=allocation['policy_step']
            group_calls=[c for c in calls if c['policy_step']==step]
            rs=[r for r in rollouts if r['policy_step']==step]
            assert len(rs)==4 and [r['reward'] for r in rs]==allocation['rewards']
            assert sum(sum(c['sampled_tokens']) for c in group_calls)==allocation['sampled_tokens']
            for r in rs:
                assert r['policy_fingerprint']==allocation['policy_fingerprint']
                assert len(r['completion_ids'])==len(r['model_token_mask'])
                assert sum(r['model_token_mask'])==r['model_tokens']
                full=r['prompt_ids']+r['completion_ids']
                assert any(full==(p+c)[:len(full)] for call in group_calls
                           for p,c in zip(call['prompt_ids'],call['completion_ids']))
    ca=[{k:c[k] for k in ['prompt_ids','completion_ids']} for c in rows(continuous/'generation_calls.jsonl') if c['policy_step']==1]
    cb=[{k:c[k] for k in ['prompt_ids','completion_ids']} for c in rows(restored/'generation_calls.jsonl')]
    metrics=rows(restored/'metrics.jsonl')
    result={'restoration_and_fresh_rollout_audit':'passed',
        'same_state_fingerprint':initial['policy_fingerprint'],
        'inherited_acquisition_tokens':initial['historical_cost'],
        'new_acquisition_tokens':ab[0]['sampled_tokens'],
        'cumulative_acquisition_tokens':b['cumulative_cost'],
        'next_raw_generation_exact_match':ca==cb,
        'next_rewards_match':ga[1]['rewards']==gb[0]['rewards'],
        'next_advantages_exact_match':ga[1]['advantages']==gb[0]['advantages'],
        'final_parameters_exact_match':a['final_fingerprint']==b['final_fingerprint'],
        'restored_fresh_gradient_steps':[m['step'] for m in metrics if float(m.get('grad_norm',0))>0],
        'verification_wall_seconds':a['wall_seconds_this_process']+b['wall_seconds_this_process'],
        'verification_new_sampled_tokens':a['cumulative_cost']+ab[0]['sampled_tokens'],
        'limitation':'One task, one GPU, completed optimizer boundary. No value/transfer effect or general dataloader-resume guarantee.'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('continuous',type=Path);p.add_argument('restored',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=audit(args.continuous,args.restored)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
