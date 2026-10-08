"""Audit allocation pilot policy provenance, budget and evaluation isolation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def rows(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def audit(path):
    run=json.loads((path/'run.json').read_text());summary=json.loads((path/'summary.json').read_text())
    tasks=json.loads((path/'tasks.json').read_text());training={t['task_id'] for t in tasks['pilot_train']}
    evaluation={t['task_id'] for t in tasks['pilot_eval']};assert not training&evaluation
    allocations=rows(path/'allocation.jsonl');updates=rows(path/'updates.jsonl')
    rollouts=rows(path/'rollouts.jsonl');groups=rows(path/'groups.jsonl');calls=rows(path/'generation_calls.jsonl')
    assert len(allocations)==len(updates)==summary['optimizer_steps']
    assert len(rollouts)==4*len(groups)
    fp={0:run['initial_fingerprint']};fp.update({u['step']:u['after_sha256'] for u in updates})
    assert fp[len(updates)]==summary['final_fingerprint']
    for u in updates:assert u['before_sha256']==fp[u['step']-1]
    indexed=defaultdict(list)
    for call in calls:indexed[(call['policy_step'],call['phase'])].append(call)
    for t in rollouts:
        assert t['policy_fingerprint']==fp[t['policy_step']]
        assert len(t['completion_ids'])==len(t['model_token_mask'])
        assert set(t['model_token_mask']) <= {0,1}
        assert sum(t['model_token_mask'])==t['model_tokens']
        full=t['prompt_ids']+t['completion_ids']
        assert any(full==(p+c)[:len(full)] for call in indexed[(t['policy_step'],t['phase'])]
                   for p,c in zip(call['prompt_ids'],call['completion_ids']))
        assert t['task_id'] in (evaluation if t['phase'].startswith('eval:') else training)
    charged=0
    for i,a in enumerate(allocations):
        assert a['policy_step']==i and a['cost_before_choice']==charged < run['token_budget']
        assert a['policy_fingerprint']==fp[i]
        selected=[t for t in rollouts if t['phase']==f'train:{i}']
        assert len(selected)==4 and [t['reward'] for t in selected]==a['rewards']
        assert all(t['task_id']==a['task_id'] for t in selected)
        cost=sum(sum(c['sampled_tokens']) for c in indexed[(i,f'train:{i}')])
        assert cost==a['sampled_tokens'];charged+=cost;assert charged==a['cost_after_group']
        assert set(a['probabilities'])==training
        assert abs(sum(a['probabilities'].values())-1)<1e-9
    for g in groups:
        batch=[t for t in rollouts if (t['policy_step'],t['phase'])==(g['policy_step'],g['phase'])]
        assert [t['reward'] for t in batch]==g['rewards']
        assert [t['task_id'] for t in batch]==g['task_ids']
    total=sum(sum(c['sampled_tokens']) for c in calls)
    assert total==summary['all_sampled_tokens_including_discarded']
    assert charged==summary['training_sampled_tokens']
    final=[t for t in rollouts if t['phase'].startswith('eval:')]
    assert len(final)==12 and all(t['policy_step']==len(updates) for t in final)
    assert sum(t['reward'] for t in final)==summary['eval_successes']
    train=[t for t in rollouts if t['phase'].startswith('train:')]
    metrics=rows(path/'metrics.jsonl')
    report={'audit':'passed','rule':run['rule'],'optimizer_steps':len(updates),
        'nonzero_gradient_steps':[m['step'] for m in metrics if float(m.get('grad_norm',0))>0],
        'groups_with_reward_variance':sum(len(set(a['rewards']))>1 for a in allocations),
        'train_successes':sum(t['reward'] for t in train),'train_episodes':len(train),
        'train_cap_hits':sum(len(t['completion_ids'])>=4096 for t in train),
        'training_sampled_tokens':charged,'budget_overshoot':summary['budget_overshoot'],
        'eval_successes':summary['eval_successes'],'eval_episodes':12,
        'retained_model_tokens':sum(t['model_tokens'] for t in rollouts),
        'discarded_sampled_tokens':total-sum(t['model_tokens'] for t in rollouts),
        'task_group_counts':dict(Counter(a['task_id'] for a in allocations)),
        'allocation_groups_after_warmup':summary['allocation_groups_after_warmup'],
        'interpretation':summary['limitation']}
    (path/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+',type=Path);args=ap.parse_args()
    for path in args.runs:print(json.dumps(audit(path)))
    if len(args.runs)==2:
        a,b=args.runs;ra=json.loads((a/'run.json').read_text());rb=json.loads((b/'run.json').read_text())
        assert ra['source_sha256']==rb['source_sha256'] and ra['initial_fingerprint']==rb['initial_fingerprint']
        assert json.loads((a/'tasks.json').read_text())==json.loads((b/'tasks.json').read_text())
        aa,ab=rows(a/'allocation.jsonl'),rows(b/'allocation.jsonl')
        compare={'scope':'single-seed engineering pilot; no confirmed effect',
            'common_warmup_task_order_matches':[r['task_id'] for r in aa[:6]]==[r['task_id'] for r in ab[:6]],
            'common_warmup_rewards_match':[r['rewards'] for r in aa[:6]]==[r['rewards'] for r in ab[:6]],
            'post_warmup_fingerprints_match':len(aa)>=7 and len(ab)>=7 and aa[6]['policy_fingerprint']==ab[6]['policy_fingerprint'],
            'arms':{p.name:json.loads((p/'audit_report.json').read_text()) for p in args.runs}}
        (a.parent/'allocation_pilot_20261008_comparison.json').write_text(json.dumps(compare,indent=2)+'\n')
        print(json.dumps(compare))


if __name__=='__main__':main()
