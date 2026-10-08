"""Combine development calibration panels without double-counting subset probes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'rl'


def load(name):
    path=BASE/f'calibration_20261008_{name}'
    run=json.loads((path/'run.json').read_text())
    summary=json.loads((path/'summary.json').read_text())
    tasks=json.loads((path/'tasks.json').read_text())
    return run,summary,tasks


def compact(summary):
    keys=['status','episodes','successes','trajectory_cap_hits','mixed_reward_groups',
          'families_with_mixed_rewards','sampled_tokens_including_discarded','discarded_tokens',
          'generation_seconds','readiness_screen_passed']
    return {k:summary[k] for k in keys}


def outcome_diagnostics(names):
    counts={'success_at_cap':0,'success_below_cap':0,
            'failure_at_cap':0,'failure_below_cap':0}
    for name in names:
        path=BASE/f'calibration_20261008_{name}'
        run=json.loads((path/'run.json').read_text())
        for line in (path/'rollouts.jsonl').read_text().splitlines():
            row=json.loads(line)
            assert row['reward'] in (0,1)
            outcome='success' if row['reward'] else 'failure'
            cap='at_cap' if len(row['completion_ids'])>=run['max_completion_length'] else 'below_cap'
            counts[f'{outcome}_{cap}']+=1
    return counts


def main():
    original=load('original');protocol=load('protocol')
    assert original[0]['policy_fingerprint']==protocol[0]['policy_fingerprint']
    for a,b in zip(original[2],protocol[2]):
        ta,tb=json.loads(a['task_json']),json.loads(b['task_json'])
        assert ta['scenario']==tb['scenario'] and ta['task_id']==tb['task_id']
    result={'scope':'development calibration; no optimizer updates or transfer estimates',
            'original_2k':compact(original[1]),'protocol_2k':compact(protocol[1])}
    for label,name in [('original_2k','original'),('protocol_2k','protocol')]:
        result[label]['outcomes_by_cap']=outcome_diagnostics([name])
    parts=[]
    part_names=[]
    for name in ['protocol4k_probe','protocol4k_extension']:
        if not (BASE/f'calibration_20261008_{name}'/'summary.json').exists():continue
        run,summary,tasks=load(name)
        assert run['condition']=='protocol' and run['max_completion_length']==4096
        assert run['policy_fingerprint']==protocol[0]['policy_fingerprint']
        assert run['source_sha256']['rl_environment.py']==protocol[0]['source_sha256']['rl_environment.py']
        offset=run.get('task_offset',0)
        assert tasks==protocol[2][offset:offset+run['task_limit']]
        if summary['status']=='complete':assert summary['policy_unchanged'] and summary['optimizer_steps']==0
        parts.append(summary)
        part_names.append(name)
    if parts:
        tasks=[t for p in parts for t in p['tasks']]
        assert len(set(t['task_id'] for t in tasks))==len(tasks)
        complete=len(tasks)==6 and all(p['status']=='complete' for p in parts)
        cap=sum(t['cap_hits'] for t in tasks)
        mixed=sum(t['mixed_reward_group'] for t in tasks)
        families=sorted({t['family'] for t in tasks if t['mixed_reward_group']})
        result['protocol_4k_adaptive']={
            'status':'complete' if complete else 'partial',
            'episodes':sum(len(t['rewards']) for t in tasks),
            'successes':sum(sum(t['rewards']) for t in tasks),
            'trajectory_cap_hits':cap,'mixed_reward_groups':mixed,
            'families_with_mixed_rewards':families,
            'sampled_tokens_including_discarded':sum(p['sampled_tokens_including_discarded'] for p in parts),
            'discarded_tokens':sum(p['discarded_tokens'] for p in parts),
            'generation_seconds':sum(p['generation_seconds'] for p in parts),
            'readiness_screen_passed': cap<=6 and mixed>=3 and len(families)>=2 if complete else None,
            'adaptive_design':True,'tasks':tasks}
        result['protocol_4k_adaptive']['outcomes_by_cap']=outcome_diagnostics(part_names)
    output=BASE/'calibration_20261008_comparison.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='protocol_4k_adaptive'},indent=2))
    if 'protocol_4k_adaptive' in result:
        print(json.dumps({k:v for k,v in result['protocol_4k_adaptive'].items() if k!='tasks'},indent=2))


if __name__=='__main__':main()
