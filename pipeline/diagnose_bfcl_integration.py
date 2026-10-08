"""Replay recorded development actions; diagnose checker failures, no policy calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
from bfcl_conversation import BFCLConversation
from bfcl_token_rollout import development_tasks
from bfcl_safe_runtime import state_checker


def diagnose(path):
    run=json.loads((path/'run.json').read_text())
    selected=run.get('task_ids') or run['protocol']['task_ids']
    tasks,answers=development_tasks(selected)
    trajectories=[json.loads(x) for x in (path/'rollouts.jsonl').read_text().splitlines()]
    outcomes=[]
    for row in trajectories:
        key=row['task_id'];c=BFCLConversation(tasks[key],answers[key]);turns=[]
        for turn in range(row['completed_user_turns']):
            for event in row['events']:
                if event['turn']==turn:
                    observed=c.tool_call(event['call']);assert observed==event['response']
            c.finish_user_turn({'role':'assistant','content':'Replay boundary; no new model call.'})
            grade=c.grades[-1]
            turns.append({'turn':turn,'official_valid':grade['valid'],
                'state_matches':bool(state_checker(c.actual.instances,c.reference.instances)['valid']),
                'error_type':grade['check'].get('error_type'),
                'error_message':grade['check'].get('error_message')})
        assert c.reward()==row['reward']
        outcomes.append({'task_id':key,'policy_step':row.get('policy_step'),
            'calibration_group':row.get('group'),'mode':row.get('mode'),'rollout_index':row['rollout_index'],
            'official_reward':c.reward(),'per_turn':turns,
            'state_matches_at_all_boundaries':len(turns)==len(tasks[key]['question']) and all(t['state_matches'] for t in turns),
            'response_mismatch_with_matching_state_turns':[t['turn'] for t in turns if t['state_matches'] and not t['official_valid']],
            'tool_error_responses':sum('error' in e['response'].lower() for e in row['events']),
            'checks_frozen_at_each_boundary':True})
    result={'scope':'Development action replay only; no generation, training or held-out inspection',
        'replayed_rewards_match':True,'trajectories':outcomes,
        'failure_type_counts':dict(Counter(t['error_type'] for o in outcomes for t in o['per_turn'] if not t['official_valid'])),
        'trajectories_matching_all_states':sum(o['state_matches_at_all_boundaries'] for o in outcomes),
        'official_successes':sum(o['official_reward'] for o in outcomes),
        'interpretation':'Matching simulator state is a diagnostic, not an alternative success reward. Read-only requests can be skipped without changing state. Official BFCL response checks constrain necessary returned values; this can reject some semantically similar operations. No reward modified.'}
    (path/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    result=diagnose(args.run);print(json.dumps({k:v for k,v in result.items() if k!='trajectories'}))
