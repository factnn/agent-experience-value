"""Two explicit trajectory-structure candidates using feature-panel draws only.

These are operational signals, not causal isolation of diversity or recovery.
No validation reward, target label, model inference, or parameter update is used.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
from statistics import mean

SIGNALS = ['within_task_call_sequence_diversity', 'explicit_error_then_valid_turn_recovery']


def call_sequence(row):
    result=[]
    for event in row['events']:
        name,sep,_=event['call'].partition('(')
        assert sep and re.fullmatch(r'[A-Za-z]\w*(?:\.[A-Za-z]\w*)?',name),event['call']
        result.append(name)
    return result


def normalized_edit_distance(left,right):
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]/max(len(left),len(right),1)


def explicit_error(response):
    assert isinstance(response,str)
    if response.startswith('Error during execution: '):return True
    try:value=json.loads(response)
    except (ValueError,TypeError):return False
    return isinstance(value,dict) and bool(value.get('error'))


def recovery_details(row):
    total=row['total_user_turns'];assert isinstance(total,int) and total>0
    valid={g['turn'] for g in row['grades'] if g['valid'] is True}
    assert all(0<=turn<total for turn in valid)
    errors=set();later_nonerrors=set()
    for event in row['events']:
        turn=event['turn'];assert isinstance(turn,int) and 0<=turn<total
        if explicit_error(event['response']):errors.add(turn)
        elif turn in errors:later_nonerrors.add(turn)
    recovered=sorted(valid & errors & later_nonerrors)
    return {'recovered_turns':recovered,'explicit_error_turns':sorted(errors),
        'valid_turns':sorted(valid),'requested_turns':total,'fraction':len(recovered)/total}


def task_features(rows):
    assert len(rows)==4 and sorted(r['rollout_index'] for r in rows)==[0,1,2,3]
    assert len({r['task_id'] for r in rows})==1 and all(r['phase']=='feature' for r in rows)
    sequences=[call_sequence(r) for r in rows]
    distances=[normalized_edit_distance(a,b) for a,b in itertools.combinations(sequences,2)]
    recoveries=[recovery_details(r) for r in rows]
    return {SIGNALS[0]:mean(distances),SIGNALS[1]:mean(x['fraction'] for x in recoveries),
        'call_sequences':sequences,'six_pair_distances':distances,'recovery_details':recoveries,
        'stop_reasons':[r['stop_reason'] for r in rows],
        'scope':'Six pairs share four trajectories; they are not six independent observations.'}


def source_features(rows,task_ids):
    selected={task:[] for task in task_ids};ignored_validation=0
    for row in rows:
        if row['phase']=='validation':ignored_validation+=1;continue
        assert row['phase']=='feature' and row['task_id'] in selected
        selected[row['task_id']].append(row)
    tasks={key:task_features(selected[key]) for key in task_ids}
    return {key:mean(task[key] for task in tasks.values()) for key in SIGNALS} | {
        'task_count':len(tasks),'feature_rollouts':4*len(tasks),'ignored_validation_rollout_count':ignored_validation,
        'tasks':tasks,'weighting':'Mean of four-trajectory task signals, then equal weight over all tasks in the source; no task/reward/stop-reason filtering.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--probe-protocol',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    protocol_path=args.root/args.probe_protocol;plan=json.loads(protocol_path.read_text());groups={};inputs={}
    for case in plan['cases']:
        path=args.root/plan['output_root']/case['name']/'rollouts.jsonl'
        rows=[json.loads(line) for line in path.open()]
        assert {x['policy_fingerprint'] for x in rows}=={plan['policy_fingerprint']}
        groups[case['source_group']]=source_features(rows,case['task_ids'])
        inputs[str(path.relative_to(args.root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={'kind':'exploratory_structural_features_from_existing_feature_draws',
        'source_protocol':{'path':args.probe_protocol,'sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest()},
        'policy_fingerprint':plan['policy_fingerprint'],'signals':SIGNALS,'groups':groups,'raw_input_sha256':inputs,
        'validation_rewards_used':False,'new_model_tokens':0,'new_gpu_jobs':0,'target_labels_used_for_extraction':False,
        'old_target_labels_already_open_when_candidates_defined':True,
        'limits':['Diversity is within-task variability of attempted function-name sequences, not semantic/task diversity or a diversity intervention.',
                  'Recovery requires an explicit execution/tool error, a later non-error tool call in the same user turn, and a valid latched turn grade; it is an observed pattern, not proof that the later action caused repair.',
                  'Malformed tool-call feedback is excluded because it lacks a reliable turn-indexed event; this is not every kind of recovery.',
                  'Recovery uses source-task turn grades and remains related to success; it is not independent of outcome value.',
                  'Signals predict source acquisition interventions; no causal isolation of trajectory properties.',
                  'Original4B heldout labels were already open; any4B fit/score is exploratory. Prospective external forecasts require separate registration and seal before target access.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'sources':len(groups),'signal_ranges':{key:[min(g[key] for g in groups.values()),max(g[key] for g in groups.values())] for key in SIGNALS},'nonzero_sources':{key:sum(g[key]>0 for g in groups.values()) for key in SIGNALS},'output':str(args.output),'new_model_tokens':0}),flush=True)


if __name__=='__main__':main()
