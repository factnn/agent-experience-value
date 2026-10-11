"""Separate conditional training and evaluation repetition; audit complete panels."""
import argparse
import hashlib
import json
import math
from statistics import mean
from pathlib import Path
from report_bfcl_intervention import ROOT, read, lines, audit_sampling, unchanged_policy_consistency
from watch_bfcl_intervention import write


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protocol(spec):
    path=ROOT/spec['path'];assert digest(path)==spec['sha256']
    return read(path)


def check_training(path,p,spec,common):
    summary=read(path/'summary.json');assert summary['status']=='complete'
    assert read(path/'run.json')['protocol_sha256']==spec['sha256']
    restored=read(path/'restored.json')
    assert restored['common_payload_sha256']==p['source_common_payload_sha256']
    assert restored['trainable_sha256']==common['trainable_sha256']
    assert restored['inherited_tokens']==common['cumulative_acquisition_tokens']
    assert restored['declared_allocator_seed_override']==p['branch_allocator_seeds']['AllTasks_dose']
    assert restored['optimizer_global_step']==0 and restored['fresh_rollout_buffer']
    allocations=lines(path/'allocation.jsonl');groups=lines(path/'groups.jsonl');updates=lines(path/'updates.jsonl')
    assert len(allocations)==len(groups)==len(updates)==summary['optimizer_steps']
    cost=audit_sampling(path);assert cost['rollouts']==4*len(allocations)
    cumulative=[];total=0
    for i,(a,g,u) in enumerate(zip(allocations,groups,updates)):
        assert a['policy_step']==i and u['step']==i+1
        assert a['task_id'] in p['conditions']['AllTasks_dose']
        assert a['seed']==p['seeds']['branch']+100*i
        assert g['task_ids']==[a['task_id']]*4 and g['rewards']==a['rewards']
        assert set(a['probabilities'])==set(p['conditions']['AllTasks_dose'])
        assert all(abs(v-1/87)<1e-12 for v in a['probabilities'].values())
        assert (len(set(a['rewards']))>1)==any(abs(v)>1e-8 for v in g['advantages'])
        total+=a['sampled_tokens'];cumulative.append(total)
        assert a['branch_cost_after_group']==total
    assert total==cost['new_model_tokens']==summary['branch_sampled_tokens']
    assert summary['all_thresholds_reached'] and summary['stop_reason']=='raw_token_budget'
    labels=read(path/'dose_checkpoints.json');assert [d['threshold'] for d in labels]==p['dose_thresholds']
    for label in labels:
        step=label['optimizer_steps'];before=cumulative[step-2] if step>1 else 0
        assert before<label['threshold']<=cumulative[step-1]==label['actual_tokens']
        assert label['overshoot']==label['actual_tokens']-label['threshold']<=32768
        checkpoint=ROOT/label['policy'];cs=read(checkpoint/'summary.json');manifest=read(checkpoint/'components/manifest.json')
        assert cs['status']=='complete' and cs['verification']['status']=='passed'
        assert cs['optimizer_backend']=='AdamW' and cs['thresholds']==label['same_step_labels']
        assert cs['policy_fingerprint']==label['policy_fingerprint']==updates[step-1]['after_sha256']
        assert manifest['progress']['cumulative_learner_optimizer_steps']==step
        assert manifest['cumulative_acquisition_tokens']==common['cumulative_acquisition_tokens']+label['actual_tokens']
        assert manifest['frozen_base_sha256']==common['frozen_base_sha256']
        assert manifest['provenance']['intervention_protocol_sha256']==spec['sha256']
        assert digest(checkpoint/'components/learning_state.pt')==manifest['payload_sha256']
    metrics=lines(path/'metrics.jsonl')
    assert all(math.isfinite(m['grad_norm']) for m in metrics if 'grad_norm' in m)
    return {'cost':cost,'labels':labels,'fresh_nonzero_gradient_steps':[m['step'] for m in metrics if m.get('grad_norm',0)>0],
            'changed_parameter_steps':[u['step'] for u in updates if u['parameters_changed']]}


def check_evaluation(path,p,spec,policy):
    summary=read(path/'summary.json');assert summary['status']=='complete' and summary['weights_unchanged']
    assert read(path/'run.json')['protocol_sha256']==spec['sha256']
    source=read(policy/'summary.json')
    assert summary['policy_fingerprint']==source['policy_fingerprint']
    outcomes=lines(path/'outcomes.jsonl')
    assert [{k:r[k] for k in p['eval_panel'][0]} for r in outcomes]==p['eval_panel']
    assert len(outcomes)==22 and all(len(r['rewards'])==1 and r['rewards'][0] in [0,1] for r in outcomes)
    assert [r['seed'] for r in outcomes]==[p['seeds']['eval']+100*i for i in range(22)]
    cost=audit_sampling(path);assert cost['rollouts']==22 and cost['new_model_tokens']==summary['sampled_tokens']
    return {'path':str(path.relative_to(ROOT)),'cost':cost,'outcomes':outcomes,
            'successes':sum(r['rewards'][0] for r in outcomes),'policy_fingerprint':summary['policy_fingerprint']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);args=ap.parse_args()
    master=read(args.protocol);root=ROOT/master['output'];old=ROOT/master['previous_wave']['path']
    assert digest(old/'comparison.json')==master['previous_wave']['comparison_sha256']
    assert not read(root/'failures.json')
    common_path=ROOT/master['source_common_path'];common=read(common_path/'components/manifest.json')
    assert common['payload_sha256']==master['source_common_payload_sha256']
    baseline={};cache={};rows=[];controls=[];training_costs=[]
    def evaluation(path,e,policy):
        if path not in cache:cache[path]=check_evaluation(path,check_protocol(e['protocol']),e['protocol'],policy)
        return cache[path]
    for e in master['evaluation_repeats']:
        path=old/'eval_common' if e['name']=='e0' else root/'eval_common_e1'
        baseline[e['name']]=evaluation(path,e,common_path)
    repeats=[{'name':'discovery','protocol':master['previous_protocol'],'path':old/'train','prospective':False}]
    repeats += [dict(x,path=root/('train_'+x['name']),prospective=True) for x in master['training_repeats']]
    for repeat in repeats:
        p=check_protocol(repeat['protocol']);training=check_training(repeat['path'],p,repeat['protocol'],common)
        if repeat['prospective']:training_costs.append(training['cost'])
        for label in training['labels']:
            policy=ROOT/label['policy'];step=label['optimizer_steps']
            for e in master['evaluation_repeats']:
                if repeat['name']=='discovery':
                    path=old/f'eval_step_{step:03d}' if e['name']=='e0' else root/f"eval_discovery_step_{step:03d}_e1"
                else:path=root/f"eval_{repeat['name']}_step_{step:03d}_{e['name']}"
                evaluated=evaluation(path,e,policy);base=baseline[e['name']]
                pairs=[{'task_id':a['task_id'],'source':b['rewards'][0],
                        'checkpoint':a['rewards'][0]} for a,b in zip(evaluated['outcomes'],base['outcomes'])]
                item={'training_repeat':repeat['name'],'prospective_training_repeat':repeat['prospective'],
                    'evaluation_repeat':e['name'],'threshold':label['threshold'],
                    'actual_tokens':label['actual_tokens'],'optimizer_steps':step,
                    'evaluation':evaluated['path'],'policy_fingerprint':label['policy_fingerprint'],
                    'successes':evaluated['successes'],'tasks':22,
                    'G':(evaluated['successes']-base['successes'])/22,'paired_outcomes':pairs,
                    'fresh_nonzero_gradient_steps':training['fresh_nonzero_gradient_steps']}
                rows.append(item)
                # Require exact parameter identity before using unchanged inference control.
                manifest=read(policy/'components/manifest.json')
                if manifest['trainable_sha256']==common['trainable_sha256']:
                    control=unchanged_policy_consistency(lines(ROOT/base['path']/'rollouts.jsonl'),lines(path/'rollouts.jsonl'))
                    controls.append({'training_repeat':repeat['name'],'threshold':label['threshold'],
                        'evaluation_repeat':e['name'],**control})
    contrasts=[]
    for repeat in repeats:
        for e in master['evaluation_repeats']:
            selected={r['threshold']:r for r in rows if r['training_repeat']==repeat['name'] and r['evaluation_repeat']==e['name']}
            middle=selected[65536];late=selected[131072]
            contrasts.append({'training_repeat':repeat['name'],'prospective':repeat['prospective'],
                'evaluation_repeat':e['name'],'late_minus_middle_successes':late['successes']-middle['successes'],
                'late_minus_middle_rate':(late['successes']-middle['successes'])/22})
    descriptions=[]
    for threshold in master['dose_thresholds']:
        for e in master['evaluation_repeats']:
            selected=[r for r in rows if r['prospective_training_repeat'] and r['threshold']==threshold and r['evaluation_repeat']==e['name']]
            assert len(selected)==3;values=[r['G'] for r in selected]
            descriptions.append({'threshold':threshold,'evaluation_repeat':e['name'],
                'training_repeats':3,'G_values':values,'G_mean':mean(values),'G_range':[min(values),max(values)]})
    launches=[read(path) for path in root.glob('*.launch.json')]
    assert all(r['gpu'] in master['allowed_gpus'] and r['hard_seconds']==master['hard_seconds_per_process'] for r in launches)
    new_evals=[v for path,v in cache.items() if path.is_relative_to(root)]
    result={'status':'audited_complete','master_protocol_sha256':digest(args.protocol),
        'new_training_repeats':3,'evaluation_repeats':2,'discovery_counted_as_prospective_repeat':False,
        'baseline_successes':{k:v['successes'] for k,v in baseline.items()},'rows':rows,
        'late_minus_middle_contrasts':contrasts,'prospective_descriptive_G':descriptions,
        'unchanged_policy_controls':controls,'unchanged_policy_control_flagged':any(c['status']!='exact_match' for c in controls),
        'new_train_generation_tokens':sum(c['new_model_tokens'] for c in training_costs),
        'new_eval_generation_tokens':sum(v['cost']['new_model_tokens'] for v in new_evals),
        'new_unique_evaluation_panels':len(new_evals),'archived_evaluation_panels_reused':len(cache)-len(new_evals),
        'inherited_probe_tokens':master['inherited_probe_tokens'],'new_probe_tokens':0,
        'gpus_used':sorted({r['gpu'] for r in launches}),'limitations':master['limitations'],
        'scope':'Conditional development diagnosis; no best dose, test transfer claim, fitted predictor or significance assertion.'}
    write(root/'comparison.json',result)
    print(json.dumps({'status':result['status'],'new_training_repeats':3,
        'new_eval_panels':len(new_evals),'unchanged_policy_control_flagged':result['unchanged_policy_control_flagged']}))


if __name__=='__main__':main()
