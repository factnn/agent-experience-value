"""Audit every reached dose, cost and complete development panel; never select a winner."""
import hashlib
import json
from pathlib import Path
from report_bfcl_intervention import read, lines, audit_sampling, ROOT
from calibrate_bfcl import write

PROTOCOL=ROOT/'rl/BFCL_DOSE_PROTOCOL_20261010.json'
OUT=ROOT/'rl/bfcl_intervention_dose_20261010'


def main():
    p=read(PROTOCOL);protocol_hash=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    train=OUT/'train';summary=read(train/'summary.json');assert summary['status']=='complete'
    assert read(train/'run.json')['protocol_sha256']==protocol_hash
    restored=read(train/'restored.json')
    assert restored['common_payload_sha256']==p['source_common_payload_sha256']
    assert restored['inherited_tokens']==90122
    assert restored['declared_allocator_seed_override']==p['branch_allocator_seeds']['AllTasks_dose']
    cost=audit_sampling(train);assert cost['new_model_tokens']==summary['branch_sampled_tokens']
    allocations=lines(train/'allocation.jsonl');groups=lines(train/'groups.jsonl');updates=lines(train/'updates.jsonl')
    assert len(allocations)==len(groups)==len(updates)==summary['optimizer_steps']
    assert cost['rollouts']==4*len(allocations)
    cumulative=[];total=0
    for i,(a,g,u) in enumerate(zip(allocations,groups,updates)):
        assert a['policy_step']==i and u['step']==i+1
        assert a['task_id'] in p['conditions']['AllTasks_dose']
        assert a['seed']==p['seeds']['branch']+100*i
        assert g['task_ids']==[a['task_id']]*4 and g['rewards']==a['rewards']
        assert set(a['probabilities'])==set(p['conditions']['AllTasks_dose'])
        assert all(abs(v-1/87)<1e-12 for v in a['probabilities'].values())
        total+=a['sampled_tokens'];cumulative.append(total)
    assert total==cost['new_model_tokens']
    labels=read(train/'dose_checkpoints.json')
    assert [d['threshold'] for d in labels]==p['dose_thresholds']
    policies={};rows=[];base_rows=None
    all_eval_tokens=0
    eval_specs=[('source',OUT/'eval_common',ROOT/p['source_common_path'])]
    for d in labels:
        if d['policy'] not in policies:
            policies[d['policy']]=OUT/f"eval_step_{d['optimizer_steps']:03d}"
            eval_specs.append((d['policy'],policies[d['policy']],ROOT/d['policy']))
    for name,evaluation,policy in eval_specs:
        es=read(evaluation/'summary.json');assert es['status']=='complete' and es['weights_unchanged']
        assert read(evaluation/'run.json')['protocol_sha256']==protocol_hash
        source=read(policy/'summary.json')
        assert es['policy_fingerprint']==source['policy_fingerprint']
        outcomes=lines(evaluation/'outcomes.jsonl')
        assert [{k:r[k] for k in p['eval_panel'][0]} for r in outcomes]==p['eval_panel']
        assert [r['seed'] for r in outcomes]==[p['seeds']['eval']+100*i for i in range(22)]
        assert all(len(r['rewards'])==1 for r in outcomes)
        ec=audit_sampling(evaluation);assert ec['new_model_tokens']==es['sampled_tokens']
        assert ec['rollouts']==22;all_eval_tokens+=ec['new_model_tokens']
        if base_rows is None: base_rows=outcomes
        successes=sum(r['rewards'][0] for r in outcomes)
        item={'policy':name,'successes':successes,'tasks':22,'rate':successes/22,
            'G':(successes-sum(r['rewards'][0] for r in base_rows))/22,
            'eval_cost':ec,'paired_outcomes':[{'task_id':r['task_id'],
                'source':b['rewards'][0],'checkpoint':r['rewards'][0]} for b,r in zip(base_rows,outcomes)]}
        rows.append(item)
    for d in labels:
        step=d['optimizer_steps'];before=cumulative[step-2] if step>1 else 0
        assert before<d['threshold']<=cumulative[step-1]==d['actual_tokens']
        assert d['overshoot']==d['actual_tokens']-d['threshold']<=32768
        checkpoint=ROOT/d['policy'];cs=read(checkpoint/'summary.json');manifest=read(checkpoint/'components/manifest.json')
        assert cs['verification']['status']=='passed' and cs['optimizer_backend']=='AdamW'
        assert cs['policy_fingerprint']==d['policy_fingerprint']==updates[step-1]['after_sha256']
        assert manifest['progress']['cumulative_learner_optimizer_steps']==step
        assert manifest['cumulative_acquisition_tokens']==90122+d['actual_tokens']
        assert hashlib.sha256((checkpoint/'components/learning_state.pt').read_bytes()).hexdigest()==manifest['payload_sha256']
        assert manifest['provenance']['intervention_protocol_sha256']==protocol_hash
        assert cs['thresholds']==d['same_step_labels']
    assert not read(OUT/'failures.json')
    result={'status':'audited_complete','scope':'One correlated development-only dose trajectory; no best-dose selection.',
        'protocol_sha256':protocol_hash,'train_cost':cost,'inherited_probe_tokens':90122,'new_probe_tokens':0,
        'new_generation_tokens_total':cost['new_model_tokens']+all_eval_tokens,
        'checkpoint_labels':labels,'unique_policy_evaluations':rows,'limitations':p['inference_limits']}
    amendment_path=ROOT/'rl/BFCL_DOSE_RESOURCE_AMENDMENT_20261010.json'
    if amendment_path.exists():
        amendment=read(amendment_path)
        assert amendment['unchanged_scientific_protocol_sha256']==protocol_hash
        launches=[read(path) for path in OUT.glob('*.launch.json')]
        assert all(r['gpu'] in amendment['allowed_gpus'] for r in launches)
        assert all(r['hard_seconds']==p['schedule']['hard_seconds_per_process'] for r in launches)
        assert read(OUT/'scheduler.json')['max_simultaneous_gpus']<=4
        result['resource_execution']={'amendment':str(amendment_path.relative_to(ROOT)),
            'amendment_sha256':hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
            'gpus_used':sorted({r['gpu'] for r in launches}),
            'maximum_authorized_simultaneous_gpus':amendment['max_simultaneous_gpu_processes'],
            'training_resampled':False,'note':'Parallel development panels; no evaluation feedback to training.'}
    write(OUT/'comparison.json',result)
    print(json.dumps({'status':result['status'],'dose_labels':len(labels),
        'unique_panels':len(rows),'development_successes':[r['successes'] for r in rows]}))


if __name__=='__main__': main()
