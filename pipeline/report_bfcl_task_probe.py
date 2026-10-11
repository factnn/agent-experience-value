"""Audit complete fixed-policy probe cases and prespecified macro Brier contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from report_bfcl_intervention import ROOT,read,lines,audit_sampling
from supervise_bfcl_stage_fork import write
from bfcl_probe_forecast import forecast,score
from bfcl_task_probe_statistics import compare_independent_panels


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_case(plan,case,protocol_sha):
    out=ROOT/plan['output_root']/case['name'];s=read(out/'summary.json');run=read(out/'run.json')
    assert s['status']=='complete' and s['weights_unchanged'] and s['optimizer_updates']==0
    assert run['protocol_sha256']==protocol_sha and run['case']==case
    assert run['policy_fingerprint']==s['policy_fingerprint']==plan['policy_fingerprint']
    feature,validation=[lines(out/(name+'.jsonl')) for name in ['feature','validation']]
    for phase,rows in [('feature',feature),('validation',validation)]:
        assert [x['task_id'] for x in rows]==case['task_ids']
        assert [x['seed'] for x in rows]==case['seeds'][phase]
        assert all(x['source_group']==case['source_group'] and x['stage']==0 and x['policy_fingerprint']==plan['policy_fingerprint'] for x in rows)
    assert compare_independent_panels(feature,validation)==s['task_statistics']
    f=forecast(feature,case['historical_signal']);assert f==read(out/'predictions.json')
    receipt=read(out/'prediction_receipt.json');start=read(out/'validation_start.json')
    assert receipt['sealed_before_validation'] and receipt['validation_rows_at_seal']==0
    assert receipt['sealed_unix']<=start['started_unix']
    assert digest(out/'predictions.json')==receipt['sha256']==start['prediction_sha256']==s['prediction_sha256']
    scores=score(f,validation);assert scores==s['scores']
    costs=audit_sampling(out);assert costs['rollouts']==8*len(case['task_ids'])
    assert costs['new_model_tokens']==s['new_sampled_tokens']==sum(x['sampled_tokens'] for x in feature+validation)<=case['token_ceiling']
    calls=lines(out/'generation_calls.jsonl');rollouts=lines(out/'rollouts.jsonl')
    assert len({r['group'] for r in rollouts})==2*len(case['task_ids'])
    n=len(case['task_ids'])
    for i,row in enumerate(feature+validation):
        raw=[x for x in rollouts if x['group']==i]
        assert len(raw)==4 and [x['rollout_index'] for x in raw]==list(range(4))
        assert [x['reward'] for x in raw]==row['rewards']
        assert all(x['task_id']==row['task_id'] and x['seed']==row['seed'] and x['draw_id']==row['draw_id'] and x['phase']==row['phase'] for x in raw)
        raw_calls=[x for x in calls if x['group']==i]
        assert all(x['task_id']==row['task_id'] and x['seed']==row['seed'] and x['draw_id']==row['draw_id'] and x['phase']==row['phase'] for x in raw_calls)
        assert sum(len(x['generated_ids']) for x in raw_calls)==row['sampled_tokens']
    return {'case':case['name'],'source_group':case['source_group'],'cost':costs,'scores':scores,'wall_seconds':s['wall_seconds'],'scope':s['scope']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--case');args=ap.parse_args();p=read(args.protocol);root=ROOT/p['output_root']
    try:
        selected=[x for x in p['cases'] if args.case is None or x['name']==args.case];assert selected
        rows=[audit_case(p,c,digest(args.protocol)) for c in selected]
        result={'status':'passed','complete_cases':len(rows),'new_sampled_tokens':sum(r['cost']['new_model_tokens'] for r in rows),'rows':rows,'macro_Brier':{k:mean(r['scores']['Brier'][k] for r in rows) for k in rows[0]['scores']['Brier']},'primary_macro_difference':mean(r['scores']['primary_difference'] for r in rows),'all_16_groups_complete':len(rows)==16,'inherited_probe_tokens_not_recharged':p['inherited_acquisition_tokens'],'scope':'Conditional local proxy calibration, source-group macro weighting; no transfer or population significance claim.'}
    except Exception as e:result={'status':'failed','type':type(e).__name__,'message':str(e),'partial_evidence_retained':True}
    target=root/(args.case+'/audit.json' if args.case else 'audit_status.json');write(target,result)
    print(json.dumps(result));raise SystemExit(0 if result['status']=='passed' else 1)

if __name__=='__main__':main()
