"""Feature-only feasibility inventory for an outcome-success candidate.

No transfer labels, validation rewards, new sampling, fitting, or changes to
the frozen main comparison. The proposed candidate is not a registered arm.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone
from bfcl_probe_forecast import forecast, beta_mixed

ROOT = Path(__file__).resolve().parents[1]
MASTER = 'rl/BFCL_SOURCE_STAGE0_EXECUTION_PROTOCOL_20261012.json'
MASTER_SHA = 'ea7c8a3f883d30a4358b05d74101dbf8653e379356ae4bf996c0075d511a98dd'


def inventory(root):
    evidence = {}
    def read(path, expected=None):
        content=(root/path).read_bytes();sha=hashlib.sha256(content).hexdigest()
        if expected is not None and sha != expected:raise ValueError('Changed source: '+path)
        evidence[path]=sha
        return json.loads(content)
    master=read(MASTER,MASTER_SHA)
    features=read(**{'path':master['features']['path'],'expected':master['features']['sha256']})
    analysis=read(master['analysis_contract']['path'],master['analysis_contract']['sha256'])
    plan=read(features['source_protocol']['path'],features['source_protocol']['sha256'])
    if features['stage'] != 0 or features['validation_labels_used'] is not False:raise ValueError('Require original feature-only measurements')
    existing={r['source_group']:r for r in features['rows']}
    rows=[];all_k=Counter();tasks_seen=set()
    for case in plan['cases']:
        source=existing[case['source_group']]
        path=str(Path(plan['output_root'])/case['name']/'feature.jsonl')
        content=(root/path).read_bytes();sha=hashlib.sha256(content).hexdigest()
        if sha != source['feature_sha256']:raise ValueError('Feature panel changed')
        evidence[path]=sha;raw=[json.loads(line) for line in content.splitlines()]
        if [r['task_id'] for r in raw] != case['task_ids'] or [r['seed'] for r in raw] != case['seeds']['feature']:raise ValueError('Feature task/seed mismatch')
        if any(r['phase'] != 'feature' or r['stage'] != 0 or r['policy_fingerprint'] != features['policy_fingerprint'] or
               r['source_group'] != case['source_group'] for r in raw):raise ValueError('Invalid feature identity')
        if tasks_seen & set(case['task_ids']):raise ValueError('Repeated source task')
        tasks_seen.update(case['task_ids'])
        f=forecast(raw,case['historical_signal'])
        saved=read(str(Path(plan['output_root'])/case['name']/'predictions.json'),source['prediction_sha256'])
        if f != saved:raise ValueError('Feature forecast cannot be replayed')
        ks=[int(sum(r['rewards'])) for r in raw]
        if any(len(r['rewards']) != 4 for r in raw):raise ValueError('Require four draws per task')
        successes=sum(ks);trials=4*len(ks);fraction=Fraction(successes,trials)
        task_beta=mean(beta_mixed(k,4) for k in ks);pooled=beta_mixed(successes,trials)
        if abs(task_beta-source['primary_task_beta']) > 1e-14 or abs(pooled-source['current_pooled_beta']) > 1e-14:
            raise ValueError('Original feature extraction differs')
        all_k.update(ks)
        rows.append({'source_group':case['source_group'],'split':'fit' if case['source_group'] in analysis['source_split']['fit'] else 'heldout',
            'task_count':len(ks),'successes':successes,'trials':trials,'candidate_success_rate':float(fraction),
            'candidate_exact_fraction':str(fraction),'task_success_count_histogram':{str(k):ks.count(k) for k in range(5)},
            'primary_task_beta':task_beta,'current_pooled_beta':pooled,
            'historical_frontier':source['historical_frontier'],'historical_coverage':source['historical_coverage']})
    if len(rows) != 16 or len(tasks_seen) != 87 or set(existing) != {r['source_group'] for r in rows}:raise ValueError('Incomplete source panel')
    matched=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if abs(a['primary_task_beta']-b['primary_task_beta']) < 1e-12 and a['candidate_exact_fraction'] != b['candidate_exact_fraction']:
                matched.append({'groups':[a['source_group'],b['source_group']], 'task_beta':a['primary_task_beta'],
                                'success_rates':[a['candidate_success_rate'],b['candidate_success_rate']]})
    for p in ['pipeline/inventory_bfcl_outcome_candidate.py','pipeline/bfcl_probe_forecast.py','pipeline/bfcl_task_probe_statistics.py']:
        evidence[p]=hashlib.sha256((root/p).read_bytes()).hexdigest()
    return {'status':'passed','kind':'outcome_candidate_feature_only_feasibility','observed_at':datetime.now(timezone.utc).isoformat(),
        'stage':0,'policy_fingerprint':features['policy_fingerprint'],'source_sha256':evidence,'rows':rows,
        'source_groups':16,'tasks':87,'existing_feature_rollouts':348,
        'task_success_count_histogram':{str(k):all_k[k] for k in range(5)},
        'candidate_source_distinct_values':len({r['candidate_exact_fraction'] for r in rows}),
        'candidate_source_range':[min(r['candidate_success_rate'] for r in rows),max(r['candidate_success_rate'] for r in rows)],
        'source_macro_success':mean(r['candidate_success_rate'] for r in rows),
        'task_weighted_success':sum(r['successes'] for r in rows)/348,
        'matched_task_beta_different_outcome_pairs':matched,
        'local_validation_rewards_read':False,'transfer_labels_read':False,'models_fitted':False,
        'new_sampled_tokens':0,'gpu_jobs_launched':0,'candidate_added_to_frozen_analysis':False,
        'scope':'Actual feature-only candidate inventory, not a transfer-value test or a causal comparison of successful versus failed trajectories. No new scientific protocol registered.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=inventory(ROOT);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'audit_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','source_groups','tasks','task_success_count_histogram',
        'candidate_source_distinct_values','candidate_source_range','matched_task_beta_different_outcome_pairs','new_sampled_tokens']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
