"""Add descriptive repeat/task/stratum evidence to the unchanged frozen full104 report."""
from pathlib import Path
from collections import Counter
from statistics import mean,stdev
import argparse,datetime,hashlib,json,math

REPORT='rl/bfcl_source_stage0_transfer_prediction_20261015/audit_status.json'
REPORT_SHA='c26424eee244968470b478a32ad6bf4b96d5563197972bc04331d6bc753bb992'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(root):
 p=root/REPORT;assert digest(p)==REPORT_SHA;report=json.loads(p.read_text())
 assert report['status']=='passed' and report['labels']['panel_count']==104
 refs=[{'path':REPORT,'sha256':REPORT_SHA}]
 panels={};outcomes={};task_strata={};cost={}
 for phase in ('fit','heldout'):
  for row in report['evidence']['evaluation'][phase]:
   entry=row['job'];job_path=root/entry['path'];assert digest(job_path)==entry['sha256']
   job=json.loads(job_path.read_text());out=root/job['output'];a=json.loads((out/'audit.json').read_text())
   assert digest(out/'audit.json')==row['audit_sha256']
   key=(job['policy_case'],job['evaluation_repeat']);assert key not in panels;panels[key]=a
   file=out/'outcomes.jsonl';assert digest(file)==row['source_files']['outcomes.jsonl']
   rows=[json.loads(line) for line in file.read_text().splitlines()];assert [x['task_id'] for x in rows]==job['task_ids']
   outcomes[key]={x['task_id']:x['rewards'][0] for x in rows}
   for x in rows:
    assert x['rewards'][0] in (0,1)
    if x['task_id'] in task_strata:assert task_strata[x['task_id']]==x['stratum']
    task_strata[x['task_id']]=x['stratum']
   for name,value in a['cost'].items():
    if isinstance(value,(int,float)):cost[name]=cost.get(name,0)+value
 assert len(panels)==104 and len(task_strata)==32
 training=report['costs']['training_rows'];uniform={x['training_repeat']:x['case_id'] for x in training if x['role']=='uniform_control'}
 assert set(uniform)=={0,1,2}
 held=set(report['sealed_predictions']['heldout_groups']);preds={x['source_group']:x for x in report['source_predictions']}
 source_rows=[];seed_diffs=[]
 for g,row in report['labels']['groups'].items():
  strata={s:{'G':[],'V':[]} for s in set(task_strata.values())};task_cells={t:{'G':[],'V':[]} for t in task_strata};repeats=[]
  for rep in row['training_repeats']:
   rr=rep['training_repeat'];case=rep['policy_case'];u=uniform[rr];branch=next(x for x in training if x['case_id']==case)
   for e in (0,1):
    for s in strata:
     strata[s]['G'].append(panels[case,e]['stratum_success'][s]-panels['common',e]['stratum_success'][s])
     strata[s]['V'].append(panels[case,e]['stratum_success'][s]-panels[u,e]['stratum_success'][s])
    for t in task_strata:
     task_cells[t]['G'].append(outcomes[case,e][t]-outcomes['common',e][t])
     task_cells[t]['V'].append(outcomes[case,e][t]-outcomes[u,e][t])
   d=rep['panels'][0]['G']-rep['panels'][1]['G'];seed_diffs.append({'source_group':g,'training_repeat':rr,'G_e0_minus_e1':d,'V_e0_minus_e1':rep['panels'][0]['V']-rep['panels'][1]['V']})
   repeats.append({'training_repeat':rr,'G':rep['G'],'V':rep['V'],'paired_evaluation_panels':rep['panels'],'training':branch})
  strata={s:{m:mean(v) for m,v in ms.items()} for s,ms in strata.items()}
  tasks=[{'task_id':t,'stratum':task_strata[t],**{m:mean(v) for m,v in ms.items()},'six_paired_cells':ms} for t,ms in task_cells.items()]
  for metric in ('G','V'):
   assert math.isclose(row[metric],mean(x[metric] for x in tasks),abs_tol=1e-12)
   assert math.isclose(row[metric],mean(x[metric] for x in strata.values()),abs_tol=1e-12)
  source_rows.append({'source_group':g,'split':'heldout' if g in held else 'fit','G':row['G'],'V':row['V'],'G_by_training_repeat':[x['G'] for x in repeats],'V_by_training_repeat':[x['V'] for x in repeats],'G_training_repeat_sample_sd_descriptive':stdev(x['G'] for x in repeats),'G_training_repeat_range':[min(x['G'] for x in repeats),max(x['G'] for x in repeats)],'V_training_repeat_range':[min(x['V'] for x in repeats),max(x['V'] for x in repeats)],'training_and_evaluation_repeats':repeats,'strata':strata,'tasks':tasks,'all_five_original_predictions':preds[g]['predictions'],'all_five_squared_errors':preds[g]['squared_errors']})
 def sign_counts(metric):return dict(Counter('positive' if x[metric]>1e-12 else 'negative' if x[metric]<-1e-12 else 'zero' for x in source_rows))
 primary=report['prediction_results']['primary'];mse=primary['MSE'];held_rows=[x for x in source_rows if x['split']=='heldout']
 contributions=[{'source_group':x['source_group'],'observed_G':x['G'],'observed_V':x['V'],'frozen_predictions':x['all_five_original_predictions'],'squared_errors':x['all_five_squared_errors'],'task_minus_mean_squared_error':x['all_five_squared_errors']['primary_task_beta']-x['all_five_squared_errors']['fit_mean'],'contribution_to_primary_difference':(x['all_five_squared_errors']['primary_task_beta']-x['all_five_squared_errors']['fit_mean'])/4} for x in held_rows]
 assert math.isclose(sum(x['contribution_to_primary_difference'] for x in contributions),primary['primary_difference'],abs_tol=1e-12)
 summary={'source_conditions':16,'fit_conditions':12,'heldout_conditions':4,'target_tasks':32,'training_repeats':3,'evaluation_seeds':2,'source_mean_G':mean(x['G'] for x in source_rows),'source_mean_V':mean(x['V'] for x in source_rows),'source_G_range':[min(x['G'] for x in source_rows),max(x['G'] for x in source_rows)],'G_sign_counts':sign_counts('G'),'V_sign_counts':sign_counts('V'),'groups_with_G_positive_all3':sum(all(g>1e-12 for g in x['G_by_training_repeat']) for x in source_rows),'groups_with_V_positive_all3':sum(all(v>1e-12 for v in x['V_by_training_repeat']) for x in source_rows),'groups_positive_mean_V':[{'source_group':x['source_group'],'V':x['V'],'V_by_training_repeat':x['V_by_training_repeat']} for x in source_rows if x['V']>1e-12],'largest_absolute_paired_eval_G_difference':max(abs(x['G_e0_minus_e1']) for x in seed_diffs),'heldout_source_G_range':[min(x['G'] for x in held_rows),max(x['G'] for x in held_rows)],'task_MSE_relative_to_fit_mean':mse['primary_task_beta']/mse['fit_mean']-1,'task_worse_than_mean_heldout_sources':sum(x['task_minus_mean_squared_error']>0 for x in contributions),'task_worse_than_mean_training_repeat_scores':sum(x['primary_difference']>0 for x in report['prediction_results']['same_sealed_predictions_by_training_repeat']),'stratum_mean_G_V':{s:{m:mean(x['strata'][s][m] for x in source_rows) for m in ('G','V')} for s in task_strata.values()}}
 for name in ['G','V']:
  assert math.isclose(mean(v[name] for v in summary['stratum_mean_G_V'].values()),summary['source_mean_'+name],abs_tol=1e-12)
 return {'kind':'stage0_full104_descriptive_supplement','computed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':refs,'analysis_code_sha256':digest(Path(__file__)),'prespecified_primary_report_unchanged':True,'summary':summary,'heldout_error_contributions':contributions,'original_five_model_MSE':mse,'original_primary_difference':primary['primary_difference'],'original_secondary_difference':primary['mechanism_secondary_difference'],'same_sealed_predictions_by_training_repeat':report['prediction_results']['same_sealed_predictions_by_training_repeat'],'paired_eval_seed_differences':seed_diffs,'source_rows':source_rows,'actual_evaluation_cost':cost,'original_costs':report['costs'],'uncertainty_scope':'Descriptive repeat ranges, sample SDs and paired task/stratum cells only; no new confirmatory statistic, predictor selection or population confidence interval. Source primitives overlap; all sources share ancestor,32 targets,baseline panels and corresponding Uniform controls. Two eval seeds are not two independent source populations; G/V rankings differ by a common shift.','interpretation_limits':['Finite-source task-Beta primary comparison did not improve on the mean baseline, although it improved on pooled Beta; this does not reject an entire value hypothesis class.','Coverage and Frontier have lower aggregate heldout MSE here; both remain separately prespecified comparators, not post-selected winners or tested acquisition policies.','Positive meanG for a source is distinct from positiveV and from repeat-stable advantage. Three training and two evaluation repeats do not establish a population ranking.','Dose, optimizer opportunities and changed parameters differ across branches; token ratios do not remove this confounding.','All results use Qwen3-4B at stage0 and existing32 target tasks; fixed late-stage, mechanism breadth, scale/family,ID,21 reserved targets and environment extrapolation remain incomplete.'],'validation_probe_rewards_used':False,'predictions_refitted_or_selected':False,'new_scientific_tokens':0,'goal_complete':False}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 result=describe(args.root)
 with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'summary':result['summary'],'heldout_error_contributions':result['heldout_error_contributions'],'actual_evaluation_cost':result['actual_evaluation_cost'],'output':str(args.output)},ensure_ascii=False,indent=2))
