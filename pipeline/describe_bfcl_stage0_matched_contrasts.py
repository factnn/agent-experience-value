"""Post-completion source contrasts at matching repeat/seed aggregation; no new test or fit."""
import argparse,datetime,hashlib,itertools,json,math
from pathlib import Path
from statistics import mean

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(root):
 rp=root/'rl/bfcl_source_stage0_transfer_prediction_20261015/audit_status.json'
 dp=rp.parent/'descriptive_supplement.json'
 assert sha(rp)=='c26424eee244968470b478a32ad6bf4b96d5563197972bc04331d6bc753bb992'
 assert sha(dp)=='107fe9c35efc4c0dd50eb5bb579ba35ae16588326e7797ca3f8222f086d0961a'
 report=json.loads(rp.read_text());desc=json.loads(dp.read_text());groups=report['labels']['groups'];held=set(report['sealed_predictions']['heldout_groups']);tasks={row['source_group']:{x['task_id']:x for x in row['tasks']} for row in desc['source_rows']}
 rows=[]
 for a,b in itertools.combinations(groups,2):
  per_repeat=[];cells=[]
  for rr in range(3):
   ar=groups[a]['training_repeats'][rr];br=groups[b]['training_repeats'][rr]
   d=[ar['panels'][e]['G']-br['panels'][e]['G'] for e in range(2)]
   assert all(math.isclose(d[e],ar['panels'][e]['V']-br['panels'][e]['V'],abs_tol=1e-12) for e in range(2))
   assert math.isclose(mean(d),ar['G']-br['G'],abs_tol=1e-12)
   cells.extend(d);per_repeat.append(mean(d))
  per_seed=[mean(cells[2*rr+e] for rr in range(3)) for e in range(2)]
  task_rows=[]
  assert set(tasks[a])==set(tasks[b]) and len(tasks[a])==32
  for t in tasks[a]:
   av=tasks[a][t]['six_paired_cells']['G'];bv=tasks[b][t]['six_paired_cells']['G'];v=[x-y for x,y in zip(av,bv)]
   task_rows.append({'task_id':t,'stratum':tasks[a][t]['stratum'],'mean_success_contrast_A_minus_B':mean(v)})
  d=mean(per_repeat)
  assert math.isclose(d,groups[a]['G']-groups[b]['G'],abs_tol=1e-12)
  assert math.isclose(d,mean(x['mean_success_contrast_A_minus_B'] for x in task_rows),abs_tol=1e-12)
  pos=all(x>1e-12 for x in per_repeat);neg=all(x<-1e-12 for x in per_repeat)
  rows.append({'source_A':a,'source_B':b,'both_original_heldout':a in held and b in held,'mean_G_contrast_A_minus_B':d,'same_training_repeat_contrasts_each_mean_of2eval_seeds':per_repeat,'training_repeat_contrast_range':[min(per_repeat),max(per_repeat)],'same_eval_seed_contrasts_each_mean_of3training_repeats':per_seed,'six_paired_repeat_seed_contrasts':cells,'strict_same_direction_all3training_repeats':pos or neg,'same_mean_direction_both_eval_seeds':all(x>1e-12 for x in per_seed) if d>1e-12 else all(x<-1e-12 for x in per_seed) if d<-1e-12 else False,'per_task_contrasts':task_rows})
 def summary(selected):return {'fixed_source_pairs':len(selected),'strict_same_direction_all3_training_repeats':sum(x['strict_same_direction_all3training_repeats'] for x in selected),'same_mean_direction_both_eval_seeds':sum(x['same_mean_direction_both_eval_seeds'] for x in selected),'both_descriptive_conditions':sum(x['strict_same_direction_all3training_repeats'] and x['same_mean_direction_both_eval_seeds'] for x in selected),'mean_contrast_zero':sum(abs(x['mean_G_contrast_A_minus_B'])<=1e-12 for x in selected)}
 return {'kind':'post_completion_matched_source_contrasts','computed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in [rp,dp]],'analysis_code_sha256':sha(Path(__file__)),'all16_source_summary':summary(rows),'four_original_heldout_summary':summary([x for x in rows if x['both_original_heldout']]),'source_pairs':rows,'original_five_model_RMSE_percentage_points':{m:100*math.sqrt(v) for m,v in report['prediction_results']['primary']['MSE'].items()},'descriptive_scope':'All120 source pairs retained, including all6 heldout pairs. For each pair compare source success at the same training-repeat index and evaluation seed, then average exactly as the original source labels. Common andUniform baselines cancel; G andV contrasts are algebraically identical. The3 repeat contrasts and2 seed means are dependent descriptive views, not independent population replication. Pairs share sources,32 tasks,ancestors and controls; counts are not120 independent tests or a discovered sampling policy. No significance test,confidence interval,new primary criterion,refitting,source deletion or additional seed.','cross_level_clarification':'Single-branch maximum eval-seed gap25pp and four-source meanG span3.6458pp are different aggregation levels. Their comparison alone cannot establish that source differences are unmeasurable. Matched contrasts report the observed source differences and their repeat/seed dependence without such an inference.','validation_rewards_used':False,'new_scientific_tokens':0,'new_gpu_jobs':0,'goal_complete':False}

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();v=describe(a.root)
 with a.output.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v[k] for k in ['all16_source_summary','four_original_heldout_summary','original_five_model_RMSE_percentage_points']},indent=2))
