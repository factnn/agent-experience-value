"""Fit the separately registered H1 early-source predictor once; no late or validation labels."""
import argparse,datetime,hashlib,json,math
from pathlib import Path
from statistics import mean

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_success(rows):
 if not rows or len({r['task_id'] for r in rows})!=len(rows):raise ValueError('Require distinct full-source feature tasks')
 if any(r['phase']!='feature' or len(r['rewards'])!=4 or any(v not in (0,1) for v in r['rewards']) for r in rows):raise ValueError('Only original4 feature outcomes per task')
 return mean(mean(r['rewards']) for r in rows)

def predict(model,x):
 return model['intercept']+model['slope']*((x-model['early_fit_mean'])/model['early_fit_population_std'] if model['early_fit_population_std'] else 0.0)

def fit(root,protocol,expected_sha,output):
 def pinned(ref):
  p=root/ref['path']
  if digest(p)!=ref['sha256']:raise ValueError('Changed H1 input: '+str(p))
  return json.loads(p.read_text())
 p=pinned({'path':str(protocol),'sha256':expected_sha})
 if p['kind']!='separate_H1_source_success_auxiliary_forward_protocol' or p['fitting']['lambda']!=1.0:raise ValueError('Wrong auxiliary protocol')
 inventory=pinned(p['early_feature_inventory']);labels=pinned(p['early_fit_labels']);analysis=pinned(p['analysis_contract'])
 assert inventory['status']=='passed' and inventory['local_validation_rewards_read'] is False
 assert labels['phase']=='fit' and labels['panel_count']==80 and p['fit_groups']==analysis['source_split']['fit']
 for name,h in inventory['source_sha256'].items():
  if digest(root/name)!=h:raise ValueError('Original feature inventory changed: '+name)
 x_by_group={}
 for row in inventory['rows']:
  g=row['source_group'];candidates=[path for path in inventory['source_sha256'] if path.endswith('/feature.jsonl')]
  matches=[]
  for path in candidates:
   rows=[json.loads(line) for line in (root/path).read_text().splitlines()]
   if rows[0]['source_group']==g:matches.append(rows)
  assert len(matches)==1 and len(matches[0])==row['task_count']
  value=source_success(matches[0]);assert math.isclose(value,row['candidate_success_rate'],abs_tol=1e-15);x_by_group[g]=value
 groups=p['fit_groups'];x=[x_by_group[g] for g in groups];y=[labels['groups'][g]['G'] for g in groups]
 location=mean(x);scale=math.sqrt(mean((v-location)**2 for v in x));intercept=mean(y)
 z=[(v-location)/scale for v in x] if scale else [0.0]*len(x)
 slope=sum(a*(b-intercept) for a,b in zip(z,y))/(sum(a*a for a in z)+1.0)
 model={'intercept':intercept,'slope':slope,'early_fit_mean':location,'early_fit_population_std':scale,'lambda':1.0,'fit_groups':groups,'future_test_groups':p['future_test_groups'],'feature':'equal-task feature source success rate','target':'G','sign_constraint':False,'direction':'positive' if slope>0 else 'negative' if slope<0 else 'zero','feature_selection':False,'prediction_clipping':False,'standardization_refitted_on_late_features':False}
 value={'kind':'early_H1_source_success_model','fitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':{'path':str(protocol),'sha256':expected_sha},'source_inventory':p['early_feature_inventory'],'fit_labels':p['early_fit_labels'],'model':model,'source_features':x_by_group,'original_stage0_feature_predictions':{g:predict(model,v) for g,v in x_by_group.items()},'stage0_auxiliary_scope':'Exploratory after all stage0 labels were exposed; no second confirmatory test and no changes to47. Only12 originalfitG were used in coefficient estimation.','stage0_heldout_labels_used_in_coefficient_estimation':False,'late_target_labels_used':False,'local_validation_rewards_used':False,'new_scientific_tokens':0,'goal_complete':False,'analysis_code_sha256':digest(Path(__file__))}
 output.mkdir(exist_ok=False)
 with (output/'early_model.json').open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
 receipt={'sealed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':{'path':str((output/'early_model.json').relative_to(root)),'sha256':digest(output/'early_model.json')},'protocol':value['protocol'],'late_actual_point_predictions_not_yet_available':True,'future_requirement':'Apply this exact early model and its early normalization to actual late feature panels; publish the4 point forecasts before any stage1 target labels.','new_scientific_tokens':0}
 with (output/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 return value,receipt

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 value,receipt=fit(a.root,a.protocol,a.protocol_sha256,a.output)
 print(json.dumps({'model':value['model'],'receipt':receipt,'new_scientific_tokens':0,'late_labels_used':False},ensure_ascii=False,indent=2))
