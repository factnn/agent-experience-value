"""Analytical CPU checks of paired E369 statistics; no real target labels."""
import json
import datetime
from bfcl_qwen8_forward15_analysis import registry,labels,error_scores,paired_resampling
groups=['cpu_A','cpu_B','cpu_C','cpu_D'];tasks=['cpu_task_'+str(i) for i in range(32)]
evaluation={'registry':{t:'stratum_'+str(i//8) for i,t in enumerate(tasks)}}
analysis={'source_split':{'heldout':groups,'all_groups':groups},'target_panel':evaluation}
jobs=[]
for repeat in range(3):
 for gi,group in enumerate([None,*groups]):
  jobs.append({'case_id':'q8_s0_uniform_r'+str(repeat) if group is None else 'cpu_group_'+str(gi)+'_r'+str(repeat),'stage':0,'source_group':group,'training_repeat':repeat})
master={'stage':0,'training_repeats':3,'jobs':[{'name':j['case_id']} for j in jobs]}
reg=registry(master,jobs,analysis);outcomes={};panels=[]
for case,info in reg.items():
 for e in range(2):
  rows=[]
  for i,t in enumerate(tasks):
   group=info['source_group'];reward=1 if group==groups[0] else 0 if group==groups[1] else i%2
   rows.append({'task_id':t,'stratum':evaluation['registry'][t],'rewards':[reward]})
  outcomes[case,e]=rows
  strata={s:sum(x['rewards'][0] for x in rows if x['stratum']==s)/8 for s in set(evaluation['registry'].values())}
  panels.append({'status':'passed','policy_case':case,'evaluation_repeat':e,'role':info['role'],'macro_success':sum(strata.values())/4,'stratum_success':strata})
measured=labels(panels,reg,analysis)
assert list(measured['G_labels'].values())==[.5,-.5,0.,0.]
points={g:{'original_five/fit_mean':.1,'zero_gain_reference':0.,'cpu_negative_slope':v} for g,v in zip(groups,[.5,-.5,0.,0.])}
scores=error_scores(points,measured['G_labels'])
assert scores['cpu_negative_slope']['MSE']==0.
assert abs(scores['original_five/fit_mean']['MSE']-.135)<1e-12
assert abs(scores['original_five/fit_mean']['squared_mean_signed_error']-.01)<1e-12
assert abs(scores['original_five/fit_mean']['centered_error_MSE']-.125)<1e-12
spec={'replicates':10000,'seed':2026101701,'source_groups_resampled':False,'frozen_predictions_refitted':False}
resampled=paired_resampling(outcomes,reg,evaluation,measured,points,spec)
assert resampled['groups'][groups[2]]['G']=={'q025':0.,'median':0.,'q975':0.}
assert resampled['groups'][groups[2]]['V']=={'q025':0.,'median':0.,'q975':0.}
assert resampled['paired_source_contrasts'][0]['paired_G_and_V_difference']=={'q025':1.,'median':1.,'q975':1.}
assert resampled==paired_resampling(outcomes,reg,evaluation,measured,points,spec)
rejections=[]
def reject(name,fn):
 try:fn()
 except (ValueError,KeyError):rejections.append(name)
 else:raise AssertionError('Accepted invalid '+name)
reject('missing_policy_seed_panel',lambda:paired_resampling(dict(list(outcomes.items())[1:]),reg,evaluation,measured,points,spec))
bad=json.loads(json.dumps(points));bad[groups[0]]['cpu_negative_slope']=float('nan')
reject('nonfinite_point',lambda:error_scores(bad,measured['G_labels']))
reject('missing_tested_source',lambda:error_scores({g:v for g,v in points.items() if g!=groups[0]},measured['G_labels']))
bad_spec=dict(spec,frozen_predictions_refitted=True)
reject('refitted_bootstrap_forecast',lambda:paired_resampling(outcomes,reg,evaluation,measured,points,bad_spec))
bad_measured=json.loads(json.dumps(measured));bad_measured['G_labels'][groups[0]]+=.01
reject('raw_task_mean_mismatch',lambda:paired_resampling(outcomes,reg,evaluation,bad_measured,points,spec))
receipt={'status':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'simulated_in_memory_CPU_outcomes_only':True,'checks':['analytical_paired_G_V','unmodified_negative_slope_points','MSE_bias_squared_and_centered_decomposition','identical_policy_zero_gain_every_resample','shared_task_train_eval_pairing','constant_source_contrast_every_resample','fixed_seed_exact_replay'],'rejections':rejections,'real_native8b_target_labels_accessed':False,'new_model_tokens':0,'gpu_jobs':0,'admission_effect':None}
print(json.dumps(receipt))
