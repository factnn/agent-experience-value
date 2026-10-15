"""Freeze the full fixed-parent stage1 intervention design, without launching GPU work."""
import copy,json,time
from pathlib import Path
from bfcl_source_eval_contract import pinned,digest
from bfcl_source_group_contract import validate_intervention

ALLOCATOR_OFFSET=100000000
ROLLOUT_OFFSET=500000000
EVALUATION_BASES=[2500000000,2510000000]

def make_jobs(root,early,probe,binding):
 jobs=[]
 for entry in early['jobs']:
  original=pinned(root,entry);job=copy.deepcopy(original)
  job.update(case_id=original['case_id'].replace('s0_','s1_',1),stage=1,
    source_checkpoint=binding['policy_path'],source_payload_sha256=binding['learning_state']['sha256'],source_step=binding['source_step'],
    source_protocol=original['algorithm_protocol'],allow_legacy_zero_optimizer_source=False,
    allocator_seed=original['allocator_seed']+ALLOCATOR_OFFSET,rollout_seed_base=original['rollout_seed_base']+ROLLOUT_OFFSET)
  job['output']='rl/bfcl_source_stage1_train_20261015/'+job['case_id']
  job['stage0_corresponding_job']=entry
  job['parent_binding']=probe['parent_binding']
  for key in ('summary','audit','manifest','learning_state','adapter_config','adapter_weights'):
   ref=binding[key];job['input_sha256'][ref['path']]=ref['sha256']
  jobs.append(job)
 return jobs

def validate_jobs(root,early,probe,binding,jobs):
 if len(jobs)!=51 or len({j['case_id'] for j in jobs})!=51:raise ValueError('Require all51 distinct late branches')
 expected=make_jobs(root,early,probe,binding)
 if jobs!=expected:raise ValueError('Late branches changed fixed source,role,pool,seed,budget or recipe')
 if binding['parent_case']!=early['prespecified_late_stage_parent'] or binding['source_step']!=7:raise ValueError('Wrong fixed late parent')
 if probe['stage']!=1 or probe['parent_binding']['sha256']!=digest(root/probe['parent_binding']['path']):raise ValueError('Wrong stage1 probe lineage')
 old_rollout=set();new_rollout=set();old_alloc=set();new_alloc=set();cells=set()
 for old_ref,job in zip(early['jobs'],jobs):
  old=pinned(root,old_ref);algorithm=pinned(root,job['algorithm_protocol'])
  validate_intervention(job,algorithm['training_registry'])
  if job['source_protocol']!=job['algorithm_protocol'] or job['allow_legacy_zero_optimizer_source'] is not False:raise ValueError('Nonzero parent requires the identical corrected recipe')
  if (job['token_budget'],job['max_updates'],job['soft_seconds'],job['hard_seconds'])!=(65536,128,12600,14400):raise ValueError('Dose or stopping design changed')
  cells.add((job['source_group'],job['training_repeat']))
  old_alloc.add(old['allocator_seed']);new_alloc.add(job['allocator_seed'])
  old_rollout.update(old['rollout_seed_base']+100*u for u in range(old['source_step'],old['source_step']+old['max_updates']))
  seeds={job['rollout_seed_base']+100*u for u in range(job['source_step'],job['source_step']+job['max_updates'])}
  if len(seeds)!=128 or new_rollout.intersection(seeds) or any(not 0<=s<2**32 for s in seeds):raise ValueError('Late rollout seeds overlap or are invalid')
  new_rollout.update(seeds)
 if len(cells)!=51 or len(new_alloc)!=51 or new_alloc.intersection(old_alloc) or new_rollout.intersection(old_rollout):raise ValueError('Missing condition or early/late seed reuse')
 if sum(j['token_budget'] for j in jobs)!=3342336 or sum(j['token_budget']+32768 for j in jobs)!=5013504:raise ValueError('Total training budget changed')
 if set(new_rollout).intersection(s for c in probe['cases'] for phase in ('feature','validation') for s in c['seeds'][phase]):raise ValueError('Rollout/probe seed collision')
 return {'jobs':51,'source_groups':16,'training_repeats':3,'uniform_controls':3,'nominal_new_training_tokens':3342336,'new_training_token_ceiling':5013504,'possible_rollout_seed_identities':len(new_rollout),'source_step':7}

def evaluation_design(early):
 evaluation=copy.deepcopy(early['evaluation'])
 evaluation['evaluation_seed_bases']=EVALUATION_BASES
 evaluation['task_seed_vectors']=[[base+i for i in range(len(evaluation['task_ids']))] for base in EVALUATION_BASES]
 evaluation['required_order']='Publish frozen early5 and separateH1 actual late forecasts before ALL late target panels; after complete51 audit/review run80fit, fit/seal/publish5within-late models, then24heldout; forward and within-stage scores separate.'
 evaluation['existing_old_recipe_eval_reuse']=False
 evaluation['stage0_uniform_parent_panel_reuse']=False
 assert not set(sum(evaluation['task_seed_vectors'],[])).intersection(sum(early['evaluation']['task_seed_vectors'],[]))
 return evaluation

def prepare(root,early_ref,probe_ref,registration,protocol_path,required_reviews):
 early=pinned(root,early_ref);probe=pinned(root,probe_ref);binding=pinned(root,probe['parent_binding'])
 jobs=make_jobs(root,early,probe,binding);checked=validate_jobs(root,early,probe,binding,jobs)
 if registration.exists() or protocol_path.exists():raise FileExistsError('Keep existing registered late jobs')
 registration.mkdir(exist_ok=False);entries=[]
 for job in jobs:
  p=registration/(job['case_id']+'.json')
  with p.open('x') as f:json.dump(job,f,ensure_ascii=False,indent=2);f.write('\n')
  entries.append({'name':job['case_id'],'path':str(p.relative_to(root)),'sha256':digest(p)})
 cap='experiments/SSH_FUTURE_GPU_CAP_V3_20261014.json'
 plan={'kind':'source_stage1_training_and_transfer_execution','stage':1,'registered_unix':time.time(),'stage0_execution_protocol':early_ref,'stage1_probe_protocol':probe_ref,'parent_binding':probe['parent_binding'],'stage_start':{'path':binding['policy_path'],'payload_sha256':binding['learning_state']['sha256'],'policy_fingerprint':binding['policy_fingerprint'],'source_step':7,'inherited_acquisition_tokens_not_recharged':160662},'analysis_contract':early['analysis_contract'],'jobs':entries,'output_root':'rl/bfcl_source_stage1_train_20261015','allowed_gpus':list(range(8)),'max_simultaneous_gpus':8,'resource_authorization':{'path':cap,'sha256':digest(root/cap)},'automatic_retry':False,'gpu_lanes':{str(g):[e['name'] for i,e in enumerate(entries) if i%8==g] for g in range(8)},'training_repeats':3,'source_groups':16,'training_jobs':51,'nominal_new_training_tokens':3342336,'new_training_token_ceiling':5013504,'evaluation':evaluation_design(early),'required_review_events':required_reviews,'input_sha256':early['input_sha256']|{ref['path']:ref['sha256'] for ref in [probe_ref,probe['parent_binding']]}|{binding[k]['path']:binding[k]['sha256'] for k in ('summary','audit','manifest','learning_state','adapter_config','adapter_weights')},'mandatory_launch_bindings':['Complete16 late probes with all original raw audits and execution-agent reviews.','Actual published frozen early5 and separateH1 late forecasts before new experience acquisition.','A separate current SSH training authority bound to these exact bytes,actual completion/forecast receipts,current stops andCAP_V3; actually idleGPU check each job.'],'actual_late_feature_binding':'The completed feature/forward receipts must be added as separate immutable admission/analysis bindings; no placeholder data and no in-place edit of this plan.','target_access':'Separate current evaluation authority, complete51 audit/review and published actual early-forward/H1 seals required for every late target panel including common/fit; additionally within-late5seal before heldout24.','failure_policy':early['failure_policy'],'stage_comparison':'Report G,V,all5 forward versus frozen earlymean and all5within-late versus latefitmean separately; compare errors/sourceG across stages with true shared ancestry/targets and dose,not opposite-significance labels.','budgets':{'late_probe_new_token_ceiling_separate':5701632,'training_new_token_ceiling':5013504,'fit80_new_token_ceiling':20971520,'heldout24_new_token_ceiling':6291456,'post_probe_training_and_evaluation_ceiling':32276480,'stage1_including_probe_new_token_ceiling':37978112,'ancestor_tokens_recharged':0,'failure_or_retry_budget':'No automatic science retries; partial cost remains within its originalattempt and recovery requires separate registration.','paid_api_calls':0},'scope':'Full predetermined fixed-learner stage contrast only. No new model/environment axis,ID22 or reserved21 evaluation authorized; cannot mark wholeGoal complete.'}
 plan['input_sha256']['pipeline/bfcl_stage1_matrix.py']=digest(root/'pipeline/bfcl_stage1_matrix.py')
 with protocol_path.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.write('\n')
 return plan,checked
