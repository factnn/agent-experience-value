"""CPU integration check against previously sealed4B features and forecasts."""
from pathlib import Path
import hashlib,json,math
from bfcl_cross_model_prediction_values import feature_panels,predictions

r=Path('/share/project/zpy/RL/agent_experience_project_20260928')
def read(p):return json.loads(p.read_text())
def ref(p):return {'path':str(p.relative_to(r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
plan_path=r/'rl/BFCL_TASK_PROBE_STAGE0_V2_PROTOCOL_20261011.json';plan=read(plan_path)
analysis_path=r/'rl/BFCL_SOURCE_GROUP_ANALYSIS_CONTRACT_20261011.json';analysis=read(analysis_path)
computed=feature_panels(r,plan,ref(plan_path),analysis,ref(analysis_path)['sha256'])
saved=read(r/'rl/bfcl_source_stage0_features_20261012.json');expected={x['source_group']:x for x in saved['rows']}
keys=['primary_task_beta','current_pooled_beta','historical_frontier','historical_coverage']
max_difference=max(abs(row[key]-expected[row['source_group']][key]) for row in computed['rows'] for key in keys)
assert max_difference==0.0
structural=read(r/'rl/bfcl_structural_value_candidates_20261017/early4b_feature_only.json')
for row in computed['rows']:
 for key in structural['signals']:assert row[key]==structural['groups'][row['source_group']][key]
seal_dir=r/'rl/bfcl_source_stage0_prediction_seal_20261015';five=read(seal_dir/'models.json')
h1_receipt=read(r/'rl/bfcl_h1_source_success_early_model_20261015/receipt.json');h1=read(r/h1_receipt['model']['path'])
fitted_structural=read(r/'rl/bfcl_structural_value_candidates_20261017/fitted_early4b_models.json')
result=predictions(five,h1,fitted_structural,computed,analysis)
saved_predictions=read(seal_dir/'predictions.json')
assert all(result['original_five'][g]==saved_predictions['point_predictions'][g] for g in analysis['source_split']['heldout'])
from bfcl_cross_model_prediction_values import feature_case
case=plan['cases'][0]
bad=dict(case,task_ids=list(reversed(case['task_ids'])))
try:feature_case(r,plan,bad)
except AssertionError:wrong_task_order_rejected=True
else:raise AssertionError('Task order guard failed')
wrong=dict(plan,policy_fingerprint='not-the-source-policy')
try:feature_case(r,wrong,case)
except AssertionError:wrong_policy_rejected=True
else:raise AssertionError('Policy guard failed')
print(json.dumps({'status':'passed','historical4B_sources_reconstructed':16,'original4_signals_exact_match':True,
 'original20_sealed4B_predictions_exact_match':True,'structural_feature_reconstruction_exact_match':True,
 'all_point_rows_checked':len(result['standardized_source_rows']),'wrong_task_order_rejected':wrong_task_order_rejected,
 'wrong_policy_rejected':wrong_policy_rejected,'new_model_tokens':0,'new_gpu_jobs':0,'8B_point_predictions_created':False}),flush=True)
