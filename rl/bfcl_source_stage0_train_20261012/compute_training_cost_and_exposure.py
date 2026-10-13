import pathlib,json,statistics,collections,datetime,hashlib
r=pathlib.Path(__file__).resolve().parents[2]
output=pathlib.Path(__file__).with_name('training_cost_and_exposure.json')
if output.exists():
 raise FileExistsError('Do not overwrite registered training cost evidence')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def reference(path):
 return {'path':str(path.relative_to(r)),'sha256':digest(path)}
read=lambda p:json.loads(p.read_text())
master=read(r/'rl/BFCL_SOURCE_STAGE0_EXECUTION_PROTOCOL_20261012.json')
rows=[]
for e in master['jobs']:
 assert digest(r/e['path'])==e['sha256']
 j=read(r/e['path']);o=r/j['output'];s=read(o/'summary.json');a=read(o/'audit.json');t=read(o/'supervisor.json');run=read(o/'run.json')
 assert s['status']=='complete' and a['status']=='passed' and a['job_sha256']==e['sha256'] and s['branch_sampled_tokens']==a['cost']['new_model_tokens']
 g=[json.loads(x) for x in (o/'allocation.jsonl').read_text().splitlines()]
 m=[json.loads(x) for x in (o/'metrics.jsonl').read_text().splitlines()]
 u=[json.loads(x) for x in (o/'updates.jsonl').read_text().splitlines()]
 fresh=[x['step'] for x in m if 'grad_norm' in x and x['grad_norm']>0]
 changed=[x['step'] for x in u if x['parameters_changed']]
 assert fresh==a['fresh_nonzero_gradient_steps'] and changed==a['changed_parameter_steps']
 assert sum(x['sampled_tokens'] for x in g)==s['branch_sampled_tokens']
 case=e['name'];rep=int(case.rsplit('_r',1)[1]);counts=collections.Counter(x['task_id'] for x in g)
 stops=collections.Counter(str(x.get('stop_reason')) for x in [json.loads(x) for x in (o/'rollouts.jsonl').read_text().splitlines()])
 rows.append({'case':case,'repeat':rep,'role':j['role'],'group':j.get('source_group'),'new_tokens':a['cost']['new_model_tokens'],'input_positions':a['cost']['input_token_positions_excluding_padding'],'external_tokens':a['cost']['external_tokens'],'rollouts':a['cost']['rollouts'],'updates':a['optimizer_steps'],'fresh_steps':len(fresh),'changed_steps':len(changed),'fresh_ids':fresh,'changed_ids':changed,'nominal':j['token_budget'],'overshoot':s['branch_sampled_tokens']-j['token_budget'],'before_last':sum(x['sampled_tokens'] for x in g[:-1]),'last':g[-1]['sampled_tokens'],'pool':j['task_pool'],'task_groups':dict(counts),'distinct_tasks':len(counts),'stops':dict(stops),'summary_wall_seconds':s['wall_seconds'],'generation_seconds':s['generation_seconds'],'supervisor_wall_seconds':t['wall_seconds'],'gpu':run['gpu'],'returncode':t['returncode'],'hard_timeout':t['hard_timeout'],'inherited':s['inherited_acquisition_tokens'],
 'original_job':e,'sources':{name:reference(o/name) for name in ['summary.json','audit.json','allocation.jsonl','groups.jsonl','metrics.jsonl','updates.jsonl','rollouts.jsonl','supervisor.json','run.json']}})
keys=['new_tokens','input_positions','external_tokens','rollouts','updates','fresh_steps','changed_steps','nominal','overshoot','summary_wall_seconds','generation_seconds','supervisor_wall_seconds']
def totals(xs):
 return {k:sum(x[k] for x in xs) for k in keys}|{'branches':len(xs),'zero_fresh_branches':sum(x['fresh_steps']==0 for x in xs),'zero_parameter_change_branches':sum(x['changed_steps']==0 for x in xs),'stops':dict(sum((collections.Counter(x['stops']) for x in xs),collections.Counter()))}
overs=[x['overshoot'] for x in rows];qs=statistics.quantiles(overs,n=4,method='inclusive')
bygroup=[]
for group in sorted({x['group'] for x in rows if x['role']=='source_group'}):
 xs=sorted([x for x in rows if x['group']==group],key=lambda x:x['repeat']);pool=set(xs[0]['pool']);assert all(set(x['pool'])==pool for x in xs)
 visits=[set(x['task_groups']) for x in xs]
 bygroup.append({'group':group,'pool_size':len(pool),'distinct_by_repeat':[len(v) for v in visits],'union_three':len(set.union(*visits)),'common_three':len(set.intersection(*visits)),'unvisited_three':sorted(pool-set.union(*visits)),'new_tokens_by_repeat':[x['new_tokens'] for x in xs],'fresh_by_repeat':[x['fresh_steps'] for x in xs],'changed_by_repeat':[x['changed_steps'] for x in xs]})
uniform=[x for x in rows if x['role']=='uniform_control']
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Descriptive recomputation from saved complete training summaries, audits, raw allocation/metrics/update/rollout records. Not full finalizer or transfer analysis.',
 'totals':totals(rows),'by_repeat':{str(rep):totals([x for x in rows if x['repeat']==rep]) for rep in range(3)},'by_role':{role:totals([x for x in rows if x['role']==role]) for role in sorted({x['role'] for x in rows})},
 'overshoot_distribution':{'min':min(overs),'q1':qs[0],'median':statistics.median(overs),'q3':qs[2],'max':max(overs),'mean':statistics.mean(overs),'all_last_group_started_below_nominal':all(x['before_last']<x['nominal'] for x in rows),'max_cases':[x['case'] for x in rows if x['overshoot']==max(overs)],'min_cases':[x['case'] for x in rows if x['overshoot']==min(overs)],'over_10pct':sum(x['overshoot']>0.1*x['nominal'] for x in rows),'over_25pct':sum(x['overshoot']>0.25*x['nominal'] for x in rows)},
 'zero_fresh_cases':[x['case'] for x in rows if x['fresh_steps']==0], 'zero_changed_cases':[x['case'] for x in rows if x['changed_steps']==0],
 'all_fresh_raw_metrics_match_saved_audits':True,'all_changed_raw_records_match_saved_audits':True,
 'all_runner_success':all(x['returncode']==0 and not x['hard_timeout'] for x in rows),'inherited_values':sorted({x['inherited'] for x in rows}),
 'group_exposure':bygroup,'uniform_exposure':[{'case':x['case'],'pool_size':len(x['pool']),'distinct_tasks':x['distinct_tasks'],'task_groups':x['task_groups'],'new_tokens':x['new_tokens'],'updates':x['updates'],'fresh_steps':x['fresh_steps'],'changed_steps':x['changed_steps']} for x in uniform],
 'gpu_supervisor_wall_seconds':{gpu:sum(x['supervisor_wall_seconds'] for x in rows if x['gpu']==gpu) for gpu in sorted({x['gpu'] for x in rows})},
 'branch_extrema':{k:{'min':min(x[k] for x in rows),'max':max(x[k] for x in rows),'median':statistics.median(x[k] for x in rows)} for k in ['new_tokens','updates','fresh_steps','changed_steps','supervisor_wall_seconds']}}
certificate_path=r/master['output_root']/'audit_status.json'
certificate=read(certificate_path)
assert certificate['status']=='passed' and certificate['complete_jobs']==51
assert certificate['new_sampled_tokens']==result['totals']['new_tokens']
assert result['totals']['new_tokens']==3708360
assert result['totals']['input_positions']==68566949
assert result['totals']['external_tokens']==494940
assert result['totals']['rollouts']==1404
assert result['totals']['updates']==351
assert result['totals']['fresh_steps']==143
assert result['totals']['changed_steps']==278
assert result['inherited_values']==[90122]
result.update(schema_version=1,complete_training_certificate=reference(certificate_path),
 original_scientific_protocol=reference(r/'rl/BFCL_SOURCE_STAGE0_EXECUTION_PROTOCOL_20261012.json'),
 computation_script=reference(pathlib.Path(__file__).resolve()),jobs=rows,
 accounting={'branch_new_model_tokens':3708360,'shared_ancestor_acquisition_tokens_counted_once':90122,
  'branch_plus_one_shared_ancestor_tokens':3798482,
  'excluded_from_this_account':['other opportunity probes','other shared measurements','evaluation panels','failed or interrupted scientific attempts outside these 51 completed identities','zero-child administrative admission failures'],
  'failure_accounting':'All 51 final scientific runner receipts returned zero without hard timeout. Historical control/admission failures remain in separate original/V2/V3/V4 execution evidence; their cost is not inferred from these successful branches.',
  'time_units':'seconds summed across individual workers',
  'time_interpretation':'Supervisor/summary/generation durations are recorded wall times, not CUDA-active time or elapsed calendar makespan. This file does not infer actual peak concurrency from GPU authorization.',
  'input_positions':'Sum of non-padding input token positions recorded by each completed branch; distinct from sampled new-model tokens.',
  'external_tokens':'Environment/external tokens excluded from the model loss and reported separately.',
  'fresh_gradient_interpretation':'Logged finite nonzero gradient-step identities cross-checked with saved audits; gradients were not independently recomputed on GPU.',
  'dependence':'Rollouts within a task group, repeated task exposure, training repeats, source groups sharing primitives and the common ancestor are not independent scientific conditions.',
  'scientific_boundary':'No transfer labels, G, V, predictor performance, or scientific value conclusions are computed.'})
with output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(output.relative_to(r)),'sha256':digest(output),'script':result['computation_script'],
 'complete_training_certificate':result['complete_training_certificate'],'totals':result['totals'],'jobs':len(rows)},indent=2))
