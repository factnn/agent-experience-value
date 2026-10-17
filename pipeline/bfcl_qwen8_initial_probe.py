"""Model-specific bindings for the first Qwen3-8B common state and local probe."""
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean
import torch
from bfcl_allocation import CombinationAllocator
from report_bfcl_intervention import audit_sampling, lines, read
from report_bfcl_task_probe import audit_case
from rl_learning_state import tensor_hash
from supervise_bfcl_stage_fork import ROOT, write

REVISION = '26028140be3ee69b82b1d1450179ab71bb1121b9'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def ref(path):return {'path':str(path.relative_to(ROOT)),'sha256':digest(path)}
def pinned(value):
    p=ROOT/value['path'];assert digest(p)==value['sha256'],value['path'];return read(p)
def exclusive(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def load_master(path,sha):
    assert digest(path)==sha
    p=read(path);assert p['kind']=='qwen3_8b_initial_common_and_full_source_probe'
    assert p['model_revision']==REVISION and p['allowed_gpus']==list(range(8)) and p['max_simultaneous_gpus']==8
    assert p['automatic_retry'] is False and p['target_evaluation_allowed'] is False and p['reserved21_allowed'] is False
    assert p['paid_api_calls']==0 and p['new_model_token_ceiling']==5963776
    assert p['common_hard_seconds']==5400 and p['maximum_sum_worker_seconds']==432600
    source=pinned(p['source_protocol']);assert source['base_revision']==REVISION
    split=read(ROOT/'rl/bfcl_research_split_candidate/manifest.json')
    train={x['id'] for x in split['train']};tasks=[];seeds=[]
    assert len(p['cases'])==16 and len({c['source_group'] for c in p['cases']})==16
    for c in p['cases']:
        assert c['task_ids']==sorted(t for t,g in source['training_registry'].items() if g==c['source_group'])
        assert c['token_ceiling']==len(c['task_ids'])*8*8192 and c['hard_seconds']==600+4800*len(c['task_ids'])
        tasks.extend(c['task_ids'])
        for phase in ['feature','validation']:
            assert len(c['seeds'][phase])==len(c['task_ids']);seeds.extend(c['seeds'][phase])
    assert len(tasks)==len(set(tasks))==87 and set(tasks)==set(source['training_registry'])<=train
    assert len(seeds)==len(set(seeds))==174 and all(0<=s<2**32 for s in seeds)
    assert set(source['common_probe_task_ids'])<=set(tasks) and len(source['common_probe_task_ids'])==8
    assert sum(c['token_ceiling'] for c in p['cases'])+262144==p['new_model_token_ceiling']
    assert sum(c['hard_seconds'] for c in p['cases'])+p['common_hard_seconds']==p['maximum_sum_worker_seconds']
    scheduled=[x for lane in p['gpu_lanes'].values() for x in lane]
    assert len(scheduled)==16 and set(scheduled)=={c['name'] for c in p['cases']}
    for name,sha in p['input_sha256'].items():assert digest(ROOT/name)==sha,name
    cap=pinned(p['resource_authorization']);assert cap['max_simultaneous_gpus_for_subsequent_batches']==8
    return p,source


def verify_model(p):
    v=pinned(p['model_verification']);assert v['immutable_commit']==REVISION
    path=Path(v['destination']);assert path.name=='Qwen3-8B'
    for row in v['files']:
        f=path/row['path'];assert f.stat().st_size==row['bytes'] and digest(f)==row['sha256'],row['path']
    return path


def audit_common(p):
    out=ROOT/p['common_output'];summary=read(out/'summary.json');cost=audit_sampling(out)
    assert summary['status']=='complete' and summary['optimizer_updates']==0
    assert cost['new_model_tokens']==summary['sampled_tokens']<=262144 and cost['rollouts']==32
    source=pinned(p['source_protocol']);manifest=read(out/'components/manifest.json')
    assert digest(out/'components/learning_state.pt')==manifest['payload_sha256']
    payload=torch.load(out/'components/learning_state.pt',map_location='cpu',weights_only=False)
    assert payload['progress']==manifest['progress'] and payload['progress']['global_step']==payload['progress']['micro_step']==0
    assert not payload['optimizer']['state'] and payload['optimizer_type']=='AdamW'
    assert tensor_hash(payload['trainable'].items())==manifest['trainable_sha256']
    assert payload['provenance']==manifest['provenance'] and payload['provenance']['base_revision']==REVISION
    assert payload['provenance']['intervention_protocol_sha256']==p['source_protocol']['sha256']
    allocator=CombinationAllocator(source['training_registry'],seed=source['seeds']['initialization'])
    history=lines(out/'probe_history.jsonl');assert len(history)==8
    for i,row in enumerate(history):
        key=source['common_probe_task_ids'][i];choice=allocator.choose(forced=key)
        assert row['task_id']==key and row['seed']==source['seeds']['common_probe']+100*i
        for k,v in choice.items():assert row[k]==v,k
        allocator.observe(key,row['rewards'],row['sampled_tokens'],f'probe:{i}')
    assert allocator.state_dict()==payload['allocator']==read(out/'allocation_state.json')
    assert allocator.total_sampled_tokens==manifest['cumulative_acquisition_tokens']==cost['new_model_tokens']
    run=read(out/'run.json');assert run['model_revision']==REVISION and run['target_evaluation'] is False
    assert run['model_verification']==p['model_verification'] and run['common_optimizer_updates']==0
    terminal=read(out/'supervisor.json');assert terminal['returncode']==0 and terminal['hard_timeout'] is False
    result={'status':'passed','cost':cost,'common_optimizer_updates':0,'common_history_groups':8,
        'policy_fingerprint':summary['policy_fingerprint'],'learning_state':ref(out/'components/learning_state.pt'),
        'manifest':ref(out/'components/manifest.json'),'adapter_config':ref(out/'adapter/adapter_config.json'),
        'adapter_weights':ref(out/'adapter/adapter_model.safetensors'),'allocation_history_replay':'exact',
        'optimizer_initial_state_entries':0,'target_evaluation':False,'reserved21_accessed':False,
        'scope':'Native8B step0, its own charged common history and empty optimizer; not a4B state or efficacy claim.'}
    exclusive(out/'audit.json',result)
    return result,allocator.signals()


def bind_probe(p,master_ref,common_audit,signals):
    root=ROOT/p['binding_root'];root.mkdir(exist_ok=False)
    binding={'master':master_ref,'common_audit':ref(ROOT/p['common_output']/'audit.json'),
        'model_verification':p['model_verification'],'model_revision':REVISION,
        'common_policy':p['common_output'],'policy_fingerprint':common_audit['policy_fingerprint'],
        'learning_state':common_audit['learning_state'],'manifest':common_audit['manifest'],
        'adapter_config':common_audit['adapter_config'],'adapter_weights':common_audit['adapter_weights'],
        'history_signals':signals,'inherited_acquisition_tokens':common_audit['cost']['new_model_tokens'],
        'scope':'Deterministic binding to completed native8B common state; no target label consulted.'}
    exclusive(root/'common_binding.json',binding)
    cases=copy.deepcopy(p['cases'])
    for case in cases:case['historical_signal']=signals[case['source_group']]
    plan={'kind':'qwen3_8b_stage0_local_opportunity_probe_bound','stage':0,'source_protocol':p['source_protocol']['path'],
        'output_root':p['probe_output'],'policy_path':p['common_output'],'policy_fingerprint':binding['policy_fingerprint'],
        'inherited_acquisition_tokens':binding['inherited_acquisition_tokens'],'input_sha256':p['input_sha256'],
        'cases':cases,'model_revision':REVISION,'native_model_path':str(Path(pinned(p['model_verification'])['destination'])),
        'master':master_ref,'common_binding':ref(root/'common_binding.json'),'gpu_lanes':p['gpu_lanes'],
        'allowed_gpus':p['allowed_gpus'],'max_simultaneous_gpus':8,'automatic_retry':False,'new_token_ceiling':5701632,
        'reserved21_allowed':False,'target_evaluation_allowed':False}
    exclusive(root/'probe_plan.json',plan)
    return plan,ref(root/'probe_plan.json')


def check_binding(plan):
    b=pinned(plan['common_binding']);assert b['model_revision']==plan['model_revision']==REVISION
    assert b['policy_fingerprint']==plan['policy_fingerprint'] and b['common_policy']==plan['policy_path']
    for key in ['learning_state','manifest','adapter_config','adapter_weights']:
        value=b[key];assert digest(ROOT/value['path'])==value['sha256'],key
    assert pinned(b['common_audit'])['status']=='passed'
    for case in plan['cases']:assert case['historical_signal']==b['history_signals'][case['source_group']]
    return b


def audit_probe(plan,case,plan_sha):
    check_binding(plan);row=audit_case(plan,case,plan_sha)
    run=read(ROOT/plan['output_root']/case['name']/'run.json')
    assert run['model_revision']==REVISION and run['native_model_path']==plan['native_model_path']
    assert run['native_master']==plan['master'] and run['native_common_binding']==plan['common_binding']
    assert run['target_evaluation'] is False and run['reserved21_accessed'] is False
    return row


def complete_probe(plan,plan_ref):
    rows=[read(ROOT/plan['output_root']/c['name']/'audit.json') for c in plan['cases']]
    assert len(rows)==16 and all(row['case']==case['name'] for row,case in zip(rows,plan['cases']))
    result={'status':'passed','model_revision':REVISION,'probe_plan':plan_ref,'complete_cases':16,
        'rows':rows,'new_sampled_tokens':sum(x['cost']['new_model_tokens'] for x in rows),
        'macro_Brier':{key:mean(x['scores']['Brier'][key] for x in rows) for key in rows[0]['scores']['Brier']},
        'primary_macro_difference':mean(x['scores']['primary_difference'] for x in rows),
        'common_acquisition_cost_counted_once_elsewhere':plan['inherited_acquisition_tokens'],
        'optimizer_updates':0,'reserved21_accessed':False,'target_evaluation':False,'goal_complete':False,
        'scope':'Native8B local opportunity measurements only; no transfer, model superiority or broad mechanism claim.'}
    exclusive(ROOT/plan['output_root']/'audit_status.json',result);return result
