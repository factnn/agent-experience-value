"""Freeze newly generated reference evidence, then audit exact full-state pairing."""
import argparse
import hashlib
from pathlib import Path
import torch
from report_bfcl_intervention import ROOT,read,lines,audit_sampling
from report_bfcl_stage_fork import compare_rows
from bfcl_dose_checkpoint import assert_equal
from bfcl_paired_fork_contract import validate_pair_boundary
from supervise_bfcl_stage_fork import write


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(folder):
    m=read(folder/'manifest.json');assert digest(folder/'learning_state.pt')==m['payload_sha256']
    return torch.load(folder/'learning_state.pt',map_location='cpu',weights_only=False)


def check_case(plan,case,protocol_sha):
    out=ROOT/plan['output_root']/case;s=read(out/'summary.json');run=read(out/'run.json')
    assert s['status']=='complete' and s['case']==case
    assert run['protocol_sha256']==protocol_sha and run['case']==case
    assert run['deterministic_algorithms'] and run['cudnn_deterministic'] and not run['cudnn_benchmark']
    assert run['cublas_workspace_config']==':4096:8' and run['historical_loss_scaling_preserved']
    count=plan['updates'][case];start=plan['source_step']+(0 if case=='continuous' else plan['save_after_local_updates'])
    assert s['branch_local_optimizer_steps']==count and s['cumulative_learner_optimizer_steps']==start+count
    assert s['micro_step']==4*(start+count)
    restored=read(out/'restored.json');assert restored['status']=='all_components_verified'
    assert restored['source_step']==start and restored['inherited_tokens']==s['inherited_tokens']
    assert restored['optimizer_nonzero_moments'] and restored['fresh_rollout_buffer'] and not restored['allocator_reset']
    cost=audit_sampling(out);assert cost['new_model_tokens']==s['new_sampled_tokens']<=plan['token_ceiling'][case]
    assert cost['rollouts']==count*4
    alloc=lines(out/'allocation.jsonl');assert len(alloc)==count
    paid=0
    for local,row in enumerate(alloc):
        assert row['policy_step']==local and row['cumulative_policy_step']==start+local
        assert row['cost_before_choice']==s['inherited_tokens']+paid
        paid+=row['sampled_tokens'];assert row['branch_cost_after_group']==paid
    assert paid==cost['new_model_tokens']
    for name,key in [('microbatches','micro_step_before'),('microgradients','micro_step_after')]:
        records=lines(out/(name+'.jsonl'));assert len(records)==4*count
        shift=int(key.endswith('after'))
        assert [r[key] for r in records]==list(range(4*start+shift,4*(start+count)+shift))
    final=payload(out/'final/components')
    assert final['progress']['inherited_acquisition_tokens']+final['progress']['branch_sampled_tokens']==s['inherited_tokens']+s['new_sampled_tokens']
    return s,cost


def freeze_reference(plan,sha):
    root=ROOT/plan['output_root'];folder=root/'continuous'
    summary,cost=check_case(plan,'continuous',sha)
    boundary=read(folder/'boundary/components/manifest.json')
    p=read(ROOT/plan['source_protocol'])
    assert validate_pair_boundary(boundary['progress'],p['algorithm'])==plan['source_step']+plan['save_after_local_updates']
    assert read(folder/'boundary/components/save_verification.json')['continuous_model_reloaded'] is False
    last=[r for r in lines(folder/'metrics.jsonl') if r.get('step')==plan['updates']['continuous'] and 'grad_norm' in r]
    assert len(last)==1 and last[0]['grad_norm']>0,'Final reference update must contain a fresh gradient'
    assert summary['inherited_tokens']==read(ROOT/plan['source_checkpoint']/'components/manifest.json')['cumulative_acquisition_tokens']
    files=[p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.pt'}]
    receipt={'status':'reference_verified_before_fork','paired_protocol_sha256':sha,'reference_cost':cost,
        'boundary_payload_sha256':boundary['payload_sha256'],
        'sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted(files)},'reference_final_grad_norm':last[0]['grad_norm']}
    target=root/'reference_receipt.json';assert not target.exists(),'Never replace frozen reference'
    write(target,receipt);return receipt


def audit_pair(plan,sha):
    root=ROOT/plan['output_root'];receipt=read(root/'reference_receipt.json')
    assert receipt['paired_protocol_sha256']==sha
    for path,expected in receipt['sha256'].items():assert digest(ROOT/path)==expected,path
    cs,cc=check_case(plan,'continuous',sha);fs,fc=check_case(plan,'restored',sha)
    boundary=read(root/'continuous/boundary/components/manifest.json')
    assert read(root/'restored/restored.json')['source_payload_sha256']==boundary['payload_sha256']
    assert fs['inherited_tokens']==boundary['cumulative_acquisition_tokens']
    assert cs['inherited_tokens']+cs['new_sampled_tokens']==fs['inherited_tokens']+fs['new_sampled_tokens']
    sections={
      'allocation':('policy_step',['task_id','combination','phase','probabilities','signals_before','cost_before_choice','seed','policy_fingerprint','rewards','sampled_tokens','stop_reasons','cumulative_policy_step']),
      'groups':('policy_step',['task_ids','rewards','advantages','model_tokens_in_loss','external_tokens_excluded']),
      'generation_calls':('policy_step',['rollout_index','task_id','input_ids','generated_ids','generation_allowance','remaining_completion_budget']),
      'rollouts':('policy_step',['task_id','enable_thinking','prompt_ids','completion_ids','model_token_mask','segments','bridges','completed_user_turns','total_user_turns','reward','stop_reason','rollout_index','policy_fingerprint']),
      'updates':('step',['before_sha256','after_sha256','parameters_changed']),
      'microbatches':('local_optimizer_step',['cumulative_optimizer_step','micro_step_before','input_tensor_hash','tensor_shapes']),
      'microgradients':('local_optimizer_step',['cumulative_optimizer_step','micro_step_after','loss','gradient_tensor_hash','gradient_tensor_count'])}
    differences=[];start=plan['save_after_local_updates']
    for section,(index,fields) in sections.items():
        old=[r for r in lines(root/'continuous'/(section+'.jsonl')) if r[index]>=start+int(index=='step')]
        new=lines(root/'restored'/(section+'.jsonl'))
        differences+=compare_rows(old,new,fields,section)
        if [r[index]-start for r in old]!=[r[index] for r in new]:differences.append({'section':section,'fields':['local/cumulative alignment']})
    old=[r for r in lines(root/'continuous/metrics.jsonl') if r.get('step')==3 and 'grad_norm' in r]
    new=[r for r in lines(root/'restored/metrics.jsonl') if 'grad_norm' in r]
    differences+=compare_rows(old,new,['grad_norm','loss'],'metrics')
    assert old and new and old[0]['grad_norm']>0 and new[0]['grad_norm']>0
    a,b=payload(root/'continuous/final/components'),payload(root/'restored/final/components')
    components=['trainable','optimizer','scheduler','allocator','buffers','module_training','rng']
    for component in components:
        try:assert_equal(a[component],b[component])
        except AssertionError:differences.append({'section':'final_state','fields':[component]})
    assert cs['final_fingerprint']==lines(root/'continuous/updates.jsonl')[-1]['after_sha256']
    assert fs['final_fingerprint']==lines(root/'restored/updates.jsonl')[-1]['after_sha256']
    return {'status':'passed' if not differences else 'mismatch_requires_review','exact_replay':not differences,
      'differences':differences,'continuous_new_cost':cc,'restored_new_cost':fc,
      'total_new_sampled_tokens':cc['new_model_tokens']+fc['new_model_tokens'],
      'ancestral_acquisition_tokens':cs['inherited_tokens'],'fork_inherited_tokens':fs['inherited_tokens'],
      'final_fingerprint':fs['final_fingerprint'],'verified_final_components':components,
      'fresh_gradient_norm':new[0]['grad_norm'],
      'scope':'Three-group deterministic continuation and one-group fresh-gradient boundary fork, historical loss scaling; no arbitrary resume or efficacy claim.',
      'original_default_replay_status':'mismatch_requires_review'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--freeze-reference',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);plan=read(args.protocol);sha=digest(args.protocol)
    try:
        result=freeze_reference(plan,sha) if args.freeze_reference else audit_pair(plan,sha)
    except Exception as error:
        result={'status':'failed_or_incomplete','type':type(error).__name__,'message':str(error),'partial_evidence_retained':True}
        if args.freeze_reference:write(ROOT/plan['output_root']/'reference_failure.json',result)
    if not args.freeze_reference:write(ROOT/plan['output_root']/'audit_status.json',result)
    print(result)
    raise SystemExit(0 if result['status'] in {'passed','reference_verified_before_fork'} else 1)
