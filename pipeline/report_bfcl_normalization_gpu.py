"""Audit frozen Qwen/LoRA single-normalization cases without accepting a full fork."""
import argparse
import json
from pathlib import Path
import torch
from bfcl_fixed_rollout_inputs import tensor_distance
from rl_learning_state import tensor_hash
from bfcl_dose_checkpoint import assert_equal
from supervise_bfcl_stage_fork import ROOT, read, write
import hashlib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors(root, label):
    folder = root/label
    manifest = read(folder/'manifest.json')
    path = folder/'learning_state.pt'
    assert digest(path) == manifest['sha256'], str(path)
    data = torch.load(path, map_location='cpu', weights_only=False)
    assert tensor_hash(data.items()) == manifest['tensor_hash']
    return data


def audit(plan, protocol_sha):
    summaries = []
    runs = {}
    for case in plan['cases']:
        folder = ROOT/plan['output_root']/case['name']
        s, run = read(folder/'summary.json'), read(folder/'run.json')
        assert s['status'] == 'complete' and s['new_sampled_tokens'] == 0
        assert s['micro_steps'] == 36 and s['fresh_gradient_updates'] == 1
        assert run['protocol_sha256'] == protocol_sha and run['case'] == case
        assert run['fixed_order'] == plan['rollout_order'] and run['generation_forbidden']
        assert run['grpo_loss_normalizer'] == 4
        assert run['accelerator_gradient_accumulation_steps'] == (1 if case['single_normalization'] else 4)
        assert run['historical_loss_scaling_preserved'] == (not case['single_normalization'])
        assert run['deterministic_algorithms'] and run['cublas_workspace_config'] == ':4096:8'
        reconstruction = read(folder/'reconstruction.json')
        assert len(reconstruction) == 2 and all(x['matches_reference'] for x in reconstruction)
        assert [x['fingerprint'] for x in reconstruction] == [x['fingerprint'] for x in plan['zero_gradient_reconstruction']]
        for label in ['inputs','start_parameters','final_parameters',*[f'micro_{i:03d}' for i in range(4)]]:
            tensors(folder,label)
        for label in ['start','final']:
            key = label+'_optimizer_sha256'
            expected = run[key] if label == 'start' else s[key]
            assert digest(folder/(label+'_optimizer')/'learning_state.pt') == expected
        micro = read(folder/'microbatches.json')
        assert len(micro) == 4 and [x['rollout_index'] for x in micro] == plan['rollout_order']
        runs[case['name']] = run
        summaries.append(s)
    anchor = ROOT/plan['output_root']/plan['cases'][0]['name']
    anchor_optimizer = torch.load(anchor/'start_optimizer/learning_state.pt',map_location='cpu',weights_only=False)
    for case in plan['cases'][1:]:
        folder = ROOT/plan['output_root']/case['name']
        for label in ['inputs','start_parameters']:
            assert tensor_distance(tensors(anchor,label),tensors(folder,label))['exact']
        assert_equal(anchor_optimizer,torch.load(folder/'start_optimizer/learning_state.pt',map_location='cpu',weights_only=False))
    comparisons = []
    for left, right in plan['comparisons']:
        a,b = [ROOT/plan['output_root']/x for x in [left,right]]
        ma,mb = read(a/'microbatches.json'),read(b/'microbatches.json')
        assert [x['input_hash'] for x in ma] == [x['input_hash'] for x in mb]
        labels = [f'micro_{i:03d}' for i in range(4)] + ['final_parameters']
        comparisons.append({'left':left,'right':right,
            'microbatch_losses_equal':[x['loss'] for x in ma] == [x['loss'] for x in mb],
            'tensor_differences':{label:tensor_distance(tensors(a,label),tensors(b,label)) for label in labels}})
    root = ROOT/plan['output_root']
    scaling = {}
    for index in range(4):
        label = f'micro_{index:03d}'
        legacy = tensors(root/'legacy', label)
        candidate = tensors(root/'single_1', label)
        scaling[label] = tensor_distance({k:v*4 for k,v in legacy.items()}, candidate)
    repeat = comparisons[0]
    assert repeat['left'] == 'single_1' and repeat['right'] == 'single_2'
    opt_a = torch.load(root/'single_1/final_optimizer/learning_state.pt', map_location='cpu', weights_only=False)
    opt_b = torch.load(root/'single_2/final_optimizer/learning_state.pt', map_location='cpu', weights_only=False)
    optimizer_exact = True
    try:
        assert_equal(opt_a, opt_b)
    except (AssertionError, ValueError):
        optimizer_exact = False
    accepted = (all(x['exact'] for x in scaling.values()) and
                all(x['exact'] for x in repeat['tensor_differences'].values()) and
                repeat['microbatch_losses_equal'] and optimizer_exact and
                all(x['grad_norm'] > 0 for x in summaries))
    return {'status':'diagnostic_complete', 'acceptance_status':'passed' if accepted else 'mismatch_requires_review',
        'four_times_legacy_vs_candidate':scaling, 'candidate_final_optimizer_exact':optimizer_exact,
        'full_trainer_loop_gpu_tested':False, 'normalization_changed_after_historical_moment_reconstruction':True,'new_sampled_tokens':0,'fresh_gradient_updates':len(summaries),
        'input_token_positions_excluding_padding':sum(x['input_token_positions_excluding_padding'] for x in summaries),
        'case_wall_seconds_sum':sum(x['wall_seconds'] for x in summaries),'comparisons':comparisons,
        'case_gradient_norms':{x['case']:x['grad_norm'] for x in summaries},
        'scientific_stage_fork_acceptance':False,'original_exact_replay_status':'mismatch_requires_review',
        'scope':'Independent fixed-input/order backward repeats; no historical microbatch-order equivalence or efficacy claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);plan=read(args.protocol)
    try:
        result=audit(plan,digest(args.protocol))
    except Exception as error:
        result={'status':'diagnostic_audit_failed','type':type(error).__name__,'message':str(error),
                'scientific_stage_fork_acceptance':False,'partial_evidence_retained':True}
    write(ROOT/plan['output_root']/'audit_status.json',result)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['status']=='diagnostic_complete' else 1)
