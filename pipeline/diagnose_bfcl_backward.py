"""Single fixed-rollout backward diagnostic; no generation or efficacy labels."""
import argparse
import copy
import json
import os
import time
import types
from pathlib import Path

import run_bfcl_intervention as r
from bfcl_fixed_rollout_inputs import pack
from bfcl_dose_checkpoint import assert_equal


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def forbidden(*args, **kwargs):
    raise RuntimeError('Generation is forbidden in the fixed-rollout diagnostic')


def snapshot(out, label, tensors):
    path = out / label / 'learning_state.pt'
    path.parent.mkdir(parents=True, exist_ok=False)
    r.torch.save(tensors, path)
    metadata = {'path':str(path.relative_to(r.ROOT)), 'sha256':r.digest(path),
                'tensor_hash':r.tensor_hash(tensors.items())}
    r.write(path.parent / 'manifest.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    plan = read(args.protocol)
    case = next(x for x in plan['cases'] if x['name'] == args.case)
    out = r.ROOT / plan['output_root'] / case['name']
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        for path, expected in plan['input_sha256'].items():
            assert r.digest(r.ROOT/path) == expected, path
        assert r.torch.cuda.is_available() and r.torch.cuda.device_count() == 1
        r.torch.set_num_threads(4)
        deterministic = case['deterministic']
        r.torch.use_deterministic_algorithms(deterministic, warn_only=False)
        r.torch.backends.cudnn.deterministic = deterministic
        r.torch.backends.cudnn.benchmark = False
        if deterministic:
            assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        protocol = read(r.ROOT/plan['source_protocol'])
        r.set_seed(protocol['seeds']['initialization'])
        tokenizer = r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model', local_files_only=True)
        tokenizer.padding_side = 'left'
        model = r.AutoModelForCausalLM.from_pretrained(r.ROOT/'smoke/model', dtype=r.torch.bfloat16,
                    attn_implementation='sdpa', local_files_only=True)
        model.generate = types.MethodType(forbidden, model)
        config = r.config_for(protocol, out)
        r.write(out/'trainer_config.json', json.loads(config.to_json_string()))
        algorithm = protocol['algorithm']
        trainer = r.BFCLTrainer(model=model, args=config, processing_class=tokenizer,
            train_dataset=r.Dataset.from_list([{'prompt':'diagnostic placeholder'}]*4),
            reward_funcs=forbidden,
            peft_config=r.LoraConfig(r=algorithm['lora_r'], lora_alpha=algorithm['lora_alpha'],
              lora_dropout=algorithm['lora_dropout'], target_modules=algorithm['lora_targets'],task_type='CAUSAL_LM'))
        assert trainer.accelerator.num_processes == 1 and not trainer.is_fsdp_enabled
        assert not trainer.is_deepspeed_enabled and trainer.accelerator.gradient_accumulation_steps == 1
        trainer.create_optimizer()
        trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs)
        trainer.model.train()
        trainer.model_wrapped, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
        trainer.create_scheduler(num_training_steps=protocol['branch_max_updates'])
        assert trainer.optimizer.scaler is None
        raw = trainer.optimizer.optimizer
        source = r.ROOT/plan['source_checkpoint']/'components'
        allocator, progress = r.restore_learning_state(source, trainer.model, raw, trainer.lr_scheduler,
                                                       r.provenance(protocol, r.ROOT/plan['source_protocol']))
        payload = r.torch.load(source/'learning_state.pt', map_location='cpu', weights_only=False)
        assert_equal(payload['optimizer'], raw.state_dict())
        assert_equal(payload['scheduler'], trainer.lr_scheduler.state_dict())
        assert_equal(payload['buffers'], dict(trainer.model.named_buffers()))
        assert_equal(payload['module_training'], {n:m.training for n,m in trainer.model.named_modules()})
        del payload
        assert progress['global_step'] == 6
        before = r.fingerprint(trainer.model)
        assert before == plan['source_fingerprint']
        named = {n:p for n,p in trainer.model.named_parameters() if p.requires_grad}
        reconstruction = []
        for target in plan['zero_gradient_reconstruction']:
            for p in named.values():
                p.grad = r.torch.zeros_like(p)
            trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), config.max_grad_norm)
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.model.zero_grad(set_to_none=True)
            fingerprint = r.fingerprint(trainer.model)
            reconstruction.append({'step':target['step'], 'fingerprint':fingerprint,
                                   'matches_reference':fingerprint == target['fingerprint']})
            r.write(out/'reconstruction.json', reconstruction)
            assert fingerprint == target['fingerprint'], 'Zero-gradient source reconstruction mismatch'
        # Source8 weights/Adam moments, with explicit diagnostic RNG and fixed microbatch order.
        # This does not reconstruct historical allocator/RNG progression or a full source8 checkpoint.
        r.set_seed(plan['fixed_backward_seed'])
        trainer._step = 32
        trainer.state.global_step = 0
        trainer.current_gradient_accumulation_steps = 4
        trainer._prepare_inputs = types.MethodType(lambda self, inputs: inputs, trainer)
        trainer._generate_and_score_completions = types.MethodType(forbidden, trainer)
        rows = [x for x in lines(r.ROOT/plan['reference_training']/'rollouts.jsonl') if x['policy_step'] == 8]
        groups = [x for x in lines(r.ROOT/plan['reference_training']/'groups.jsonl') if x['policy_step'] == 8]
        assert len(groups) == 1
        inputs = pack(rows, groups[0], tokenizer.pad_token_id)
        packed = snapshot(out, 'inputs', inputs)
        start_tensors = {n:p.detach().cpu().clone() for n,p in named.items()}
        start_snapshot = snapshot(out, 'start_parameters', start_tensors)
        # Optimizer snapshots are local trusted artifacts; digest persisted for later audit.
        optimizer_path = out/'start_optimizer'/'learning_state.pt'
        optimizer_path.parent.mkdir()
        r.torch.save(copy.deepcopy(raw.state_dict()), optimizer_path)
        r.write(out/'run.json', {'protocol_sha256':r.digest(args.protocol),'case':case,
            'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'], 'fixed_order':plan['rollout_order'],
            'new_sampled_tokens':0,'generation_forbidden':True,'torch':r.torch.__version__,
            'deterministic_algorithms':r.torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'sdpa':{'flash':r.torch.backends.cuda.flash_sdp_enabled(),
                    'memory_efficient':r.torch.backends.cuda.mem_efficient_sdp_enabled(),
                    'math':r.torch.backends.cuda.math_sdp_enabled()},
            'start_fingerprint':r.fingerprint(trainer.model),'inputs':packed,
            'start_parameters':start_snapshot,'start_optimizer_sha256':r.digest(optimizer_path),
            'scope':'Fixed archived inputs and explicit RNG/order; historical allocator and sampling RNG are not replayed.'})
        micro = []
        compute_start = time.perf_counter()
        for index in plan['rollout_order']:
            batch = {k:v[index:index+1].to(trainer.accelerator.device) for k,v in inputs.items()}
            batch_hash = r.tensor_hash(batch.items())
            # Official GRPO/Transformers training_step, loss normalization, AMP and backward.
            loss = trainer.training_step(trainer.model_wrapped, batch, None)
            r.torch.cuda.synchronize()
            grads = {n:p.grad.detach().cpu().clone() for n,p in named.items() if p.grad is not None}
            assert grads.keys() == named.keys()
            assert all(r.torch.isfinite(v).all() for v in grads.values())
            evidence = snapshot(out, f'micro_{len(micro):03d}', grads)
            micro.append({'rollout_index':index,'input_hash':batch_hash,'loss':float(loss),
                          'gradient':evidence})
            r.write(out/'microbatches.json', micro)
        norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), config.max_grad_norm)
        assert r.torch.isfinite(norm)
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.model.zero_grad(set_to_none=True)
        r.torch.cuda.synchronize()
        compute_seconds = time.perf_counter()-compute_start
        final_tensors = {n:p.detach().cpu().clone() for n,p in named.items()}
        final_snapshot = snapshot(out, 'final_parameters', final_tensors)
        path = out/'final_optimizer'/'learning_state.pt'
        path.parent.mkdir()
        r.torch.save(copy.deepcopy(raw.state_dict()), path)
        summary = {'status':'complete','case':case['name'],'diagnostic_only':True,
            'new_sampled_tokens':0,'fresh_gradient_updates':1,'zero_gradient_reconstruction_updates':2,
            'grad_norm':float(norm),'input_token_positions_excluding_padding':int(inputs['prompt_mask'].sum()+inputs['completion_mask'].sum()),
            'padded_forward_token_positions':4*(inputs['prompt_ids'].shape[1]+inputs['completion_ids'].shape[1]),
            'model_tokens_in_loss':groups[0]['model_tokens_in_loss'],
            'gradient_checkpoint_recomputation_not_in_token_position_count':True,
            'training_compute_and_evidence_seconds':compute_seconds,'wall_seconds':time.perf_counter()-started,
            'final_fingerprint':r.fingerprint(trainer.model),'final_parameters':final_snapshot,
            'final_optimizer_sha256':r.digest(path),'micro_steps':trainer._step,
            'scientific_stage_fork_acceptance':False}
        assert trainer._step == 36 and len(micro) == 4
        r.write(out/'summary.json',summary)
        print(json.dumps(summary),flush=True)
    except Exception as error:
        r.write(out/'failure.json',{'type':type(error).__name__,'message':str(error),
            'wall_seconds':time.perf_counter()-started,'new_sampled_tokens':0,
            'partial_evidence_retained':True,'automatic_retry':False})
        raise


if __name__ == '__main__':
    main()
