"""New deterministic continuous-versus-restored control, preserving old failures."""
import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path
import run_bfcl_intervention as r
from accept_bfcl_stage_fork import ForkTrainer
from bfcl_stage_fork import validate_boundary, learner_position
from bfcl_paired_fork_contract import validate_pair_boundary
from bfcl_dose_checkpoint import assert_equal, rng_state


def read(path):
    return json.loads(path.read_text())


def pair_provenance(p, plan_path):
    return r.provenance(p,r.ROOT/read(plan_path)['source_protocol']) | {'paired_protocol_sha256':r.digest(plan_path)}


class RecordedForkTrainer(ForkTrainer):
    def _prepare_inputs(self, inputs):
        batch = super()._prepare_inputs(inputs)
        tensors = {k:v for k,v in batch.items() if r.torch.is_tensor(v)}
        row={'local_optimizer_step':self.state.global_step,
             'cumulative_optimizer_step':self.source_step+self.state.global_step,
             'micro_step_before':self._step,'input_tensor_hash':r.tensor_hash(tensors.items()),
             'tensor_shapes':{k:list(v.shape) for k,v in tensors.items()}}
        r.append(self.evidence_dir/'microbatches.jsonl',row)
        return batch

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss=super().training_step(model,inputs,num_items_in_batch)
        grads={n:p.grad for n,p in self.model.named_parameters() if p.requires_grad and p.grad is not None}
        r.append(self.evidence_dir/'microgradients.jsonl',{
            'local_optimizer_step':self.state.global_step,
            'cumulative_optimizer_step':self.source_step+self.state.global_step,
            'micro_step_after':self._step,'loss':float(loss),
            'gradient_tensor_hash':r.tensor_hash(grads.items()),'gradient_tensor_count':len(grads)})
        return loss


def save_without_restore(t, path, plan_path):
    """Save and verify no side effects; continuous reference is never reloaded."""
    progress={'global_step':t.state.global_step,'branch_local_optimizer_steps':t.state.global_step,
        'source_branch_start_step':t.source_step,
        'cumulative_learner_optimizer_steps':t.source_step+t.state.global_step,
        'micro_step':t._step,'at_optimizer_boundary':True,'fresh_rollout_on_next_group':True,
        'inherited_acquisition_tokens':t.inherited_tokens,'branch_sampled_tokens':t.sampled_tokens}
    validate_pair_boundary(progress,t.protocol['algorithm'])
    assert all(p.grad is None for p in t.model.parameters())
    before_rng=rng_state()
    before_opt=copy.deepcopy(t.optimizer.optimizer.state_dict())
    before_scheduler=copy.deepcopy(t.lr_scheduler.state_dict())
    before_allocator=copy.deepcopy(t.allocator.state_dict())
    before_buffers={n:b.detach().cpu().clone() for n,b in t.model.named_buffers()}
    before_modes={n:m.training for n,m in t.model.named_modules()}
    before_fingerprint=r.fingerprint(t.model)
    manifest=r.save_learning_state(path,t.model,t.optimizer.optimizer,t.lr_scheduler,t.allocator,
                                   progress,pair_provenance(t.protocol,plan_path))
    assert_equal(before_rng,rng_state())
    assert_equal(before_opt,t.optimizer.optimizer.state_dict())
    assert_equal(before_scheduler,t.lr_scheduler.state_dict())
    assert_equal(before_allocator,t.allocator.state_dict())
    assert_equal(before_buffers,dict(t.model.named_buffers()))
    assert_equal(before_modes,{n:m.training for n,m in t.model.named_modules()})
    assert before_fingerprint==r.fingerprint(t.model)
    r.write(path/'save_verification.json',{'status':'passed','continuous_model_reloaded':False,
        'fingerprint':before_fingerprint,'checks':['parameters','optimizer','scheduler','allocator','buffers','modes','RNG']})
    return manifest


class PairState(r.TrainerCallback):
    def __init__(self,t,plan,plan_path,case):
        self.t,self.plan,self.plan_path,self.case=t,plan,plan_path,case

    def on_train_begin(self,args,state,control,**kwargs):
        t,plan=self.t,self.plan
        assert state.global_step==0 and t._step==0 and t._buffered_inputs is None
        assert t.optimizer.scaler is None
        assert t.accelerator.gradient_accumulation_steps==4
        if self.case=='continuous':
            source=r.ROOT/plan['source_checkpoint']/'components'
            expected=r.provenance(t.protocol,r.ROOT/plan['source_protocol'])
            assert read(source/'manifest.json')['payload_sha256']==plan['source_payload_sha256']
        else:
            receipt=read(r.ROOT/plan['output_root']/'reference_receipt.json')
            assert receipt['paired_protocol_sha256']==r.digest(self.plan_path)
            for path,sha in receipt['sha256'].items():assert r.digest(r.ROOT/path)==sha,path
            source=r.ROOT/plan['output_root']/'continuous/boundary/components'
            expected=pair_provenance(t.protocol,self.plan_path)
        manifest=read(source/'manifest.json')
        allocator,progress=r.restore_learning_state(source,t.model,t.optimizer.optimizer,t.lr_scheduler,expected)
        t.source_step=(validate_boundary(progress,t.protocol['algorithm']) if self.case=='continuous'
                       else validate_pair_boundary(progress,t.protocol['algorithm']))
        assert t.source_step==plan['source_step']+(0 if self.case=='continuous' else plan['save_after_local_updates'])
        t.allocator=allocator;t.inherited_tokens=allocator.total_sampled_tokens
        assert t.inherited_tokens==manifest['cumulative_acquisition_tokens']
        t._step=progress['micro_step'];t.current_fingerprint=r.fingerprint(t.model)
        payload=r.torch.load(source/'learning_state.pt',map_location='cpu',weights_only=False)
        assert_equal(payload['trainable'],{n:p for n,p in t.model.named_parameters() if p.requires_grad})
        assert_equal(payload['optimizer'],t.optimizer.optimizer.state_dict())
        assert_equal(payload['scheduler'],t.lr_scheduler.state_dict())
        assert_equal(payload['allocator'],allocator.state_dict())
        assert_equal(payload['buffers'],dict(t.model.named_buffers()))
        assert_equal(payload['module_training'],{n:m.training for n,m in t.model.named_modules()})
        rr=payload['rng'];assert_equal({'python':rr['python'],'numpy':rr['numpy'],'cpu':rr['torch_cpu'],'cuda':rr['torch_cuda']},rng_state())
        assert any(r.torch.count_nonzero(s['exp_avg']).item() for s in payload['optimizer']['state'].values())
        del payload
        r.write(t.evidence_dir/'restored.json',{'status':'all_components_verified','source_step':t.source_step,
            'source_payload_sha256':manifest['payload_sha256'],'inherited_tokens':t.inherited_tokens,
            'policy_fingerprint':t.current_fingerprint,'micro_step':t._step,'optimizer_nonzero_moments':True,
            'allocator_reset':False,'fresh_rollout_buffer':True,'historical_accelerator_accumulation':4,
            'checks':['parameters','optimizer','scheduler','allocator','buffers','modes','RNG']})

    def on_step_end(self,args,state,control,**kwargs):
        t,plan=self.t,self.plan
        assert t._step==4*(t.source_step+state.global_step)
        assert len(t.allocation_records)==state.global_step
        if self.case=='continuous' and state.global_step==plan['save_after_local_updates']:
            save_without_restore(t,t.evidence_dir/'boundary/components',self.plan_path)
        r.published_progress(t.evidence_dir,'deterministic_pair',case=self.case,
            **learner_position(t.source_step,state.global_step),new_sampled_tokens=t.sampled_tokens,
            inherited_tokens=t.inherited_tokens)
        return control


def make_trainer(p,out,tokenizer,model,case,plan):
    registry=r.load_tasks(p,list(p['training_registry']),'train')
    rows={key:{'task_id':key,'prompt':r.initial_prompt(pair[0])} for key,pair in registry.items()}
    config=r.config_for(p,out);config.max_steps=plan['updates'][case]
    r.write(out/'trainer_config.json',json.loads(config.to_json_string()))
    a=p['algorithm']
    def reward(**kwargs):return [e.conversation.reward() for e in trainer.episodes]
    trainer=RecordedForkTrainer(model=model,args=config,processing_class=tokenizer,
        train_dataset=r.Dataset.from_list(list(rows.values())),reward_funcs=reward,
        peft_config=r.LoraConfig(r=a['lora_r'],lora_alpha=a['lora_alpha'],lora_dropout=a['lora_dropout'],
                                target_modules=a['lora_targets'],task_type='CAUSAL_LM'))
    assert trainer.accelerator.num_processes==1 and not trainer.is_fsdp_enabled and not trainer.is_deepspeed_enabled
    trainer.registry=registry;trainer.training_rows=rows;trainer.evidence_dir=out;trainer.protocol=p
    trainer.segment_cap=a['segment_cap'];trainer.episode_limits={'max_calls':a['max_calls'],'max_segments':a['max_segments']}
    trainer.sampled_tokens=0;trainer.generation_seconds=trainer.train_compute_seconds=0.
    trainer.allocation_records=[];trainer.stop_reason=None
    return trainer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--case',choices=['continuous','restored'],required=True);args=parser.parse_args()
    plan=read(args.protocol);out=r.ROOT/plan['output_root']/args.case;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        for path,sha in plan['input_sha256'].items():assert r.digest(r.ROOT/path)==sha,path
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        assert r.torch.cuda.is_available() and r.torch.cuda.device_count()==1
        r.torch.use_deterministic_algorithms(True,warn_only=False)
        r.torch.backends.cudnn.deterministic=True;r.torch.backends.cudnn.benchmark=False
        r.torch.set_num_threads(4)
        p=read(r.ROOT/plan['source_protocol'])
        assert r.digest(r.ROOT/'rl/bfcl_research_split_candidate/manifest.json')==p['manifest_sha256']
        assert r.digest(r.PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json')==p['data_sha256']
        r.set_seed(p['seeds']['initialization'])
        r.write(out/'run.json',{'protocol_sha256':r.digest(args.protocol),'case':args.case,'pid':os.getpid(),
            'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip(),
            'deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,
            'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'historical_loss_scaling_preserved':True})
        tokenizer=r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
        model=r.AutoModelForCausalLM.from_pretrained(r.ROOT/'smoke/model',dtype=r.torch.bfloat16,
                                                   attn_implementation='sdpa',local_files_only=True)
        t=make_trainer(p,out,tokenizer,model,args.case,plan);t.started=started
        state=PairState(t,plan,args.protocol,args.case)
        # Restore must precede UpdateEvidence.on_train_begin so its initial hash is restored.
        # The snapshot computes its own current hash and never reloads the live model.
        t.add_callback(state);t.add_callback(r.UpdateEvidence(t))
        t.train()
        assert t.state.global_step==plan['updates'][args.case]
        assert t._step==4*(t.source_step+t.state.global_step)
        assert t.sampled_tokens<=plan['token_ceiling'][args.case]
        save_without_restore(t,out/'final/components',args.protocol)
        r.write(out/'summary.json',{'status':'complete','case':args.case,
            **learner_position(t.source_step,t.state.global_step),'new_sampled_tokens':t.sampled_tokens,
            'inherited_tokens':t.inherited_tokens,'final_fingerprint':r.fingerprint(t.model),
            'generation_seconds':t.generation_seconds,'wall_seconds':time.perf_counter()-started,
            'scientific_claim':False,'pair_audit_required':True})
    except Exception as error:
        r.write(out/'failure.json',{'type':type(error).__name__,'message':str(error),
            'wall_seconds':time.perf_counter()-started,'partial_evidence_retained':True,'automatic_retry':False})
        raise


if __name__=='__main__':main()
