"""One real Qwen/LoRA Trainer.train update against the frozen manual-loop reference."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import types
import run_bfcl_intervention as r
from bfcl_dose_checkpoint import assert_equal
from bfcl_loss_scaling_contract import configure_single_normalization
from diagnose_bfcl_backward import read, forbidden, snapshot


def load(path):
    return r.torch.load(path, map_location='cpu', weights_only=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);args=ap.parse_args()
    plan=read(args.protocol);out=r.ROOT/plan['output_root']/'loop';out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        for path,sha in plan['input_sha256'].items():assert r.digest(r.ROOT/path)==sha,path
        assert r.torch.cuda.is_available() and r.torch.cuda.device_count()==1
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        r.torch.set_num_threads(4);r.torch.use_deterministic_algorithms(True,warn_only=False)
        r.torch.backends.cudnn.deterministic=True;r.torch.backends.cudnn.benchmark=False
        source_protocol=read(r.ROOT/plan['source_protocol'])
        r.set_seed(source_protocol['seeds']['initialization'])
        tokenizer=r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
        model=r.AutoModelForCausalLM.from_pretrained(r.ROOT/'smoke/model',dtype=r.torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model.generate=types.MethodType(forbidden,model)
        local=copy.deepcopy(source_protocol);local['branch_max_updates']=1
        config=r.config_for(local,out);algorithm=local['algorithm']
        reference=r.ROOT/plan['manual_reference'];inputs=load(reference/'inputs/learning_state.pt')
        named={};seen=[];observations={}
        class FixedTrainer(r.BFCLTrainer):
            def _prepare_inputs(self,ignored):
                i=self._step-32
                assert i==len(seen)<4
                assert self.current_gradient_accumulation_steps==4
                assert self.accelerator.gradient_accumulation_steps==1
                seen.append(i)
                return {k:v[i:i+1].to(self.accelerator.device) for k,v in inputs.items()}
            def _generate_and_score_completions(self,*a,**kw):return forbidden()
            def training_step(self,*a,**kw):
                loss=super().training_step(*a,**kw)
                grads={n:p.grad.detach().cpu().clone() for n,p in named.items()}
                index=len(seen)-1
                assert_equal(grads,load(reference/f'micro_{index:03d}/learning_state.pt'))
                snapshot(out,f'micro_{index:03d}',grads)
                r.append(out/'microbatches.jsonl',{'index':index,'micro_step':self._step,'loss':float(loss),'gradient_tensor_hash':r.tensor_hash(grads.items())})
                return loss
        trainer=FixedTrainer(model=model,args=config,processing_class=tokenizer,
            train_dataset=r.Dataset.from_list([{'prompt':'fixed archived input'}]*4),reward_funcs=forbidden,
            peft_config=r.LoraConfig(r=algorithm['lora_r'],lora_alpha=algorithm['lora_alpha'],lora_dropout=algorithm['lora_dropout'],target_modules=algorithm['lora_targets'],task_type='CAUSAL_LM'))
        normalization=configure_single_normalization(trainer)
        named.update({n:p for n,p in trainer.model.named_parameters() if p.requires_grad})
        class Restore(r.TrainerCallback):
            def on_train_begin(self,args,state,control,**kw):
                assert trainer.accelerator.gradient_accumulation_steps==1 and args.gradient_accumulation_steps==4
                assert trainer.optimizer.scaler is None
                raw=trainer.optimizer.optimizer
                allocator,progress=r.restore_learning_state(r.ROOT/plan['source_checkpoint']/'components',trainer.model,raw,trainer.lr_scheduler,r.provenance(source_protocol,r.ROOT/plan['source_protocol']))
                assert progress['global_step']==6
                assert r.fingerprint(trainer.model)==plan['source_fingerprint']
                # Reconstruct historical moments before the candidate update; no gradients are sampled.
                for target in plan['zero_gradient_reconstruction']:
                    for p in named.values():p.grad=r.torch.zeros_like(p)
                    trainer.accelerator.clip_grad_norm_(trainer.model.parameters(),args.max_grad_norm)
                    trainer.optimizer.step();trainer.lr_scheduler.step();trainer.model.zero_grad(set_to_none=True)
                    assert r.fingerprint(trainer.model)==target['fingerprint']
                current={n:p.detach().cpu().clone() for n,p in named.items()}
                assert_equal(current,load(reference/'start_parameters/learning_state.pt'))
                assert_equal(raw.state_dict(),load(reference/'start_optimizer/learning_state.pt'))
                r.set_seed(plan['fixed_backward_seed']);trainer._step=32
                observations.update(start_fingerprint=r.fingerprint(trainer.model),accelerator_divisor_at_train_begin=trainer.accelerator.gradient_accumulation_steps,hf_accumulation=args.gradient_accumulation_steps)
                r.write(out/'restored.json',observations|{'reference_start_parameters_and_optimizer_exact':True})
            def on_pre_optimizer_step(self,args,state,control,**kw):
                assert state.global_step==0 and len(seen)==4 and trainer._step==36
                assert r.fingerprint(trainer.model)==observations['start_fingerprint']
                grads={n:p.grad.detach().cpu().clone() for n,p in named.items()}
                assert_equal(grads,load(reference/'micro_003/learning_state.pt'))
                observations['pre_optimizer_gradient_exact']=True
        trainer.add_callback(Restore())
        r.write(out/'trainer_config.json',json.loads(config.to_json_string()))
        r.write(out/'run.json',{'protocol_sha256':r.digest(args.protocol),'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'normalization':normalization,'new_sampled_tokens':0,'generation_forbidden':True,'manual_reference':plan['manual_reference']})
        trainer.train()
        assert trainer.state.global_step==1 and trainer._step==36 and seen==[0,1,2,3]
        assert trainer.accelerator.gradient_accumulation_steps==1
        final={n:p.detach().cpu().clone() for n,p in named.items()}
        assert_equal(final,load(reference/'final_parameters/learning_state.pt'))
        assert_equal(trainer.optimizer.optimizer.state_dict(),load(reference/'final_optimizer/learning_state.pt'))
        snapshot(out,'final_parameters',final)
        path=out/'final_optimizer/learning_state.pt';path.parent.mkdir();r.torch.save(trainer.optimizer.optimizer.state_dict(),path)
        expected=read(reference/'summary.json');assert r.fingerprint(trainer.model)==expected['final_fingerprint']
        assert all(p.grad is None for p in trainer.model.parameters())
        summary={'status':'complete','acceptance_status':'passed','new_sampled_tokens':0,'generation_forbidden':True,
            'actual_trainer_train':True,'microbatch_order':seen,'fresh_optimizer_updates':1,'historical_reconstruction_updates':2,
            'manual_reference_exact':['start_parameters','start_optimizer','four_cumulative_gradients','pre_optimizer_gradient','final_parameters','final_optimizer'],
            'accelerator_divisor_after_train':trainer.accelerator.gradient_accumulation_steps,'final_fingerprint':r.fingerprint(trainer.model),
            'final_optimizer_sha256':r.digest(path),'wall_seconds':time.perf_counter()-started,'observations':observations,
            'scope':'Real Qwen/LoRA corrected full Trainer.train, fixed archived inputs; sampling/full-state history verified separately under historical normalization, no efficacy claim.'}
        r.write(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    except Exception as error:
        r.write(out/'failure.json',{'status':'failed','type':type(error).__name__,'message':str(error),'wall_seconds':time.perf_counter()-started,'new_sampled_tokens':0,'automatic_retry':False})
        raise


if __name__=='__main__':main()
