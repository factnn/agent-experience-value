"""Actual CPU Trainer.train loop acceptance for single normalization; no generation."""
import argparse
import json
from pathlib import Path
import tempfile
import types
import run_bfcl_intervention as r
from transformers import GPT2Config,GPT2LMHeadModel
from bfcl_loss_scaling_contract import configure_single_normalization
from bfcl_fixed_rollout_inputs import tensor_distance


def forbidden(*args,**kwargs):raise RuntimeError('No generation allowed in CPU objective check')


def one_case(corrected):
    torch=r.torch;r.set_seed(2026101140)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_embd=8,n_layer=1,n_head=1,n_positions=32,
                                    attn_pdrop=0.,embd_pdrop=0.,resid_pdrop=0.))
    ids=torch.tensor([[1,2,3,4,5],[1,3,6,7,0],[1,4,8,9,10],[1,5,11,12,13]])
    cm=torch.tensor([[1,1,1],[1,1,0],[1,1,1],[1,1,1]])
    tm=torch.tensor([[1,0,1],[1,1,1],[0,1,1],[1,1,1]]);pm=torch.ones((4,2),dtype=torch.long)
    rewards=torch.tensor([1.,1.,1.,0.]);adv=(rewards-rewards.mean())/(rewards.std()+1e-4)
    inputs={'prompt_ids':ids[:,:2],'prompt_mask':pm,'completion_ids':ids[:,2:],
            'completion_mask':cm,'tool_mask':tm,'advantages':adv}
    model.train();logits=model(input_ids=ids,attention_mask=torch.cat([pm,cm],1),use_cache=False).logits
    logps=logits[:,:-1,:][:,-3:,:].log_softmax(-1).gather(-1,ids[:,2:].unsqueeze(-1)).squeeze(-1)
    loss=-((logps*cm*tm).sum(-1)/(cm*tm).sum(-1)*adv).mean();loss.backward()
    expected={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True);before=r.fingerprint(model)
    observations={};seen=[]
    class Observe(r.TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            observations['hf_accumulation']=args.gradient_accumulation_steps
            observations['accelerator_divisor_at_train_begin']=trainer.accelerator.gradient_accumulation_steps
            assert r.fingerprint(model)==before
        def on_pre_optimizer_step(self,args,state,control,**kw):
            actual={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
            assert len(seen)==4 and trainer._step==4 and state.global_step==0
            multiplier=1 if corrected else 4
            for name in expected:torch.testing.assert_close(multiplier*actual[name],expected[name],rtol=2e-5,atol=2e-7)
            observations['gradient_difference']=tensor_distance(expected,actual)
            observations['scaled_gradient_difference']=tensor_distance(expected,{k:multiplier*v for k,v in actual.items()})
            observations['before_optimizer_fingerprint']=r.fingerprint(model)
            assert observations['before_optimizer_fingerprint']==before
    with tempfile.TemporaryDirectory() as out:
        tokenizer=r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True)
        args=r.GRPOConfig(output_dir=out,use_cpu=True,bf16=False,per_device_train_batch_size=1,
            gradient_accumulation_steps=4,steps_per_generation=4,num_generations=4,num_iterations=1,
            max_steps=1,max_grad_norm=0.,beta=0.,loss_type='grpo',temperature=1.,
            learning_rate=1e-5,lr_scheduler_type='constant',optim='adamw_torch',save_strategy='no',
            logging_steps=1,report_to='none',disable_tqdm=True,dataloader_pin_memory=False)
        trainer=r.BFCLTrainer(model=model,args=args,processing_class=tokenizer,
            train_dataset=r.Dataset.from_list([{'prompt':'fixture'}]*4),reward_funcs=forbidden)
        def prepare(self,ignored):
            i=self._step
            assert i==len(seen)<4
            seen.append(i)
            return {k:v[i:i+1] for k,v in inputs.items()}
        trainer._prepare_inputs=types.MethodType(prepare,trainer)
        trainer._generate_and_score_completions=types.MethodType(forbidden,trainer)
        trainer.model.generate=types.MethodType(forbidden,trainer.model)
        if corrected:configure_single_normalization(trainer)
        trainer.add_callback(Observe())
        trainer.train()
        assert trainer.state.global_step==1 and trainer._step==4
        assert trainer.accelerator.gradient_accumulation_steps==(1 if corrected else 4)
        observations.update(microbatch_order=seen,optimizer_updates=trainer.state.global_step,
            final_fingerprint=r.fingerprint(model),parameters_changed=before!=r.fingerprint(model),
            accelerator_divisor_after_train=trainer.accelerator.gradient_accumulation_steps)
        trainer.accelerator.gradient_accumulation_steps=4
    return observations


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists(),'Preserve original check'
    r.torch.set_num_threads(4)
    try:
        historical=one_case(False);candidate=one_case(True)
        assert historical['before_optimizer_fingerprint']==candidate['before_optimizer_fingerprint']
        result={'status':'complete','audit':'passed','new_sampled_tokens':0,'gpu_work':False,
            'historical':historical,'candidate':candidate,
            'tolerance':{'rtol':2e-5,'atol':2e-7},
            'scope':'Two one-update CPU tiny-GPT2 Trainer.train loops with real AdamW, synthetic masked inputs, no generation; no GPU/fork acceptance.',
            'source_sha256':{n:r.digest(r.ROOT/n) for n in ['pipeline/check_bfcl_loss_normalization_loop.py','pipeline/bfcl_loss_scaling_contract.py']}}
    except Exception as error:
        result={'status':'failed','type':type(error).__name__,'message':str(error),'new_sampled_tokens':0,'gpu_work':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);r.write(args.output,result)
    print(json.dumps(result));raise SystemExit(0 if result['status']=='complete' else 1)
