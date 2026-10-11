"""Compare actual installed GRPO backward with a full-group CPU policy-gradient objective."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import tempfile
import types

import run_bfcl_intervention as r
from transformers import GPT2Config,GPT2LMHeadModel
from bfcl_loss_scaling_contract import configure_single_normalization
from bfcl_fixed_rollout_inputs import tensor_distance
from accelerate import Accelerator


def grads(model):
    return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}


def gradient_check():
    torch=r.torch;torch.set_num_threads(4);r.set_seed(2026101140)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_embd=8,n_layer=1,n_head=1,n_positions=32,
                                   attn_pdrop=0.,embd_pdrop=0.,resid_pdrop=0.))
    model.train()
    # Per-sequence averaging, unequal token masks, and excluded external feedback.
    ids=torch.tensor([[1,2,3,4,5],[1,3,6,7,0],[1,4,8,9,10],[1,5,11,12,13]])
    completion_mask=torch.tensor([[1,1,1],[1,1,0],[1,1,1],[1,1,1]])
    tool_mask=torch.tensor([[1,0,1],[1,1,1],[0,1,1],[1,1,1]])
    prompt_mask=torch.ones((4,2),dtype=torch.long)
    attention=torch.cat([prompt_mask,completion_mask],dim=1)
    rewards=torch.tensor([1.,1.,1.,0.]);advantages=(rewards-rewards.mean())/(rewards.std()+1e-4)
    batch={'prompt_ids':ids[:,:2],'prompt_mask':prompt_mask,'completion_ids':ids[:,2:],
           'completion_mask':completion_mask,'tool_mask':tool_mask,'advantages':advantages}
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    # Independent full-group differentiable target: sum_i A_i * mean_token log pi / four.
    output=model(input_ids=ids,attention_mask=attention,use_cache=False).logits
    logps=output[:,:-1,:][:,-3:,:].log_softmax(-1).gather(-1,ids[:,2:].unsqueeze(-1)).squeeze(-1)
    mask=completion_mask*tool_mask
    target=-((logps*mask).sum(-1)/mask.sum(-1)*advantages).mean()
    target.backward();reference=grads(model);model.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory() as out:
        tokenizer=r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True)
        config=r.GRPOConfig(output_dir=out,use_cpu=True,bf16=False,per_device_train_batch_size=1,
            gradient_accumulation_steps=4,steps_per_generation=4,num_generations=4,num_iterations=1,
            beta=0.,loss_type='grpo',temperature=1.,report_to='none',disable_tqdm=True)
        trainer=r.BFCLTrainer(model=model,args=config,processing_class=tokenizer,
            train_dataset=r.Dataset.from_list([{'prompt':'fixed CPU fixture'}]*4),
            reward_funcs=lambda **kwargs:[0.]*4)
        trainer.current_gradient_accumulation_steps=4
        trainer._prepare_inputs=types.MethodType(lambda self,inputs:inputs,trainer)
        assert trainer.accelerator.gradient_accumulation_steps==4
        results={}
        for mode in ['historical','single_normalization']:
            if mode=='single_normalization':contract=configure_single_normalization(trainer)
            model.zero_grad(set_to_none=True);trainer._step=0
            losses=[]
            for i in range(4):
                loss=trainer.training_step(model,{k:v[i:i+1] for k,v in batch.items()},None)
                losses.append(float(loss))
            results[mode]={'grads':grads(model),'micro_losses':losses,
                           'accelerator_divisor':trainer.accelerator.gradient_accumulation_steps}
        trainer.accelerator.gradient_accumulation_steps=4
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters()),'No optimizer update allowed in this check'
    historical=results['historical']['grads'];corrected=results['single_normalization']['grads']
    # CPU matrix kernels can change summation order between 4xB1 and B4; use a stated numerical tolerance.
    for name in reference:
        torch.testing.assert_close(4*historical[name],reference[name],rtol=2e-5,atol=2e-7)
        torch.testing.assert_close(corrected[name],reference[name],rtol=2e-5,atol=2e-7)
    comparison={
        'historical_vs_full_group':tensor_distance(reference,historical),
        'four_times_historical_vs_full_group':tensor_distance(reference,{k:4*v for k,v in historical.items()}),
        'candidate_vs_full_group':tensor_distance(reference,corrected)}
    assert comparison['historical_vs_full_group']['relative_l2']>.7
    assert comparison['candidate_vs_full_group']['relative_l2']<2e-5
    return {'status':'passed','new_sampled_tokens':0,'gpu_work':False,
        'fixture':'One dropout-free tiny GPT2, four unequal masked completions; no optimizer steps.',
        'target':'-mean_i(advantage_i * mean_model_token(log pi_i)); gradients compared at on-policy ratio=1, beta=0.',
        'cpu_tolerance':{'rtol':2e-5,'atol':2e-7},'configuration':contract,
        'comparisons':comparison,'micro_losses':{k:v['micro_losses'] for k,v in results.items()},
        'objective_mask_counts':mask.sum(-1).tolist(),
        'source_sha256':{str(Path(inspect.getfile(c)).relative_to(r.ROOT)):r.digest(Path(inspect.getfile(c)))
                         for c in [type(trainer),Accelerator]},
        'limitations':['CPU numerical tolerance is not a relaxation of the running exact GPU fork criterion.',
                      'No claim of full GPU-loop, AdamW-update or resumed-state equivalence from this CPU test.',
                      'Historical weights/optimizer trajectories are retained; no retrospective rescaling.',
                      'Changing gradients can change AdamW eps/clipping/moments; fourfold gradients do not imply fourfold parameter updates.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists(),'Preserve each completed check'
    try:result=gradient_check()
    except Exception as error:
        result={'status':'failed','type':type(error).__name__,'message':str(error),'new_sampled_tokens':0,'gpu_work':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);r.write(args.output,result)
    print(json.dumps(result));raise SystemExit(0 if result['status']=='passed' else 1)
