"""Bounded allocation pilot; task selection changes before official GRPO sampling."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time

from train_rl_smoke import (EvidenceTrainer, UpdateEvidence, GRPOConfig, LoraConfig,
    AutoTokenizer, AutoModelForCausalLM, Dataset, TrainerCallback, append,
    fingerprint, set_seed, torch)
from rl_environment import ROOT, MessageEnv, make_task, oracle
from calibrate_rl import PROTOCOL
from rl_allocation import TaskAllocator


def manifest():
    result={}
    for split,seeds in [('pilot_train',[92000,92001]),('pilot_eval',[93000])]:
        tasks=[]
        for seed in seeds:
            for family in ['send','new_contact','replace']:
                t=make_task(seed,family,split)
                t['prompt'][0]['content']+=PROTOCOL
                env=MessageEnv();env.reset(json.dumps(t));assert env._reward()==0
                oracle(env);assert env._reward()==1
                tasks.append({'task_id':t['task_id'],'prompt':t['prompt'],'task_json':json.dumps(t)})
        result[split]=tasks
    assert not ({t['task_id'] for t in result['pilot_train']} & {t['task_id'] for t in result['pilot_eval']})
    return result


class AllocationTrainer(EvidenceTrainer):
    def _generate_and_score_completions(self, inputs):
        if self.evidence_phase.startswith('eval:'):
            return super()._generate_and_score_completions(inputs)
        assert self.model.training
        assert len(inputs)==4, 'one four-rollout group per update required'
        assert self.state.global_step==len(self.allocation_records), 'unexpected replay or multiple groups per update'
        assert self.allocator.total_sampled_tokens < self.token_budget
        choice=self.allocator.choose()
        selected=self.training_rows[choice['task_id']]
        inputs=[copy.deepcopy(selected) for _ in range(4)]
        self.evidence_phase=f'train:{self.state.global_step}'
        # Per-update sampling RNG is independent of the allocation RNG and previous trajectory lengths.
        set_seed(self.pilot_seed+100*self.state.global_step)
        before=self.sampled_tokens
        result=super()._generate_and_score_completions(inputs)
        cost=self.sampled_tokens-before
        rewards=[e._reward() for e in self.environments]
        self.allocator.observe(choice['task_id'],rewards,cost,self.state.global_step)
        row=choice | {'policy_step':self.state.global_step,'policy_fingerprint':self.current_fingerprint,
                      'rewards':rewards,'sampled_tokens':cost,
                      'cost_after_group':self.allocator.total_sampled_tokens}
        self.allocation_records.append(row)
        append(self.evidence_dir/'allocation.jsonl',row)
        return result


class BudgetProgress(TrainerCallback):
    def __init__(self,trainer):self.trainer=trainer
    def on_step_end(self,args,state,control,**kwargs):
        t=self.trainer
        assert len(t.allocation_records)==state.global_step
        reached=t.allocator.total_sampled_tokens>=t.token_budget
        if reached:control.should_training_stop=True
        progress={'status':'training','optimizer_steps':state.global_step,
                  'training_sampled_tokens':t.allocator.total_sampled_tokens,
                  'token_budget':t.token_budget,'budget_reached':reached,
                  'allocation_groups':len(t.allocation_records),
                  'latest':t.allocation_records[-1]}
        (t.evidence_dir/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(progress),flush=True)
        return control


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rule',choices=['uniform','frontier'],required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();seed=20261008;set_seed(seed);torch.set_num_threads(4)
    tasks=manifest();(out/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    run=vars(args)|{'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'purpose':'single-seed allocation engineering pilot','seed':seed,'token_budget':120000,
        'max_updates':16,'max_completion_length':4096,'exploration':0.2,'history_window':16,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{n:hashlib.sha256((ROOT/'pipeline'/n).read_bytes()).hexdigest() for n in
            ['train_rl_allocation.py','rl_allocation.py','train_rl_smoke.py','rl_environment.py','calibrate_rl.py']},
        'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','trl','peft']}}
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(ROOT/'smoke/model',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True)
    config=GRPOConfig(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=4,
        steps_per_generation=4,num_generations=4,per_device_eval_batch_size=4,max_steps=16,
        max_completion_length=4096,max_tool_calling_iterations=8,learning_rate=1e-5,
        lr_scheduler_type='constant',bf16=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},beta=0.0,loss_type='grpo',
        scale_rewards='group',num_iterations=1,temperature=1.0,top_p=1.0,top_k=0,
        seed=seed,data_seed=seed,chat_template_kwargs={'enable_thinking':True},logging_steps=1,
        save_strategy='no',report_to='none',disable_tqdm=True,dataloader_num_workers=0,
        use_vllm=False,mask_truncated_completions=False)
    (out/'trainer_config.json').write_text(config.to_json_string())
    def reward(environments,**kwargs):return [e._reward() for e in environments]
    trainer=AllocationTrainer(model=model,args=config,processing_class=tokenizer,
        train_dataset=Dataset.from_list(tasks['pilot_train']),reward_funcs=reward,
        environment_factory=MessageEnv,peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,
            target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    assert not trainer.is_fsdp_enabled and trainer.accelerator.num_processes==1
    assert config.steps_per_generation==config.gradient_accumulation_steps==4
    trainer.evidence_dir=out;trainer.evidence_phase='train:0';trainer.generation_seconds=0.0
    trainer.sampled_tokens=0;trainer.train_compute_seconds=0.0
    trainer.training_rows={t['task_id']:t for t in tasks['pilot_train']}
    trainer.allocator=TaskAllocator(trainer.training_rows,args.rule,seed=seed,exploration=.2,window=16)
    trainer.allocation_records=[];trainer.token_budget=120000;trainer.pilot_seed=seed
    run['initial_fingerprint']=fingerprint(trainer.model)
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    trainer.add_callback(UpdateEvidence(trainer));trainer.add_callback(BudgetProgress(trainer))
    torch.cuda.reset_peak_memory_stats();training=trainer.train()
    train_generation=trainer.generation_seconds;train_sampled=trainer.sampled_tokens
    assert train_sampled==trainer.allocator.total_sampled_tokens
    trainer.save_model(str(out/'adapter'));final_fp=fingerprint(trainer.model)
    trainer.model.eval();eval_results=[]
    for i,row in enumerate(tasks['pilot_eval']):
        trainer.evidence_phase=f'eval:{i}';set_seed(seed+100000+i)
        before=trainer.sampled_tokens
        with torch.no_grad():trainer._generate_and_score_completions([copy.deepcopy(row) for _ in range(4)])
        assert fingerprint(trainer.model)==final_fp
        eval_results.append({'task_id':row['task_id'],'rewards':[e._reward() for e in trainer.environments],
                             'sampled_tokens':trainer.sampled_tokens-before})
        (out/'evaluation.json').write_text(json.dumps(eval_results,indent=2)+'\n')
        print(json.dumps({'status':'evaluating','completed_tasks':len(eval_results),'latest':eval_results[-1]}),flush=True)
    wall=time.perf_counter()-start
    summary={'status':'complete','rule':args.rule,'optimizer_steps':trainer.state.global_step,
        'training_sampled_tokens':train_sampled,'token_budget':120000,
        'budget_overshoot':max(0,train_sampled-120000),
        'stop_reason':'token_budget' if train_sampled>=120000 else '16_update_safety_limit',
        'allocation_groups_after_warmup':max(0,len(trainer.allocation_records)-6),
        'initial_fingerprint':run['initial_fingerprint'],'final_fingerprint':final_fp,
        'all_sampled_tokens_including_discarded':trainer.sampled_tokens,
        'evaluation_sampled_tokens':trainer.sampled_tokens-train_sampled,
        'eval_successes':sum(sum(e['rewards']) for e in eval_results),'eval_episodes':12,
        'train_generation_and_environment_seconds':train_generation,
        'eval_generation_seconds':trainer.generation_seconds-train_generation,
        'train_compute_seconds':trainer.train_compute_seconds,'total_wall_seconds':wall,
        'single_gpu_reserved_hours':wall/3600,'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gb':torch.cuda.max_memory_reserved()/2**30,'training_metrics':training.metrics,
        'limitation':'Single seed; small same-simulator instance split; adaptive developmental configuration; endpoint costs may differ. No confirmed allocation/transfer effect.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
