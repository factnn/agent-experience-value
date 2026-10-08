"""Qwen3/official GRPO completed-boundary resume acceptance, not efficacy.

One engineering task, two uninterrupted updates, then resume update two from the
step-one checkpoint. Native HF checkpoint plus full component/history snapshot.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from dataclasses import asdict

from train_rl_allocation import AllocationTrainer
from train_rl_smoke import (UpdateEvidence,TrainerCallback,GRPOConfig,LoraConfig,AutoTokenizer,
    AutoModelForCausalLM,Dataset,set_seed,torch,fingerprint)
from rl_environment import ROOT,MessageEnv,make_task,oracle
from rl_allocation import TaskAllocator
from rl_learning_state import save_learning_state,restore_learning_state
from calibrate_rl import PROTOCOL


class ResumeEvidence(TrainerCallback):
    def __init__(self,trainer,provenance,resume):
        self.trainer=trainer;self.provenance=provenance;self.resume=resume
    def on_train_begin(self,args,state,control,**kwargs):
        t=self.trainer
        if self.resume:
            allocator,progress=restore_learning_state(self.resume/'components',t.model,
                t.optimizer,t.lr_scheduler,self.provenance)
            assert state.global_step==progress['global_step']
            t.allocator=allocator;t.sampled_tokens=allocator.total_sampled_tokens
            t.allocation_records=json.loads((self.resume/'allocation_records.json').read_text())
            assert len(t.allocation_records)==state.global_step
            assert t._step==0 and t._buffered_inputs is None
            t.current_fingerprint=fingerprint(t.model)
            (t.evidence_dir/'restored.json').write_text(json.dumps({
                'global_step':state.global_step,'historical_cost':allocator.total_sampled_tokens,
                'policy_fingerprint':t.current_fingerprint,'fresh_buffer':True},indent=2)+'\n')
    def on_save(self,args,state,control,**kwargs):
        t=self.trainer;checkpoint=t.evidence_dir/f'checkpoint-{state.global_step}'
        assert t._step%4==0
        save_learning_state(checkpoint/'components',t.model,t.optimizer,t.lr_scheduler,
            t.allocator,{'global_step':state.global_step,'micro_step':t._step,
                'at_optimizer_boundary':True,'trainer_state':asdict(state)},self.provenance)
        (checkpoint/'allocation_records.json').write_text(json.dumps(t.allocation_records,indent=2)+'\n')
        print(json.dumps({'checkpoint_saved':str(checkpoint),'step':state.global_step}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--resume',type=Path);args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    set_seed(20260930);torch.set_num_threads(4)
    task=make_task(91000,'send','resume_engineering');task['prompt'][0]['content']+=PROTOCOL
    env=MessageEnv();env.reset(json.dumps(task));oracle(env);assert env._reward()==1
    row={'task_id':task['task_id'],'prompt':task['prompt'],'task_json':json.dumps(task)}
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True)
    tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(ROOT/'smoke/model',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True)
    config=GRPOConfig(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=4,
        steps_per_generation=4,num_generations=4,max_steps=2,max_completion_length=2048,
        max_tool_calling_iterations=8,learning_rate=1e-5,lr_scheduler_type='constant',bf16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        beta=0.0,loss_type='grpo',scale_rewards='group',num_iterations=1,temperature=1.0,
        top_p=1.0,top_k=0,seed=20260930,data_seed=20260930,
        chat_template_kwargs={'enable_thinking':True},logging_steps=1,save_strategy='steps',
        save_steps=1,report_to='none',disable_tqdm=True,dataloader_num_workers=0,use_vllm=False,
        mask_truncated_completions=False,optim='adamw_torch')
    algorithm=json.loads(config.to_json_string());algorithm.pop('output_dir')
    provenance={'base_revision':'1cfa9a7208912126459214e8b04321603b3df60c',
        'task_manifest_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
        'algorithm_config_sha256':hashlib.sha256(json.dumps(algorithm,sort_keys=True).encode()).hexdigest()}
    (out/'config.json').write_text(config.to_json_string());(out/'tasks.json').write_text(json.dumps([row],indent=2)+'\n')
    (out/'run.json').write_text(json.dumps({'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'purpose':'completed-boundary resume engineering acceptance','provenance':provenance,
        'resume':str(args.resume) if args.resume else None,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},indent=2)+'\n')
    def reward(environments,**kwargs):return [e._reward() for e in environments]
    trainer=AllocationTrainer(model=model,args=config,processing_class=tokenizer,
        train_dataset=Dataset.from_list([row]),reward_funcs=reward,environment_factory=MessageEnv,
        peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    assert trainer.accelerator.num_processes==1 and not trainer.is_fsdp_enabled
    trainer.evidence_dir=out;trainer.evidence_phase='train:0';trainer.generation_seconds=0.
    trainer.sampled_tokens=0;trainer.train_compute_seconds=0.;trainer.training_rows={row['task_id']:row}
    trainer.allocator=TaskAllocator([row['task_id']],seed=20260930)
    trainer.allocation_records=[];trainer.token_budget=120000;trainer.pilot_seed=20260930
    trainer.add_callback(UpdateEvidence(trainer));trainer.add_callback(ResumeEvidence(trainer,provenance,args.resume))
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    (out/'summary.json').write_text(json.dumps({'status':'complete','global_step':trainer.state.global_step,
        'final_fingerprint':fingerprint(trainer.model),'cumulative_cost':trainer.sampled_tokens,
        'generation_seconds_this_process':trainer.generation_seconds,
        'wall_seconds_this_process':time.perf_counter()-started,
        'limitation':'One engineering task; resume acceptance only. No allocation or learning efficacy claim.'},indent=2)+'\n')


if __name__=='__main__':main()
