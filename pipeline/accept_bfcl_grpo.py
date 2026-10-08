"""Bounded development-only BFCL/Qwen multi-user-turn GRPO acceptance."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from train_rl_smoke import (GRPOTrainer,GRPOConfig,AutoTokenizer,AutoModelForCausalLM,
    LoraConfig,Dataset,UpdateEvidence,append,fingerprint,set_seed,torch)
from trl.models import unwrap_model_for_generation
from bfcl_safe_runtime import ROOT
from bfcl_token_rollout import TokenEpisode,initial_prompt,development_tasks,generation_buckets


class BFCLTrainer(GRPOTrainer):
    def _generate_single_turn(self,prompts):
        episodes=self.episodes
        assert [e.prompt for e in episodes]==prompts
        with unwrap_model_for_generation(self.model_wrapped,self.accelerator,
                generation_kwargs=self.generation_kwargs) as model,torch.no_grad():
            while any(e.active for e in episodes):
                for budget,active in generation_buckets(episodes):
                    sequences=[e.input_ids for e in active];width=max(map(len,sequences))
                    ids=torch.tensor([[self.pad_token_id]*(width-len(s))+s for s in sequences],device=self.accelerator.device)
                    attention=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences],device=ids.device)
                    config=copy.deepcopy(self.generation_config);config.max_new_tokens=budget
                    before=time.perf_counter()
                    output=model.generate(input_ids=ids,attention_mask=attention,
                        generation_config=config,disable_compile=True)[:,width:].tolist()
                    self.generation_seconds+=time.perf_counter()-before
                    for e,generated in zip(active,output):
                        if self.eos_token_id in generated:generated=generated[:generated.index(self.eos_token_id)+1]
                        self.sampled_tokens+=len(generated)
                        append(self.evidence_dir/'generation_calls.jsonl',{'policy_step':self.state.global_step,
                            'task_id':e.conversation.task['id'],'input_ids':e.input_ids,'generated_ids':generated,
                            'generation_allowance':budget,'remaining_completion_budget':e.budget-len(e.completion_ids)})
                        e.accept(generated)
        for i,e in enumerate(episodes):
            append(self.evidence_dir/'rollouts.jsonl',e.evidence()|{'rollout_index':i,
                'policy_step':self.state.global_step,'policy_fingerprint':self.current_fingerprint})
        return ([e.prompt_ids for e in episodes],[e.completion_ids for e in episodes],
            None,{'env_mask':[e.mask for e in episodes]})

    def _generate_and_score_completions(self,inputs):
        self.episodes=[TokenEpisode(self.processing_class,*self.registry[x['task_id']],
            budget=self.args.max_completion_length,
            enable_thinking=self.chat_template_kwargs.get('enable_thinking',False)) for x in inputs]
        result=super()._generate_and_score_completions(inputs)
        mask=result['tool_mask'];loss_mask=mask*result['completion_mask']
        for i,e in enumerate(self.episodes):
            assert mask[i,:len(e.mask)].tolist()==e.mask
            assert int(loss_mask[i].sum())==sum(e.mask)
        append(self.evidence_dir/'groups.jsonl',{'policy_step':self.state.global_step,
            'task_ids':[x['task_id'] for x in inputs],'rewards':[e.conversation.reward() for e in self.episodes],
            'advantages':result['advantages'].tolist(),'model_tokens_in_loss':int(loss_mask.sum()),
            'external_tokens_excluded':sum(len(e.mask)-sum(e.mask) for e in self.episodes)})
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--thinking',action='store_true');args=p.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    selected=['multi_turn_base_26','multi_turn_base_70'];tasks,answers=development_tasks(selected)
    rows=[{'task_id':key,'prompt':initial_prompt(tasks[key])} for key in selected]
    set_seed(20261008);torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(ROOT/'smoke/model',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True)
    config=GRPOConfig(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=4,
        steps_per_generation=4,num_generations=4,max_steps=2,max_completion_length=4096,
        learning_rate=1e-5,lr_scheduler_type='constant',bf16=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},beta=0.0,scale_rewards='group',
        loss_type='grpo',num_iterations=1,temperature=1.,top_p=1.,top_k=0,
        seed=20261008,data_seed=20261008,chat_template_kwargs={'enable_thinking':args.thinking},
        logging_steps=1,save_strategy='no',report_to='none',disable_tqdm=True,
        dataloader_num_workers=0,use_vllm=False,mask_truncated_completions=False)
    (out/'config.json').write_text(config.to_json_string())
    (out/'run.json').write_text(json.dumps({'purpose':'BFCL generation/loss engineering acceptance; development only',
        'task_ids':selected,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pid':os.getpid(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'max_updates':2,'whole_completion_budget':4096,'per_segment_generation_cap':1024,
        'max_calls':16,'max_segments':16,'thinking':args.thinking,
        'source_sha256':{name:hashlib.sha256((ROOT/'pipeline'/name).read_bytes()).hexdigest() for name in
            ['accept_bfcl_grpo.py','bfcl_token_rollout.py','bfcl_conversation.py','bfcl_safe_runtime.py']}},indent=2)+'\n')
    def reward(**kwargs):return [e.conversation.reward() for e in trainer.episodes]
    trainer=BFCLTrainer(model=model,args=config,processing_class=tokenizer,
        train_dataset=Dataset.from_list(rows),reward_funcs=reward,
        peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    assert trainer.accelerator.num_processes==1 and not trainer.is_fsdp_enabled and not trainer.tools
    trainer.registry={key:(tasks[key],answers[key]) for key in selected};trainer.evidence_dir=out
    trainer.generation_seconds=0.;trainer.sampled_tokens=0;trainer.add_callback(UpdateEvidence(trainer))
    trainer.train()
    (out/'summary.json').write_text(json.dumps({'status':'complete','global_step':trainer.state.global_step,
        'sampled_tokens':trainer.sampled_tokens,'generation_seconds':trainer.generation_seconds,
        'wall_seconds':time.perf_counter()-start,'final_fingerprint':fingerprint(trainer.model),
        'limitation':'Development-only engineering updates; no held-out evaluation or value-effect comparison.'},indent=2)+'\n')


if __name__=='__main__':main()
