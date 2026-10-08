"""Frozen common-base probes, controlled official GRPO branches, and paired eval."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from train_rl_smoke import (GRPOConfig,LoraConfig,AutoTokenizer,AutoModelForCausalLM,
    Dataset,TrainerCallback,UpdateEvidence,fingerprint,set_seed,torch,append)
from accept_bfcl_grpo import BFCLTrainer
from bfcl_token_rollout import TokenEpisode,initial_prompt,ROOT,PKG
from bfcl_sampling import sample_episodes
from bfcl_allocation import CombinationAllocator
from rl_learning_state import save_learning_state,restore_learning_state,tensor_hash
from peft import PeftModel
from calibrate_bfcl import write

PROTOCOL=ROOT/'rl/BFCL_INTERVENTION_PROTOCOL_20261008.json'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_tasks(protocol,ids,split):
    m=json.loads((ROOT/'rl/bfcl_research_split_candidate/manifest.json').read_text())
    assert set(ids)<={r['id'] for r in m[split]}
    tasks={r['id']:r for r in map(json.loads,(PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in ids}
    answers={r['id']:r['ground_truth'] for r in map(json.loads,(PKG/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in ids}
    for key,row in tasks.items():
        assert hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==protocol['record_sha256'][key]
    assert set(tasks)==set(answers)==set(ids)
    return {key:(tasks[key],answers[key]) for key in ids}

def config_for(protocol,out):
    a=protocol['algorithm']
    return GRPOConfig(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=4,
        steps_per_generation=4,num_generations=a['num_generations'],max_steps=protocol['branch_max_updates'],
        max_completion_length=a['whole_completion_budget'],learning_rate=a['learning_rate'],
        lr_scheduler_type='constant',bf16=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},beta=a['beta'],scale_rewards=a['scale_rewards'],
        loss_type=a['loss_type'],num_iterations=a['num_iterations'],temperature=a['temperature'],
        top_p=a['top_p'],top_k=a['top_k'],seed=protocol['seeds']['initialization'],
        data_seed=protocol['seeds']['initialization'],chat_template_kwargs={'enable_thinking':a['thinking']},
        logging_steps=1,save_strategy='no',report_to='none',disable_tqdm=True,
        dataloader_num_workers=0,use_vllm=False,mask_truncated_completions=False,optim='adamw_torch')

def provenance(protocol):
    return {'base_revision':protocol['base_revision'],'task_manifest_sha256':protocol['manifest_sha256'],
        'algorithm_config_sha256':hashlib.sha256(json.dumps(protocol['algorithm'],sort_keys=True).encode()).hexdigest(),
        'intervention_protocol_sha256':digest(PROTOCOL)}

def episode(tokenizer,pair,p):
    a=p['algorithm']
    return TokenEpisode(tokenizer,*pair,budget=a['whole_completion_budget'],
        max_calls=a['max_calls'],max_segments=a['max_segments'],enable_thinking=a['thinking'])

def published_progress(out,phase,**fields):
    row={'phase':phase,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),**fields}
    write(out/'progress.json',row);print(json.dumps(row),flush=True)


class InterventionTrainer(BFCLTrainer):
    def _generate_and_score_completions(self,inputs):
        assert len(inputs)==4 and self.state.global_step==len(self.allocation_records)
        assert self.allocator.total_sampled_tokens-self.inherited_tokens<self.protocol['branch_raw_token_budget']
        choice=self.allocator.choose();selected=self.training_rows[choice['task_id']]
        inputs=[copy.deepcopy(selected) for _ in range(4)]
        seed=self.protocol['seeds']['branch']+10000*self.condition_index+100*self.state.global_step
        set_seed(seed);before=self.sampled_tokens
        result=super()._generate_and_score_completions(inputs)
        rewards=[e.conversation.reward() for e in self.episodes];cost=self.sampled_tokens-before
        self.allocator.observe(choice['task_id'],rewards,cost,f"{self.condition}:{self.rule}:{self.state.global_step}")
        row=choice|{'policy_step':self.state.global_step,'seed':seed,'policy_fingerprint':self.current_fingerprint,
            'rewards':rewards,'sampled_tokens':cost,'branch_cost_after_group':self.sampled_tokens,
            'stop_reasons':[e.stop_reason for e in self.episodes]}
        self.allocation_records.append(row);append(self.evidence_dir/'allocation.jsonl',row)
        return result

    def training_step(self,model,inputs,num_items_in_batch=None):
        torch.cuda.synchronize();start=time.perf_counter();before=self.generation_seconds
        result=super().training_step(model,inputs,num_items_in_batch)
        torch.cuda.synchronize()
        self.train_compute_seconds+=time.perf_counter()-start-(self.generation_seconds-before)
        return result


class BranchState(TrainerCallback):
    def __init__(self,t,common):self.t=t;self.common=common
    def on_train_begin(self,args,state,control,**kwargs):
        t=self.t
        allocator,progress=restore_learning_state(self.common/'components',t.model,t.optimizer,
            t.lr_scheduler,provenance(t.protocol))
        assert state.global_step==progress['global_step']==0 and t._step==0 and t._buffered_inputs is None
        # Pool restriction and rule are the only declared interventions; all
        # weights/moments/RNG/history remain inherited from one source payload.
        t.allocator=CombinationAllocator.from_state_dict(allocator.state_dict(),rule=t.rule,
            pool=t.protocol['conditions'][t.condition])
        t.inherited_tokens=t.allocator.total_sampled_tokens
        t.current_fingerprint=fingerprint(t.model)
        manifest=json.loads((self.common/'components/manifest.json').read_text())
        assert tensor_hash((n,p) for n,p in t.model.named_parameters() if p.requires_grad)==manifest['trainable_sha256']
        write(t.evidence_dir/'restored.json',{'common_payload_sha256':manifest['payload_sha256'],
            'trainable_sha256':manifest['trainable_sha256'],'policy_fingerprint':t.current_fingerprint,
            'optimizer_global_step':0,'fresh_rollout_buffer':True,'inherited_tokens':t.inherited_tokens,
            'initial_signals':t.allocator.signals(),'initial_probabilities':t.allocator.probabilities(),
            'inherited_allocator_state':allocator.state_dict()})
    def on_step_end(self,args,state,control,**kwargs):
        t=self.t;assert len(t.allocation_records)==state.global_step
        reason=None
        if t.sampled_tokens>=t.protocol['branch_raw_token_budget']:reason='raw_token_budget'
        elif time.perf_counter()-t.started>=t.protocol['branch_soft_wall_seconds']:reason='wall_limit_after_update'
        elif state.global_step>=t.protocol['branch_max_updates']:reason='max_updates'
        if reason:control.should_training_stop=True;t.stop_reason=reason
        published_progress(t.evidence_dir,'training',optimizer_steps=state.global_step,
            branch_sampled_tokens=t.sampled_tokens,stop_reason=reason,latest=t.allocation_records[-1])
        return control


def make_trainer(p,out,tokenizer,model):
    registry=load_tasks(p,list(p['training_registry']),'train')
    rows={key:{'task_id':key,'prompt':initial_prompt(pair[0])} for key,pair in registry.items()}
    config=config_for(p,out);write(out/'trainer_config.json',json.loads(config.to_json_string()))
    a=p['algorithm']
    def reward(**kwargs):return [e.conversation.reward() for e in trainer.episodes]
    trainer=InterventionTrainer(model=model,args=config,processing_class=tokenizer,
        train_dataset=Dataset.from_list(list(rows.values())),reward_funcs=reward,
        peft_config=LoraConfig(r=a['lora_r'],lora_alpha=a['lora_alpha'],lora_dropout=a['lora_dropout'],
            target_modules=a['lora_targets'],task_type='CAUSAL_LM'))
    assert trainer.accelerator.num_processes==1 and not trainer.is_fsdp_enabled and not trainer.tools
    trainer.registry=registry;trainer.training_rows=rows;trainer.evidence_dir=out;trainer.protocol=p
    trainer.segment_cap=a['segment_cap'];trainer.episode_limits={'max_calls':a['max_calls'],'max_segments':a['max_segments']}
    trainer.sampled_tokens=0;trainer.generation_seconds=0.;trainer.train_compute_seconds=0.
    trainer.allocation_records=[];trainer.stop_reason=None
    return trainer


def build_common(p,out,tokenizer,model,started):
    t=make_trainer(p,out,tokenizer,model)
    t.create_optimizer_and_scheduler(num_training_steps=p['branch_max_updates'])
    t.model.to('cuda:0');t.model.eval()
    allocator=CombinationAllocator(p['training_registry'],seed=p['seeds']['initialization'])
    initial=fingerprint(t.model);total=0;seconds=0.
    with torch.inference_mode():
        for index,key in enumerate(p['common_probe_task_ids']):
            if time.perf_counter()-started>=p['common_probe_soft_wall_seconds']:
                write(out/'summary.json',{'status':'incomplete_common_probe','groups':index,'tokens':total})
                raise RuntimeError('common probes incomplete; do not fork')
            choice=allocator.choose(forced=key);seed=p['seeds']['common_probe']+100*index;set_seed(seed)
            episodes=[episode(tokenizer,t.registry[key],p) for _ in range(4)]
            def record(row):append(out/'generation_calls.jsonl',row|{'group':index,'phase':'common_probe','seed':seed})
            stats=sample_episodes(t.model,tokenizer,episodes,t.generation_config,record,segment_cap=p['algorithm']['segment_cap'])
            rewards=[e.conversation.reward() for e in episodes]
            allocator.observe(key,rewards,stats['sampled_tokens'],f'probe:{index}')
            total+=stats['sampled_tokens'];seconds+=stats['generation_seconds']
            for i,e in enumerate(episodes):append(out/'rollouts.jsonl',e.evidence()|{'group':index,'rollout_index':i,'phase':'common_probe','seed':seed})
            append(out/'probe_history.jsonl',choice|{'group':index,'seed':seed,'rewards':rewards,'sampled_tokens':stats['sampled_tokens']})
            published_progress(out,'common_probe',groups=index+1,planned_groups=len(p['common_probe_task_ids']),
                sampled_tokens=total,latest_task=key,rewards=rewards)
    assert total==allocator.total_sampled_tokens and total<=p['common_probe_raw_token_ceiling']
    assert initial==fingerprint(t.model) and not t.optimizer.state and all(x.grad is None for x in t.model.parameters())
    t.model.train()
    manifest=save_learning_state(out/'components',t.model,t.optimizer,t.lr_scheduler,allocator,
        {'global_step':0,'micro_step':0,'at_optimizer_boundary':True,'probe_groups':len(p['common_probe_task_ids'])},provenance(p))
    t.model.save_pretrained(out/'adapter');write(out/'allocation_state.json',allocator.state_dict())
    write(out/'summary.json',{'status':'complete','optimizer_updates':0,'sampled_tokens':total,
        'generation_seconds':seconds,'wall_seconds':time.perf_counter()-started,
        'policy_fingerprint':initial,'common_manifest':manifest,'all_probes_charged':True})
    published_progress(out,'common_complete',sampled_tokens=total,optimizer_updates=0)


def train_branch(p,out,tokenizer,model,args,started):
    t=make_trainer(p,out,tokenizer,model);t.condition=args.condition;t.rule=args.rule
    t.condition_index=list(p['conditions']).index(args.condition);t.started=started
    t.add_callback(BranchState(t,args.common));t.add_callback(UpdateEvidence(t))
    # Restoration callback precedes UpdateEvidence, so its first hash is the
    # restored common state, rather than independently initialized adapters.
    t.train()
    assert t._step%4==0 and all(x.grad is None for x in t.model.parameters())
    manifest=save_learning_state(out/'components',t.model,t.optimizer,t.lr_scheduler,t.allocator,
        {'global_step':t.state.global_step,'micro_step':t._step,'at_optimizer_boundary':True},provenance(p))
    t.model.save_pretrained(out/'adapter')
    write(out/'summary.json',{'status':'complete','stop_reason':t.stop_reason,'optimizer_steps':t.state.global_step,
        'condition':t.condition,'rule':t.rule,'inherited_probe_tokens':t.inherited_tokens,
        'branch_sampled_tokens':t.sampled_tokens,'budget':p['branch_raw_token_budget'],
        'overshoot':max(0,t.sampled_tokens-p['branch_raw_token_budget']),
        'generation_seconds':t.generation_seconds,'non_generation_training_seconds':t.train_compute_seconds,
        'wall_seconds':time.perf_counter()-started,'final_fingerprint':fingerprint(t.model),'final_manifest':manifest})
    published_progress(out,'branch_complete',optimizer_steps=t.state.global_step,branch_sampled_tokens=t.sampled_tokens)


def evaluate(p,out,tokenizer,model,args,started):
    source=json.loads((args.policy/'summary.json').read_text());assert source['status']=='complete'
    model=PeftModel.from_pretrained(model,args.policy/'adapter',is_trainable=True).to('cuda:0').eval()
    initial=fingerprint(model)
    expected=source.get('final_fingerprint',source.get('policy_fingerprint'));assert initial==expected
    from transformers import GenerationConfig
    a=p['algorithm'];config=GenerationConfig(do_sample=True,temperature=a['temperature'],top_p=a['top_p'],
        top_k=a['top_k'],eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None,use_cache=True,max_new_tokens=a['segment_cap'])
    total=0;seconds=0.;outcomes=[]
    with torch.inference_mode():
        for index,row in enumerate(p['eval_panel']):
            if time.perf_counter()-started>=p['eval_soft_wall_seconds']:break
            key=row['task_id'];pair=load_tasks(p,[key],row['split'])[key]
            seed=p['seeds']['eval']+100*index;set_seed(seed)
            episodes=[episode(tokenizer,pair,p) for _ in range(p['eval_rollouts_per_task'])]
            def record(call):append(out/'generation_calls.jsonl',call|{'group':index,'phase':'evaluation','seed':seed})
            stats=sample_episodes(model,tokenizer,episodes,config,record,segment_cap=a['segment_cap'])
            total+=stats['sampled_tokens'];seconds+=stats['generation_seconds']
            for i,e in enumerate(episodes):append(out/'rollouts.jsonl',e.evidence()|{'group':index,'rollout_index':i,'phase':'evaluation','seed':seed})
            outcome=row|{'seed':seed,'rewards':[e.conversation.reward() for e in episodes],
                'sampled_tokens':stats['sampled_tokens'],'stop_reasons':[e.stop_reason for e in episodes]}
            outcomes.append(outcome);append(out/'outcomes.jsonl',outcome)
            published_progress(out,'evaluation',completed_tasks=index+1,planned_tasks=len(p['eval_panel']),
                sampled_tokens=total,latest=outcome)
    assert fingerprint(model)==initial and all(x.grad is None for x in model.parameters())
    write(out/'summary.json',{'status':'complete' if len(outcomes)==len(p['eval_panel']) else 'incomplete_panel',
        'policy':str(args.policy.relative_to(ROOT)),'policy_fingerprint':initial,'completed_tasks':len(outcomes),
        'sampled_tokens':total,'generation_seconds':seconds,'wall_seconds':time.perf_counter()-started,
        'optimizer_updates':0,'weights_unchanged':True})
    published_progress(out,'evaluation_finished',completed_tasks=len(outcomes),sampled_tokens=total)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['common','train','eval'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--common',type=Path);ap.add_argument('--policy',type=Path)
    ap.add_argument('--condition');ap.add_argument('--rule',choices=['uniform','frontier','coverage']);args=ap.parse_args()
    started=time.perf_counter();p=json.loads(PROTOCOL.read_text());out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    assert digest(ROOT/'rl/bfcl_research_split_candidate/manifest.json')==p['manifest_sha256']
    assert digest(PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json')==p['data_sha256']
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    if args.mode=='train':assert args.common and args.condition in p['conditions'] and args.rule in p['rules']
    if args.mode=='eval':assert args.policy
    names=['run_bfcl_intervention.py','bfcl_allocation.py','rl_learning_state.py','accept_bfcl_grpo.py',
        'bfcl_sampling.py','bfcl_token_rollout.py','bfcl_conversation.py','bfcl_safe_runtime.py','train_rl_smoke.py']
    write(out/'run.json',{'mode':args.mode,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(PROTOCOL),
        'condition':args.condition,'rule':args.rule,'policy':str(args.policy) if args.policy else None,
        'common':str(args.common) if args.common else None,'provenance':provenance(p),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{n:digest(ROOT/'pipeline'/n) for n in names}})
    set_seed(p['seeds']['initialization']);torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(ROOT/'smoke/model',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True)
    try:
        if args.mode=='common':build_common(p,out,tokenizer,model,started)
        elif args.mode=='train':train_branch(p,out,tokenizer,model,args,started)
        else:evaluate(p,out,tokenizer,model,args,started)
    except Exception as error:
        write(out/'failure.json',{'type':type(error).__name__,'message':str(error),
            'wall_seconds':time.perf_counter()-started,'partial_generation_evidence_retained':True})
        raise
if __name__=='__main__':main()
