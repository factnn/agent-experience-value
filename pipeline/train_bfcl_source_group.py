"""Corrected GRPO source-group intervention from a declared full learner state.

This runner does not choose scientific groups, stages, budgets, or held-out splits.
Those belong to a separately frozen job and algorithm protocol before execution.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import run_bfcl_intervention as r
from bfcl_dose_checkpoint import assert_equal,rng_state
from bfcl_loss_scaling_contract import configure_single_normalization
from bfcl_source_group_contract import validate_source,validate_intervention,learner_progress


def read(path):return json.loads(path.read_text())


class SourceTrainer(r.InterventionTrainer):
    def _generate_and_score_completions(self,inputs):
        local=self.state.global_step
        assert len(inputs)==4 and local==len(self.allocation_records)
        assert self._step==4*(self.source_step+local)
        assert self.accelerator.gradient_accumulation_steps==1
        assert self.sampled_tokens<self.job['token_budget']
        choice=self.allocator.choose();selected=self.training_rows[choice['task_id']]
        inputs=[copy.deepcopy(selected) for _ in range(4)]
        seed=self.job['rollout_seed_base']+100*(self.source_step+local)
        assert seed<2**32;r.set_seed(seed);before=self.sampled_tokens
        result=r.BFCLTrainer._generate_and_score_completions(self,inputs)
        rewards=[e.conversation.reward() for e in self.episodes];cost=self.sampled_tokens-before
        self.allocator.observe(choice['task_id'],rewards,cost,f"{self.job['case_id']}:{self.source_step+local}")
        row=choice|{'policy_step':local,'cumulative_policy_step':self.source_step+local,
            'seed':seed,'policy_fingerprint':self.current_fingerprint,'rewards':rewards,
            'sampled_tokens':cost,'branch_cost_after_group':self.sampled_tokens,
            'stop_reasons':[e.stop_reason for e in self.episodes]}
        self.allocation_records.append(row);r.append(self.evidence_dir/'allocation.jsonl',row)
        return result

    def training_step(self,*args,**kwargs):
        assert self.accelerator.gradient_accumulation_steps==1 and self.args.gradient_accumulation_steps==4
        result=super().training_step(*args,**kwargs)
        assert self.current_gradient_accumulation_steps==4
        r.append(self.evidence_dir/'microsteps.jsonl',{'micro_step_after':self._step,
            'local_optimizer_step':self.state.global_step,'loss':float(result),
            'accelerate_backward_divisor':self.accelerator.gradient_accumulation_steps,
            'grpo_loss_divisor':self.current_gradient_accumulation_steps})
        return result


class SourceState(r.TrainerCallback):
    def __init__(self,t):self.t=t
    def on_train_begin(self,args,state,control,**kwargs):
        t=self.t;job=t.job
        assert state.global_step==0 and t._step==0 and t._buffered_inputs is None
        assert t.accelerator.gradient_accumulation_steps==1 and args.gradient_accumulation_steps==4
        raw=t.optimizer.optimizer;assert t.optimizer.scaler is None
        source=r.ROOT/job['source_checkpoint']/'components';manifest=read(source/'manifest.json')
        assert manifest['payload_sha256']==job['source_payload_sha256']
        source_protocol=r.ROOT/job['source_protocol']['path'];assert r.digest(source_protocol)==job['source_protocol']['sha256']
        prior=read(source_protocol)
        expected=r.provenance(prior,source_protocol)
        allocator,progress=r.restore_learning_state(source,t.model,raw,t.lr_scheduler,expected)
        payload=r.torch.load(source/'learning_state.pt',map_location='cpu',weights_only=False)
        t.source_step=validate_source(prior['algorithm'],t.protocol['algorithm'],progress,payload['optimizer'],job['allow_legacy_zero_optimizer_source'])
        assert t.source_step==job['source_step']
        assert_equal(payload['optimizer'],raw.state_dict());assert_equal(payload['scheduler'],t.lr_scheduler.state_dict())
        assert_equal(payload['buffers'],dict(t.model.named_buffers()))
        assert_equal(payload['module_training'],{n:m.training for n,m in t.model.named_modules()})
        saved=payload['rng'];assert_equal({'python':saved['python'],'numpy':saved['numpy'],'cpu':saved['torch_cpu'],'cuda':saved['torch_cuda']},rng_state())
        original=allocator.state_dict();assert_equal(payload['allocator'],original)
        assert original['registry']==t.protocol['training_registry']
        # Pool restriction and independent private allocator seed are declared interventions.
        t.allocator=r.CombinationAllocator.from_state_dict(original,rule='uniform',pool=job['task_pool'])
        t.allocator.rng.seed(job['allocator_seed'])
        changed=t.allocator.state_dict()
        for key in original:
            if key not in ['pool','rule','rng_state']:assert_equal(original[key],changed[key])
        t.inherited_tokens=t.allocator.total_sampled_tokens
        assert t.inherited_tokens==manifest['cumulative_acquisition_tokens']
        t._step=4*t.source_step;t.current_fingerprint=r.fingerprint(t.model)
        assert r.tensor_hash((n,p) for n,p in t.model.named_parameters() if p.requires_grad)==manifest['trainable_sha256']
        r.write(t.evidence_dir/'restored.json',{'status':'verified','source_step':t.source_step,
            'source_payload_sha256':manifest['payload_sha256'],'source_provenance':expected,
            'policy_fingerprint':t.current_fingerprint,'inherited_tokens':t.inherited_tokens,
            'history_preserved':True,'fresh_rollout_buffer':True,'accelerator_backward_divisor':1,
            'declared_changes':{'task_pool':job['task_pool'],'allocator_seed':job['allocator_seed'],'rule':'uniform'},
            'initial_signals':t.allocator.signals(),'initial_probabilities':t.allocator.probabilities(),
            'optimizer_state_entries':len(raw.state),'legacy_zero_optimizer_transition':prior['algorithm']!=t.protocol['algorithm']})
        del payload
    def on_step_end(self,args,state,control,**kwargs):
        t=self.t;assert t._step==4*(t.source_step+state.global_step)
        assert len(t.allocation_records)==state.global_step
        if t.sampled_tokens>=t.job['token_budget']:t.stop_reason='token_budget_after_complete_group'
        elif state.global_step>=args.max_steps:t.stop_reason='max_updates_before_budget'
        elif time.perf_counter()-t.started>=t.job['soft_seconds']:t.stop_reason='soft_wall_before_budget'
        if t.stop_reason:control.should_training_stop=True
        r.published_progress(t.evidence_dir,'source_group_training',**learner_progress(t.source_step,state.global_step,t.inherited_tokens,t.sampled_tokens),stop_reason=t.stop_reason)
        return control


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);ap.add_argument('--job-sha256',required=True);args=ap.parse_args()
    assert r.digest(args.job)==args.job_sha256
    job=read(args.job);out=r.ROOT/job['output'];out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        for path,sha in job['input_sha256'].items():assert r.digest(r.ROOT/path)==sha,path
        protocol_path=r.ROOT/job['algorithm_protocol']['path'];assert r.digest(protocol_path)==job['algorithm_protocol']['sha256']
        p=read(protocol_path);validate_intervention(job,p['training_registry'])
        assert r.torch.cuda.is_available() and r.torch.cuda.device_count()==1
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        r.torch.set_num_threads(4);r.torch.use_deterministic_algorithms(True,warn_only=False)
        r.torch.backends.cudnn.deterministic=True;r.torch.backends.cudnn.benchmark=False
        r.set_seed(p['seeds']['initialization'])
        tokenizer=r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
        model=r.AutoModelForCausalLM.from_pretrained(r.ROOT/'smoke/model',dtype=r.torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        registry=r.load_tasks(p,list(p['training_registry']),'train')
        rows={k:{'task_id':k,'prompt':r.initial_prompt(pair[0])} for k,pair in registry.items()}
        config=r.config_for(p,out);config.max_steps=job['max_updates'];a=p['algorithm']
        def reward(**kwargs):return [e.conversation.reward() for e in t.episodes]
        t=SourceTrainer(model=model,args=config,processing_class=tokenizer,train_dataset=r.Dataset.from_list(list(rows.values())),reward_funcs=reward,
            peft_config=r.LoraConfig(r=a['lora_r'],lora_alpha=a['lora_alpha'],lora_dropout=a['lora_dropout'],target_modules=a['lora_targets'],task_type='CAUSAL_LM'))
        configure_single_normalization(t)
        assert not t.tools
        t.registry=registry;t.training_rows=rows;t.evidence_dir=out;t.protocol=p;t.job=job
        t.segment_cap=a['segment_cap'];t.episode_limits={'max_calls':a['max_calls'],'max_segments':a['max_segments']}
        t.sampled_tokens=0;t.generation_seconds=0.;t.train_compute_seconds=0.;t.allocation_records=[];t.stop_reason=None;t.started=started
        r.write(out/'run.json',{'job_sha256':args.job_sha256,'algorithm_protocol_sha256':r.digest(protocol_path),'case_id':job['case_id'],'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'role':job['role'],'normalization':'single','deterministic_algorithms':True})
        r.write(out/'trainer_config.json',json.loads(config.to_json_string()))
        t.add_callback(SourceState(t));t.add_callback(r.UpdateEvidence(t));t.train()
        assert t._step==4*(t.source_step+t.state.global_step)
        assert all(x.grad is None for x in t.model.parameters())
        progress=learner_progress(t.source_step,t.state.global_step,t.inherited_tokens,t.sampled_tokens)
        assert t.allocator.total_sampled_tokens==t.inherited_tokens+t.sampled_tokens
        manifest=r.save_learning_state(out/'components',t.model,t.optimizer.optimizer,t.lr_scheduler,t.allocator,progress,r.provenance(p,protocol_path))
        t.model.save_pretrained(out/'adapter')
        complete=t.stop_reason=='token_budget_after_complete_group'
        if complete:assert 0<=t.sampled_tokens-job['token_budget']<=4*a['whole_completion_budget']
        summary={'status':'complete' if complete else 'incomplete','case_id':job['case_id'],'role':job['role'],**progress,
            'new_sampled_tokens':t.sampled_tokens,'nominal_token_budget':job['token_budget'],'stop_reason':t.stop_reason,
            'final_fingerprint':r.fingerprint(t.model),'final_manifest':manifest,'generation_seconds':t.generation_seconds,
            'wall_seconds':time.perf_counter()-started,'evaluation_performed':False,
            'scope':'Source-group RL training only; transfer G/V requires frozen paired evaluation. Shared measurement costs are separate from ancestor/training lineage.'}
        r.write(out/'summary.json',summary)
        if not complete:raise RuntimeError('Training stopped before complete token budget; retained incomplete snapshot')
    except Exception as e:
        r.write(out/'failure.json',{'type':type(e).__name__,'message':str(e),'wall_seconds':time.perf_counter()-started,'partial_evidence_retained':True,'automatic_retry':False});raise

if __name__=='__main__':main()
