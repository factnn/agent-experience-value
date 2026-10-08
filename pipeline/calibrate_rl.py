"""Frozen-policy development calibration; no optimizer or held-out test use."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

from train_rl_smoke import (EvidenceTrainer, GRPOConfig, LoraConfig, AutoModelForCausalLM,
                            AutoTokenizer, Dataset, append, fingerprint, set_seed, torch)
from rl_environment import ROOT, MessageEnv, make_task, oracle

PROTOCOL = (' Contact names are not contact IDs. Obtain IDs with lookup or add_contact, '
            'then wait for the returned values before making dependent calls. '
            'Never use guessed IDs or placeholders. Log in using the sender ID returned by lookup. '
            'Inspect each tool result and correct errors. Do not repeat a successful send. '
            'Keep reasoning brief and move to the next tool action promptly.')


def tasks_for(condition):
    tasks = []
    for seed in [91010, 91011]:
        for family in ['send', 'new_contact', 'replace']:
            task = make_task(seed, family, split='development_calibration')
            if condition == 'protocol':
                task['prompt'][0]['content'] += PROTOCOL
            tasks.append(task)
    return tasks


def summarize(path):
    rows = [json.loads(line) for line in (path/'rollouts.jsonl').read_text().splitlines()]
    calls = [json.loads(line) for line in (path/'generation_calls.jsonl').read_text().splitlines()]
    run = json.loads((path/'run.json').read_text())
    results = []
    for task_id in dict.fromkeys(row['task_id'] for row in rows):
        batch = [r for r in rows if r['task_id'] == task_id]
        phase = batch[0]['phase']
        raw_calls = [g for g in calls if g['phase'] == phase]
        for r in batch:
            sequence = r['prompt_ids'] + r['completion_ids']
            candidates = [p+c for g in raw_calls for p,c in zip(g['prompt_ids'],g['completion_ids'])]
            assert any(sequence == c[:len(sequence)] for c in candidates)
            assert r['policy_fingerprint'] == run['policy_fingerprint']
            assert len(r['completion_ids']) == len(r['model_token_mask'])
        rewards = [r['reward'] for r in batch]
        results.append({'task_id': task_id, 'family': task_id.split(':')[1],
                        'rewards': rewards, 'mixed_reward_group': len(set(rewards)) > 1,
                        'cap_hits': sum(len(r['completion_ids']) >= run['max_completion_length'] for r in batch),
                        'model_tokens_retained': sum(r['model_tokens'] for r in batch),
                        'sampled_tokens': sum(sum(g['sampled_tokens']) for g in raw_calls),
                        'generation_seconds': batch[0]['seconds'],
                        'environment_error_results': sum('error' in e['result'] or any(v is False for k,v in e['result'].items() if k.endswith('_status')) for r in batch for e in r['events'])})
    expected = run.get('task_limit', 6)
    cap_hits = sum(r['cap_hits'] for r in results)
    mixed = sum(r['mixed_reward_group'] for r in results)
    mixed_families = sorted({r['family'] for r in results if r['mixed_reward_group']})
    total_tokens = sum(r['sampled_tokens'] for r in results)
    return {'condition': run['condition'], 'status': 'complete' if len(results)==expected else 'partial',
            'episodes': len(rows), 'successes': sum(r['reward'] for r in rows),
            'trajectory_cap_hits': cap_hits, 'mixed_reward_groups': mixed,
            'families_with_mixed_rewards': mixed_families,
            'sampled_tokens_including_discarded': total_tokens,
            'discarded_tokens': total_tokens - sum(r['model_tokens_retained'] for r in results),
            'generation_seconds': sum(r['generation_seconds'] for r in results),
            'readiness_screen_passed': (len(results)==6 and cap_hits <= 6 and mixed >= 3 and len(mixed_families)>=2) if expected==6 else None,
            'raw_token_history_audit': 'passed', 'tasks': results,
            'interpretation': 'Development calibration only; no RL update or allocation/transfer effect estimate.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition', choices=['original','protocol'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-completion-length',type=int,default=2048)
    ap.add_argument('--task-limit',type=int,choices=range(1,7),default=6)
    args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();set_seed(20261008);torch.set_num_threads(4)
    tasks=tasks_for(args.condition)[:args.task_limit]
    for task in tasks:
        env=MessageEnv();env.reset(json.dumps(task));assert env._reward()==0
        oracle(env);assert env._reward()==1
    rows=[{'prompt':t['prompt'],'task_json':json.dumps(t),'task_id':t['task_id']} for t in tasks]
    (out/'tasks.json').write_text(json.dumps(rows,indent=2)+'\n')
    run=vars(args)|{'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'purpose':'fixed-base-policy development calibration; no training',
                   'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   'source_sha256':{n:hashlib.sha256((ROOT/'pipeline'/n).read_bytes()).hexdigest() for n in ['calibrate_rl.py','train_rl_smoke.py','rl_environment.py']},
                   'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','trl','peft']}}
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True)
    tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(ROOT/'smoke/model',dtype=torch.bfloat16,
                                             attn_implementation='sdpa',local_files_only=True)
    config=GRPOConfig(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=4,
                      num_generations=4,per_device_eval_batch_size=4,max_steps=1,
                      max_completion_length=args.max_completion_length,max_tool_calling_iterations=8,
                      bf16=True,beta=0.0,loss_type='grpo',scale_rewards='group',num_iterations=1,
                      temperature=1.0,top_p=1.0,top_k=0,seed=20261008,
                      chat_template_kwargs={'enable_thinking':True},save_strategy='no',report_to='none',
                      disable_tqdm=True,use_vllm=False,mask_truncated_completions=False)
    (out/'trainer_config.json').write_text(config.to_json_string())
    def reward(environments,**kwargs):return [env._reward() for env in environments]
    trainer=EvidenceTrainer(model=model,args=config,processing_class=tokenizer,train_dataset=Dataset.from_list(rows),
                            reward_funcs=reward,environment_factory=MessageEnv,
                            peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    trainer.model.eval();trainer.evidence_dir=out;trainer.generation_seconds=0.;trainer.sampled_tokens=0
    trainer.current_fingerprint=fingerprint(trainer.model)
    run['policy_fingerprint']=trainer.current_fingerprint
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    for i,row in enumerate(rows):
        set_seed(20261008+i)
        trainer.evidence_phase=f"calibration:{i}"
        with torch.no_grad():
            trainer._generate_and_score_completions([copy.deepcopy(row) for _ in range(4)])
        assert fingerprint(trainer.model)==trainer.current_fingerprint
        progress=summarize(out);progress['main_wall_seconds']=time.perf_counter()-start
        (out/'summary.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps({'completed_tasks':i+1,'condition':args.condition,'latest':progress['tasks'][-1]}),flush=True)
    progress['policy_unchanged']=True
    progress['optimizer_steps']=0
    progress['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    progress['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    (out/'summary.json').write_text(json.dumps(progress,indent=2)+'\n')


if __name__=='__main__':main()
