"""Frozen development-only, fixed-base sampling. No optimizer or GRPO update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig,set_seed
import torch
from bfcl_token_rollout import ROOT,TokenEpisode,development_tasks
from bfcl_sampling import sample_episodes
from rl_learning_state import tensor_hash


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def append(path,value):
    with path.open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def publish(out,label):
    # Explicit artifacts only; do not add checkpoint binaries or raw docs.
    files=[str(p.relative_to(ROOT)) for p in out.iterdir() if p.suffix in {'.json','.jsonl'}]
    files+=['rl/STATUS.json']
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True,capture_output=True)
    change=subprocess.run(['git','diff','--cached','--quiet','--',*files],cwd=ROOT)
    if change.returncode:
        subprocess.run(['git','commit','--only','-m',label,'--',*files],cwd=ROOT,check=True,capture_output=True)
    result=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True)
    # Do not log remote/error strings, which could include credentialed URLs.
    print(json.dumps({'publication':'pushed' if result.returncode==0 else 'push_failed',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'returncode':result.returncode}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--out',required=True);p.add_argument('--push',action='store_true');args=p.parse_args()
    start=time.perf_counter();protocol=json.loads(args.protocol.read_text())
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256((ROOT/'rl/bfcl_research_split_candidate/manifest.json').read_bytes()).hexdigest()==protocol['development_manifest_sha256']
    tasks,answers=development_tasks(protocol['task_ids'])
    for key in tasks:
        digest=hashlib.sha256(json.dumps(tasks[key],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert digest==protocol['task_record_sha256'][key]
    source_names=['calibrate_bfcl.py','bfcl_sampling.py','bfcl_token_rollout.py','bfcl_conversation.py','bfcl_safe_runtime.py']
    run={'purpose':'Fixed-base development calibration; zero optimizer updates',
        'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':protocol,
        'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{name:hashlib.sha256((ROOT/'pipeline'/name).read_bytes()).hexdigest() for name in source_names}}
    write(out/'run.json',run)
    status_path=ROOT/'rl/STATUS.json';status=json.loads(status_path.read_text())
    status.update({'status':'bfcl_calibration_running','active_gpu_count':1,
        'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    status['bfcl_calibration']={'path':str(out.relative_to(ROOT)),'pid':os.getpid(),
        'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'status':'loading_fixed_base',
        'protocol':str(args.protocol)}
    write(status_path,status)
    set_seed(protocol['seed']);torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Select exactly one idle GPU'
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(ROOT/'smoke/model',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
    assert next(model.parameters()).device.type=='cuda'
    run['execution_device']=str(next(model.parameters()).device)
    model.requires_grad_(False)
    initial_hash=tensor_hash(model.named_parameters());run['initial_weights_sha256']=initial_hash
    write(out/'run.json',run)
    config=GenerationConfig(do_sample=True,temperature=protocol['temperature'],top_p=protocol['top_p'],
        top_k=protocol['top_k'],eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None,use_cache=True,max_new_tokens=protocol['segment_cap'])
    write(out/'generation_config.json',config.to_dict())
    sampled=0;generation_seconds=0.;groups=[];stop_reason=None
    def limit():
        if sampled>=protocol['raw_token_budget']:return 'global_token_limit'
        if time.perf_counter()-start>=protocol['soft_wall_seconds']:return 'global_wall_limit'
        return None
    with torch.inference_mode():
        for index,key in enumerate(protocol['task_ids']):
            for mode in protocol['modes']:
                if limit():stop_reason=limit();break
                group=len(groups);seed=protocol['seed']+100*index;set_seed(seed)
                episodes=[TokenEpisode(tokenizer,tasks[key],answers[key],budget=protocol['whole_completion_budget'],
                    max_calls=protocol['max_calls'],max_segments=protocol['max_segments'],
                    enable_thinking=(mode=='thinking')) for _ in range(protocol['rollouts_per_group'])]
                expected='<|im_start|>assistant\n'+('' if mode=='thinking' else '<think>\n\n</think>\n\n')
                assert all(tokenizer.decode(e.prompt_ids).endswith(expected) for e in episodes)
                before=sampled
                def record(row):
                    nonlocal sampled
                    sampled+=len(row['generated_ids'])
                    append(out/'generation_calls.jsonl',row|{'group':group,'mode':mode,'seed':seed})
                stats=sample_episodes(model,tokenizer,episodes,config,record,
                    segment_cap=protocol['segment_cap'],should_stop=limit)
                generation_seconds+=stats['generation_seconds']
                assert sampled-before==stats['sampled_tokens']
                for i,e in enumerate(episodes):
                    for bridge in e.bridges:
                        assert tokenizer.decode(bridge['ids']).endswith(expected)
                    append(out/'rollouts.jsonl',e.evidence()|{'group':group,'mode':mode,'seed':seed,
                        'rollout_index':i,'weights_sha256':initial_hash})
                rewards=[e.conversation.reward() for e in episodes]
                row={'group':group,'task_id':key,'mode':mode,'seed':seed,'rewards':rewards,
                    'sampled_tokens':stats['sampled_tokens'],'generation_seconds':stats['generation_seconds'],
                    'censored':stats['censored'],'mixed_rewards':len(set(rewards))>1,
                    'completed_conversations':sum(e.conversation.completed for e in episodes),
                    'stop_reasons':[e.stop_reason for e in episodes]}
                groups.append(row);append(out/'groups.jsonl',row)
                progress={'status':'sampling','complete_groups':len(groups),'planned_groups':len(protocol['task_ids'])*len(protocol['modes']),
                    'sampled_tokens':sampled,'wall_seconds':time.perf_counter()-start,'latest':row}
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                status_path=ROOT/'rl/STATUS.json';status=json.loads(status_path.read_text())
                status.update({'status':'bfcl_calibration_running','active_gpu_count':1,
                    'updated_at_utc':datetime.now(timezone.utc).isoformat()})
                status['bfcl_calibration']={'path':str(out.relative_to(ROOT)),'pid':os.getpid(),
                    'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'progress':progress,
                    'protocol':str(args.protocol.relative_to(ROOT)) if args.protocol.is_absolute() else str(args.protocol)}
                write(status_path,status)
                if args.push:publish(out,f'Publish BFCL fixed-base calibration group {group}: {mode} {key}')
                if stats['censored']:stop_reason=next(e.stop_reason for e in episodes if e.stop_reason.startswith('global_'));break
            if stop_reason:break
    final_hash=tensor_hash(model.named_parameters());assert final_hash==initial_hash
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    write(out/'summary.json',{'status':'complete' if len(groups)==len(protocol['task_ids'])*len(protocol['modes']) and not stop_reason else 'bounded_stop',
        'stop_reason':stop_reason or 'fixed_manifest_complete','groups':len(groups),'planned_groups':len(protocol['task_ids'])*len(protocol['modes']),
        'sampled_tokens':sampled,'generation_seconds':generation_seconds,'wall_seconds':time.perf_counter()-start,
        'initial_weights_sha256':initial_hash,'final_weights_sha256':final_hash,'optimizer_updates':0,
        'limitation':'Fixed-policy development only. Mode/budget calibration, not RL efficacy or selector/transfer evidence.'})
    status_path=ROOT/'rl/STATUS.json';status=json.loads(status_path.read_text())
    status.update({'status':'bfcl_calibration_sampling_complete','active_gpu_count':1})
    status['bfcl_calibration'].update({'status':'sampling_finished_process_exiting','summary':str((out/'summary.json').relative_to(ROOT))})
    write(status_path,status)
    if args.push:publish(out,'Publish bounded BFCL fixed-base calibration sampling result')


if __name__=='__main__':main()
