"""Stage1 fixed-ancestor source-group probe: feature panel, sealed forecasts, validation panel."""
import argparse
import json
import os
from pathlib import Path
import time
import run_bfcl_intervention as r
from transformers import GenerationConfig
from bfcl_probe_forecast import forecast,score
from bfcl_stage1_probe_contract import load_protocol, load_design
from bfcl_task_probe_statistics import compare_independent_panels


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--case',required=True);ap.add_argument('--protocol-sha256',required=True);args=ap.parse_args()
    p,binding=load_protocol(r.ROOT,args.protocol,args.protocol_sha256);case=next(x for x in p['cases'] if x['name']==args.case)
    out=r.ROOT/p['output_root']/case['name'];out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cost=0
    try:
        for path,sha in p['input_sha256'].items():assert r.digest(r.ROOT/path)==sha,path
        assert r.torch.cuda.is_available() and r.torch.cuda.device_count()==1
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        r.torch.set_num_threads(4);r.torch.use_deterministic_algorithms(True,warn_only=False)
        r.torch.backends.cudnn.deterministic=True;r.torch.backends.cudnn.benchmark=False
        source=json.loads((r.ROOT/p['source_protocol']['path']).read_text());a=source['algorithm']
        tokenizer=r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True);tokenizer.padding_side='left'
        model=r.AutoModelForCausalLM.from_pretrained(r.ROOT/'smoke/model',dtype=r.torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model=r.PeftModel.from_pretrained(model,r.ROOT/p['policy_path']/'adapter',is_trainable=True).to('cuda:0').eval()
        initial=r.fingerprint(model);assert initial==p['policy_fingerprint']
        registry=r.load_tasks(source,case['task_ids'],'train')
        config=GenerationConfig(do_sample=True,temperature=a['temperature'],top_p=a['top_p'],top_k=a['top_k'],eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,bos_token_id=None,use_cache=True,max_new_tokens=a['segment_cap'])
        r.write(out/'run.json',{'protocol_sha256':r.digest(args.protocol),'case':case,'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'policy_fingerprint':initial,'policy_stage':1,'parent_binding':p['parent_binding'],'parent_payload_sha256':binding['learning_state']['sha256'],'source_step':binding['source_step'],'started_unix':time.time(),'optimizer_updates':0,'deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']})
        panels={};generation_seconds=0
        with r.torch.inference_mode():
            for phase in ['feature','validation']:
                rows=[]
                if phase=='validation':
                    r.write(out/'validation_start.json',{'started_unix':time.time(),'prediction_sha256':r.digest(out/'predictions.json')})
                for index,key in enumerate(case['task_ids']):
                    seed=case['seeds'][phase][index];r.set_seed(seed)
                    group_index=index+(0 if phase=='feature' else len(case['task_ids']))
                    episodes=[r.episode(tokenizer,registry[key],source) for _ in range(4)]
                    draw_id=case['name']+':'+phase+':'+key
                    def record(call):r.append(out/'generation_calls.jsonl',call|{'phase':phase,'group':group_index,'seed':seed,'draw_id':draw_id})
                    stats=r.sample_episodes(model,tokenizer,episodes,config,record,segment_cap=a['segment_cap'])
                    cost+=stats['sampled_tokens'];generation_seconds+=stats['generation_seconds'];assert cost<=case['token_ceiling']
                    for i,e in enumerate(episodes):r.append(out/'rollouts.jsonl',e.evidence()|{'phase':phase,'group':group_index,'rollout_index':i,'seed':seed,'draw_id':draw_id,'policy_fingerprint':initial,'stage':1})
                    row={'task_id':key,'source_group':case['source_group'],'policy_fingerprint':initial,'stage':1,'draw_id':draw_id,'phase':phase,'seed':seed,'rewards':[e.conversation.reward() for e in episodes],'sampled_tokens':stats['sampled_tokens'],'stop_reasons':[e.stop_reason for e in episodes]}
                    rows.append(row);r.append(out/(phase+'.jsonl'),row)
                    r.write(out/'progress.json',{'phase':phase,'tasks_complete':index+1,'tasks_per_panel':len(case['task_ids']),'new_sampled_tokens':cost,'pid':os.getpid()})
                panels[phase]=rows
                assert r.fingerprint(model)==initial
                if phase=='feature':
                    prediction=forecast(rows,case['historical_signal']);r.write(out/'predictions.json',prediction)
                    sealed=r.digest(out/'predictions.json');r.write(out/'prediction_receipt.json',{'sha256':sealed,'sealed_before_validation':True,'feature_rows':len(rows),'validation_rows_at_seal':0,'sealed_unix':time.time()})
            assert r.digest(out/'predictions.json')==sealed
        assert all(x.grad is None for x in model.parameters())
        # Probe inference must not write back into the inherited learning state.
        load_design(r.ROOT,p['design_inputs'])
        paired=compare_independent_panels(panels['feature'],panels['validation']);scores=score(prediction,panels['validation'])
        r.write(out/'summary.json',{'status':'complete','case':case['name'],'source_group':case['source_group'],'new_sampled_tokens':cost,'generation_seconds':generation_seconds,'wall_seconds':time.perf_counter()-started,'optimizer_updates':0,'policy_stage':1,'parent_binding':p['parent_binding'],'parent_payload_sha256':binding['learning_state']['sha256'],'parent_state_unchanged':True,'weights_unchanged':r.fingerprint(model)==initial,'policy_fingerprint':initial,'prediction_sha256':sealed,'panels':{k:len(v) for k,v in panels.items()},'scores':scores,'task_statistics':paired,'scope':'Fixed stage1 policy local opportunity calibration on training tasks; no RL transfer label or efficacy claim.'})
    except Exception as error:
        r.write(out/'failure.json',{'status':'failed','type':type(error).__name__,'message':str(error),'completed_group_tokens':cost,'partial_generation_cost_requires_audit':True,'wall_seconds':time.perf_counter()-started,'automatic_retry':False});raise

if __name__=='__main__':main()
