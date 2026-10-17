"""Run native8B common/probe through the existing unchanged measurement routines."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import run_bfcl_intervention as r
from bfcl_qwen8_initial_probe import load_master,pinned,verify_model,check_binding,REVISION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master',type=Path,required=True);parser.add_argument('--master-sha256',required=True)
    parser.add_argument('--mode',choices=['common','probe'],required=True);parser.add_argument('--case')
    parser.add_argument('--probe-plan',type=Path);parser.add_argument('--probe-plan-sha256');args=parser.parse_args()
    master,source=load_master(args.master,args.master_sha256);model_path=verify_model(master)
    assert r.torch.cuda.is_available() and r.torch.cuda.device_count()==1 and os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    r.torch.set_num_threads(4);r.torch.use_deterministic_algorithms(True,warn_only=False)
    r.torch.backends.cudnn.deterministic=True;r.torch.backends.cudnn.benchmark=False
    master_ref={'path':str(args.master.relative_to(r.ROOT)),'sha256':args.master_sha256}
    if args.mode=='common':
        out=r.ROOT/master['common_output'];out.mkdir(exist_ok=False);started=time.perf_counter()
        try:
            r.PROTOCOL=r.ROOT/master['source_protocol']['path'];r.set_seed(source['seeds']['initialization'])
            r.write(out/'run.json',{'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'started_unix':time.time(),
                'native_master':master_ref,'source_protocol':master['source_protocol'],'model_verification':master['model_verification'],
                'model_revision':REVISION,'native_model_path':str(model_path),'common_optimizer_updates':0,
                'target_evaluation':False,'reserved21_accessed':False,
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip(),
                'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','trl','peft','accelerate','datasets','tokenizers']}})
            tokenizer=r.AutoTokenizer.from_pretrained(model_path,local_files_only=True);tokenizer.padding_side='left'
            model=r.AutoModelForCausalLM.from_pretrained(model_path,dtype=r.torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
            r.build_common(source,out,tokenizer,model,started)
        except Exception as error:
            r.write(out/'failure.json',{'type':type(error).__name__,'message':str(error),'wall_seconds':time.perf_counter()-started,'partial_evidence_retained':True,'automatic_retry':False});raise
        return
    assert args.case and args.probe_plan and args.probe_plan_sha256
    plan_ref={'path':str(args.probe_plan.relative_to(r.ROOT)),'sha256':args.probe_plan_sha256}
    plan=pinned(plan_ref);assert plan['master']==master_ref and plan['native_model_path']==str(model_path)
    check_binding(plan)
    # The legacy entry point hard-codes smoke/model in exactly two factory calls.
    # Redirect only those calls in this process; templates, tokenization, generation,
    # prediction seals, masks and scoring run in the unchanged original routines.
    tokenizer_factory=r.AutoTokenizer;model_factory=r.AutoModelForCausalLM;original_write=r.write
    class NativeTokenizer:
        @staticmethod
        def from_pretrained(path,**kwargs):
            assert Path(path)==r.ROOT/'smoke/model'
            return tokenizer_factory.from_pretrained(model_path,**kwargs)
    class NativeModel:
        @staticmethod
        def from_pretrained(path,**kwargs):
            assert Path(path)==r.ROOT/'smoke/model'
            return model_factory.from_pretrained(model_path,**kwargs)
    def annotated_write(path,value):
        if path.name=='run.json':
            value=dict(value,model_revision=REVISION,native_model_path=str(model_path),native_master=master_ref,
                native_common_binding=plan['common_binding'],target_evaluation=False,reserved21_accessed=False,
                model_factory_override='Process-local native8B factories; original4B files unchanged')
        return original_write(path,value)
    r.AutoTokenizer=NativeTokenizer;r.AutoModelForCausalLM=NativeModel;r.write=annotated_write
    import run_bfcl_task_probe
    sys.argv=[sys.argv[0],'--protocol',str(args.probe_plan),'--case',args.case]
    run_bfcl_task_probe.main()
    check_binding(plan)


if __name__=='__main__':main()
