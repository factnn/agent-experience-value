"""One bounded corrected full-GPU-loop integration, without generation/retries."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from supervise_bfcl_stage_fork import ROOT,read,write,supervise
from bfcl_diagnostic_preflight import preflight


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True);ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args()
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest()==args.protocol_sha256
    p=read(args.protocol)
    for path,sha in p['input_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    for eid in p['required_review_events']:assert read(ROOT/'experiments/reviews'/(eid+'.json'))['status']=='reviewed'
    check=preflight(args.gpu,p);root=ROOT/p['output_root'];root.mkdir(parents=True,exist_ok=False)
    write(root/'launch.json',{'supervisor_pid':os.getpid(),'gpu':args.gpu,'started_unix':time.time(),'protocol_sha256':args.protocol_sha256,'preflight':check})
    try:
        result=supervise([str(ROOT/'.venv-rl/bin/python'),'-u',str(ROOT/'pipeline/check_bfcl_normalization_gpu_loop.py'),'--protocol',str(args.protocol.resolve())],root/'loop',root/'loop.console.log',p['hard_seconds'],dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        assert result['returncode']==0 and not result['hard_timeout'],'Full-loop integration failed'
        s=read(root/'loop/summary.json');assert s['acceptance_status']=='passed'
        write(root/'audit_status.json',{'status':'passed','new_sampled_tokens':0,'actual_trainer_train':True,'summary_sha256':hashlib.sha256((root/'loop/summary.json').read_bytes()).hexdigest(),'scope':s['scope']})
    except Exception as error:
        write(root/'failures.json',[{'type':type(error).__name__,'message':str(error),'new_sampled_tokens':0,'automatic_retry':False}]);raise


if __name__=='__main__':main()
