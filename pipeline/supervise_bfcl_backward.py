"""One bounded diagnostic batch, four predetermined cases, no automatic retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from supervise_bfcl_stage_fork import ROOT, read, write, preflight, supervise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest()==args.protocol_sha256
    plan=read(args.protocol)
    for path,expected in plan['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    for eid in plan.get('required_review_events',['E030']):
        assert read(ROOT/'experiments/reviews'/(eid+'.json'))['status']=='reviewed'
    assert read(ROOT/'rl/bfcl_stage_fork_acceptance_20261011/replay/replay_audit.json')['status']=='mismatch_requires_review'
    preflight(args.gpu,plan)
    root=ROOT/plan['output_root'];root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write(root/'launch.json',{'supervisor_pid':os.getpid(),'gpu':args.gpu,'started_unix':time.time(),
        'protocol_sha256':args.protocol_sha256,'hard_seconds_total':plan['hard_seconds_total']})
    try:
        for case in plan['cases']:
            preflight(args.gpu,plan)
            remaining=plan['hard_seconds_total']-(time.monotonic()-started)
            if remaining <= 30:
                raise TimeoutError('Diagnostic total deadline reached')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu))
            # Separate processes: deterministic mode config is set before CUDA initialization.
            env.pop('CUBLAS_WORKSPACE_CONFIG',None)
            if case['deterministic']:env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            command=[str(ROOT/'.venv-rl/bin/python'),'-u',str(ROOT/'pipeline/diagnose_bfcl_backward.py'),
                '--protocol',str(args.protocol.resolve()),'--case',case['name']]
            receipt=supervise(command,root/case['name'],root/(case['name']+'.console.log'),
                              min(plan['hard_seconds_per_case'],remaining-30),env,grace_seconds=30)
            if receipt['returncode']!=0 or receipt['hard_timeout'] or not (root/case['name']/'summary.json').exists():
                raise RuntimeError('Predetermined diagnostic case failed: '+case['name'])
        command=[str(ROOT/'.venv-rl/bin/python'),str(ROOT/'pipeline/report_bfcl_backward.py'),
                 '--protocol',str(args.protocol.resolve())]
        with (root/'audit.console.log').open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        assert result.returncode==0,'Diagnostic evidence audit failed'
        write(root/'supervisor.json',{'status':'diagnostic_complete_pending_review','wall_seconds':time.monotonic()-started,
              'new_sampled_tokens':0,'automatic_retry':False})
    except Exception as error:
        write(root/'failures.json',[{'type':type(error).__name__,'message':str(error),'new_sampled_tokens':0,
              'wall_seconds':time.monotonic()-started,'partial_evidence_retained':True,'automatic_retry':False}])
        raise


if __name__=='__main__':main()
