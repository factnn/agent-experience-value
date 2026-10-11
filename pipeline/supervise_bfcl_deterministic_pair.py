"""Sequential deterministic reference and fork, with immutable reference receipt."""
import argparse
import hashlib
import os
import subprocess
import time
from pathlib import Path
from supervise_bfcl_stage_fork import ROOT,read,write,supervise
from bfcl_diagnostic_preflight import preflight


def report(protocol,root,reference=False):
    command=[str(ROOT/'.venv-rl/bin/python'),str(ROOT/'pipeline/report_bfcl_deterministic_pair.py'),
             '--protocol',str(protocol)]
    if reference:command.append('--freeze-reference')
    with (root/('reference_audit.console.log' if reference else 'pair_audit.console.log')).open('x') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=180,
                              env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    if result.returncode:raise RuntimeError('Reference audit failed' if reference else 'Pair audit failed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();protocol=args.protocol.resolve()
    assert hashlib.sha256(protocol.read_bytes()).hexdigest()==args.protocol_sha256
    plan=read(protocol)
    for path,sha in plan['input_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    for eid in plan['required_review_events']:assert read(ROOT/'experiments/reviews'/(eid+'.json'))['status']=='reviewed'
    assert read(ROOT/'rl/bfcl_backward_diagnostic_v3_20261011/audit_status.json')['status']=='diagnostic_complete'
    check=preflight(args.gpu,plan)
    root=ROOT/plan['output_root'];root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write(root/'launch.json',{'supervisor_pid':os.getpid(),'gpu':args.gpu,'started_unix':time.time(),
        'protocol_sha256':args.protocol_sha256,'preflight':check})
    try:
        for case in ['continuous','restored']:
            check=preflight(args.gpu,plan);write(root/(case+'.preflight.json'),check)
            receipt=supervise([str(ROOT/'.venv-rl/bin/python'),'-u',str(ROOT/'pipeline/accept_bfcl_deterministic_pair.py'),
                '--protocol',str(protocol),'--case',case],root/case,root/(case+'.console.log'),
                plan['hard_seconds'][case],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            if receipt['status']!='runner_complete_pending_replay_audit':raise RuntimeError(case+' failed or incomplete')
            if case=='continuous':report(protocol,root,reference=True)
        report(protocol,root)
        write(root/'supervisor.json',{'status':'pair_audited_pending_review','wall_seconds':time.monotonic()-started,
                                     'automatic_retry':False})
    except Exception as error:
        write(root/'failures.json',[{'type':type(error).__name__,'message':str(error),
            'wall_seconds':time.monotonic()-started,'partial_evidence_retained':True,'automatic_retry':False}])
        raise


if __name__=='__main__':main()
