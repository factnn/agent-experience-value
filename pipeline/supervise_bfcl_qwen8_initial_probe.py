"""Own one8B common state then reuse the existing eight-lane fixed probe scheduler."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from bfcl_qwen8_initial_probe import load_master,pinned,ref,exclusive,audit_common,bind_probe,audit_probe,complete_probe,check_binding
from bfcl_stage1_probe_ssh import execute_lanes
from supervise_bfcl_stage_fork import ROOT,read,write,supervise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master',type=Path,required=True);parser.add_argument('--master-sha256',required=True)
    parser.add_argument('--publication',type=Path,required=True);args=parser.parse_args()
    p,source=load_master(args.master,args.master_sha256)
    control=ROOT/p['control_root'];master_ref={'path':str(args.master.relative_to(ROOT)),'sha256':args.master_sha256}
    publication=read(args.publication)
    assert publication['commit']==publication['remote_main'] and publication['scientific_child_started'] is False
    assert publication['files'][master_ref['path']]==master_ref['sha256']
    for name,sha in publication['files'].items():assert ref(ROOT/name)['sha256']==sha,name
    assert not (control/'launch.json').exists() and not (ROOT/p['common_output']).exists()
    def admitted(gpu):
        # These markers close NEW admission; already launched children keep their
        # original hard limits. Explicit termination acts only on owned PGIDs.
        for name in [p['stop_file'],p['global_stop_file']]:
            if (ROOT/name).exists():raise RuntimeError('Current SSH owner stopped new admission: '+name)
        assert gpu in p['allowed_gpus']
        info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if int(info[1].strip())>64 or info[0].strip() in apps:raise RuntimeError('GPU occupied; no new scientific child launched')
        return {'gpu':gpu,'memory_mib':int(info[1].strip()),'no_compute_process':True,'master':master_ref,'publication':ref(args.publication),'current_owner':p['current_ssh_owner']}
    def launch(gpu,out,log,hard,tail):
        command=[str(ROOT/'.venv-rl/bin/python'),'-u',str(ROOT/'pipeline/run_bfcl_qwen8_initial_probe.py'),
            '--master',str(args.master),'--master-sha256',args.master_sha256,*tail]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        return supervise(command,out,log,hard,env)
    phase='common_admission'
    try:
        free=[]
        for gpu in p['allowed_gpus']:
            try:checked=admitted(gpu);free.append((gpu,checked))
            except RuntimeError as error:
                if 'GPU occupied' not in str(error):raise
        if not free:raise RuntimeError('No actually free GPU; no common child launched')
        gpu,preflight=free[0]
        exclusive(control/'launch.json',{'supervisor_pid':os.getpid(),'started_unix':time.time(),'master':master_ref,'publication':ref(args.publication),'common_gpu':gpu,'common_preflight':preflight,'maximum_simultaneous_gpus':8})
        phase='common_runner'
        result=launch(gpu,ROOT/p['common_output'],control/'common.console.log',p['common_hard_seconds'],['--mode','common'])
        assert result['returncode']==0 and result['hard_timeout'] is False and result['status']=='runner_complete_pending_replay_audit',result
        phase='common_audit';audit,signals=audit_common(p)
        plan,plan_ref=bind_probe(p,master_ref,audit,signals)
        exclusive(control/'bound_probe_plan.json',plan_ref)
        (ROOT/p['probe_output']).mkdir(exist_ok=False)
        phase='source_probe_execution'
        def check(gpu):
            check_binding(plan);return admitted(gpu)
        def run(gpu,case):
            return launch(gpu,ROOT/p['probe_output']/case['name'],control/(case['name']+'.console.log'),case['hard_seconds'],
                ['--mode','probe','--case',case['name'],'--probe-plan',str(ROOT/plan_ref['path']),'--probe-plan-sha256',plan_ref['sha256']])
        def audit_case(name):
            if name is None:complete_probe(plan,plan_ref);return
            case=next(c for c in plan['cases'] if c['name']==name)
            exclusive(ROOT/p['probe_output']/name/'audit.json',audit_probe(plan,case,plan_ref['sha256']))
        result=execute_lanes(control,plan,check,run,audit_case)
        exclusive(control/'result.json',{'phase':phase,'queue':result,'automatic_retry':False,'goal_complete':False})
    except Exception as error:
        exclusive(control/'control_failure.json',{'phase':phase,'type':type(error).__name__,'message':str(error),'scientific_retry':False,'partial_evidence_retained':True})
        raise


if __name__=='__main__':main()
