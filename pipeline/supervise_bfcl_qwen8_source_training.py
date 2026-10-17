"""Use existing eight-lane single-attempt supervision for the E369 source jobs."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from bfcl_qwen8_source_execution import (exclusive,load_execution,owner,prerequisites,published,ref,SCOPE)
from bfcl_source_eval_contract import pinned
from bfcl_stage1_probe_ssh import execute_lanes
from supervise_bfcl_stage_fork import read,supervise


def run(root,execution_ref,authority_ref,publication_ref):
    master,design,jobs=load_execution(root,execution_ref)
    authority=owner(root,authority_ref,execution_ref,master,live=True,started=time.time())
    prerequisites(root,master);published(root,publication_ref,execution_ref,authority_ref,master,authority)
    control=root/authority['control_root'];output=root/master['output_root']
    if (authority['control_root']!=master['control_root'] or control==output or control.is_relative_to(output)
            or control.exists() or any((root/job['output']).exists() for job in jobs)):
        raise FileExistsError('Prior source/control evidence exists or paths overlap; no repeat or overwrite')
    watched=read(root/'experiments/watch_roots.json')
    if master['output_root'] not in watched or authority['control_root'] not in watched:
        raise ValueError('Source and current control roots must be registered before admission')
    control.mkdir(exist_ok=False)
    def check(gpu):
        owner(root,authority_ref,execution_ref,master,live=True,started=time.time())
        published(root,publication_ref,execution_ref,authority_ref,master,authority)
        if gpu not in master['allowed_gpus']:raise ValueError('GPU outside authorized native8B lanes')
        info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if int(info[1].strip())>64 or info[0].strip() in apps:
            raise RuntimeError('GPU occupied; no unrelated process may be stopped')
        return {'gpu':gpu,'memory_mib':int(info[1].strip()),'no_compute_process':True,
                'gpu_uuid':info[0].strip(),
                'execution':execution_ref,'authority':authority_ref,'publication':publication_ref,'scope':SCOPE}
    try:
        initial={g:check(g) for g in master['allowed_gpus']}
    except Exception as error:
        exclusive(control/'admission_failure.json',{'phase':'current_source_admission','type':type(error).__name__,
            'message':str(error),'scientific_children_launched':0,'new_model_tokens':0})
        exclusive(control/'supervisor.json',{'status':'incomplete_requires_review','complete_cases':0,'automatic_retry':False})
        raise
    exclusive(control/'launch.json',{'supervisor_pid':os.getpid(),'started_unix':time.time(),
        'execution':execution_ref,'authority':authority_ref,'publication':publication_ref,
        'initial_free_gpu_checks':initial,'scientific_jobs':15,'max_simultaneous_gpus':8})
    entries={e['name']:e for e in master['jobs']};by_name={j['case_id']:j for j in jobs}
    schedule={'cases':[e|{'hard_seconds':by_name[e['name']]['hard_seconds']} for e in master['jobs']],
              'gpu_lanes':master['gpu_lanes'],'allowed_gpus':master['allowed_gpus'],'max_simultaneous_gpus':8}
    def launch(gpu,case):
        entry=entries[case['name']];job=by_name[case['name']]
        if (root/job['output']).exists():raise FileExistsError('Never overwrite a registered native8B branch')
        command=[str(root/'.venv-rl/bin/python'),'-u',str(root/'pipeline/run_bfcl_qwen8_source_training.py'),
            '--job',str(root/entry['path']),'--job-sha256',entry['sha256'],
            '--execution',execution_ref['path'],'--execution-sha256',execution_ref['sha256']]
        return supervise(command,root/job['output'],control/(case['name']+'.console.log'),job['hard_seconds'],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4'))
    def audit(case):
        if case is None:
            rows=[read(root/job['output']/'audit.json') for job in jobs]
            if any(row['status']!='passed' or row['job_sha256']!=entry['sha256'] for row,entry in zip(rows,master['jobs'])):
                raise ValueError('Incomplete original15 full native8B raw training audits')
            tokens=sum(row['cost']['new_model_tokens'] for row in rows)
            seconds=sum(row['actual_supervised_worker_wall_seconds'] for row in rows)
            if tokens>master['new_training_token_ceiling'] or seconds>master['maximum_training_worker_seconds']:
                raise ValueError('Actual source cost exceeds registered finite ceiling; retain overrun evidence')
            certificate={'kind':'complete_native8b_forward15_source_training_audit','status':'passed','stage':0,
                'complete_jobs':15,'execution_protocol':execution_ref,'scientific_design':master['design'],
                'source_prediction_binding':master['source_prediction_binding'],'new_sampled_tokens':tokens,
                'actual_supervised_worker_wall_seconds':seconds,'jobs':rows,'evaluation_performed':False,
                'ancestor_tokens_recharged':0,'goal_complete':False}
            path=output/'audit_status.json';exclusive(path,certificate)
            exclusive(control/'complete_training_audit_binding.json',{'complete_audit':ref(root,path)})
            return
        entry=entries[case];job=by_name[case]
        command=[str(root/'.venv-rl/bin/python'),str(root/'pipeline/report_bfcl_qwen8_source_training.py'),
            '--job',str(root/entry['path']),'--execution',execution_ref['path'],
            '--execution-sha256',execution_ref['sha256']]
        with (control/(case+'.audit.console.log')).open('x') as log:
            result=subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                stdout=log,stderr=subprocess.STDOUT,timeout=300)
        saved=read(root/job['output']/'audit.json')
        if result.returncode or saved['status']!='passed' or saved['job_sha256']!=entry['sha256']:
            raise ValueError('Original native8B full raw training audit failed: '+case)
        if saved['cost']['new_model_tokens']>job['token_budget']+32768:
            raise ValueError('Source full-group acquisition ceiling exceeded')
    result=execute_lanes(control,schedule,check,launch,audit)
    failures=[]
    for failure in result['failures']:
        phase=failure['phase'];category={'preflight':'control_or_gpu_query_admission',
            'audit':'raw_training_audit_failure','batch_audit':'full_training_batch_audit_or_cost_failure'}.get(phase,'scientific_worker_failure')
        if phase=='runner':
            name=failure['case'];out=root/by_name[name]['output']
            if not (control/(name+'.console.process.json')).exists():category='process_launch_dependency_before_scientific_child'
            elif not (out/'run.json').exists():category='execution_or_model_initialization_before_rollout'
        failures.append(failure|{'classification':category,'not_a_transfer_negative_result':True})
    exclusive(control/'source_control_result.json',{'complete_jobs':len(result['complete']),
        'pending_original_jobs':result['pending'],'classified_failures':failures,
        'automatic_scientific_retry':False,'target_evaluation_started':False})
    if result['failures'] or result['pending']:
        raise RuntimeError('Original native8B cohort incomplete; preserve evidence and register any continuation separately')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['execution','authority','publication']:
        p.add_argument('--'+name,required=True);p.add_argument('--'+name+'-sha256',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    run(root,*[{'path':getattr(a,n),'sha256':getattr(a,n+'_sha256')} for n in ['execution','authority','publication']])
