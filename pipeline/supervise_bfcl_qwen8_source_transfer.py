"""Current separately owned native8B32 target panels through existing supervision."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from bfcl_qwen8_source_execution import exclusive,load_execution,ref
from bfcl_qwen8_transfer_contract import (RUNTIME,complete_training,forecasts,load_job,owner,publication_time)
from bfcl_source_eval_contract import digest,pinned,validate_panel
from bfcl_stage1_probe_ssh import execute_lanes
from prepare_bfcl_qwen8_source_transfer import cells,name,policy,terminal_training
from supervise_bfcl_stage_fork import read,supervise


def load_manifest(root,reference):
    manifest=pinned(root,reference)
    master,design,_=load_execution(root,manifest['execution_protocol'])
    binding=pinned(root,manifest['admission_binding'])
    batch=complete_training(root,manifest['execution_protocol'],master,binding)
    terminal_training(root,binding,master)
    forecasts(root,binding,master,time.time())
    if (manifest['kind']!='native8b_forward_target_manifest' or manifest['stage']!=0
            or manifest['phase']!='forward_validation' or manifest['automatic_retry'] is not False
            or manifest['reserved21_allowed'] is not False or manifest['within_native8b_fitting_performed'] is not False
            or manifest['same_weight_policy_alias_reuse'] is not False
            or manifest['allowed_gpus']!=master['allowed_gpus'] or manifest['max_simultaneous_gpus']!=8
            or manifest['resource_authorization']!=master['resource_authorization']
            or manifest['new_token_ceiling']!=master['evaluation']['new_token_ceiling']
            or manifest['hard_seconds_per_panel']!=master['evaluation']['hard_seconds_per_panel']
            or manifest['maximum_worker_seconds']!=design['budgets']['maximum_evaluation_worker_seconds']
            or manifest['output_root']!=design['output_roots']['evaluation']):
        raise ValueError('Wrong finite native8B target manifest, dose or scope')
    runtime={path:digest(root/path) for path in RUNTIME}
    if manifest['runtime_sha256']!=runtime:raise ValueError('Frozen native8B target runtime changed')
    jobs=[pinned(root,e) for e in manifest['jobs']];expected=cells(master)
    if (len(jobs)!=32 or [(j['policy_case'],j['evaluation_repeat']) for j in jobs]!=expected
            or [e['name'] for e in manifest['jobs']]!=[name(case,e) for case,e in expected]):
        raise ValueError('Original32 target policy/seed identities substituted or omitted')
    lanes={str(g):[e['name'] for i,e in enumerate(manifest['jobs']) if i%8==g] for g in range(8)}
    if manifest['gpu_lanes']!=lanes:raise ValueError('Frozen target lane schedule changed')
    analysis=pinned(root,master['analysis_contract'])
    for job in jobs:
        if (job['execution_protocol']!=manifest['execution_protocol'] or job['admission_binding']!=manifest['admission_binding']
                or job['prediction_seal']!=binding['cross_size_seal'] or job['stage']!=0):
            raise ValueError('Target job belongs to another frozen learner/admission binding')
        validate_panel(job,master['evaluation'])
        path,fingerprint,role,required=policy(root,job['policy_case'],master,batch,analysis)
        if (job['policy_fingerprint']!=fingerprint or job['role']!=role
                or job['policy_artifacts']!={n:digest(path/n) for n in required}
                or any(job['input_sha256'].get(n)!=sha for n,sha in runtime.items())
                or job['output']!=manifest['output_root']+'/'+name(job['policy_case'],job['evaluation_repeat'])):
            raise ValueError('Frozen target policy or input identity changed')
    return manifest,master,binding,jobs


def run(root,manifest_ref,authority_ref,publication_ref):
    manifest,master,binding,jobs=load_manifest(root,manifest_ref)
    first=jobs[0];authority,bound_manifest=owner(root,authority_ref,first,master,live=True,started=time.time())
    if authority['manifest']!=manifest_ref or bound_manifest!=manifest:
        raise ValueError('Current target owner belongs to another registered32 manifest')
    publication=pinned(root,publication_ref)
    if (publication['scientific_child_started'] is not False or publication_time(publication)>time.time()
            or subprocess.run(['git','merge-base','--is-ancestor',publication['commit'],publication['remote_main']],
                              cwd=root,capture_output=True,timeout=30).returncode):
        raise ValueError('Actual remote target publication required before inference')
    expected={e['path']:e['sha256'] for e in [manifest_ref,authority_ref,manifest['execution_protocol'],
              manifest['admission_binding'],master['design'],*manifest['jobs']]}|authority['control_input_sha256']
    if any(publication['files'].get(n)!=sha for n,sha in expected.items()):
        raise ValueError('Actual publication omits target code/manifest/jobs/current owner')
    # The shared loader verifies all actual forecast/train/terminal publication
    # references before any target output, then each worker repeats its own gate.
    load_job(root,root/manifest['jobs'][0]['path'],manifest['jobs'][0]['sha256'],
             authority_ref=authority_ref,publication_ref=publication_ref,admit=True)
    control=root/authority['control_root'];output=root/manifest['output_root']
    relative=Path(authority['control_root'])
    if (relative.is_absolute() or '..' in relative.parts or relative.parts[:1]!=('rl',)
            or control==output or control.is_relative_to(output) or control.exists()
            or any((root/j['output']).exists() for j in jobs)):
        raise FileExistsError('Prior target/control evidence exists or paths overlap; never repeat original panels')
    watched=read(root/'experiments/watch_roots.json')
    if authority['control_root'] not in watched or manifest['output_root'] not in watched:
        raise ValueError('Target and control roots were not registered')
    control.mkdir(exist_ok=False)
    def check(gpu):
        owner(root,authority_ref,first,master,live=True,started=time.time())
        for n,sha in publication['files'].items():
            if digest(root/n)!=sha:raise ValueError('Published target input changed: '+n)
        if gpu not in manifest['allowed_gpus']:raise ValueError('GPU outside authorized native8B target lanes')
        info=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()
        if int(info[1].strip())>64 or info[0].strip() in apps:raise RuntimeError('GPU occupied; do not stop unrelated work')
        return {'gpu':gpu,'gpu_uuid':info[0].strip(),'memory_mib':int(info[1].strip()),'no_compute_process':True,
                'manifest':manifest_ref,'authority':authority_ref,'publication':publication_ref}
    try:initial={g:check(g) for g in manifest['allowed_gpus']}
    except Exception as error:
        exclusive(control/'admission_failure.json',{'phase':'target_admission_before_inference','type':type(error).__name__,
            'message':str(error),'scientific_children_launched':0,'new_model_tokens':0})
        exclusive(control/'supervisor.json',{'status':'incomplete_requires_review','complete_cases':0,'automatic_retry':False})
        raise
    exclusive(control/'launch.json',{'supervisor_pid':os.getpid(),'started_unix':time.time(),'manifest':manifest_ref,
        'authority':authority_ref,'publication':publication_ref,'initial_free_gpu_checks':initial,
        'scientific_panels':32,'max_simultaneous_gpus':8})
    entries={e['name']:e for e in manifest['jobs']};by_name={e['name']:j for e,j in zip(manifest['jobs'],jobs)}
    def launch(gpu,case):
        entry=entries[case['name']];job=by_name[case['name']]
        if (root/job['output']).exists():raise FileExistsError('Never overwrite a native8B target panel')
        command=[str(root/'.venv-rl/bin/python'),'-u',str(root/'pipeline/run_bfcl_qwen8_source_transfer.py'),
            '--job',str(root/entry['path']),'--job-sha256',entry['sha256'],
            '--authority',authority_ref['path'],'--authority-sha256',authority_ref['sha256'],
            '--publication',publication_ref['path'],'--publication-sha256',publication_ref['sha256']]
        return supervise(command,root/job['output'],control/(case['name']+'.console.log'),manifest['hard_seconds_per_panel'],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4'))
    def audit(case):
        if case is None:
            rows=[read(root/j['output']/'audit.json') for j in jobs]
            if any(row['status']!='passed' or row['job_sha256']!=e['sha256'] for row,e in zip(rows,manifest['jobs'])):
                raise ValueError('All original32 complete native8B target raw audits required')
            tokens=sum(row['cost']['new_model_tokens'] for row in rows)
            seconds=sum(row['actual_supervised_worker_wall_seconds'] for row in rows)
            if tokens>manifest['new_token_ceiling'] or seconds>manifest['maximum_worker_seconds']:
                raise ValueError('Actual target costs exceed the finite registered budget; retain overrun evidence')
            certificate={'status':'passed','stage':0,'complete_panels':32,'manifest':manifest_ref,
                'execution_protocol':manifest['execution_protocol'],'admission_binding':manifest['admission_binding'],
                'new_sampled_tokens':tokens,'actual_supervised_worker_wall_seconds':seconds,'rows':rows,
                'within_native8b_fitting_performed':False,'reserved21_accessed':False,'goal_complete':False}
            path=output/'audit_status.json';exclusive(path,certificate)
            exclusive(control/'target_audit_binding.json',{'complete_audit':ref(root,path)})
            return
        entry=entries[case];job=by_name[case]
        command=[str(root/'.venv-rl/bin/python'),str(root/'pipeline/report_bfcl_qwen8_source_transfer.py'),
            '--job',str(root/entry['path'])]
        with (control/(case+'.audit.console.log')).open('x') as log:
            result=subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                stdout=log,stderr=subprocess.STDOUT,timeout=300)
        saved=read(root/job['output']/'audit.json')
        if result.returncode or saved['status']!='passed' or saved['job_sha256']!=entry['sha256']:
            raise ValueError('Original native8B target raw replay failed: '+case)
        if saved['cost']['new_model_tokens']>262144:raise ValueError('Native8B panel token ceiling exceeded')
    schedule={'cases':manifest['jobs'],'gpu_lanes':manifest['gpu_lanes'],
              'allowed_gpus':manifest['allowed_gpus'],'max_simultaneous_gpus':8}
    result=execute_lanes(control,schedule,check,launch,audit)
    failures=[]
    for failure in result['failures']:
        category={'preflight':'control_or_gpu_query_admission','audit':'raw_target_audit_failure',
                  'batch_audit':'full_target_batch_audit_or_cost_failure'}.get(failure['phase'],'scientific_target_worker_failure')
        if failure['phase']=='runner':
            out=root/by_name[failure['case']]['output'];terminal=out/'supervisor.json'
            if terminal.exists() and read(terminal)['returncode']==3:category='target_admission_before_model_load'
            elif (out/'native_target_failure.json').exists():category=read(out/'native_target_failure.json')['classification']
            elif not (control/(failure['case']+'.console.process.json')).exists():category='process_launch_dependency_before_child'
        failures.append(failure|{'classification':category,'not_a_transfer_negative_result':True})
    exclusive(control/'target_control_result.json',{'complete_panels':len(result['complete']),
        'pending_original_panels':result['pending'],'classified_failures':failures,'automatic_scientific_retry':False})
    if result['failures'] or result['pending']:raise RuntimeError('Native8B target cohort incomplete; preserve all original evidence')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['manifest','authority','publication']:
        p.add_argument('--'+field,required=True);p.add_argument('--'+field+'-sha256',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    run(root,*[{'path':getattr(a,n),'sha256':getattr(a,n+'_sha256')} for n in ['manifest','authority','publication']])
