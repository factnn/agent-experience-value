"""Create the fixed32 native8B target registrations only after reviewed training."""
import argparse
import time
from pathlib import Path

from bfcl_qwen8_source_execution import exclusive,load_execution,process_state,ref
from bfcl_qwen8_transfer_contract import RUNTIME,SCOPE,complete_training,forecasts
from bfcl_source_eval_contract import digest,pinned,policy_role,validate_panel


def cells(master):
    return [(case,e) for case in ['common',*[j['name'] for j in master['jobs']]] for e in range(2)]


def name(case,repeat):return ('q8_s0_common' if case=='common' else case)+'_e'+str(repeat)


def terminal_training(root,binding,master):
    proof=pinned(root,binding['prior_training_terminal_proof'])
    if (not proof['controllers'] or len(proof['children'])!=15
            or sorted(x['case'] for x in proof['children'])!=sorted(x['name'] for x in master['jobs'])):
        raise ValueError('Require original controller and all15 source process identities')
    for row in [*proof['controllers'],*proof['children']]:
        record=pinned(root,row['process_record']);pid=record.get('pid',record.get('supervisor_pid'))
        if pid!=row['pid'] or type(pid) is not int or pid<=0 or process_state(pid) not in [None,'Z','X']:
            raise ValueError('Original source process is not confirmed terminal: '+str(row['pid']))
    return proof


def policy(root,case,master,batch,analysis):
    common=pinned(root,master['native_common_binding']);training=None
    if case=='common':
        path=root/common['common_policy'];fingerprint=common['policy_fingerprint']
        required=['summary.json','adapter/adapter_config.json','adapter/adapter_model.safetensors']
    else:
        entry=next(x for x in master['jobs'] if x['name']==case);training=pinned(root,entry)
        path=root/training['output'];summary=pinned(root,ref(root,path/'summary.json'))
        audit=pinned(root,ref(root,path/'audit.json'))
        if (summary['status']!='complete' or audit!=next((x for x in batch['jobs'] if x['job_sha256']==entry['sha256']),None)
                or summary['final_manifest']['payload_sha256']!=audit['final_payload_sha256']):
            raise ValueError('Source endpoint differs from its complete reviewed native8B cohort')
        fingerprint=summary['final_fingerprint']
        required=['summary.json','audit.json','adapter/adapter_config.json','adapter/adapter_model.safetensors']
    return path,fingerprint,policy_role(case,training,analysis),required


def prepare(root,execution_ref,binding_ref,authority_path,control_root):
    master,design,_=load_execution(root,execution_ref)
    binding=pinned(root,binding_ref)
    batch=complete_training(root,execution_ref,master,binding)
    terminal_training(root,binding,master)
    forecasts(root,binding,master,time.time())
    registration=root/design['output_roots']['evaluation_jobs'];output=root/design['output_roots']['evaluation']
    control=root/control_root
    if any(p.exists() for p in [registration,output,control,authority_path]):
        raise FileExistsError('Preserve original target registrations/results; no automatic rerun')
    if (Path(control_root).is_absolute() or '..' in Path(control_root).parts or Path(control_root).parts[:1]!=('rl',)
            or control==output or control.is_relative_to(output)):
        raise ValueError('Separate native8B target control and scientific output paths')
    for marker in [design['evaluation_stop_file'],design['global_stop_file']]:
        if (root/marker).exists():raise RuntimeError('Current SSH owner stopped NEW target registration/admission')
    analysis=pinned(root,master['analysis_contract'])
    runtime={path:digest(root/path) for path in RUNTIME}
    inputs=master['source_input_sha256']|runtime
    jobs=[]
    for case,repeat in cells(master):
        path,fingerprint,role,required=policy(root,case,master,batch,analysis)
        if role not in ['baseline','control','heldout']:raise ValueError('This forward cohort cannot add old-fit source endpoints')
        job={'kind':'native8b_frozen_forward_target_panel','stage':0,'execution_protocol':execution_ref,
             'admission_binding':binding_ref,'prediction_seal':binding['cross_size_seal'],
             'policy_case':case,'role':role,'evaluation_repeat':repeat,
             'task_ids':master['evaluation']['task_ids'],'task_seeds':master['evaluation']['task_seed_vectors'][repeat],
             'policy_fingerprint':fingerprint,'policy_artifacts':{n:digest(path/n) for n in required},
             'input_sha256':inputs,'output':design['output_roots']['evaluation']+'/'+name(case,repeat)}
        validate_panel(job,master['evaluation']);jobs.append(job)
    if len(jobs)!=32 or len({j['output'] for j in jobs})!=32:raise ValueError('Require exactly32 original panels')
    registration.mkdir(exist_ok=False);entries=[]
    for job in jobs:
        identifier=name(job['policy_case'],job['evaluation_repeat']);path=registration/(identifier+'.json')
        exclusive(path,job);entries.append({'name':identifier,**ref(root,path)})
    manifest={'kind':'native8b_forward_target_manifest','stage':0,'phase':'forward_validation',
        'registered_unix':time.time(),'execution_protocol':execution_ref,'admission_binding':binding_ref,
        'jobs':entries,'output_root':design['output_roots']['evaluation'],'runtime_sha256':runtime,
        'allowed_gpus':master['allowed_gpus'],'max_simultaneous_gpus':8,'resource_authorization':master['resource_authorization'],
        'gpu_lanes':{str(g):[e['name'] for i,e in enumerate(entries) if i%8==g] for g in range(8)},
        'new_token_ceiling':master['evaluation']['new_token_ceiling'],
        'hard_seconds_per_panel':master['evaluation']['hard_seconds_per_panel'],
        'maximum_worker_seconds':design['budgets']['maximum_evaluation_worker_seconds'],
        'automatic_retry':False,'reserved21_allowed':False,'within_native8b_fitting_performed':False,
        'same_weight_policy_alias_reuse':False,'goal_complete':False}
    manifest_path=registration/'manifest.json';exclusive(manifest_path,manifest);manifest_ref=ref(root,manifest_path)
    authority={'kind':'current_native8b_target_evaluation_authority','mode':'human_authorized_ssh_takeover',
        'status':'authorized','scope':SCOPE,'registered_unix':time.time(),'execution_protocol':execution_ref,
        'admission_binding':binding_ref,'manifest':manifest_ref,'current_ssh_owner':master['current_ssh_owner'],
        'resource_authorization':master['resource_authorization'],'control_root':control_root,
        'stop_file':design['evaluation_stop_file'],'global_stop_file':design['global_stop_file'],
        'control_input_sha256':runtime,'automatic_scientific_retry':False,
        'basis':'Current autonomous research and eight-free-card authorization; separate target scope, no old-thread Goal queries.'}
    exclusive(authority_path,authority)
    import json
    watched_path=root/'experiments/watch_roots.json';watched=pinned(root,ref(root,watched_path))
    for relative in [design['output_roots']['evaluation'],control_root,design['output_roots']['evaluation_jobs']]:
        if relative not in watched:watched.append(relative)
    watched_path.write_text(json.dumps(watched,ensure_ascii=False,indent=2)+'\n')
    return {'manifest':manifest_ref,'authority':ref(root,authority_path),'watch_roots':ref(root,watched_path),
            'target_panels_registered':32,'scientific_children_launched':0}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['execution','binding']:
        p.add_argument('--'+field,required=True);p.add_argument('--'+field+'-sha256',required=True)
    p.add_argument('--authority',required=True);p.add_argument('--control-root',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    print(json.dumps(prepare(root,{'path':a.execution,'sha256':a.execution_sha256},
        {'path':a.binding,'sha256':a.binding_sha256},root/a.authority,a.control_root),indent=2))
