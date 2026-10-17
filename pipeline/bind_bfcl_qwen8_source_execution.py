"""Bind the fixed E369 native8B jobs after complete measured/forecast evidence."""
import argparse
import copy
import importlib.metadata
import platform
from pathlib import Path
import sys
import time

from bfcl_qwen8_forward_design import validate
from bfcl_qwen8_source_execution import (KIND,RUNTIME,SCOPE,VERSIONS,exclusive,load_execution,
    owner,prerequisites,ref)
from bfcl_source_eval_contract import digest,pinned
from seal_bfcl_qwen8_cross_size_predictions import native_target_record_check


def build(root,design_ref,binding_ref,execution_path,authority_path,control_root):
    design=pinned(root,design_ref);validate(root,design)
    core_protocol=pinned(root,design['prediction_protocols']['core128'])
    audit=root/core_protocol['complete_probe_audit_path']
    if not audit.exists():
        raise ValueError('Complete original16 native8B probe required; no source execution binding created')
    for marker in [design['training_stop_file'],design['global_stop_file']]:
        if (root/marker).exists():raise RuntimeError('Current SSH owner stopped NEW source binding/admission')
    master={key:copy.deepcopy(design[key]) for key in ['model_revision','model_verification','native_common_binding',
        'analysis_contract','algorithm_protocol','source_groups','training_repeats','training','evaluation',
        'allowed_gpus','max_simultaneous_gpus','resource_authorization','current_ssh_owner']}
    master.update(kind=KIND,stage=0,design=design_ref,bound_unix=time.time(),automatic_retry=False,
        reserved21_allowed=False,automatic_target_launch=False,source_prediction_binding=binding_ref,
        auxiliary_candidate_design=design['prediction_protocols']['auxiliary32_design'],
        new_training_token_ceiling=design['budgets']['training_new_token_ceiling'],
        maximum_training_worker_seconds=design['budgets']['maximum_training_worker_seconds'],
        output_root=design['output_roots']['training'],control_root=control_root)
    prerequisites(root,master)
    native_target_record_check(root)
    job_root=root/design['output_roots']['training_jobs']
    if any(p.exists() for p in [job_root,execution_path,authority_path,root/control_root,root/master['output_root']]):
        raise FileExistsError('A native8B source binding or output exists; preserve it and register continuation if needed')
    if Path(control_root).is_absolute() or '..' in Path(control_root).parts or Path(control_root).parts[:1]!=('rl',):
        raise ValueError('Source control root must be inside project rl')
    if (root/control_root).resolve().is_relative_to((root/master['output_root']).resolve()):
        raise ValueError('Keep scientific source outputs separate from control receipts')
    code={name:digest(root/name) for name in RUNTIME}
    initial=pinned(root,design['initial_native8b_execution_protocol'])
    inputs=copy.deepcopy(initial['input_sha256'])|code
    common=pinned(root,design['native_common_binding'])
    for item in [design_ref,design['algorithm_protocol'],design['native_common_binding'],design['model_verification'],
                 *[common[k] for k in ['common_audit','manifest','learning_state','adapter_config','adapter_weights']]]:
        inputs[item['path']]=item['sha256']
    for name,sha in inputs.items():
        if digest(root/name)!=sha:raise ValueError('Frozen source input differs: '+name)
    planned=[]
    for blueprint in design['training_job_blueprints']:
        job=copy.deepcopy(blueprint)
        job.update(output=master['output_root']+'/'+job['case_id'],native_cohort_design=design_ref,
            native_common_binding=design['native_common_binding'],native_model_verification=design['model_verification'],
            native_model_revision=design['model_revision'],algorithm_protocol=design['algorithm_protocol'],
            source_protocol=initial['source_protocol'],source_checkpoint=common['common_policy'],
            source_payload_sha256=common['learning_state']['sha256'],input_sha256=inputs)
        planned.append(job)
    master.update(runtime_code_sha256=code,runtime_versions={n:importlib.metadata.version(n) for n in VERSIONS},
        python_version=platform.python_version(),python_executable=str(Path(sys.executable).resolve()),
        source_input_sha256=inputs,registered_scientific_jobs=15,targets_opened=False)
    job_root.mkdir(exist_ok=False)
    entries=[]
    for job in planned:
        path=job_root/(job['case_id']+'.json');exclusive(path,job)
        entries.append({'name':job['case_id'],**ref(root,path)})
    master['jobs']=entries
    master['gpu_lanes']={str(g):[e['name'] for i,e in enumerate(entries) if i%8==g] for g in range(8)}
    exclusive(execution_path,master);execution_ref=ref(root,execution_path)
    load_execution(root,execution_ref)
    authority={'kind':'current_native8b_source_training_authority','mode':'human_authorized_ssh_takeover',
        'status':'authorized','scope':SCOPE,'registered_unix':time.time(),'execution_protocol':execution_ref,
        'current_ssh_owner':design['current_ssh_owner'],'resource_authorization':design['resource_authorization'],
        'control_root':control_root,'stop_file':design['training_stop_file'],'global_stop_file':design['global_stop_file'],
        'control_input_sha256':code,'automatic_scientific_retry':False,
        'basis':'Current human autonomous research/eight-free-card authorization; E369 execution binding, no invented new per-job approval.'}
    exclusive(authority_path,authority);authority_ref=ref(root,authority_path)
    owner(root,authority_ref,execution_ref,master,live=True,started=time.time())
    import json
    watched_path=root/'experiments/watch_roots.json';watched=pinned(root,ref(root,watched_path))
    for relative in [master['output_root'],control_root,design['output_roots']['training_jobs']]:
        if relative not in watched:watched.append(relative)
    watched_path.write_text(json.dumps(watched,ensure_ascii=False,indent=2)+'\n')
    return {'execution':execution_ref,'authority':authority_ref,'watch_roots':ref(root,watched_path),
            'registered_jobs':15,'scientific_children_launched':0}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['design','binding']:
        p.add_argument('--'+name,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ['execution','authority','control-root']:p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    print(json.dumps(build(root,{'path':a.design,'sha256':a.design_sha256},
        {'path':a.binding,'sha256':a.binding_sha256},root/a.execution,root/a.authority,a.control_root),indent=2))
