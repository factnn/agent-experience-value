"""Current native8B source-RL execution binding; retain E369 scientific choices."""
import importlib.metadata
from pathlib import Path
import subprocess
import time

from bfcl_source_eval_contract import digest,pinned
from bfcl_qwen8_forward_design import validate
from bfcl_qwen8_source_training import identity,load_job,REVISION,REQUIRED as REQUIRED_JOB_CODE
from bfcl_qwen8_transfer_contract import forecasts,publication_time
from supervise_bfcl_stage_fork import read

SELF='pipeline/bfcl_qwen8_source_execution.py'
SCOPE='qwen3_8b_stage0_source_training'
KIND='qwen3_8b_stage0_forward_training_and_transfer'
RUNTIME={SELF,'pipeline/bind_bfcl_qwen8_source_execution.py','pipeline/supervise_bfcl_qwen8_source_training.py',
 'pipeline/bfcl_qwen8_forward_design.py','pipeline/bfcl_qwen8_source_training.py',
 'pipeline/run_bfcl_qwen8_source_training.py','pipeline/report_bfcl_qwen8_source_training.py',
 'pipeline/train_bfcl_source_group.py','pipeline/report_bfcl_source_training.py',
 'pipeline/bfcl_qwen8_transfer_contract.py','pipeline/bfcl_stage1_probe_ssh.py',
 'pipeline/supervise_bfcl_stage1_probe.py','pipeline/supervise_bfcl_stage_fork.py'}
RUNTIME|=REQUIRED_JOB_CODE
VERSIONS=['torch','transformers','trl','peft','accelerate','datasets','tokenizers']


def exclusive(path,value):
    import json
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def ref(root,path):return {'path':str(path.resolve().relative_to(root.resolve())),'sha256':digest(path)}


def process_state(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.read_text().rsplit(')',1)[1].split()[0] if path.exists() else None


def load_execution(root,reference):
    master=pinned(root,reference);design=pinned(root,master['design']);validate(root,design)
    if (master['kind']!=KIND or master['model_revision']!=REVISION or master['stage']!=0
            or master['automatic_retry'] is not False or master['reserved21_allowed'] is not False
            or master['automatic_target_launch'] is not False):
        raise ValueError('Wrong registered native8B source execution')
    for field in ['model_revision','model_verification','native_common_binding','analysis_contract','algorithm_protocol',
                  'source_groups','training_repeats','training','evaluation','allowed_gpus','max_simultaneous_gpus',
                  'resource_authorization','current_ssh_owner']:
        if master[field]!=design[field]:raise ValueError('Execution changed frozen E369 choice: '+field)
    if (master['new_training_token_ceiling']!=design['budgets']['training_new_token_ceiling']
            or master['maximum_training_worker_seconds']!=design['budgets']['maximum_training_worker_seconds']
            or master['auxiliary_candidate_design']!=design['prediction_protocols']['auxiliary32_design']):
        raise ValueError('Wrong finite dose or auxiliary candidate registration')
    expected=design['training_job_blueprints'];entries=master['jobs']
    if len(entries)!=15 or len({e['name'] for e in entries})!=15:
        raise ValueError('Exact original15 source/control job identities required')
    jobs=[]
    for entry,blueprint in zip(entries,expected):
        job,_,_,_=load_job(root,root/entry['path'],entry['sha256'],reference)
        if entry['name']!=blueprint['case_id'] or any(job[k]!=v for k,v in blueprint.items()):
            raise ValueError('Original source,seed,pool or paired repeat changed')
        if (identity(job['native_cohort_design'])!=identity(master['design'])
                or job['output']!=design['output_roots']['training']+'/'+blueprint['case_id']):
            raise ValueError('Wrong immutable cohort definition or output identity')
        jobs.append(job)
    lanes={str(g):[entry['name'] for i,entry in enumerate(entries) if i%8==g] for g in range(8)}
    if master['gpu_lanes']!=lanes:raise ValueError('Frozen source lane schedule changed')
    if not RUNTIME<=master['runtime_code_sha256'].keys():raise ValueError('Source runtime omitted a required dependency')
    for name,sha in master['runtime_code_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('Source execution code changed: '+name)
    if set(master['runtime_versions'])!=set(VERSIONS):raise ValueError('Incomplete actual runtime version binding')
    for name,version in master['runtime_versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Runtime library changed: '+name)
    return master,design,jobs


def owner(root,reference,execution_ref,master,*,live=True,started=None):
    authority=pinned(root,reference);design=pinned(root,master['design'])
    if (authority['mode']!='human_authorized_ssh_takeover' or authority['status']!='authorized'
            or authority['scope']!=SCOPE or authority['current_ssh_owner']!=master['current_ssh_owner']
            or identity(authority['execution_protocol'])!=identity(execution_ref)
            or authority['resource_authorization']!=master['resource_authorization']
            or authority['stop_file']!=design['training_stop_file'] or authority['global_stop_file']!=design['global_stop_file']):
        raise ValueError('Wrong separately registered current native8B training owner')
    if started is not None and authority['registered_unix']>started:raise ValueError('Training authority registered after use')
    for key in ['stop_file','global_stop_file']:
        path=(root/authority[key]).resolve()
        if path.parent!=(root/'experiments/.runtime').resolve():raise ValueError('Training stop outside project runtime')
        if live and path.exists():raise RuntimeError('Current SSH owner stopped new native8B training admission')
    if not RUNTIME<=authority['control_input_sha256'].keys():raise ValueError('Missing current source control pins')
    for name,sha in authority['control_input_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('Current source control changed: '+name)
    return authority


def prerequisites(root,master):
    binding=pinned(root,master['source_prediction_binding'])
    if (binding['kind']!='native8b_forward_source_prediction_binding'
            or identity(binding['scientific_design'])!=identity(master['design'])
            or identity(binding['native_common_binding'])!=identity(master['native_common_binding'])):
        raise ValueError('Wrong native8B pre-acquisition binding')
    batch=pinned(root,binding['complete_native_probe_audit'])
    review=pinned(root,binding['complete_native_probe_review'])
    if (batch['status']!='passed' or batch['complete_cases']!=16 or batch['model_revision']!=REVISION
            or review['status']!='passed' or review['complete_cases']!=16
            or review['raw_replay_equals_saved_audits'] is not True
            or identity(review['batch_audit'])!=identity(binding['complete_native_probe_audit'])):
        raise ValueError('Complete original16 raw probe audits and execution review required')
    for field in ['prior_native_probe_terminal_proof','prior_auxiliary_terminal_proof']:
        proof=pinned(root,binding[field])
        if proof['all_owned_scientific_children_terminal'] is not True:
            raise ValueError('Previous scientific work is still live: '+field)
        if process_state(proof['controller_pid']) not in [None,'Z','X']:
            raise ValueError('Previous controller remains live: '+field)
        for row in proof['children']:
            if process_state(row['pid']) not in [None,'Z','X']:
                raise ValueError('Previous scientific child remains live: '+str(row['pid']))
    native=pinned(root,binding['prior_native_probe_terminal_proof'])
    if (identity(native['native_probe_audit'])!=identity(binding['complete_native_probe_audit'])
            or sorted(x['case'] for x in native['children'])!=sorted(x['case'] for x in batch['rows'])):
        raise ValueError('Native terminal proof omits or substitutes an original probe case')
    core,aux=forecasts(root,binding,master,time.time())
    auxiliary=pinned(root,binding['prior_auxiliary_terminal_proof'])
    disposition=binding['auxiliary_forecast_disposition']
    if identity(auxiliary['outcome_receipt'])!=identity(disposition['receipt']):
        raise ValueError('Auxiliary terminal proof belongs to another sealed/failed outcome')
    auxiliary_design=pinned(root,master['auxiliary_candidate_design'])
    control=root/auxiliary_design['control_root']
    actual={p.name.removesuffix('.console.process.json'):p for p in control.glob('*.console.process.json')}
    if sorted(actual)!=sorted(x['case'] for x in auxiliary['children']):
        raise ValueError('Auxiliary terminal proof omits a launched scientific process')
    for row in auxiliary['children']:
        if (identity(row['process_record'])!=identity(ref(root,actual[row['case']]))
                or read(actual[row['case']])['pid']!=row['pid']):
            raise ValueError('Auxiliary process identity differs from original launch evidence')
    if disposition['status']=='sealed':
        completed=pinned(root,aux['complete_measurement_audit'])
        if len(auxiliary['children'])!=32 or sorted(actual)!=sorted(x['case'] for x in completed['rows']):
            raise ValueError('Completed auxiliary outcome needs all original32 terminal workers')
    path=(root/binding['cross_size_seal']['path']).parent/'complete_probe_review.json'
    if identity(binding['complete_native_probe_review'])!=identity(ref(root,path)):
        raise ValueError('Native probe review differs from the actual cross-size seal')
    if core['files']['complete_probe_review.json']!=binding['complete_native_probe_review']['sha256']:
        raise ValueError('Cross-size forecast does not contain its full probe review')
    return binding,core,aux


def published(root,publication_ref,execution_ref,authority_ref,master,authority):
    value=pinned(root,publication_ref)
    if (value['scientific_child_started'] is not False or publication_time(value)>time.time()
            or subprocess.run(['git','merge-base','--is-ancestor',value['commit'],value['remote_main']],cwd=root,capture_output=True,timeout=30).returncode):
        raise ValueError('Actual remote publication was not verified before source admission')
    binding=pinned(root,master['source_prediction_binding'])
    required=[execution_ref,authority_ref,master['design'],master['algorithm_protocol'],master['source_prediction_binding'],
        binding['complete_native_probe_audit'],binding['complete_native_probe_review'],binding['prior_native_probe_terminal_proof'],
        binding['prior_auxiliary_terminal_proof'],binding['cross_size_seal'],binding['cross_size_publication'],
        binding['auxiliary_forecast_disposition']['receipt'],binding['auxiliary_forecast_disposition']['publication'],*master['jobs']]
    expected={item['path']:item['sha256'] for item in required}|master['runtime_code_sha256']|authority['control_input_sha256']
    if any(value['files'].get(name)!=sha for name,sha in expected.items()):raise ValueError('Verified publication omits a source prerequisite')
    for name,sha in value['files'].items():
        if digest(root/name)!=sha:raise ValueError('Published source bytes changed: '+name)
    return value
