"""CPU temporary15-job registry and current-incomplete admission checks."""
import copy
import importlib.metadata
import json
from pathlib import Path
import tempfile

from bfcl_qwen8_source_execution import KIND,RUNTIME,SCOPE,VERSIONS,load_execution,owner,ref
from bfcl_qwen8_transfer_contract import publication_time
from bfcl_source_eval_contract import digest,pinned
from bind_bfcl_qwen8_source_execution import build
from supervise_bfcl_stage_fork import ROOT,read


def main():
    design_ref=ref(ROOT,ROOT/'rl/QWEN3_8B_STAGE0_FORWARD15_SCIENTIFIC_DESIGN_20261017.json')
    design=pinned(ROOT,design_ref);initial=pinned(ROOT,design['initial_native8b_execution_protocol'])
    common=pinned(ROOT,design['native_common_binding']);checks=[]
    with tempfile.TemporaryDirectory(prefix='native8b-source-execution-cpu-',dir=ROOT/'experiments/.runtime') as temporary:
        directory=Path(temporary)
        def save(name,value):
            path=directory/name
            path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
            return ref(ROOT,path)
        code={name:digest(ROOT/name) for name in RUNTIME}
        inputs=initial['input_sha256']|code
        for item in [design_ref,design['algorithm_protocol'],design['native_common_binding'],design['model_verification'],
                     *[common[k] for k in ['common_audit','manifest','learning_state','adapter_config','adapter_weights']]]:
            inputs[item['path']]=item['sha256']
        originals=[];entries=[]
        for i,blueprint in enumerate(design['training_job_blueprints']):
            job=copy.deepcopy(blueprint)
            job.update(output=design['output_roots']['training']+'/'+job['case_id'],native_cohort_design=design_ref,
                native_common_binding=design['native_common_binding'],native_model_verification=design['model_verification'],
                native_model_revision=design['model_revision'],algorithm_protocol=design['algorithm_protocol'],
                source_protocol=initial['source_protocol'],source_checkpoint=common['common_policy'],
                source_payload_sha256=common['learning_state']['sha256'],input_sha256=inputs)
            originals.append(job);entries.append({'name':job['case_id'],**save('job'+str(i)+'.json',job)})
        master={key:copy.deepcopy(design[key]) for key in ['model_revision','model_verification','native_common_binding',
            'analysis_contract','algorithm_protocol','source_groups','training_repeats','training','evaluation',
            'allowed_gpus','max_simultaneous_gpus','resource_authorization','current_ssh_owner']}
        master.update(kind=KIND,stage=0,design=design_ref,automatic_retry=False,reserved21_allowed=False,
            automatic_target_launch=False,auxiliary_candidate_design=design['prediction_protocols']['auxiliary32_design'],
            new_training_token_ceiling=design['budgets']['training_new_token_ceiling'],
            maximum_training_worker_seconds=design['budgets']['maximum_training_worker_seconds'],
            jobs=entries,gpu_lanes={str(g):[e['name'] for i,e in enumerate(entries) if i%8==g] for g in range(8)},
            runtime_code_sha256=code,runtime_versions={n:importlib.metadata.version(n) for n in VERSIONS},
            synthetic_fixture_only=True)
        execution=save('execution.json',master)
        load_execution(ROOT,execution)
        checks.append('All15 exact E369 native8B branch identities, real common artifacts, inputs and actual package versions accepted in temporary structural reader fixture')
        def reject(name,action):
            try:action()
            except (ValueError,AssertionError,FileNotFoundError,RuntimeError):checks.append(name)
            else:raise AssertionError('Invalid source execution accepted: '+name)
        bad=copy.deepcopy(master);bad['jobs']=bad['jobs'][:-1]
        reject('Missing original source/control branch rejected',lambda:load_execution(ROOT,save('bad_execution.json',bad)))
        bad=copy.deepcopy(master);bad['model_revision']='4B'
        reject('Wrong learner revision rejected',lambda:load_execution(ROOT,save('bad_execution.json',bad)))
        bad=copy.deepcopy(master);bad['new_training_token_ceiling']+=1
        reject('Finite cost ceiling change rejected',lambda:load_execution(ROOT,save('bad_execution.json',bad)))
        bad=copy.deepcopy(master);bad['runtime_versions']['torch']='wrong-version'
        reject('Actual runtime version mismatch rejected',lambda:load_execution(ROOT,save('bad_execution.json',bad)))
        bad=copy.deepcopy(master);job=copy.deepcopy(originals[0]);job['rollout_seed_base']+=100
        bad['jobs'][0]={'name':job['case_id'],**save('changed_job.json',job)}
        reject('Seed substitution rejected even after job and execution SHA are recomputed',lambda:load_execution(ROOT,save('bad_execution.json',bad)))
        authority={'mode':'human_authorized_ssh_takeover','status':'authorized','scope':SCOPE,
            'execution_protocol':execution,'current_ssh_owner':master['current_ssh_owner'],
            'resource_authorization':master['resource_authorization'],'stop_file':design['training_stop_file'],
            'global_stop_file':design['global_stop_file'],'registered_unix':1,'control_input_sha256':code,
            'synthetic_fixture_only':True}
        authority_ref=save('authority.json',authority)
        owner(ROOT,authority_ref,execution,master,live=False,started=20)
        bad_authority=authority|{'scope':'qwen3_8b_stage0_forward_target_evaluation'}
        reject('Target ownership cannot admit source training',lambda:owner(ROOT,save('bad_authority.json',bad_authority),execution,master,live=False,started=20))
        assert publication_time({'verified_unix':15})==publication_time({'verified_at':'1970-01-01T00:00:15+00:00'})==15.0
        reject('Nonfinite publication observation rejected',lambda:publication_time({'verified_unix':float('nan')}))
        reject('Boolean publication clock rejected',lambda:publication_time({'verified_unix':True}))
        checks.append('Existing numeric and native ISO publication receipt clocks interpreted identically')
    execution_path=ROOT/'rl/QWEN3_8B_STAGE0_FORWARD15_EXECUTION_20261017.json'
    authority_path=ROOT/'rl/QWEN3_8B_STAGE0_FORWARD15_TRAIN_SSH_AUTHORITY_20261017.json'
    job_root=ROOT/design['output_roots']['training_jobs']
    assert not execution_path.exists() and not authority_path.exists() and not job_root.exists()
    missing={'path':'rl/qwen3_8b_stage0_forward15_binding_20261017/prediction_binding.json','sha256':'0'*64}
    try:
        build(ROOT,design_ref,missing,execution_path,authority_path,'rl/qwen3_8b_stage0_forward15_train_ssh_20261017')
    except (ValueError,FileNotFoundError) as error:
        actual_refusal={'type':type(error).__name__,'message':str(error)}
    else:raise AssertionError('Current incomplete real evidence admitted a source binding')
    assert not execution_path.exists() and not authority_path.exists() and not job_root.exists()
    print(json.dumps({'status':'passed','checks':checks,'actual_current_binding_refusal':actual_refusal,
        'fixtures':'Temporary complete15 structural registry linked to real E369/native common inputs; deleted. No live authority or actual scientific job registered.',
        'new_gpu_jobs':0,'new_model_tokens':0,'actual_source_execution_created':False,'target_labels_read':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
