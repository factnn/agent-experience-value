"""CPU synthetic admission checks; no real8B targets, model loads or publication."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import bfcl_qwen8_transfer_contract as contract
from bfcl_source_eval_contract import digest
from bfcl_qwen8_source_training import REVISION
from supervise_bfcl_stage_fork import ROOT, read


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='native8b-target-gates-cpu-') as temporary:
        root=Path(temporary)
        def save(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
            return {'path':name,'sha256':digest(p)}
        common=save('fixture/common.json',{'synthetic_fixture_only':True})
        jobs=[save('fixture/job'+str(i)+'.json',{'case_id':'fixture'+str(i)})|{'name':'fixture'+str(i)} for i in range(2)]
        master={'jobs':jobs,'native_common_binding':common,'new_training_token_ceiling':196608,
                'current_ssh_owner':{'thread_id':'cpu_fixture_only','mode':'human_authorized_ssh_takeover'},
                'resource_authorization':common}
        master['design']=save('fixture/target_design.json',{'evaluation_stop_file':'experiments/.runtime/cpu_target_stop',
                          'global_stop_file':'experiments/.runtime/cpu_global_stop'})
        execution=save('fixture/execution.json',master)
        batch={'status':'passed','complete_jobs':2,'execution_protocol':execution,
               'jobs':[{'status':'passed','native_model_revision':REVISION,'job_sha256':j['sha256'],
                        'cost':{'new_model_tokens':65536}} for j in jobs], 'new_sampled_tokens':131072}
        original_review={'kind':'complete_native8b_source_training_execution_review','status':'reviewed',
                        'execution_protocol':execution,'fresh_raw_audits_match_saved':True,'no_failed_or_omitted_active_branches':True}
        terminal={'all_owned_scientific_children_terminal':True,'execution_protocol':execution,
                  'scientific_job_sha256s':[j['sha256'] for j in jobs]}
        def binding_for(b=None,r=None,t=None):
            br=save('fixture/batch.json',batch if b is None else b)
            review=copy.deepcopy(original_review if r is None else r);review['complete_audit']=br
            return {'kind':'qwen3_8b_stage0_forward_target_admission','execution_protocol':execution,
                    'native_common_binding':common,'complete_training_audit':br,
                    'complete_training_review':save('fixture/review.json',review),
                    'prior_training_terminal_proof':save('fixture/terminal.json',terminal if t is None else t)}
        binding=binding_for()
        contract.complete_training(root,execution,master,binding)
        checks.append('Synthetic complete-cohort audit/review/terminal identities accepted; no science jobs represented')
        def reject(name,action):
            try:action()
            except (ValueError,RuntimeError,AssertionError,FileNotFoundError):checks.append(name)
            else:raise AssertionError('Invalid target prerequisite accepted: '+name)
        bad=copy.deepcopy(batch);bad['jobs']=bad['jobs'][:-1]
        reject('Omitted native8B training branch rejected',lambda:contract.complete_training(root,execution,master,binding_for(b=bad)))
        bad=copy.deepcopy(batch);bad['jobs'][0]['native_model_revision']='4B'
        reject('4B training audit rejected',lambda:contract.complete_training(root,execution,master,binding_for(b=bad)))
        bad=copy.deepcopy(batch);bad['new_sampled_tokens']+=1
        reject('Actual training cost inconsistency rejected',lambda:contract.complete_training(root,execution,master,binding_for(b=bad)))
        bad=copy.deepcopy(original_review);bad['fresh_raw_audits_match_saved']=False
        reject('Missing full raw execution review rejected',lambda:contract.complete_training(root,execution,master,binding_for(r=bad)))
        bad=copy.deepcopy(terminal);bad['all_owned_scientific_children_terminal']=False
        reject('Still-live prior scientific work rejected',lambda:contract.complete_training(root,execution,master,binding_for(t=bad)))
        binding=binding_for()
        b_ref=save('fixture/binding.json',binding)
        manifest_ref=save('fixture/manifest.json',{'kind':'native8b_forward_target_manifest','execution_protocol':execution,
                         'admission_binding':b_ref,'jobs':[]})
        runtime={}
        for name in contract.RUNTIME:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic CPU fixture only\n');runtime[name]=digest(p)
        auth={'mode':'human_authorized_ssh_takeover','status':'authorized','scope':contract.SCOPE,
              'current_ssh_owner':master['current_ssh_owner'],'execution_protocol':execution,'admission_binding':b_ref,
              'resource_authorization':common,'registered_unix':5,'stop_file':'experiments/.runtime/cpu_target_stop',
              'global_stop_file':'experiments/.runtime/cpu_global_stop','control_input_sha256':runtime,'manifest':manifest_ref}
        auth_ref=save('fixture/authority.json',auth)
        target={'execution_protocol':execution,'admission_binding':b_ref}
        contract.owner(root,auth_ref,target,master,live=True,started=20)
        wrong_stop=auth|{'stop_file':'experiments/.runtime/old_phase_stop'}
        reject('Wrong prior-phase stop marker rejected',lambda:contract.owner(root,save('fixture/wrong_stop_authority.json',wrong_stop),target,master,live=True,started=20))
        stop=root/auth['stop_file'];stop.parent.mkdir(parents=True,exist_ok=True);stop.touch()
        reject('Current stop rejects NEW target admission',lambda:contract.owner(root,auth_ref,target,master,live=True,started=20))
        contract.owner(root,auth_ref,target,master,live=False,started=20)
        checks.append('Historical owner replay does not reinterpret a later admission stop as failed science')
        stop.unlink()
        core_ref=save('fixture/core_receipt.json',{'synthetic_fixture_only':True})
        pub_ref=save('fixture/core_publication.json',{'synthetic_fixture_only':True})
        design=save('fixture/aux_design.json',{'kind':'distinct_model_conditioned_value_candidates_design',
            'profiles':{'qwen8b':{'native_common_binding':common}},'maximum_worker_seconds':115200})
        master['auxiliary_candidate_design']=design
        captured=[]
        core_fn,aux_fn=contract.verify_cross_size_seal,contract.verify_auxiliary_seal
        def core_spy(*args,**kwargs):captured.append(('core',args,kwargs));return {'sealed_unix':10}
        def aux_spy(*args,**kwargs):captured.append(('aux',args,kwargs));return {'design':design}
        try:
            contract.verify_cross_size_seal,contract.verify_auxiliary_seal=core_spy,aux_spy
            fb={'cross_size_seal':core_ref,'cross_size_publication':pub_ref,
                'auxiliary_forecast_disposition':{'status':'sealed','receipt':core_ref,'publication':pub_ref}}
            master['analysis_contract']=common
            contract.forecasts(root,fb,master,20)
            assert [x[0] for x in captured]==['core','aux']
            assert all(x[2]['evaluation_started_unix']==20 and x[2]['check_remote'] is False for x in captured)
            checks.append('Both actual-seal verifiers receive the exact native parent/split/publication and pre-target timestamp')
        finally:
            contract.verify_cross_size_seal,contract.verify_auxiliary_seal=core_fn,aux_fn
        # Exercise real git-show byte checks inside an isolated local fixture
        # repository. This never queries or publishes to any remote.
        failure=save('fixture/worker_failure.json',{'classification':'synthetic_measurement_batch_audit_failure'})
        ledger={'failures':[{'phase':'batch_audit','evidence':failure}], 'raw_evidence_refs':[failure],
                'all_owned_scientific_children_terminal':True,'completed_cases':32,
                'complete_measurement_audit_passed':False,'scientific_retries':0,'native8b_target_labels_used':False}
        ledger_ref=save('disposition/terminal_and_cost.json',ledger)
        receipt={'kind':'e368_bounded_measurement_failure_disposition','target_model_revision':REVISION,
                 'native_common_binding':common,'bounded_design':design,'native8b_target_labels_used':False,
                 'all_owned_scientific_children_terminal':True,'automatic_retry':False,
                 'actual_auxiliary_points_available':False,'post_label_repair_can_be_called_prospective':False,
                 'files':{'terminal_and_cost.json':ledger_ref['sha256']},'partial_cost_and_terminal_evidence':ledger_ref,
                 'sealed_unix':10,'publication_required_before_any_native8b_target_evaluation':True}
        receipt_ref=save('disposition/receipt.json',receipt)
        subprocess.run(['git','init','-q'],cwd=root,check=True,capture_output=True)
        subprocess.run(['git','add','.'],cwd=root,check=True,capture_output=True)
        subprocess.run(['git','-c','user.name=CPU fixture','-c','user.email=cpu-fixture@example.invalid',
                        'commit','-qm','Synthetic target admission fixture only'],cwd=root,check=True,capture_output=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        publication=save('disposition/publication.json',{'receipt_sha256':receipt_ref['sha256'],
                         'verified_remote_at_unix':15,'commit':commit,'synthetic_no_remote_query_or_publication':True})
        disposition={'receipt':receipt_ref,'publication':publication,'status':'unavailable_after_bounded_measurement'}
        contract.verify_unavailable_auxiliary(root,disposition,master,20)
        checks.append('Bounded auxiliary failure may retain all32 executed cases with failed aggregate audit; local Git fixture verifies exact failure bytes')
        reject('Failure disposition published after target time rejected',lambda:contract.verify_unavailable_auxiliary(root,disposition,master,12))
        (root/failure['path']).write_text('altered synthetic failure evidence\n')
        reject('Changed raw auxiliary failure evidence rejected',lambda:contract.verify_unavailable_auxiliary(root,disposition,master,20))
    # A real currently absent seal must refuse. No placeholder or target output
    # is created to make the check pass.
    actual_seal=ROOT/'rl/qwen3_8b_cross_size_prediction_seal_20261017/receipt.json'
    if not actual_seal.exists():
        initial=read(ROOT/'rl/QWEN3_8B_STAGE0_INITIAL_PROBE_EXECUTION_20261017.json')
        actual_master={'native_common_binding':{'path':'rl/qwen3_8b_stage0_initial_binding_20261017/common_binding.json',
                         'sha256':digest(ROOT/'rl/qwen3_8b_stage0_initial_binding_20261017/common_binding.json')},
                       'analysis_contract':{'path':'rl/BFCL_SOURCE_GROUP_ANALYSIS_CONTRACT_20261011.json',
                         'sha256':digest(ROOT/'rl/BFCL_SOURCE_GROUP_ANALYSIS_CONTRACT_20261011.json')}}
        actual_binding={'cross_size_seal':{'path':str(actual_seal.relative_to(ROOT)),'sha256':'0'*64},
                        'cross_size_publication':{'path':str(actual_seal.parent.relative_to(ROOT))+'/publication.json','sha256':'0'*64}}
        reject('Actual incomplete native8B prediction seal refuses target prerequisites without creating outputs',
               lambda:contract.forecasts(ROOT,actual_binding,actual_master,20))
        assert not actual_seal.parent.exists()
    print(json.dumps({'status':'passed','checks':checks,
        'fixtures':'Synthetic audit/control metadata and isolated temporary local Git fixture; all deleted; actual absent-seal rejection checked where applicable.',
        'real_native8b_target_labels_accessed':False,'actual_target_manifest_registered':False,
        'model_weights_loaded':False,'gpu_jobs':0,'new_model_tokens':0,'remote_publications':0},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
