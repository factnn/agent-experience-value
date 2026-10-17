"""CPU fixed32 identities and real-incomplete target-registration refusal."""
import json
import os
from pathlib import Path
import tempfile

from bfcl_qwen8_source_execution import exclusive,ref
from bfcl_source_eval_contract import pinned
from prepare_bfcl_qwen8_source_transfer import cells,name,prepare,terminal_training
from supervise_bfcl_stage_fork import ROOT


def main():
    d=pinned(ROOT,ref(ROOT,ROOT/'rl/QWEN3_8B_STAGE0_FORWARD15_SCIENTIFIC_DESIGN_20261017.json'))
    master={'jobs':[{'name':j['case_id']} for j in d['training_job_blueprints']]}
    matrix=cells(master)
    assert len(matrix)==len(set(matrix))==32
    assert {case for case,_ in matrix}=={'common',*[j['case_id'] for j in d['training_job_blueprints']]}
    assert all({e for c,e in matrix if c==case}=={0,1} for case in {c for c,_ in matrix})
    assert len({name(case,e) for case,e in matrix})==32
    checks=['Exactly common plus all15 frozen endpoint identities, each original evaluation repeat once;32unique outputs']
    with tempfile.TemporaryDirectory(prefix='native8b-target-registration-cpu-',dir=ROOT/'experiments/.runtime') as temporary:
        directory=Path(temporary)
        process_file=directory/'live_cpu_fixture.json';exclusive(process_file,{'pid':os.getpid(),'synthetic_fixture_only':True})
        process_ref=ref(ROOT,process_file)
        proof={'controllers':[{'pid':os.getpid(),'process_record':process_ref}],
            'children':[{'case':j['case_id'],'pid':os.getpid(),'process_record':process_ref} for j in d['training_job_blueprints']],
            'synthetic_rejection_fixture_only':True}
        proof_file=directory/'proof.json';exclusive(proof_file,proof)
        binding={'prior_training_terminal_proof':ref(ROOT,proof_file)}
        try:terminal_training(ROOT,binding,master)
        except ValueError:checks.append('A known live CPU fixture process prevents early target registration; no process signalled')
        else:raise AssertionError('Live process accepted as terminal')
        proof['children']=proof['children'][:-1]
        proof_file=directory/'incomplete_proof.json';exclusive(proof_file,proof)
        try:terminal_training(ROOT,{'prior_training_terminal_proof':ref(ROOT,proof_file)},master)
        except ValueError:checks.append('Missing source child identity prevents target registration')
        else:raise AssertionError('Missing child accepted')
    execution_path=ROOT/'rl/QWEN3_8B_STAGE0_FORWARD15_EXECUTION_20261017.json'
    registration=ROOT/d['output_roots']['evaluation_jobs'];output=ROOT/d['output_roots']['evaluation']
    authority=ROOT/'rl/QWEN3_8B_STAGE0_FORWARD32_EVAL_SSH_AUTHORITY_20261017.json'
    assert not execution_path.exists() and not registration.exists() and not output.exists() and not authority.exists()
    try:
        prepare(ROOT,{'path':str(execution_path.relative_to(ROOT)),'sha256':'0'*64},
            {'path':'rl/qwen3_8b_stage0_forward32_binding_20261017/admission.json','sha256':'0'*64},
            authority,'rl/qwen3_8b_stage0_forward32_eval_ssh_20261017')
    except (FileNotFoundError,ValueError) as error:
        refusal={'type':type(error).__name__,'message':str(error)}
    else:raise AssertionError('Actual incomplete training evidence admitted native8B targets')
    assert not registration.exists() and not output.exists() and not authority.exists()
    print(json.dumps({'status':'passed','checks':checks,'actual_current_registration_refusal':refusal,
        'fixtures':'Only temporary structural/rejection metadata using this live CPU checker PID; deleted. No positive simulated training or native8B target outcomes.',
        'actual_target_manifest_created':False,'new_gpu_jobs':0,'new_model_tokens':0,'native8b_target_labels_accessed':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
