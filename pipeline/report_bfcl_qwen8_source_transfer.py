"""Raw native8B target replay with all-role prospective publication and base identity."""
import argparse
import json
from pathlib import Path

from bfcl_qwen8_transfer_contract import load_job
from bfcl_qwen8_source_training import REVISION, identity
from bfcl_source_eval_contract import digest, pinned
from report_bfcl_source_eval import audit_panel
from report_bfcl_intervention import read, lines, audit_sampling
from supervise_bfcl_stage_fork import ROOT


def audit_job(root, path):
    raw = read(path); output = root/raw['output']
    run, summary = read(output/'run.json'), read(output/'summary.json')
    job, master, _, _, _ = load_job(root, path, digest(path),
        authority_ref=run['current_eval_authority'], publication_ref=run['execution_publication'],
        admit=False, evaluation_started_unix=run['evaluation_started_unix'])
    common = pinned(root, master['native_common_binding']); common_manifest = pinned(root, common['manifest'])
    integrity = read(output/'native_base_integrity.json')
    expected = common_manifest['frozen_base_sha256']
    if (run['job_sha256'] != digest(path) or run['stage'] != 0 or run['native_model_revision'] != REVISION
            or run['target_evaluation'] is not True or run['reserved21_accessed'] is not False
            or run['within_native8b_fitting_performed'] is not False or run['heldout_gate'] != 'verified'
            or identity(run['native_execution_protocol']) != identity(job['execution_protocol'])
            or identity(run['native_common_binding']) != identity(master['native_common_binding'])
            or identity(run['native_model_verification']) != identity(master['model_verification'])
            or identity(run['admission_binding']) != identity(job['admission_binding'])
            or run['forward_verification'] != job['_runtime_forward_verification']):
        raise ValueError('Raw native8B run lacks its exact pre-target model, forecast or owner identity')
    if run['native_model_path'] != pinned(root, master['model_verification'])['destination']:
        raise ValueError('Target child used another model cache')
    if run['versions']!=master['runtime_versions']:
        raise ValueError('Actual native8B target libraries differ from the frozen learner execution')
    if (any(integrity[key] != expected for key in ['expected_frozen_base_sha256', 'initial_frozen_base_sha256', 'final_frozen_base_sha256'])
            or integrity['frozen_base_unchanged'] is not True or integrity['optimizer_updates'] != 0):
        raise ValueError('Native8B frozen base identity was not preserved')
    cost = audit_sampling(output)
    scores = audit_panel(job, master['evaluation'], summary, run,
                        lines(output/'outcomes.jsonl'),lines(output/'rollouts.jsonl'),lines(output/'generation_calls.jsonl'),cost)
    terminal = read(output/'supervisor.json')
    if terminal['returncode'] != 0 or terminal['hard_timeout'] is not False:
        raise ValueError('A failed or interrupted native8B target cannot count as a complete panel')
    return {'status': 'passed', 'stage': 0, 'native_model_revision': REVISION,
        'policy_case': job['policy_case'], 'evaluation_repeat': job['evaluation_repeat'], 'role': job['role'],
        'job_sha256': digest(path), 'policy_fingerprint': job['policy_fingerprint'], 'cost': cost, **scores,
        'actual_supervised_worker_wall_seconds': terminal['wall_seconds'],
        'native_common_binding': master['native_common_binding'], 'execution_protocol': job['execution_protocol'],
        'admission_binding': job['admission_binding'], 'forward_verification': job['_runtime_forward_verification'],
        'forward_publication_timing_verified_for_every_role': True, 'frozen_base_sha256': expected,
        'source_files': {name: digest(output/name) for name in
            ['summary.json','run.json','outcomes.jsonl','rollouts.jsonl','generation_calls.jsonl','native_base_integrity.json','supervisor.json']},
        'within_native8b_fitting_performed': False, 'reserved21_accessed': False,
        'scope': 'One complete native8B panel; full paired G/V, uncertainty and prediction error require the entire registered cohort.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args();job=read(args.job);output=ROOT/job['output']/'audit.json'
    if output.exists():raise FileExistsError('Preserve original target audit; use pure replay for later review')
    try:
        result=audit_job(ROOT,args.job)
    except Exception as error:
        result={'status':'failed','type':type(error).__name__,'message':str(error),'partial_evidence_retained':True,'automatic_retry':False}
    with output.open('x') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['status']=='passed' else 1)
