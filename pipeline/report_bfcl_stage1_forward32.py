"""Raw forward32 target replay,including all-role forecast timing and active15 lineage."""
import argparse
import json
from pathlib import Path

from bfcl_source_eval_contract import digest
from bfcl_stage1_forward15_contract import MODE
from bfcl_stage1_forward32_contract import load_job, check_owner, registered_job
from report_bfcl_source_eval import audit_panel
from report_bfcl_intervention import read, lines, audit_sampling
from supervise_bfcl_stage_fork import ROOT


def audit_job(root, path):
    raw_job = read(path)
    out = root/raw_job['output']; run, summary = read(out/'run.json'), read(out/'summary.json')
    job, master, _, _, seal = load_job(root, path, digest(path), admit=False,
                                      evaluation_started_unix=run['evaluation_started_unix'])
    if (run['job_sha256'] != digest(path) or run['stage'] != 1 or run['batch_mode'] != MODE
            or run['forward_bundle'] != job['forward_bundle'] or run['admission_binding'] != job['admission_binding']
            or run['within_late_fitting_performed'] is not False or run['heldout_gate'] != 'verified'
            or run['prediction_seal'] is not None):
        raise ValueError('Raw target run lacks its actual all-role forward admission')
    # The live proof has additional routing fields. Replay compares the exact
    # actual publication identity and timestamp below for every target role.
    proof = job['_runtime_forward_verification']
    for name in ('status', 'receipt', 'publication_commit', 'verified_remote_at_unix'):
        if run['forward_verification'][name] != proof[name]:
            raise ValueError('Raw target forecast publication identity changed')
    authority = check_owner(root, run['current_eval_authority'], job['execution_protocol'],
                            live=False, started_unix=run['evaluation_started_unix'])
    if authority['admission_binding'] != job['admission_binding']:
        raise ValueError('Raw target owner belongs to another active batch')
    registered_job(root, path, digest(path), authority)
    cost = audit_sampling(out)
    scores = audit_panel(job, master['evaluation'], summary, run, lines(out/'outcomes.jsonl'),
                         lines(out/'rollouts.jsonl'), lines(out/'generation_calls.jsonl'), cost)
    return {'status': 'passed', 'stage': 1, 'batch_mode': MODE, 'policy_case': job['policy_case'],
            'evaluation_repeat': job['evaluation_repeat'], 'role': job['role'], 'job_sha256': digest(path),
            'policy_fingerprint': job['policy_fingerprint'], 'cost': cost, **scores,
            'source_files': {name: digest(out/name) for name in ['summary.json', 'run.json', 'outcomes.jsonl', 'rollouts.jsonl', 'generation_calls.jsonl']},
            'forward_bundle': job['forward_bundle'], 'admission_binding': job['admission_binding'],
            'forward_publication_timing_verified_for_every_role': True,
            'historical_current_eval_owner_verified': True, 'within_late_fitting_performed': False,
            'scope': 'One complete forward-only panel. Shared start/Uniform/task/ancestor dependence remains;G/V require all32.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--job', type=Path, required=True); args = parser.parse_args()
    job = read(args.job); output = ROOT/job['output']/'audit.json'
    if output.exists(): raise FileExistsError('Preserve the original raw audit; use pure replay for later review')
    try:
        result = audit_job(ROOT, args.job)
    except Exception as error:
        result = {'status': 'failed', 'type': type(error).__name__, 'message': str(error), 'partial_evidence_retained': True}
    with output.open('x') as handle: json.dump(result, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
