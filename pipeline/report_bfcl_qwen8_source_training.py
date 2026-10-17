"""Retain the original full RL raw audit and add native8B identity evidence."""
import argparse
import json
from pathlib import Path

from bfcl_qwen8_source_training import load_job, REVISION, identity
from bfcl_source_eval_contract import digest, pinned
from report_bfcl_source_training import audit_job as original_audit
from supervise_bfcl_stage_fork import ROOT, read


def audit_job(root, path, execution_ref):
    job, master, common, common_manifest = load_job(root, path, digest(path), execution_ref)
    out = root/job['output']
    result = original_audit(job, digest(path))
    run, summary = read(out/'run.json'), read(out/'summary.json')
    terminal=read(out/'supervisor.json')
    if terminal['returncode']!=0 or terminal['hard_timeout'] is not False:
        raise ValueError('Failed or interrupted native8B source process cannot pass as a complete branch')
    if run['versions']!=master['runtime_versions']:
        raise ValueError('Actual native8B training libraries differ from the frozen execution')
    verification = pinned(root, master['model_verification'])
    if (run['native_model_revision'] != REVISION
            or run['native_model_path'] != verification['destination']
            or run['target_evaluation'] is not False or run['reserved21_accessed'] is not False):
        raise ValueError('Scientific child did not use the registered native8B learner')
    if identity(run['native_execution_protocol']) != identity(execution_ref):
        raise ValueError('Native8B run belongs to another registered execution')
    for field in ['native_cohort_design', 'native_common_binding', 'native_model_verification']:
        if identity(run[field]) != identity(job[field]):
            raise ValueError('Native8B run identity changed: '+field)
    if summary['final_manifest']['frozen_base_sha256'] != common_manifest['frozen_base_sha256']:
        raise ValueError('Frozen native8B base weights differ from their own common state')
    if read(out/'restored.json')['policy_fingerprint'] != common['policy_fingerprint']:
        raise ValueError('Training did not begin at its own native8B policy')
    return result | {'native_model_revision': REVISION,
        'native_execution_protocol': execution_ref,
        'native_common_binding': job['native_common_binding'],
        'frozen_base_sha256': common_manifest['frozen_base_sha256'],
        'actual_supervised_worker_wall_seconds':terminal['wall_seconds'],
        'target_evaluation': False, 'reserved21_accessed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--execution-sha256', required=True)
    args = parser.parse_args()
    job = read(args.job)
    path = ROOT/job['output']/'audit.json'
    if path.exists():
        raise FileExistsError('Preserve prior native8B training audit; no overwrite')
    try:
        result = audit_job(ROOT, args.job,
                           {'path': args.execution, 'sha256': args.execution_sha256})
    except Exception as error:
        result = {'status': 'failed', 'type': type(error).__name__, 'message': str(error),
                  'partial_evidence_retained': True, 'automatic_retry': False}
    with path.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
