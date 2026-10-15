"""Use the original unchanged inference engine after forward32 admission.

Exit3 means target admission failed before output creation/model inference;
it is distinct from a scientifically started evaluation that later fails.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from bfcl_source_eval_contract import digest
from bfcl_stage1_forward15_contract import MODE
from bfcl_stage1_forward32_contract import load_job, check_owner
from supervise_bfcl_stage_fork import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True); parser.add_argument('--job-sha256', required=True)
    parser.add_argument('--authority', required=True); parser.add_argument('--authority-sha256', required=True)
    args = parser.parse_args()
    authority = {'path': args.authority, 'sha256': args.authority_sha256}
    try:
        checked = load_job(ROOT, args.job.resolve(), args.job_sha256, admit=True, authority_ref=authority)
    except Exception as error:
        print(json.dumps({'classification': 'target_admission_before_inference', 'type': type(error).__name__,
                          'message': str(error), 'new_scientific_tokens': 0, 'scientific_inference_started': False}), file=sys.stderr, flush=True)
        raise SystemExit(3) from error
    job = checked[0]
    import eval_bfcl_source_transfer as engine
    original_loader, original_write, original_argv = engine.load_job, engine.r.write, sys.argv
    def loader(root, path, expected_sha):
        if root != ROOT or path.resolve() != args.job.resolve() or expected_sha != args.job_sha256 or digest(path) != expected_sha:
            raise ValueError('Admitted target job changed before output creation')
        check_owner(ROOT, authority, job['execution_protocol'], live=True, started_unix=time.time())
        return checked
    def write(path, value):
        if path.name == 'run.json':
            value = value | {'stage': 1, 'batch_mode': MODE, 'forward_bundle': job['forward_bundle'],
                             'admission_binding': job['admission_binding'], 'current_eval_authority': authority,
                             'forward_verification': job['_runtime_forward_verification'],
                             'gate_scope': 'Published early-five and separateH1 forward predictions for EVERY target role',
                             'within_late_fitting_performed': False}
        return original_write(path, value)
    engine.load_job, engine.r.write = loader, write
    sys.argv = [str(Path(engine.__file__)), '--job', str(args.job.resolve()), '--job-sha256', args.job_sha256]
    try:
        engine.main()
    except Exception as error:
        if not (ROOT/job['output']/'run.json').exists():
            print(json.dumps({'classification': 'target_admission_before_inference', 'type': type(error).__name__,
                              'message': str(error), 'new_scientific_tokens': 0, 'scientific_inference_started': False}), file=sys.stderr, flush=True)
            raise SystemExit(3) from error
        raise
    finally:
        engine.load_job, engine.r.write, sys.argv = original_loader, original_write, original_argv


if __name__ == '__main__': main()
