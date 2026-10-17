"""Run a registered native8B source branch through the unchanged GRPO trainer."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

import train_bfcl_source_group as core
from bfcl_qwen8_source_training import load_job, model_path, native_factories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--job-sha256', required=True)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--execution-sha256', required=True)
    args = parser.parse_args()
    execution = {'path': args.execution, 'sha256': args.execution_sha256}
    job, master, _, _ = load_job(core.r.ROOT, args.job, args.job_sha256, execution)
    path = model_path(master)
    metadata = {name: job[name] for name in ['native_cohort_design',
                'native_common_binding', 'native_model_verification']}
    metadata['native_execution_protocol'] = execution
    metadata.update(native_training_started_unix=time.time(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=core.r.ROOT, text=True).strip(),
        versions={n: importlib.metadata.version(n) for n in
                  ['torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets', 'tokenizers']})
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], '--job', str(args.job), '--job-sha256', args.job_sha256]
        with native_factories(core.r, path, metadata):
            core.main()
    finally:
        sys.argv = original_argv
    # Exact parent input identities remain checked after the scientific process.
    load_job(core.r.ROOT, args.job, args.job_sha256, execution)


if __name__ == '__main__':
    main()
