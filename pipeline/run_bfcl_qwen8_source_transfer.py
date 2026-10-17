"""Run admitted native8B targets with the unchanged paired inference engine."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from bfcl_qwen8_source_training import model_path, native_factories
from bfcl_qwen8_transfer_contract import load_job, owner
from bfcl_source_eval_contract import digest, pinned
from supervise_bfcl_stage_fork import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--job-sha256', required=True)
    for field in ['authority', 'publication']:
        parser.add_argument('--'+field, required=True)
        parser.add_argument('--'+field+'-sha256', required=True)
    args = parser.parse_args()
    authority = {'path': args.authority, 'sha256': args.authority_sha256}
    publication = {'path': args.publication, 'sha256': args.publication_sha256}
    try:
        checked = load_job(ROOT, args.job.resolve(), args.job_sha256,
                           authority_ref=authority, publication_ref=publication, admit=True)
        job, master = checked[:2]
        path = model_path(master)
    except Exception as error:
        print(json.dumps({'classification': 'native8b_target_admission_before_model_load',
            'type': type(error).__name__, 'message': str(error), 'scientific_generation_started': False,
            'new_scientific_tokens': 0}), file=sys.stderr, flush=True)
        raise SystemExit(3) from error
    import eval_bfcl_source_transfer as engine
    original_loader, original_argv = engine.load_job, sys.argv
    original_peft, original_sampling = engine.r.PeftModel, engine.r.sample_episodes
    common = pinned(ROOT, master['native_common_binding'])
    common_manifest = pinned(ROOT, common['manifest'])
    expected_base = common_manifest['frozen_base_sha256']
    initial_hash, models, generation_started = [], [], False
    def loader(root, job_path, expected_sha):
        if root != ROOT or job_path.resolve() != args.job.resolve() or expected_sha != args.job_sha256 or digest(job_path) != expected_sha:
            raise ValueError('Previously admitted native8B target job changed')
        owner(ROOT, authority, job, master, live=True, started=time.time())
        return checked
    class NativePolicy:
        @staticmethod
        def from_pretrained(model, adapter_path, **kwargs):
            if Path(adapter_path).resolve() != (checked[3]/'adapter').resolve() or kwargs.get('is_trainable') is not True:
                raise ValueError('Inference must load the exact admitted native8B adapter')
            policy = original_peft.from_pretrained(model, adapter_path, **kwargs)
            actual = engine.r.tensor_hash((n,p) for n,p in policy.named_parameters() if not p.requires_grad)
            if actual != expected_base:
                raise ValueError('Loaded frozen base differs from its native8B common state')
            initial_hash.append(actual); models.append(policy)
            return policy
    def sample(*positional, **kwargs):
        nonlocal generation_started
        generation_started = True
        return original_sampling(*positional, **kwargs)
    metadata = {'stage': 0, 'native_execution_protocol': job['execution_protocol'],
        'native_common_binding': master['native_common_binding'], 'native_model_verification': master['model_verification'],
        'current_eval_authority': authority, 'execution_publication': publication,
        'admission_binding': job['admission_binding'], 'forward_verification': job['_runtime_forward_verification'],
        'within_native8b_fitting_performed': False,
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ['torch','transformers','trl','peft','accelerate','datasets','tokenizers']}}
    engine.load_job, engine.r.PeftModel, engine.r.sample_episodes = loader, NativePolicy, sample
    sys.argv = [str(Path(engine.__file__)), '--job', str(args.job.resolve()), '--job-sha256', args.job_sha256]
    try:
        with native_factories(engine.r, path, metadata, target_evaluation=True):
            engine.main()
            if len(models) != 1 or initial_hash != [expected_base]:
                raise ValueError('Expected exactly one admitted native8B policy')
            final = engine.r.tensor_hash((n,p) for n,p in models[0].named_parameters() if not p.requires_grad)
            if final != expected_base:
                raise ValueError('Frozen native8B base parameters changed during target inference')
            output = ROOT/job['output']/'native_base_integrity.json'
            with output.open('x') as handle:
                json.dump({'expected_frozen_base_sha256': expected_base, 'initial_frozen_base_sha256': initial_hash[0],
                    'final_frozen_base_sha256': final, 'frozen_base_unchanged': True,
                    'trainable_fingerprint_checked_by_original_engine': True, 'optimizer_updates': 0},handle,indent=2)
                handle.write('\n')
    except Exception as error:
        output = ROOT/job['output']
        if output.exists() and not (output/'native_target_failure.json').exists():
            with (output/'native_target_failure.json').open('x') as handle:
                json.dump({'type': type(error).__name__, 'message': str(error),
                    'classification': 'scientific_target_execution_failure' if generation_started else 'native_target_dependency_or_identity_before_generation',
                    'scientific_generation_started': generation_started, 'partial_evidence_retained': True,
                    'automatic_retry': False},handle,indent=2)
                handle.write('\n')
        raise
    finally:
        engine.load_job, engine.r.PeftModel, engine.r.sample_episodes, sys.argv = original_loader, original_peft, original_sampling, original_argv


if __name__ == '__main__':
    main()
