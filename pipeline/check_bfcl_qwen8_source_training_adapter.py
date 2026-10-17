"""CPU identity/redirect check using actual native common artifacts; no RL job."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from bfcl_qwen8_source_training import load_job, native_factories, REVISION, REQUIRED
from bfcl_source_eval_contract import digest
from supervise_bfcl_stage_fork import ROOT, read


def main():
    common_path = 'rl/qwen3_8b_stage0_initial_binding_20261017/common_binding.json'
    common = read(ROOT/common_path)
    initial = read(ROOT/common['master']['path'])
    source = read(ROOT/initial['source_protocol']['path'])
    checks = []
    with tempfile.TemporaryDirectory(prefix='native8b-training-identity-cpu-') as temporary:
        root = Path(temporary)
        def link(relative):
            path = root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.symlink_to(ROOT/relative)
        def reference(relative):
            return {'path': relative, 'sha256': digest(root/relative)}
        def save(relative, value):
            path = root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
            return reference(relative)
        for relative in [common_path, initial['source_protocol']['path'], common['master']['path'],
                         initial['model_verification']['path'], *[common[x]['path'] for x in
                         ['common_audit', 'manifest', 'learning_state', 'adapter_config', 'adapter_weights']],
                         *REQUIRED]:
            link(relative)
        # These unpublished temporary fixtures exercise the reader only. The
        # helper is not an execution admission check and no trainer is invoked.
        group = 'MathAPI+TradingBot'
        design = {'kind': 'qwen3_8b_stage0_source_forward_design', 'model_revision': REVISION,
            'stage': 0, 'source_groups': [group], 'training_repeats': 3,
            'training': {'token_budget': 65536, 'max_updates': 128, 'soft_seconds': 12600, 'hard_seconds': 14400},
            'synthetic_fixture_only': True}
        design_ref = save('fixture/design.json', design)
        target = copy.deepcopy(source)
        target['branch_raw_token_budget'] = 65536
        target['branch_soft_wall_seconds'] = 12600
        target_ref = save('fixture/target_algorithm.json', target)
        original_job = {'case_id': 'cpu_fixture_only', 'output': 'fixture/not_created',
            'role': 'source_group', 'source_group': group, 'stage': 0, 'source_step': 0,
            'training_repeat': 0, 'allow_legacy_zero_optimizer_source': False,
            'native_cohort_design': design_ref, 'native_common_binding': reference(common_path),
            'native_model_verification': initial['model_verification'], 'native_model_revision': REVISION,
            'source_checkpoint': common['common_policy'], 'source_payload_sha256': common['learning_state']['sha256'],
            'source_protocol': initial['source_protocol'], 'algorithm_protocol': target_ref,
            'task_pool': sorted(t for t, g in source['training_registry'].items() if g == group),
            'allocator_seed': 12345, 'rollout_seed_base': 123450000,
            'input_sha256': {name: digest(root/name) for name in REQUIRED}, **design['training']}
        original_master = {'kind': 'qwen3_8b_stage0_forward_training_and_transfer',
            'model_revision': REVISION, 'stage': 0, 'automatic_retry': False, 'reserved21_allowed': False,
            'design': design_ref, 'native_common_binding': reference(common_path),
            'model_verification': initial['model_verification'], 'algorithm_protocol': target_ref,
            **{key: design[key] for key in ['source_groups', 'training_repeats', 'training']}}
        def bind_and_read(job, master=None, register_job=True):
            job_ref = save('fixture/job.json', job)
            master = copy.deepcopy(original_master if master is None else master)
            master['jobs'] = [job_ref | {'name': job['case_id']}] if register_job else []
            execution = save('fixture/execution.json', master)
            return load_job(root, root/job_ref['path'], job_ref['sha256'], execution)
        bind_and_read(original_job)
        checks.append('Actual native8B common manifest, empty-step identity and immutable inputs accepted in temporary reader fixture')
        mutations = [
            ('wrong learner revision rejected', lambda j: j.update(native_model_revision='4B')),
            ('4B parent rejected', lambda j: j.update(source_checkpoint='rl/bfcl_intervention_20261008/common')),
            ('unregistered budget rejected', lambda j: j.update(token_budget=32768)),
            ('unregistered repeat rejected', lambda j: j.update(training_repeat=3)),
            ('partial source pool rejected', lambda j: j.update(task_pool=j['task_pool'][:-1])),
        ]
        for name, mutate in mutations:
            job = copy.deepcopy(original_job)
            mutate(job)
            try:
                bind_and_read(job)
            except (ValueError, AssertionError):
                checks.append(name)
            else:
                raise AssertionError('Invalid fixture accepted: '+name)
        try:
            bind_and_read(original_job, register_job=False)
        except ValueError:
            checks.append('Job absent from exact execution registry rejected')
        else:
            raise AssertionError('Unregistered job accepted')
        changed = copy.deepcopy(target)
        changed['algorithm']['learning_rate'] *= 2
        changed_ref = save('fixture/changed_algorithm.json', changed)
        job = original_job | {'algorithm_protocol': changed_ref}
        master = original_master | {'algorithm_protocol': changed_ref}
        try:
            bind_and_read(job, master)
        except ValueError:
            checks.append('Undeclared scientific algorithm transition rejected')
        else:
            raise AssertionError('Algorithm transition accepted')
        calls, writes = [], []
        class CapturedFactory:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append((str(path), kwargs))
                return 'captured-only-no-model-loaded'
        writer = lambda path, value: writes.append((path.name, value))
        runtime = SimpleNamespace(ROOT=root, AutoTokenizer=CapturedFactory, AutoModelForCausalLM=CapturedFactory, write=writer)
        destination = Path(read(ROOT/initial['model_verification']['path'])['destination'])
        with native_factories(runtime, destination, {'native_common_binding': reference(common_path)}):
            runtime.AutoTokenizer.from_pretrained(root/'smoke/model', local_files_only=True)
            runtime.AutoModelForCausalLM.from_pretrained(root/'smoke/model', dtype='bf16', attn_implementation='sdpa')
            runtime.write(root/'run.json', {'case_id': 'captured'})
            try:
                runtime.AutoModelForCausalLM.from_pretrained(root/'other-model')
            except ValueError:
                checks.append('Unexpected factory input rejected')
            else:
                raise AssertionError('Unexpected model path accepted')
        assert runtime.AutoModelForCausalLM is CapturedFactory and runtime.AutoTokenizer is CapturedFactory and runtime.write is writer
        assert len(calls) == 2 and all(path == str(destination) for path, _ in calls)
        assert calls[1][1] == {'dtype': 'bf16', 'attn_implementation': 'sdpa'}
        assert writes[0][1]['native_model_revision'] == REVISION and writes[0][1]['target_evaluation'] is False
        checks.append('Both legacy factories redirect to the actual8B cache; precision kwargs and process-local restoration retained')
        assert not (root/original_job['output']).exists()
    print(json.dumps({'status': 'passed', 'checks': checks,
        'fixtures': 'Temporary synthetic job/design/execution registry with links to actual native8B common artifacts; deleted after CPU check.',
        'actual_execution_protocol_registered': False, 'gpu_jobs': 0, 'model_weights_loaded': False,
        'new_model_tokens': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
