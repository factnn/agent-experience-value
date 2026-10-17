"""Bind E368 only after the original native8B probe is audited and exited."""
import argparse
import copy
import importlib.metadata
from pathlib import Path
import time

from bfcl_model_conditioned_contract import (EXECUTION_KIND, check_bytes, digest, exclusive,
    identity_ref, load_execution, pinned, read, ref)
from seal_bfcl_qwen8_cross_size_predictions import native_target_record_check

RUNTIME = [
    'bfcl_model_conditioned_contract.py', 'bfcl_model_conditioned_feature_inputs.py',
    'bfcl_model_conditioned_likelihood.py', 'run_bfcl_model_conditioned_measurement.py',
    'audit_bfcl_model_conditioned_measurement.py', 'supervise_bfcl_model_conditioned_measurements.py',
    'bind_bfcl_model_conditioned_execution.py', 'seal_bfcl_model_conditioned_predictions.py',
    'fit_bfcl_structural_candidates.py', 'seal_bfcl_qwen8_cross_size_predictions.py',
    'train_rl_smoke.py', 'rl_learning_state.py', 'bfcl_stage1_probe_ssh.py',
    'supervise_bfcl_stage1_probe.py', 'supervise_bfcl_stage_fork.py']
VERSIONS = ['torch', 'transformers', 'peft', 'trl', 'numpy', 'tokenizers', 'accelerate']
METADATA = ['config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
            'vocab.json', 'merges.txt', 'model.safetensors.index.json', 'special_tokens_map.json']


def process_state(pid):
    path = Path('/proc') / str(pid) / 'stat'
    return path.read_text().split(') ', 1)[1].split()[0] if path.exists() else None


def build(root, design_ref, native_control, execution_path, binding_root):
    design = pinned(root, design_ref)
    design_ref = identity_ref(design_ref)
    native_plan = pinned(root, design['native8b_probe_plan'])
    audit_path = root / native_plan['output_root'] / 'audit_status.json'
    if not audit_path.exists():
        raise ValueError('Prior native8B16-source probe is not complete; no execution binding created')
    batch = read(audit_path)
    expected = [x['name'] for x in native_plan['cases']]
    if (batch['status'] != 'passed' or batch['complete_cases'] != 16
            or [x['case'] for x in batch['rows']] != expected):
        raise ValueError('Prior native probe lacks complete original16 raw audits')
    queue = read(native_control / 'queue.json')
    status = read(native_control / 'supervisor.json')
    if (queue['active'] or queue['pending'] or queue['failures']
            or sorted(queue['complete']) != sorted(expected) or status['status'] != 'audited_complete'):
        raise ValueError('Prior native probe queue has not fully finished')
    actual_plan_ref = read(native_control / 'bound_probe_plan.json')
    if any(actual_plan_ref[k] != design['native8b_probe_plan'][k] for k in ['path', 'sha256']):
        raise ValueError('Wrong prior native probe control root')
    launch = read(native_control / 'launch.json')
    controller_state = process_state(launch['supervisor_pid'])
    if controller_state not in [None, 'Z', 'X']:
        raise ValueError('Prior native controller remains live')
    children = []
    for name in expected:
        process_file = native_control / (name + '.console.process.json')
        process = read(process_file)
        state = process_state(process['pid'])
        terminal_file = root / native_plan['output_root'] / name / 'supervisor.json'
        terminal = read(terminal_file)
        if state not in [None, 'Z', 'X'] or terminal['returncode'] != 0 or terminal['hard_timeout'] is not False:
            raise ValueError('Prior native scientific process not complete/exited: ' + name)
        children.append({'case': name, 'pid': process['pid'], 'state': state,
                         'process_record': ref(root, process_file), 'terminal': ref(root, terminal_file)})
    access = native_target_record_check(root)
    master = pinned(root, native_plan['master'])
    cases = copy.deepcopy(design['cases'])
    reference_plan = pinned(root, design['reference_probe_plan'])
    for case in cases:
        own_plan = pinned(root, design['profiles'][case['profile']]['own_probe_plan'])
        case['own_raw'] = ref(root, root / own_plan['output_root'] / case['own_case'] / 'rollouts.jsonl')
        case['reference_raw'] = ref(root, root / reference_plan['output_root'] / case['reference_case'] / 'rollouts.jsonl')
    assets, asset_by_name = {}, {}
    for name, profile in design['profiles'].items():
        directory = Path(profile['base_model_path'])
        files = {filename: directory / filename for filename in METADATA if (directory / filename).exists()}
        if not {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt'} <= files.keys():
            raise ValueError('Incomplete cached model metadata: ' + name)
        assets[name] = [ref(root, file) for file in files.values()]
        asset_by_name[name] = {filename: ref(root, file) for filename, file in files.items()}
        if 'model_verification' in profile:
            old = pinned(root, profile['model_verification'])
            expected_files = {x['path']: x['sha256'] for x in old['files']}
            for filename, file in files.items():
                if filename in expected_files and digest(file) != expected_files[filename]:
                    raise ValueError('Cached native model metadata differs from original verification: ' + filename)
    tokenizer_files = ['tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']
    for filename in tokenizer_files:
        if asset_by_name['qwen4b'][filename]['sha256'] != asset_by_name['qwen8b'][filename]['sha256']:
            raise ValueError('Reference likelihood requires exact shared-tokenizer bytes')
    code = {'pipeline/' + name: digest(root / 'pipeline' / name) for name in RUNTIME}
    check_bytes(root, design['signals']['tool_feedback_content_sensitivity']['mask_implementation'])
    lanes, loads = {str(i): [] for i in range(8)}, [0] * 8

    def cost(case):
        multiplier = 2 if case['profile'] == 'qwen8b' else 1
        return multiplier * (64 + case['maximum_forward_calls'])

    for case in sorted(cases, key=lambda x: (-cost(x), x['name'])):
        gpu = min(range(8), key=lambda i: (loads[i], i))
        lanes[str(gpu)].append(case['name'])
        loads[gpu] += cost(case)
    if execution_path.exists() or binding_root.exists():
        raise ValueError('Execution binding already exists; preserve it')
    proof = {'kind': 'native8b_complete_probe_terminal_observation', 'observed_unix': time.time(),
             'native_probe_audit': ref(root, audit_path), 'controller_pid': launch['supervisor_pid'],
             'controller_state': controller_state, 'children': children,
             'all_owned_scientific_children_terminal': True, 'target_record_check': access}
    binding_root.mkdir(exist_ok=False)
    exclusive(binding_root / 'native_probe_terminal.json', proof)
    execution = {'kind': EXECUTION_KIND, 'bound_unix': time.time(), 'design': design_ref,
        'profiles': design['profiles'], 'cases': cases, 'allowed_gpus': list(range(8)),
        'max_simultaneous_gpus': 8, 'automatic_retry': False, 'gpu_lanes': lanes,
        'schedule_cost_rule': 'Fixed LPT by(64+maximum_forward_calls)*model-size proxy2for8B/1for4B; no outcomes used',
        'schedule_proxy_loads': loads, 'current_ssh_owner': master['current_ssh_owner'],
        'runtime_code_sha256': code, 'runtime_versions': {name: importlib.metadata.version(name) for name in VERSIONS},
        'model_metadata_assets': assets, 'shared_tokenizer_bytes_verified': True,
        'complete_native_probe_audit': proof['native_probe_audit'],
        'prior_native_execution_terminal_proof': ref(root, binding_root / 'native_probe_terminal.json'),
        'target_evaluation_allowed': False, 'new_model_tokens': 0, 'new_scientific_children_launched': 0,
        'publication_required_before_scoring': True, 'goal_complete': False}
    exclusive(execution_path, execution)
    result = identity_ref(ref(root, execution_path))
    load_execution(root, result)
    return result


def main():
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--design', required=True)
    ap.add_argument('--design-sha256', required=True)
    ap.add_argument('--native-control-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--binding-root', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(build(root, {'path': args.design, 'sha256': args.design_sha256},
          root / args.native_control_root, root / args.output, root / args.binding_root), indent=2))


if __name__ == '__main__':
    main()
