"""Immutable data/model bindings for the finite E368 forward-only measurement."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

DESIGN_KIND = 'distinct_model_conditioned_value_candidates_design'
EXECUTION_KIND = 'e368_model_conditioned_forward_execution'
SIGNALS = ('tool_feedback_content_sensitivity', 'reference_experience_learner_compatibility')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def check_bytes(root, reference):
    path = root / reference['path']
    if 'bytes' in reference and path.stat().st_size != reference['bytes']:
        raise ValueError('Pinned size changed: ' + reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError('Pinned bytes changed: ' + reference['path'])
    return path


def pinned(root, reference):
    return read(check_bytes(root, reference))


def ref(root, path):
    return {'path': str(path.relative_to(root)), 'sha256': digest(path), 'bytes': path.stat().st_size}


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def load_execution(root, reference):
    execution = pinned(root, reference)
    if execution['kind'] != EXECUTION_KIND:
        raise ValueError('Not an E368 scientific execution binding')
    design = pinned(root, execution['design'])
    if design['kind'] != DESIGN_KIND or len(design['cases']) != 32:
        raise ValueError('Wrong fixed32-case design')
    if execution['profiles'] != design['profiles']:
        raise ValueError('Registered model/ancestor identity changed')
    if ([{k: c[k] for k in original} for c, original in zip(execution['cases'], design['cases'])]
            != design['cases'] or len(execution['cases']) != 32):
        raise ValueError('Original source/task/case/budget identities changed')
    if (execution['allowed_gpus'] != list(range(8)) or execution['max_simultaneous_gpus'] != 8
            or execution['automatic_retry'] is not False):
        raise ValueError('Require the current eight-free-GPU cap and single attempts')
    cap = pinned(root, design['resource_authorization'])
    if cap['max_simultaneous_gpus_for_subsequent_batches'] != 8:
        raise ValueError('Wrong resource authorization')
    names = [c['name'] for c in design['cases']]
    scheduled = [name for lane in execution['gpu_lanes'].values() for name in lane]
    if (set(execution['gpu_lanes']) != {str(i) for i in range(8)}
            or sorted(scheduled) != sorted(names) or len(scheduled) != 32):
        raise ValueError('Missing/repeated/science-dependent job schedule')
    if execution['current_ssh_owner']['mode'] != 'human_authorized_ssh_takeover':
        raise ValueError('Wrong current SSH execution owner')
    for name, sha in execution['runtime_code_sha256'].items():
        if digest(root / name) != sha:
            raise ValueError('Published runtime code changed: ' + name)
    for name, version in execution['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Runtime library version changed: ' + name)
    check_bytes(root, design['signals'][SIGNALS[0]]['mask_implementation'])
    batch = pinned(root, execution['complete_native_probe_audit'])
    plan = pinned(root, design['native8b_probe_plan'])
    native_master = pinned(root, plan['master'])
    if execution['current_ssh_owner'] != native_master['current_ssh_owner']:
        raise ValueError('Execution owner differs from current human-authorized SSH takeover')
    if (batch['status'] != 'passed' or batch['model_revision'] != plan['model_revision']
            or batch['complete_cases'] != 16
            or [x['case'] for x in batch['rows']] != [c['name'] for c in plan['cases']]):
        raise ValueError('Current native8B probe is not completely audited')
    terminal = pinned(root, execution['prior_native_execution_terminal_proof'])
    if (terminal['all_owned_scientific_children_terminal'] is not True
            or terminal['native_probe_audit'] != execution['complete_native_probe_audit']):
        raise ValueError('Prior native probe still owns live scientific work')
    return execution, design


def feature_episodes(root, plan_reference, case_name, raw_reference):
    plan = pinned(root, plan_reference)
    case = next(x for x in plan['cases'] if x['name'] == case_name)
    expected = root / plan['output_root'] / case_name / 'rollouts.jsonl'
    path = check_bytes(root, raw_reference)
    if path != expected:
        raise ValueError('Raw trajectories belong to another case')
    result = {key: {} for key in case['task_ids']}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] != 'feature':
                continue
            key = row['task_id']
            if key not in result:
                raise ValueError('Nonregistered training task')
            index = case['task_ids'].index(key)
            if (row['seed'] != case['seeds']['feature'][index] or row['group'] != index
                    or row['draw_id'] != case_name + ':feature:' + key
                    or row['policy_fingerprint'] != plan['policy_fingerprint']):
                raise ValueError('Wrong feature draw/seed/policy lineage')
            draw = row['rollout_index']
            if draw not in range(4) or draw in result[key]:
                raise ValueError('Missing/repeated feature rollout identity')
            result[key][draw] = row
    if any(sorted(rows) != [0, 1, 2, 3] for rows in result.values()):
        raise ValueError('Require all original4 feature draws per task')
    return plan, case, result


def summarize_episodes(case, rows):
    if [(x['task_id'], x['rollout_index']) for x in rows] != [
            (key, draw) for key in case['task_ids'] for draw in range(4)]:
        raise ValueError('Scored episodes differ from all fixed source feature draws')
    tasks = []
    for key in case['task_ids']:
        values = [x for x in rows if x['task_id'] == key]
        tasks.append({'task_id': key,
                      **{signal: sum(x['values'][signal] for x in values) / 4 for signal in SIGNALS}})
    return {'task_rows': tasks,
            'source_features': {signal: sum(x[signal] for x in tasks) / len(tasks) for signal in SIGNALS}}
