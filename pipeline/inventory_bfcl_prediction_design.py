"""Metadata/history-only inventory for prospective group-held-out prediction.

Never reads evaluation rewards, fits a predictor, or selects a winning learner.
Existing online allocator history is reconstructed before any new intervention.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from bfcl_allocation import CombinationAllocator

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_lines(path):
    # Current evaluation writers may be partway through the last line.
    rows = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(rows):
        try:
            yield json.loads(line)
        except ValueError:
            if index == len(rows) - 1 and not line.endswith('\n'):
                continue
            raise


def main():
    manifest_path = ROOT / 'rl/bfcl_research_split_candidate/manifest.json'
    manifest = read(manifest_path)
    protocol_path = ROOT / 'rl/BFCL_DOSE_PROTOCOL_20261010.json'
    protocol = read(protocol_path)
    assert digest(manifest_path) == protocol['manifest_sha256']
    common_path = ROOT / 'rl/bfcl_intervention_20261008/common/allocation_state.json'
    common = CombinationAllocator.from_state_dict(read(common_path))
    old = ROOT / 'rl/bfcl_intervention_dose_20261010/train'
    allocator = CombinationAllocator.from_state_dict(common.state_dict(), rule='uniform',
        pool=protocol['conditions']['AllTasks_dose'])
    allocator.rng.seed(protocol['branch_allocator_seeds']['AllTasks_dose'])
    stages = [{'name': 'source_step0', 'learner_updates': 0,
               'cumulative_acquisition_tokens': common.total_sampled_tokens,
               'signals': common.signals()}]
    for row in valid_lines(old / 'allocation.jsonl'):
        choice = allocator.choose()
        for key in ['task_id', 'probabilities', 'signals_before', 'cost_before_choice']:
            assert choice[key] == row[key], (row['policy_step'], key)
        allocator.observe(choice['task_id'], row['rewards'], row['sampled_tokens'],
                          f"AllTasks_dose:uniform:{row['policy_step']}")
        step = row['policy_step'] + 1
        if step in (6, 11):
            saved = read(old / f'doses/step_{step:03d}/components/manifest.json')
            assert allocator.total_sampled_tokens == saved['cumulative_acquisition_tokens']
            stages.append({'name': f'discovery_step{step}', 'learner_updates': step,
                'cumulative_acquisition_tokens': allocator.total_sampled_tokens,
                'source_payload_sha256': saved['payload_sha256'], 'signals': allocator.signals()})
    groups = defaultdict(list)
    for task in manifest['train']:
        combination = protocol['training_registry'][task['id']]
        assert combination == '+'.join(sorted(task['available_tool_classes']))
        groups[combination].append(task)
    group_rows = []
    for combination, rows in sorted(groups.items()):
        group_rows.append({'combination': combination, 'tasks': [r['id'] for r in rows],
                          'task_count': len(rows), 'user_turn_counts': dict(Counter(r['user_turns'] for r in rows)),
                          'single_task_pool': len(rows) == 1,
                          'stage_signals': {s['name']: s['signals'][combination] for s in stages}})
    for stage in stages:
        vectors = [tuple(s[k] for k in ['posterior_success', 'frontier', 'coverage_deficit', 'observed_episodes'])
                   for s in stage['signals'].values()]
        stage['unique_signal_vectors'] = len(set(vectors))
        stage['combinations'] = len(vectors)
        stage['groups_without_history'] = sum(s['observed_episodes'] == 0 for s in stage['signals'].values())
    # Record selection metadata only, never scores. "Unseen" is scoped to these sources.
    reserved, observed = defaultdict(set), defaultdict(set)
    for path in sorted((ROOT / 'rl').glob('BFCL*PROTOCOL*.json')):
        for row in read(path).get('eval_panel', []):
            if 'task_id' in row:
                reserved[row['task_id']].add(str(path.relative_to(ROOT)))
    for path in sorted((ROOT / 'rl').rglob('outcomes.jsonl')):
        for row in valid_lines(path):
            if 'task_id' in row:
                observed[row['task_id']].add(str(path.relative_to(ROOT)))
    panels = {}
    for split in ['id_eval', 'composition_transfer_eval']:
        rows = []
        for task in manifest[split]:
            key = task['id']
            rows.append({'task_id': key, 'combination': '+'.join(sorted(task['available_tool_classes'])),
                         'registered_in': sorted(reserved[key]), 'observed_in': sorted(observed[key]),
                         'not_registered_or_observed_here': not reserved[key] and not observed[key]})
        untouched = [r for r in rows if r['not_registered_or_observed_here']]
        panels[split] = {'total_tasks': len(rows), 'not_registered_or_observed_count': len(untouched),
                        'untouched_by_combination': dict(Counter(r['combination'] for r in untouched)), 'tasks': rows}
    result = {'status': 'design_inventory_not_experiment_protocol',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': digest(manifest_path), 'reference_protocol_sha256': digest(protocol_path),
        'training_tasks': sum(len(x) for x in groups.values()), 'training_combinations': len(groups),
        'stages_are_candidates_not_selected_by_performance': stages, 'groups': group_rows,
        'evaluation_exposure': panels,
        'exposure_scope': 'Top-level registered BFCL protocols and rl/**/outcomes.jsonl; not a proof against pretraining exposure, semantic overlap, or unrecorded external use.',
        'signal_scope': 'Exact allocator decision/history reconstruction and cumulative-cost check; not an independent reload of optimizer tensors.',
        'new_model_calls': 0, 'predictor_fitted': False, 'new_experiment_launched': False}
    output = ROOT / 'rl/bfcl_prediction_design_inventory_20261011.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'groups': len(groups), 'stages': [{k: s[k] for k in ['name', 'unique_signal_vectors', 'groups_without_history']} for s in stages],
                      'unexposed_here': {k: v['not_registered_or_observed_count'] for k, v in panels.items()},
                      'output': str(output.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
