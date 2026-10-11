"""Report the complete old discovery curve at both frozen evaluation seeds."""
import json
import report_bfcl_dose_repeats as r
from report_bfcl_intervention import unchanged_policy_consistency


def main():
    root = r.ROOT
    master = r.read(root / 'rl/BFCL_DOSE_REPEATS_PROTOCOL_20261011.json')
    old, new = root / master['previous_wave']['path'], root / master['output']
    labels = r.read(old / 'train/dose_checkpoints.json')
    result = {'scope': 'Old discovery trajectory only, two predeclared evaluation seeds. New training repetitions not yet complete.', 'rows': []}
    for spec in master['evaluation_repeats']:
        name = spec['name']
        protocol = r.check_protocol(spec['protocol'])
        baseline = old / 'eval_common' if name == 'e0' else new / 'eval_common_e1'
        base = r.check_evaluation(baseline, protocol, spec['protocol'], root / master['source_common_path'])
        panels = [base]
        for label in labels:
            step = label['optimizer_steps']
            path = old / f'eval_step_{step:03d}' if name == 'e0' else new / f'eval_discovery_step_{step:03d}_e1'
            panels.append(r.check_evaluation(path, protocol, spec['protocol'], root / label['policy']))
        result['rows'].append({
            'eval_seed_name': name, 'eval_seed': protocol['seeds']['eval'],
            'successes': [p['successes'] for p in panels], 'n_tasks': 22,
            'G_task_counts': [p['successes'] - base['successes'] for p in panels],
            'late_minus_middle_task_count': panels[3]['successes'] - panels[2]['successes'],
            'panel_paths': [p['path'] for p in panels],
            'late_vs_base_changes': [{'task_id': x['task_id'], 'base': b['rewards'][0], 'late': x['rewards'][0]}
                                    for b, x in zip(base['outcomes'], panels[3]['outcomes']) if b['rewards'] != x['rewards']],
            'unchanged_policy_control': unchanged_policy_consistency(
                r.lines(baseline / 'rollouts.jsonl'), r.lines(root / panels[1]['path'] / 'rollouts.jsonl'))})
    result.update(all_panel_audits='passed', new_model_calls_for_report=0,
                  dose_actual_tokens=[0] + [x['actual_tokens'] for x in labels])
    (root / 'rl/bfcl_discovery_two_eval_seeds_20261011.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
