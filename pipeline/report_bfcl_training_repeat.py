"""Audit one complete training-repeat curve without summarizing unfinished repeats."""
import argparse
import json
from pathlib import Path
import report_bfcl_dose_repeats as r


def report(protocol, name):
    master = r.read(protocol)
    assert name in [x['name'] for x in master['training_repeats']]
    root = r.ROOT / master['output']
    labels = r.read(root / f'train_{name}/dose_checkpoints.json')
    assert [x['threshold'] for x in labels] == master['dose_thresholds']
    curves, unique_panels = [], {}
    for e in master['evaluation_repeats']:
        p = r.check_protocol(e['protocol'])
        base_path = (r.ROOT / master['previous_wave']['path'] / 'eval_common'
                     if e['name'] == 'e0' else root / 'eval_common_e1')
        base = r.check_evaluation(base_path, p, e['protocol'], r.ROOT / master['source_common_path'])
        panels = []
        for label in labels:
            path = root / f"eval_{name}_step_{label['optimizer_steps']:03d}_{e['name']}"
            evaluated = r.check_evaluation(path, p, e['protocol'], r.ROOT / label['policy'])
            unique_panels[str(path)] = evaluated
            panels.append({k: v for k, v in evaluated.items() if k != 'outcomes'} | {
                'threshold': label['threshold'], 'actual_training_tokens': label['actual_tokens'],
                'G_task_count': evaluated['successes'] - base['successes'],
                'paired_changes': [{'task_id': x['task_id'], 'base': b['rewards'][0], 'trained': x['rewards'][0]}
                                   for x, b in zip(evaluated['outcomes'], base['outcomes']) if x['rewards'] != b['rewards']]})
        curves.append({'evaluation_repeat': e['name'], 'baseline_path': str(base_path.relative_to(r.ROOT)),
                       'baseline_successes': base['successes'], 'panels': panels,
                       'successes': [base['successes']] + [x['successes'] for x in panels],
                       'G_task_counts': [0] + [x['G_task_count'] for x in panels],
                       'late_minus_middle_successes': panels[-1]['successes'] - panels[-2]['successes']})
    return {'status': 'complete_repeat_evaluation_audited', 'training_repeat': name,
            'master_protocol_sha256': r.digest(protocol), 'curves': curves,
            'unique_checkpoint_evaluation_panels': len(unique_panels),
            'checkpoint_evaluation_tokens': sum(x['cost']['new_model_tokens'] for x in unique_panels.values()),
            'baseline_cost_accounting': 'Shared baseline costs remain in batch ledger; not charged again in this repeat subtotal.',
            'new_model_calls_for_report': 0,
            'scope': 'One conditional trajectory, development panel, two evaluation seeds. No cross-training stability, variance-component, best-dose or transfer claim.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--repeat', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = report(args.protocol, args.repeat)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'curves'}))
