"""Plot complete audited curves only; unfinished repeats remain visibly pending."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    panels, sources = [], []
    for name, filename in [('Discovery (previous trajectory)', 'bfcl_discovery_two_eval_seeds_20261011.json'),
                           *[(f'Training repeat {n}', f'bfcl_repeat_{n}_two_eval_seeds_20261011.json')
                             for n in (1, 2, 3)]]:
        path = ROOT / 'rl' / filename
        if not path.exists():
            panels.append({'name': name, 'status': 'pending_complete_audited_curve'})
            continue
        raw = path.read_bytes()
        data = json.loads(raw)
        curves = []
        if 'rows' in data:
            assert data['all_panel_audits'] == 'passed'
            for row in data['rows']:
                assert row['n_tasks'] == 22
                curves.append({'seed': row['eval_seed_name'], 'tokens': data['dose_actual_tokens'],
                               'successes': row['successes'], 'G': row['G_task_counts']})
        else:
            assert data['status'] == 'complete_repeat_evaluation_audited'
            for row in data['curves']:
                curves.append({'seed': row['evaluation_repeat'],
                               'tokens': [0] + [x['actual_training_tokens'] for x in row['panels']],
                               'successes': row['successes'], 'G': row['G_task_counts']})
        assert [x['seed'] for x in curves] == ['e0', 'e1']
        for curve in curves:
            assert len(curve['tokens']) == len(curve['G']) == len(curve['successes']) == 4
            assert curve['tokens'] == sorted(curve['tokens'])
            assert curve['G'] == [x - curve['successes'][0] for x in curve['successes']]
        panels.append({'name': name, 'status': 'complete_curve', 'curves': curves})
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest()})
    values = [g for panel in panels for c in panel.get('curves', []) for g in c['G']]
    bound = max(3, max(abs(x) for x in values)) + 0.5
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        for ax, panel in zip(axes.flat, panels):
            ax.set_title(panel['name'], loc='left', fontweight='bold')
            ax.axhline(0, color='#777777', lw=0.8)
            ax.grid(alpha=0.18)
            ax.set_xlim(-3, 150)
            ax.set_ylim(-bound, bound)
            ax.set_xticks([0, 40, 80, 120])
            ax.set_yticks(range(int(-bound), int(bound) + 1))
            if panel['status'] != 'complete_curve':
                ax.set_facecolor('#f5f5f5')
                ax.text(0.5, 0.53, 'Complete curve pending', transform=ax.transAxes,
                        ha='center', va='center', fontsize=13, color='#555555')
                ax.text(0.5, 0.40, 'Partial scores are not plotted', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='#666666')
                continue
            for c, color, marker, style in zip(panel['curves'], ['#1565a9', '#c06412'], ['o', 's'], ['-', '--']):
                ax.plot([x / 1000 for x in c['tokens']], c['G'], color=color,
                        marker=marker, linestyle=style, label=f"Evaluation seed {c['seed']}",
                        linewidth=1.7, markersize=6, markerfacecolor='white')
            ax.legend(frameon=False, fontsize=9, loc='upper right')
        fig.supxlabel('Actual newly generated training tokens (thousands)', y=0.10)
        fig.supylabel('Net successes vs. same-seed baseline (out of 22 tasks)', x=0.025)
        fig.suptitle('BFCL development dose curves: evaluation-seed dependence', fontsize=14, y=0.98)
        fig.text(0.08, 0.025,
                 'Shared ancestor and development tasks; trajectories are conditional repeats. No confidence intervals.\n'
                 'Lines connect observed checkpoints only. Discovery first checkpoint has unchanged parameters.',
                 fontsize=9, color='#444444')
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.91, hspace=0.27, wspace=0.14)
        output = ROOT / 'figures'
        output.mkdir(exist_ok=True)
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'bfcl_completed_dose_curves_20261011.{suffix}', dpi=180)
        plt.close(fig)
        (output / 'bfcl_completed_dose_curves_20261011.sources.json').write_text(
            json.dumps({'source_reports': sources, 'panels': panels, 'new_model_calls': 0,
                        'scope': 'Visualization of complete audited curves; no incomplete-panel outcomes read.'}, indent=2) + '\n')
    print(json.dumps({'complete_curves': len(sources), 'pending_curves': len(panels) - len(sources)}))


if __name__ == '__main__':
    main()
