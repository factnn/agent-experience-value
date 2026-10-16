"""Plot saved Stage1 results; no fitting, model loading, or new evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads(args.input.read_text())
    groups = data['source_groups']
    labels = ['Math + Trading', 'Math + Travel', 'Message + Vehicle', 'Travel']
    names = ['Frozen mean', 'Task Beta', 'Pooled Beta', 'Frontier', 'Coverage']
    assert groups == ['MathAPI+TradingBot', 'MathAPI+TravelAPI', 'MessageAPI+VehicleControlAPI', 'TravelAPI']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ['#2563eb', '#d97706', '#059669']
    x = np.arange(len(groups))
    ax = axes[0, 0]
    for r, color in enumerate(colors):
        for i, group in enumerate(groups):
            row = next(row for row in data['gains'] if row['source_group'] == group and row['training_repeat'] == r)
            pos = i + (r - 1) * .19
            endpoints = [100 * row['evaluation0_G'], 100 * row['evaluation1_G']]
            ax.plot([pos, pos], endpoints, color=color, alpha=.55, linewidth=1.6)
            ax.scatter([pos] * 2, endpoints, marker='_', color=color, s=60)
            ax.scatter(pos, 100 * row['G'], color=color, s=35,
                       label=f'Training repeat {r}' if i == 0 else None)
    ax.scatter(x, [100 * data['G'][g] for g in groups], color='#111827', marker='D', s=35, label='Mean across training')
    ax.axhline(0, color='#64748b', linewidth=.8)
    ax.set(title='A. Late G: paired evaluation seeds and training repeats', ylabel='Transfer gain G (percentage points)', xticks=x, xticklabels=labels, ylim=(-10, 12))
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.text(.01, .98, 'Vertical lines: two observed seeds; not confidence intervals', transform=ax.transAxes, va='top', fontsize=8)
    ax = axes[0, 1]
    ax.bar(x - .19, [100 * data['G'][g] for g in groups], .36, label='G: vs fixed parent', color='#2563eb')
    ax.bar(x + .19, [100 * data['V'][g] for g in groups], .36, label='V: vs matched Uniform', color='#059669')
    ax.axhline(0, color='#64748b', linewidth=.8)
    ax.set(title='B. G and V use different references', ylabel='Mean gain (percentage points)', xticks=x, xticklabels=labels, ylim=(-.5, 7.5))
    ax.text(.02, .97, f"Late Uniform mean G = {100 * data['mean_uniform_G']:.4f} pp", transform=ax.transAxes, va='top', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    models = data['models']; xx = np.arange(len(models))
    ax = axes[1, 0]
    ax.bar(xx - .19, [10000 * data['early_MSE'][m] for m in models], .36, color='#94a3b8', label='Stage0: original heldout four')
    ax.bar(xx + .19, [10000 * data['late_MSE'][m] for m in models], .36, color='#2563eb', label='Stage1: sealed forward four')
    ax.set(title='C. Original five predictors, unchanged forecasts', ylabel='MSE (percentage points squared)', xticks=xx, xticklabels=names)
    ax.legend(loc='upper left', fontsize=8)
    ax.text(.02, .79, f"Separate auxiliary H1 late MSE: {10000 * data['separate_H1_MSE']:.3f} pp²", transform=ax.transAxes, fontsize=8)
    ax = axes[1, 1]
    bias = np.array([10000 * data['late_components'][m]['overall_squared_bias'] for m in models])
    centered = np.array([10000 * data['late_components'][m]['centered_between_source_error'] for m in models])
    assert np.allclose(bias + centered, [10000 * data['late_MSE'][m] for m in models], rtol=1e-12, atol=1e-12)
    ax.bar(xx, bias, .6, color='#2563eb', label='Squared overall signed bias')
    ax.bar(xx, centered, .6, bottom=bias, color='#f59e0b', label='Centered between-source error')
    ax.set(title='D. Original late MSE: descriptive identity', ylabel='MSE (percentage points squared)', xticks=xx, xticklabels=names, ylim=(0, 19))
    ax.legend(loc='upper left', fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Frozen early experience-value predictions at the fixed later learner state', fontsize=15, x=.5, y=.985)
    fig.text(.5, .015, 'Four sources × three training repeats × two evaluation seeds; shared tasks, policies and ancestor. No late refit or intercept correction.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96), h_pad=2.5, w_pad=2)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / ('gains_and_prediction_errors.' + ext) for ext in ['png', 'svg']]
    assert not any(path.exists() for path in paths), 'Refuse to overwrite published figures'
    for path in paths: fig.savefig(path, dpi=180, metadata={'Creator': 'Matplotlib; saved original results only'})
    plt.close(fig)
    receipt = {'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
               'source_report': data['source_report'], 'source_error_description': data['source_error_description'],
               'python': platform.python_version(), 'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
               'files': {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in paths},
               'new_scientific_tokens': 0, 'original_predictions_changed': False,
               'environment_scope': 'Local plotting dependencies; remote RL environment was not modified'}
    with (args.output / 'render_metadata.json').open('x') as handle: json.dump(receipt, handle, indent=2); handle.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
