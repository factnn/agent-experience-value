"""Sensitivity analysis over BFCL per-item results.

Answers the question the pilot depends on: with this many questions and this
evaluation budget, what is the smallest bin-to-bin difference we could detect?

  python pipeline/eval_sensitivity.py --results smoke/eval_bfcl/base_sweep_a.jsonl \
      --results smoke/eval_bfcl/base_sweep_b.jsonl --out smoke/sensitivity
"""
import argparse, collections, json, math, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parents[1]
Z_ALPHA = 1.959964   # 95% two-sided
Z_BETA = 0.841621    # 80% power


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z_ALPHA ** 2 / n
    centre = (p + Z_ALPHA ** 2 / (2 * n)) / d
    half = Z_ALPHA * math.sqrt(p * (1 - p) / n + Z_ALPHA ** 2 / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(flags, iters=5000, seed=20260929):
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def mde_unpaired(p, n, power_z=Z_BETA):
    """Smallest detectable difference between two independent runs of n questions."""
    if n == 0:
        return None
    return (Z_ALPHA + power_z) * math.sqrt(2 * p * (1 - p) / n)


def mde_paired(discordance, n, power_z=Z_BETA):
    """Smallest detectable difference for paired questions with the given discordance rate."""
    if n == 0 or discordance <= 0:
        return None
    return (Z_ALPHA + power_z) * math.sqrt(discordance / n)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', action='append', required=True)
    ap.add_argument('--out', default='smoke/sensitivity')
    ap.add_argument('--discordance', type=float, default=0.15,
                    help='assumed fraction of questions where two bins disagree')
    args = ap.parse_args()

    by_tag = {}
    for path in args.results:
        rows = load(path)
        tag = pathlib.Path(path).stem
        by_tag[tag] = rows

    report = {'inputs': args.results, 'tags': {}, 'mde': {}, 'cross_tag': []}
    for tag, rows in by_tag.items():
        per_cat = collections.defaultdict(list)
        for r in rows:
            per_cat[r['category']].append(int(r['valid']))
        entry = {}
        for cat, flags in per_cat.items():
            n, k = len(flags), sum(flags)
            lo, hi = wilson(k, n)
            blo, bhi = bootstrap_ci(flags)
            entry[cat] = {'n': n, 'correct': k, 'accuracy': k / n,
                          'wilson_95': [round(lo, 4), round(hi, 4)],
                          'bootstrap_95': [round(blo, 4), round(bhi, 4)],
                          'decode_failures': sum(1 for r in rows
                                                 if r['category'] == cat and not r['decodable'])}
        all_flags = [int(r['valid']) for r in rows]
        entry['_all'] = {'n': len(all_flags), 'correct': sum(all_flags),
                         'accuracy': sum(all_flags) / len(all_flags),
                         'wilson_95': [round(x, 4) for x in wilson(sum(all_flags), len(all_flags))]}
        report['tags'][tag] = entry

    # MDE at the observed overall accuracy for several evaluation-set sizes
    base_tag = list(by_tag)[0]
    p = report['tags'][base_tag]['_all']['accuracy']
    report['mde'] = {
        'baseline_accuracy': round(p, 4),
        'unpaired': {str(n): round(mde_unpaired(p, n), 4) for n in [20, 50, 100, 200, 400, 1000]},
        'paired': {str(n): round(mde_paired(args.discordance, n), 4) for n in [20, 50, 100, 200, 400, 1000]},
        'assumed_discordance': args.discordance,
        'note': 'Paired assumes every bin is evaluated on the same question set; the pilot design '
                'is paired, so the paired column is the relevant one once discordance is measured.',
    }

    # Budget sensitivity: same category+ids scored under different tags
    tags = list(by_tag)
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            index_a = {(r['category'], r['id']): int(r['valid']) for r in by_tag[a]}
            index_b = {(r['category'], r['id']): int(r['valid']) for r in by_tag[b]}
            shared = sorted(set(index_a) & set(index_b))
            if not shared:
                continue
            diff = sum(index_a[k] - index_b[k] for k in shared)
            flipped = sum(1 for k in shared if index_a[k] != index_b[k])
            report['cross_tag'].append({
                'a': a, 'b': b, 'shared_questions': len(shared),
                'accuracy_a': round(sum(index_a[k] for k in shared) / len(shared), 4),
                'accuracy_b': round(sum(index_b[k] for k in shared) / len(shared), 4),
                'accuracy_delta': round(diff / len(shared), 4),
                'discordant_questions': flipped,
                'observed_discordance': round(flipped / len(shared), 4),
            })

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
