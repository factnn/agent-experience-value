"""Paired comparison of two BFCL runs on the same question set.

Reports accuracy, decodability and P(correct | decodable) separately, because
terminal-agent SFT drifts the output dialect (see 09_evaluation_sensitivity.md 2.8)
and an accuracy-only comparison cannot tell capability change from format change.

  python pipeline/compare_runs.py --a smoke/eval_bfcl/java_js_b3072.jsonl \
      --b smoke/eval_bfcl/java_js_trained.jsonl --out smoke/compare
"""
import argparse, collections, json, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {(r['category'], r['id']): r for r in rows}


def paired_ci(deltas, iters=5000, seed=20260930):
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='baseline per-item jsonl')
    ap.add_argument('--b', required=True, help='comparison per-item jsonl')
    ap.add_argument('--out', default='smoke/compare')
    args = ap.parse_args()

    a, b = load(ROOT / args.a), load(ROOT / args.b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit('no shared (category,id) pairs')

    per_category = collections.defaultdict(list)
    for key in shared:
        per_category[key[0]].append(key)

    report = {'a': args.a, 'b': args.b, 'shared_questions': len(shared), 'categories': {}}
    for category, keys in sorted(per_category.items()):
        n = len(keys)
        acc_a = sum(int(a[k]['valid']) for k in keys) / n
        acc_b = sum(int(b[k]['valid']) for k in keys) / n
        dec_a = sum(int(a[k]['decodable']) for k in keys) / n
        dec_b = sum(int(b[k]['decodable']) for k in keys) / n
        # P(correct | decodable): denominator is decodable responses only
        cond_a = [int(a[k]['valid']) for k in keys if a[k]['decodable']]
        cond_b = [int(b[k]['valid']) for k in keys if b[k]['decodable']]
        pcond_a = (sum(cond_a) / len(cond_a)) if cond_a else None
        pcond_b = (sum(cond_b) / len(cond_b)) if cond_b else None
        deltas = [int(b[k]['valid']) - int(a[k]['valid']) for k in keys]
        lo, hi = paired_ci(deltas)
        report['categories'][category] = {
            'n': n,
            'accuracy_a': round(acc_a, 4), 'accuracy_b': round(acc_b, 4),
            'delta_accuracy': round(acc_b - acc_a, 4),
            'delta_accuracy_ci95': [round(lo, 4), round(hi, 4)],
            'decodable_a': round(dec_a, 4), 'decodable_b': round(dec_b, 4),
            'delta_decodable': round(dec_b - dec_a, 4),
            'p_correct_given_decodable_a': None if pcond_a is None else round(pcond_a, 4),
            'p_correct_given_decodable_b': None if pcond_b is None else round(pcond_b, 4),
            'delta_p_correct_given_decodable': (None if pcond_a is None or pcond_b is None
                                                else round(pcond_b - pcond_a, 4)),
            'discordant': sum(1 for d in deltas if d != 0),
            'flips_up': sum(1 for d in deltas if d > 0),
            'flips_down': sum(1 for d in deltas if d < 0),
        }

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'compare.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
