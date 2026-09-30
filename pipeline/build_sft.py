"""Tokenize selected candidate trajectories into a loss-masked SFT tensor file.

CPU only; reads pinned parquet and never executes trajectory content.
Renders explicit ChatML that preserves every original content block (including
historic thoughts), and supervises assistant content only.

Usage:
  .venv-train/bin/python pipeline/build_sft.py --split train --limit 32 --out smoke/sft_smoke
"""
import argparse, collections, glob, json, pathlib, random, statistics
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'


def select_rows(split, limit, seed, randomize):
    ids = []
    for line in (ROOT / 'prepared/candidate_manifest.jsonl').open():
        m = json.loads(line)
        if m['provisional_split'] == split:
            ids.append(m['row_id'])
    if randomize:
        random.Random(seed).shuffle(ids)
    else:
        ids.sort()
    return set(ids[:limit])


def stream_rows(wanted):
    """Yield (row_id, conversations) in the exact order used by prepare_pool.py."""
    files = sorted((ROOT / 'data/SFT-100K').glob('*.parquet'))
    assert len(files) == 10, 'pinned pool incomplete'
    index = 0
    remaining = set(wanted)
    for file in files:
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=64, use_threads=False):
            for row in batch.to_pylist():
                if index in remaining:
                    remaining.discard(index)
                    yield index, row.get('conversations') or []
                    if not remaining:
                        return
                index += 1


def encode(tok, messages, supervise_im_end):
    """Explicit ChatML with an assistant-only loss mask. Returns (ids, labels, mismatches)."""
    ids, labels, mismatches = [], [], 0
    for m in messages:
        role, content = m['role'], m.get('content') or ''
        framing = f'{IM_START}{role}\n'
        f_ids = tok.encode(framing, add_special_tokens=False)
        c_ids = tok.encode(content, add_special_tokens=False)
        e_ids = tok.encode(f'{IM_END}\n', add_special_tokens=False)
        if tok.encode(framing + content, add_special_tokens=False) != f_ids + c_ids:
            mismatches += 1
        ids += f_ids + c_ids + e_ids
        if role == 'assistant':
            supervised = c_ids + (e_ids if supervise_im_end else [])
            labels += [-100] * len(f_ids) + supervised + [-100] * (len(e_ids) - len(supervised))
        else:
            labels += [-100] * (len(f_ids) + len(c_ids) + len(e_ids))
    return ids, labels, mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--random', action='store_true', help='seeded random instead of lowest row_id')
    ap.add_argument('--no-supervise-im-end', action='store_true')
    ap.add_argument('--tokenizer', default='smoke/model')
    ap.add_argument('--out', default='smoke/sft_smoke')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    wanted = select_rows(args.split, args.limit, args.seed, args.random)
    print(f'selected {len(wanted)} {args.split} candidate rows', flush=True)

    samples, dropped, mismatches, roles = [], [], 0, collections.Counter()
    for row_id, messages in stream_rows(wanted):
        if not messages:
            dropped.append({'row_id': row_id, 'reason': 'empty'})
            continue
        ids, labels, mm = encode(tok, messages, not args.no_supervise_im_end)
        mismatches += mm
        for m in messages:
            roles[m['role']] += 1
        if len(ids) > args.max_len:
            dropped.append({'row_id': row_id, 'reason': 'over_max_len', 'tokens': len(ids)})
            continue
        samples.append({'row_id': row_id, 'input_ids': torch.tensor(ids), 'labels': torch.tensor(labels)})

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out / f'{args.split}.pt')
    lens = sorted(len(s['input_ids']) for s in samples)
    sup = [int((s['labels'] != -100).sum()) for s in samples]
    report = {
        'split': args.split, 'tokenizer': args.tokenizer, 'max_len': args.max_len,
        'requested': len(wanted), 'kept': len(samples), 'dropped': dropped,
        'roles': dict(roles), 'framing_tokenization_mismatches': mismatches,
        'loss_mask': 'assistant content' + (' + <|im_end|>' if not args.no_supervise_im_end else ''),
        'chat_format': 'explicit ChatML preserving all original content including historic thoughts',
        'total_tokens': sum(lens), 'total_supervised_tokens': sum(sup),
        'supervised_fraction': (sum(sup) / sum(lens)) if lens else 0.0,
        'length_min': lens[0] if lens else None, 'length_median': statistics.median(lens) if lens else None,
        'length_max': lens[-1] if lens else None, 'supervised_median': statistics.median(sup) if sup else None,
        'note': 'Loss-mask and ChatML are engineering choices, not a validated training recipe. '
                'Over-length trajectories are dropped, not truncated; report the dropped set.',
    }
    (out / f'{args.split}_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'dropped'}, ensure_ascii=False, indent=2), flush=True)
    if dropped:
        print('dropped:', len(dropped), dropped[:5], flush=True)


if __name__ == '__main__':
    main()
