"""Minimal single-GPU SFT smoke run over loss-masked agent trajectories.

Purpose: prove the train loop works at 32K context and measure real cost
(step time, tokens/s, peak memory). Not a final training recipe.

Usage:
  CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py \
      --data smoke/sft_smoke/train.pt --out smoke/run_lora --mode lora --max-steps 8
  CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py \
      --probe --probe-lengths 8192,16384,32768
"""
import argparse, json, pathlib, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def load_samples(path):
    samples = torch.load(path, weights_only=False)
    assert samples, f'no samples in {path}'
    return samples


def collate(batch):
    return batch[0]


def backbone_and_head(model):
    """Unwrap peft if present; return (backbone, lm_head)."""
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return base.model, base.lm_head


def forward_loss(model, input_ids, labels, loss_chunk, backbone, head):
    """Causal-LM loss. loss_chunk>0 computes logits in position chunks.

    Materializing all logits costs seq_len * vocab bf16 plus a float32 upcast
    inside cross_entropy; at 32K that alone is ~10 GB bf16 / ~20 GB fp32.
    Chunking keeps the same value while bounding that term.
    """
    hidden = backbone(input_ids=input_ids).last_hidden_state
    shifted_labels = labels[:, 1:]
    if loss_chunk <= 0:
        logits = head(hidden[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               shifted_labels.reshape(-1), ignore_index=-100)
    hidden = hidden[:, :-1]
    total, count = 0.0, 0
    for start in range(0, hidden.shape[1], loss_chunk):
        h = hidden[:, start:start + loss_chunk]
        lab = shifted_labels[:, start:start + loss_chunk]
        n = int((lab != -100).sum())
        if n == 0:
            continue
        logits = head(h)
        total = total + F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                        lab.reshape(-1), ignore_index=-100, reduction='sum')
        count += n
    if count == 0:
        raise ValueError('no supervised tokens in batch')
    return total / count


def build_model(args):
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / args.model), dtype=torch.bfloat16)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to('cuda')
    if args.mode == 'lora':
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                         target_modules=TARGET_MODULES, task_type='CAUSAL_LM', bias='none')
        model = get_peft_model(model, cfg)
    return model


def memory_probe(args):
    """Forward+backward memory at several sequence lengths, no optimizer step."""
    model = build_model(args)
    backbone, head = backbone_and_head(model)
    model.train()
    rows = []
    for length in [int(x) for x in args.probe_lengths.split(',')]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, 1000, (1, length), device='cuda')
        labels = ids.clone()
        t0 = time.time()
        try:
            loss = forward_loss(model, ids, labels, args.loss_chunk, backbone, head)
            loss.backward()
            torch.cuda.synchronize()
            rows.append({'seq_len': length, 'ok': True, 'seconds': round(time.time() - t0, 2),
                         'loss': round(float(loss), 4),
                         'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2**30, 2)})
        except torch.cuda.OutOfMemoryError as e:
            rows.append({'seq_len': length, 'ok': False, 'error': 'CUDA OOM', 'detail': str(e)[:200]})
        model.zero_grad(set_to_none=True)
        del ids, labels
        print(rows[-1], flush=True)
    return {'mode': args.mode, 'gradient_checkpointing': args.gradient_checkpointing,
            'loss_chunk': args.loss_chunk, 'probe': rows}


def train(args):
    torch.manual_seed(args.seed)
    samples = load_samples(ROOT / args.data)
    model = build_model(args)
    backbone, head = backbone_and_head(model)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    loader = DataLoader(samples, batch_size=1, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'train_log.jsonl').open('w')
    model.train()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    step, micro, tokens_seen, started = 0, 0, 0, time.time()
    losses = []
    done = False
    while not done:
        for sample in loader:
            ids = sample['input_ids'].unsqueeze(0).to('cuda')
            labels = sample['labels'].unsqueeze(0).to('cuda')
            torch.cuda.synchronize(); t0 = time.time()
            loss = forward_loss(model, ids, labels, args.loss_chunk, backbone, head) / args.grad_accum
            loss.backward()
            micro += 1
            tokens_seen += int((labels != -100).sum())
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                step += 1
                torch.cuda.synchronize()
                record = {'step': step, 'loss': round(float(loss) * args.grad_accum, 4),
                          'step_seconds': round(time.time() - t0, 2),
                          'supervised_tokens': tokens_seen,
                          'supervised_tokens_per_s': round(tokens_seen / (time.time() - started), 1),
                          'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                          'row_id': sample['row_id'], 'seq_len': int(ids.shape[1])}
                losses.append(record['loss'])
                log.write(json.dumps(record) + '\n'); log.flush()
                print(json.dumps(record), flush=True)
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
            del ids, labels
        if not args.max_steps:
            done = True

    if args.mode == 'lora':
        model.save_pretrained(out / 'adapter')
    else:
        model.save_pretrained(out / 'model')
    summary = {'mode': args.mode, 'lora_r': args.lora_r if args.mode == 'lora' else None,
               'gradient_checkpointing': args.gradient_checkpointing,
               'trainable_params': trainable, 'total_params': total,
               'trainable_fraction': trainable / total,
               'steps': step, 'lr': args.lr, 'grad_accum': args.grad_accum,
               'samples': len(samples), 'supervised_tokens_seen': tokens_seen,
               'wall_seconds': round(time.time() - started, 1),
               'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
               'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 2**30, 2),
               'loss_first': losses[0] if losses else None,
               'loss_last': losses[-1] if losses else None,
               'gpu': torch.cuda.get_device_name(0)}
    (out / 'train_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='smoke/model')
    ap.add_argument('--data', default='smoke/sft_smoke/train.pt')
    ap.add_argument('--out', default='smoke/run_lora')
    ap.add_argument('--mode', choices=['lora', 'full'], default='lora')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--max-steps', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--gradient-checkpointing', action='store_true', default=True)
    ap.add_argument('--no-gradient-checkpointing', dest='gradient_checkpointing', action='store_false')
    ap.add_argument('--loss-chunk', type=int, default=512,
                    help='positions per lm_head chunk; 0 materializes all logits (memory heavy)')
    ap.add_argument('--probe', action='store_true', help='memory probe only, no optimizer step')
    ap.add_argument('--probe-lengths', default='8192,16384,32768')
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    report = memory_probe(args) if args.probe else train(args)
    if args.probe:
        out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
        (out / 'memory_probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
