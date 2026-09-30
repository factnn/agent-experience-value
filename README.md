# Agent Experience Value

**Which properties of interactive agent experience predict post-training transfer to unseen
tasks, tools and environments?**

This is a *measurement* project, not a method paper. Existing agent-training work defines
valuable experience through different local signals — success, diversity, information gain,
recovery structure, learner compatibility, uncertainty — but those signals are rarely
compared against the same causal target: how much transferable capability the agent actually
gains after training on that experience.

We define that target as

```
V_transfer(S ; π) = M_OOD( Train(π, S) ) − M_OOD( π )
```

and ask which experience properties predict it, under one protocol: same base checkpoint,
same token budget, same optimizer, same held-out evaluation.

## Status — please read before judging the contents

**There is no capability result in this repository yet.** Nothing here shows that any
experience property predicts transfer, because that experiment has not been run.

What does exist:

| Area | State |
|---|---|
| Data reconnaissance over four public pools | done, with pinned revisions and hashes |
| Full-pool scan of the primary pool (94,334 rows) | done |
| Engineering candidate manifest (71,255 trajectories) | done, `outcome=unknown` for every row |
| Verifier / success-label recovery | **blocked** — released data carries no usable success label |
| Tokenizer + loss-mask + token-budget measurement | done |
| SFT training loop (single GPU, 32K context) | built and validated end to end |
| BFCL evaluation (official prompt/parser/checker, single- and multi-turn) | built and validated |
| Evaluation sensitivity study | done — see below |
| Four-bin token-matched intervention pilot | **not started** (gated on labels and on the sensitivity result) |

The one genuinely decision-relevant result so far is negative in a useful way, and is
explained under [Evaluation sensitivity](#evaluation-sensitivity-two-results-that-change-the-pilot-design).
Whether the project should be invested in at all is a separate question, answered in
`10_investment_verdict.md`: the bottleneck is not compute, pipeline or evaluation — it is
that **no verifier gives oracle outcomes for the training pool's own tasks**, which makes both
the success hypothesis and the local-to-transfer gap unmeasurable. TaskTrove contains
executable, verifier-carrying tasks for 3 of the pool's 4 sources (66% of candidates), and a
chroot-based harness was shown to run the task verifiers on this machine despite the absence
of any container runtime — but the trajectory-to-task join is still unverified.

## Hypotheses under test

These are treated as competing hypotheses to be measured, not as territories to avoid:

| | Signal | Line of work it comes from |
|---|---|---|
| H0 | random token-matched sample | baseline |
| H1 | success / outcome | outcome filtering |
| H2 | structural diversity | diversity scaling for tool use |
| H3 | recovery / interaction topology | topology-aware curation |
| H4 | information / PMI per action | information-based credit assignment |
| H5 | learner compatibility (student NLL) | capacity-aligned distillation |
| H6 | learnability / capability frontier | prioritized level replay, curricula |
| H7 | gradient influence | influence-function data selection |

Successful outcomes are of four kinds, any of which can carry the paper: a local-to-transfer
gap, one proxy that robustly predicts transfer, a rank reversal across scale or training
stage, or small-model rankings that predict a larger model's best data. Kill criteria and
continue criteria are written down in `04_decision_log_and_kill_criteria.md`.

## Evaluation sensitivity: two results that change the pilot design

Full detail, method and raw numbers: **`09_evaluation_sensitivity.md`**.

**1. Evaluation configuration is a bigger effect than anything we intend to measure.**
On the same 120 questions, changing only the generation budget:

| budget | accuracy | decodable | questions hitting the cap |
|---:|---:|---:|---:|
| 256 | 0.758 | 0.800 | 25 |
| 640 | 0.917 | 0.967 | 4 |
| 1024 | 0.950 | 1.000 | 0 |

That is a **19.2 point** swing from a harness setting. The effect replicates on a second
axis (simple_java, budget 640 → 1536: 0.510 → 0.650). By contrast, re-running the identical
configuration on a different physical GPU reproduces the outputs **byte for byte** — the
evaluation itself is deterministic, so configuration, not noise, is the thing to control.

**2. The obvious axes are saturated, so they cannot detect transfer.**

Base Qwen3-4B, official BFCL prompt/parser/checker:

| Axis | n | base accuracy | decodable | usable? |
|---|---:|---:|---:|---|
| simple_python | 400 | 0.932 | 0.968 | no — ceiling |
| multiple | 200 | 0.950 | 0.975 | no — ceiling |
| irrelevance | 240 | 0.887 | 1.000 | marginal |
| parallel_multiple | 200 | 0.865 | 0.905 | secondary |
| parallel | 200 | 0.860 | 0.910 | secondary |
| simple_java | 100 | 0.650 | 0.920 | yes |
| simple_javascript | 50 | 0.640 | 0.880 | yes |
| multi_turn_base | 16 | 0.250 | see note (2) | **primary candidate** |

`multi_turn_base` is the only axis the base model clearly cannot do (0.250 on 16 items; stable
across 2048 → 4096). Its failures are mixed rather than uniform: of 12 failures, 4 are
`instance_state_mismatch`, 5 are `force_terminated` (the model retries inside one turn until it is
cut off), 2 are `empty_turn_model_response` and 1 is `execution_response_mismatch`. Those mean
different things for experience value and must be reported separately. It remains the closest match
to the "stateful, long-horizon, unseen environment" transfer target.

**Correction (2026-09-30) — this is note (2) in the table above.** An earlier version of this README
and of `09_evaluation_sensitivity.md` claimed multi-turn decodability was 100%, and used that to
argue the axis measures capability rather than format. **That claim is withdrawn.** The 100% came
from `score_multi_turn()` hardcoding `True` in both the success and the failure branch, so the
metric was a tautology rather than a measurement.

Re-measured over all 16 items by classifying every model response through the official decoder:
**221 responses, 180 executable calls, 38 parse failures, 3 legitimate non-calls → decodable rate
0.814, not 1.000.** Accuracy reproduced exactly at 4/16 = 0.250 across three independent runs, so
the inference path is unchanged and deterministic. Only 1 of the 16 items had every response
decodable, and that item was still wrong; per-item decodability (0.50-1.00) does not track
correctness. So format failure is real (17.2% of responses) but is **not** the dominant failure
mode - task-level failures remain mostly `force_terminated` and `instance_state_mismatch`.
Note the units differ: multi-turn decodability is per-response, single-turn is per-item.

The same fix stops truncating each trajectory at 4000 characters: full per-item payloads (raw
responses, parse samples, execution results, stop reason) now go to `<tag>_full/<id>.json`.

With paired evaluation on a shared question set, the minimum detectable effect is
7.7 points at n=200 and 5.4 points at n=400 — which is why the pilot must freeze its
evaluation configuration before comparing anything.

## Repository map

| Path | Contents |
|---|---|
| `01_scope.md` | Paper story, the measurement target, allowed and disallowed claims, scope freeze. |
| `02_literature_review.md` | Related work organised as competing definitions of value, plus a claim-occupancy test. |
| `03_experiment_plan.md` | Interventions, benchmarks, main table/figures, statistics, pilot design. |
| `04_decision_log_and_kill_criteria.md` | Why this problem, kill criteria K1–K5, continue criteria C1–C5. |
| `05_data_pool_audit.md` | First field-level audit of four public pools (400 rows over the HF viewer). |
| `06_pilot_protocol.md` | Pilot protocol revised against the real schema. |
| `07_recovery_annotation_guide.md` | Event-level recovery annotation rubric. |
| `08_full_pool_preparation.md` | Pinned full-pool scan, label hunt, token measurements. |
| `09_evaluation_sensitivity.md` | Noise, configuration bias, MDE, per-axis headroom, cost. |
| `10_investment_verdict.md` | **Investment decision record**: what is proven, what is fatal, platform constraints, and the bounded experiment that decides go/no-go. |
| `audit/` | Data-pool reconnaissance scripts and findings. |
| `pipeline/` | Download, scan, review-page, tokenize, train, evaluate, analyse. |
| `prepared/` | Scan summaries and the 299-trajectory HTML review reader. |
| `smoke/` | Training and evaluation smoke results, including measured memory and cost. |

## Findings so far

**The released pool has no usable success label.** `OpenThoughts-Agent-SFT-100K` (94,334
rows at pinned revision `45fb28f...`) has no `reward` or `verifier_output` column; its
`result` field holds runtime exceptions (`AgentTimeoutError` 32,764, `ContextLengthExceededError`
916, empty 60,296) and an empty value does not mean the verifier passed. Upstream code shows
where real labels live — `verifier_result.rewards` inside trial artifacts — but 11 public
candidates were probed and no trial-level artifact was found (3 returned 401, cause
undetermined; recorded, not assumed private). See `05_data_pool_audit.md`, `08_full_pool_preparation.md`.

**Pool structure is known.** 94,334 conversations, 75,879 main-branch, 1,187,572 assistant
turns of which 98.675% yield a parseable command JSON. After excluding non-main and
unparseable rows, 71,255 trajectories remain, grouped into a provisional 64,062/7,193
train/dev split with no task-family, instruction-hash, conversation-hash or run+trial key
crossing the boundary. Median trajectory length is 13.7K tokens (random 200), and 38% of
tokens are supervised assistant content.

**Training is capacity-limited, and we measured where.** On one 40 GB A100, full-parameter
SFT at 32K context does not fit at any length: forward+backward alone peaks at 34.1 GB, and
AdamW's fp32 state needs another ~32 GB. Materialising the lm_head logits is itself 10 GB at
32K, so the loss must be computed in position chunks (16K and 32K both OOM without it). A
LoRA run at 32K peaks at 32.9 GB and trains at ~787 supervised tokens/s.
See `smoke/README.md`.

**The evaluation harness is official, not a local approximation.** The `bfcl_eval` wheel is
unpacked and imported rather than vendored, and its single- and multi-turn inference loops
run locally against the official mock APIs — no Docker, no external service, no API cost.
Only the generation call is replaced (`pipeline/eval_bfcl_local.py`).

**Terminal-agent SFT causes function-call dialect drift.** After 16 LoRA steps on
terminus-2 terminal trajectories, a small BFCL regression (0.95 → 0.80 on 20 items) turned
out to be mostly *format*, not capability: three of the four failures are the harness dialect
leaking into BFCL, e.g. `[func_name=solve_quadratic, params={"a": 2, "b": 5, "c": 3}]` instead
of `[solve_quadratic(a=2, b=5, c=3)]`. Any BFCL comparison must therefore be decomposed into
**decodability** and **P(correct | decodable)**, or a bin-to-bin gap could be pure dialect
drift. See `09_evaluation_sensitivity.md` §2.8.

## Reproducing

```bash
# 0. environment (torch + transformers + peft; see pipeline/runtime_versions_train.json)
python -m venv --system-site-packages .venv-train
#    install: torch, torchvision (match your CUDA), transformers, tokenizers,
#             huggingface_hub, safetensors, peft, accelerate

# 1. third-party assets this repo does not commit
.venv-train/bin/python pipeline/fetch_assets.py

# 2. pinned dataset shards (~1.7 GB) + full-pool scan + review pages (CPU only)
python pipeline/download_pool.py
python pipeline/prepare_pool.py
python pipeline/build_review.py

# 3. tokenizer measurements and tests
.venv-audit/bin/python pipeline/tokenize_review.py
.venv-train/bin/python -m unittest discover -s pipeline -p 'test_*.py'

# 4. training smoke run on one explicitly chosen idle GPU
.venv-train/bin/python pipeline/build_sft.py --split train --limit 32 --out smoke/sft_smoke
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py \
    --probe --mode lora --probe-lengths 8192,16384,32768 --out smoke/probe_lora
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py \
    --data smoke/sft_smoke/train.pt --out smoke/run_lora --mode lora --max-steps 16

# 5. evaluation, and the sensitivity analysis
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl_local.py \
    --category multi_turn_base --limit 20 --tag base_mt20
python pipeline/eval_sensitivity.py --results smoke/eval_bfcl/base_sweep_a.jsonl
```

`pipeline/README.md` documents the scripts in order; `prepared/README.md` and
`audit/README.md` state exactly which artifacts are committed and which are regenerated.

## What is deliberately not in this repository

| Excluded | Size | Why / how to get it |
|---|---:|---|
| `data/` pinned parquet shards | 1.7 GB | public dataset; `pipeline/download_pool.py` verifies every shard against its LFS SHA256 |
| `smoke/model/` Qwen3-4B weights | 7.6 GB | public model; `pipeline/fetch_assets.py` |
| `.third_party/bfcl_eval_pkg/` | 14 MB | official BFCL package; `pipeline/fetch_assets.py` downloads the pinned wheel |
| `prepared/features.jsonl`, manifests, annotation packets | 147 MB | derived; regenerate with `pipeline/prepare_pool.py`, `build_review.py` |
| `smoke/run_lora/adapter/*.safetensors` | 127 MB | 3 minutes of training; `pipeline/train_sft.py` |
| raw audit row dumps | 20 MB | public dataset rows; `audit/audit_pool.py` |

## Open problems

1. **Success labels.** The four-bin pilot as designed needs an oracle outcome per trajectory.
   Either trial-level verifier artifacts are found, or rollouts are regenerated against
   executable tasks with local verifiers. Until then only Random / Diversity / Recovery can
   be trained, with outcomes left as `unknown`.
2. **Training-seed variance is unmeasured.** This is the other half of the noise budget and
   it directly sets the discordance rate that the MDE calculation depends on. Measuring it
   should come before adding more bins.
3. **A second transfer axis.** BFCL is a cheap diagnostic; the scope calls for unseen
   tools and stateful environments. Whether that axis can be built without a container
   runtime on the target machine is unresolved.

## License and attribution

This repository is Apache-2.0 (`LICENSE`). It builds on OpenThoughts-Agent and on the
Berkeley Function Calling Leaderboard (gorilla), both Apache-2.0. See `NOTICE` for exactly
what is redistributed, what is fetched, and how derived text is attributed.
