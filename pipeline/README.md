# Data preparation pipeline

Run from the project directory. These commands only use CPU and network; no model weights are loaded, no task commands are executed.

```bash
python pipeline/download_pool.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pipeline/prepare_pool.py
python pipeline/build_review.py
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 .venv-audit/bin/python pipeline/tokenize_review.py
python -m unittest discover -s pipeline -p 'test_*.py'
```

Third-party assets that are not committed:

```bash
.venv-train/bin/python pipeline/fetch_assets.py          # Qwen3-4B weights + official bfcl_eval
```

Order matters: download all 10 verified shards before scanning; finish scanning before building review pages or measuring annotation-sample token lengths. The raw files total about 1.75 GB. prepare_pool uses one CPU process and Arrow batches of 64, not all available CPU cores.

## Training and evaluation (GPU, smoke stage)

These load Qwen3-4B weights and require `.venv-train` plus an explicitly chosen idle GPU. See [../smoke/README.md](../smoke/README.md) for measured memory, cost, and the current limits.

```bash
# tokenize + assistant-only loss mask (CPU)
.venv-train/bin/python pipeline/build_sft.py --split train --limit 32 --out smoke/sft_smoke
# memory probe only, no optimizer step
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py --probe --mode lora --probe-lengths 8192,16384,32768
# LoRA SFT
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py --data smoke/sft_smoke/train.pt --out smoke/run_lora --mode lora --max-steps 16
# BFCL simple_python, base or adapter (local generation, official scoring)
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl.py --limit 20 --tag base20_repeat
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl.py --limit 20 --tag lora16_official --adapter smoke/run_lora/adapter
# multi-turn categories need the official inference loop (executes BFCL's mock APIs)
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl_local.py --category multi_turn_base --limit 16 --tag base_mt16
# sensitivity analysis over any set of per-item result files
python pipeline/eval_sensitivity.py --results smoke/eval_bfcl/base_sweep_a.jsonl
```

`train_sft.py` chunks the lm_head logits; without `--loss-chunk` the 16K and 32K settings run out of memory on a 40 GB card. `eval_bfcl.py` scores with a local re-implementation of BFCL's PYTHON simple-function checker, not the official runner.

`download_pool.py` pins the existing audit metadata SHA and verifies every shard against HF LFS SHA256. `prepare_pool.py` regenerates features and sampling packets, so preserve a copy before manually editing annotations. Keep actual annotations in a separate file keyed by pinned dataset/row/hash; the initial packets deliberately mark rows unreviewed.

`build_review.py` writes a local HTML reader (prepared/review/index.html), escapes trace text, and never executes it. Provisional train/dev manifests group rows connected by exact instructions, task-family hints, conversations or run/trial. It is not a final OOD split and does not certify repository or near-duplicate isolation.

`probe_outcomes.py` separately probes public upstream schemas and file lists; it resolves renamed datasets and records 401/unavailable responses without assuming they are private. Absence in these probes does not prove absence everywhere. It never uses private registry credentials.

Tokenizer environment: `.venv-audit` inherits existing system packages; only the tokenizer stack is overlaid locally. Global transformers/tokenizers were not modified. Exact observed versions are in runtime_versions.json; requirements-audit.txt pins the overlay. The inherited development pyarrow build is recorded, not advertised as an installable public wheel. Qwen3-4B tokenizer is pinned to audit/tokenizer_model_metadata.json. Primary token counts use explicit ChatML preserving all original thoughts; a second count uses the official template, which removes earlier thoughts. Assistant-content counts are diagnostics, not a training loss mask.

Current schema does not supply trustworthy terminal outcomes. All candidate records retain outcome=unknown and training_ready=false. Runtime exception, self-reported completion, successful local test, and independently verified terminal success are distinct.

User resource policy: future GPU jobs may use at most four currently idle GPUs; fewer are preferred when sufficient. Recheck occupancy immediately before launch, explicitly set CUDA_VISIBLE_DEVICES, and do not disturb other users' processes. No GPU allocation is needed for this pipeline.
