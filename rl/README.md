# Online RL engineering progress

2026-09-30: environment acceptance passed; GPU RL update not yet run.

- Research direction: [PROJECT_STATE.md](../PROJECT_STATE.md).
- Environment: pinned BFCL MessageAPI implementation, **independent synthetic engineering tasks**, no BFCL question/answer files loaded. Shared simulator means this is not independent OOD evidence.
- Three task types: send to existing contact, add a contact then send, replace an old message. Twelve deterministic task instances, not a research benchmark.
- Terminal reward: exact relevant state, correct logged-in identity, preservation of unrelated contacts/messages, no duplicate sends. No reward for textual claims. No model judge or paid service.
- CPU acceptance: `python pipeline/rl_environment.py`; evidence: [environment_acceptance.json](environment_acceptance.json). Checks oracle success, reset, instance isolation, invalid/duplicate actions, wrong identity, collateral deletion and tool limit.
- Trainer integration underway: TRL 0.29.0 GRPO with its official `environment_factory` and environment-feedback loss masking. Transformers 5.2.0 required by that API. Isolated `.venv-rl` leaves SFT environment intact.
- Reference: [pinned TRL agent-training documentation](https://huggingface.co/docs/trl/v0.29.0/en/grpo_trainer#agent-training).
- Next: bounded single-GPU online rollout/update/resample; record state/action/feedback, policy version, parameter changes, reward-group variance, tokens and timing. No allocation comparison yet.

## Reproduce the bounded run

Use a separate Python 3.10+ venv with `pipeline/requirements-rl.txt`; install the CUDA 12.1 build of torch 2.5.1 for this host. Locally `.venv-rl` reuses only `.venv-train/lib/python3.10/site-packages` through a `.pth` file (read-only); **do not inherit the system site-packages**, whose old vLLM/Transformer Engine conflict with the new stack. No system packages were changed.

After checking a GPU is free:

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 .venv-rl/bin/python -u pipeline/train_rl_smoke.py \
  --out rl/smoke_001 --steps 4 --max-completion-length 768
```

The GPU number is an example and must be rechecked. Four completions per task group, one gradient update per fresh group, LoRA rank 8 on q/v projections, binary state reward, no KL term, original `grpo` loss normalization. This is an engineering configuration, not the frozen allocation protocol. Token budget includes feedback in the upstream rollout length limit; `model_token_mask` distinguishes generated vs environment tokens. No old trajectory replay and no SFT initialization. An explicit final-policy resample follows training. A zero-variance reward group cannot establish an effective RL update.

Artifacts: `run.json`, versions/config/tasks, full token IDs and masks in `rollouts.jsonl`, state/action/feedback events, advantages in `groups.jsonl`, parameter fingerprints in `updates.jsonl`, training metrics and `summary.json`. Adapter weights stay local and are ignored by git. Single-GPU wall reservation includes initialization and saving; generation timing includes environment work; training-compute timing includes forward/backward and scoring, not optimizer overhead. Do not sum overlapping components as independent totals.

Compatibility note: TRL 0.29.0 imports `torch.distributed.fsdp.FSDPModule` unconditionally. Under torch 2.5 the runner aliases the **real** class from `torch.distributed._composable.fsdp` to that public name before importing TRL. No installed library is patched, and the runner asserts single-process / FSDP disabled. This path is for the single-GPU engineering run only.

## First attempt: no learning signal (smoke_001)

Four optimizer steps completed in 136.8 s including initialization and final resampling. All 16 training trajectories and 4 final resamples failed; all groups had zero reward variance, gradients were zero and parameter fingerprints unchanged. This **does not pass** the effective-update acceptance gate. Peak allocated 9.60 GiB, reserved 11.26 GiB. Generation/environment accounted for 92.1 s of training; forward/backward/scoring 5.94 s. These are engineering costs, not an estimate for a research-scale RL run.

A concrete integration bug was found: TRL passes `padding_side='left'` to `apply_chat_template`, but Transformers 5.2 treats that as a template kwarg, not a tokenizer kwarg. The tokenizer defaults to right padding, corrupting variable-length batched continuations. The runner now explicitly sets `tokenizer.padding_side='left'`. The second attempt keeps tasks/rewards/algorithm unchanged to check this correction. First-attempt logs are retained. Its token totals count retained trajectories only, not suffixes discarded by the tool loop; the corrected runner records every generation call to charge discarded tokens too.
