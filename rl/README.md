2026-10-08 latest: real Qwen3/GRPO checkpoint continuation passed with exactly matching fresh token sequences and final parameters ([result](../19_qwen3_resume_acceptance.md)). BFCL multi-user generation and external-token loss masking also passed actual-model auditing: 8 complete conversations, 0/8 terminal successes, zero gradients and unchanged weights ([result and diagnostics](../20_bfcl_model_integration_result.md)). GPU 4 released. Execution/reward calibration is the next bottleneck; no scientific allocation or transfer result.

Earlier component record:

2026-10-08 CPU follow-up: common-state component restoration passed 5 checks (including exact stochastic AdamW continuation); BFCL user-turn controller passed 3 checks over all 22 development oracle conversations. Full Qwen3/TRL fork and generation/mask integration remain. No new GPUs used. See [acceptance and next bounded check](../18_learning_state_and_conversation_acceptance.md).

2026-10-08 completed: both allocation arms passed final audits, GPUs released. Uniform: 123,163 training tokens, 10 updates, 9/12 new-instance successes. Frontier: 125,791 tokens, 11 updates, 10/12. First four post-warmup task choices were identical; post-warmup policy states differed. This pilot validates the engineering chain, not an allocation effect. See [full result](../17_allocation_pilot_result.md) and [next scientific design](../16_benchmark_and_fork_design.md).

Earlier launch record:

2026-10-08 ongoing: uniform/frontier online allocation pilot launched on GPUs 4/5. Both first groups produced rewards `[1,1,0,0]`, nonzero gradients (~0.0262), and changed parameters. Each charged 15,863 raw generated tokens. Still in common warmup; allocation-effect results pending. See [pilot protocol](ALLOCATION_PILOT_PROTOCOL.md), [first-update audit](allocation_pilot_20261008_first_update_audit.json), and [live status](STATUS.json). A CPU watcher pushes completed-update milestones and performs final audits.

# Online RL engineering progress

2026-10-08: all 72 fixed-policy development calibration episodes completed; no optimizer updates. Protocol/4K achieved 14/24 successes with 7/24 cap hits; all panels failed the declared cap screen. See [calibration report](../14_rl_calibration_result.md) and [comparison](calibration_20261008_comparison.json). Allocation code is tested but has not trained a model. GPUs released.

2026-09-30: online multi-turn RL engineering loop **verified**; full four-step reasoning-mode run and final-policy resampling completed, audit passed, GPUs released. No allocation-effect or transfer result yet. Latest machine-readable state: [STATUS.json](STATUS.json).

- Research direction: [PROJECT_STATE.md](../PROJECT_STATE.md).
- Environment: pinned BFCL MessageAPI implementation, **independent synthetic engineering tasks**, no BFCL question/answer files loaded. Shared simulator means this is not independent OOD evidence.
- Three task types: send to existing contact, add a contact then send, replace an old message. Twelve deterministic task instances, not a research benchmark.
- Terminal reward: exact relevant state, correct logged-in identity, preservation of unrelated contacts/messages, no duplicate sends. No reward for textual claims. No model judge or paid service.
- CPU acceptance: `python pipeline/rl_environment.py`; evidence: [environment_acceptance.json](environment_acceptance.json). Checks oracle success, reset, instance isolation, invalid/duplicate actions, wrong identity, collateral deletion and tool limit.
- Trainer integration underway: TRL 0.29.0 GRPO with its official `environment_factory` and environment-feedback loss masking. Transformers 5.2.0 required by that API. Isolated `.venv-rl` leaves SFT environment intact.
- Reference: [pinned TRL agent-training documentation](https://huggingface.co/docs/trl/v0.29.0/en/grpo_trainer#agent-training).
- Next: freeze a bounded allocation-pilot contract and connect the tested allocator to online training. No allocation comparison yet.

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

## Second attempt: padding fixed, still zero signal (smoke_002)

The same four groups still have zero reward variance and unchanged parameter hashes. Total wall 145.5 s; peak allocated 9.60 GiB. This rules out padding correction alone as sufficient to get learning signal. Do not attribute the first attempt's entire failure to that bug. A same-input old/new runtime diagnostic follows before changing task difficulty or reasoning settings.

CPU regression checks passed: one real-tokenizer batch-padding test; three tests of the **official TRL loss** verify zero feedback/padding gradients, correct reward-sign gradients, invariance to feedback log-probabilities, and zero gradients for zero advantages. Commands: `.venv-rl/bin/python -m unittest discover -s pipeline -p 'test_rl_*.py'`. Environment acceptance is separate (`python pipeline/rl_environment.py`). `.venv-rl/bin/pip check` also passes.

Runtime parity: `runtime_compare_transformers457.json` and `runtime_compare_transformers520.json` contain identical greedy token sequences and identical first-token entropy (0.0012726) on the same saved prompt. This is one-input evidence, not a global equivalence guarantee. Third attempt enables thinking and raises the completion cap to 2048; tasks and rewards remain identical. It is calibration, not a controlled research comparison of allocation policies.

## Third attempt: first effective update observed (smoke_003, in progress)

First group rewards `[0, 1, 0, 0]`, advantages approximately `[-0.5, 1.5, -0.5, -0.5]`, gradient norm `0.04073`; trainable-parameter fingerprint changed after step 1. The successful trajectory initially used incorrect IDs, received failure feedback, then logged in with the looked-up ID and sent exactly one correct message. Reward is based on actual state and execution identity. This is a verified recovery example, not evidence that recovery-focused allocation improves transfer. The near-zero scalar loss is expected for centered advantages at an on-policy update; nonzero gradients and changed weights are the effective-update evidence. Full-run/final-resample audit pending.


## Completed acceptance: smoke_003

The final [audit_report.json](smoke_003/audit_report.json) passes policy-version provenance, group/trajectory reward consistency, feedback masks, and retained-token equality against actual generation histories. [summary.json](smoke_003/summary.json) records runtime. Console logs and full trajectories for all three attempts are retained.

| Item | Observed |
|---|---:|
| Optimizer steps | 4 |
| Groups with reward variance / nonzero gradients | 2 (steps 1 and 4) |
| Training successes | 4 / 16 episodes |
| Final-policy resample successes | 2 / 4 episodes |
| Training trajectories reaching the 2,048-token cap | 12 / 16 |
| All sampled model tokens, including discarded suffixes | 44,789 |
| Retained model tokens | 37,497 |
| Discarded sampled tokens (still charged) | 7,292 |
| Retained external-feedback tokens | 2,043 |
| Simulator calls / error results | 91 / 40 |
| Main-run wall time, one GPU | 988.05 s (16.47 min) |
| Training generation + environment time | 780.62 s |
| Training forward/backward + scoring time | 11.07 s |
| Final-policy resampling time | 184.08 s |
| Peak allocated / reserved GPU memory | 12.74 / 16.93 GiB |

All four optimizer steps changed weights; steps 2 and 3 had zero fresh gradient and moved due to optimizer state, so this is **two informative reward groups**, not four. The final-policy resample reuses an engineering task and is neither a held-out test nor evidence of improvement. The three smoke runs sum to about 21.2 minutes of single-GPU main-run wall time, excluding dependency installation/imports and the separate short runtime diagnostics. Two GPUs were briefly used concurrently for training and parity diagnosis; the training itself used only GPU 4. GPUs 4 and 5 are released.

Cost timing starts in the runner's `main()`, after imports. `single_gpu_reserved_hours` is a wall-time reservation proxy, not metered utilization or a financial bill. Simulator execution time excludes separate reward-check/logging overhead; generation timings include the tool loop. Raw sampled-token accounting avoids claiming savings from discarded suffixes.

**Next:** calibrate multiple train/development tasks and horizon so learnability does not come solely from one easy family and truncation does not dominate; define independent held-out tasks; then freeze [the first allocation comparison](../13_rl_experiment_contract_draft.md). No more SFT gate or teacher-label recovery is required to proceed with this online path.
