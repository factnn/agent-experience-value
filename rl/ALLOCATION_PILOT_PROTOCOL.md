# Single-seed allocation engineering pilot — 2026-10-08

Declared before either arm samples. This pilot tests integration and practical costs. It does not freeze a paper-level experiment or establish allocation effectiveness.

## Fixed configuration

Original local Qwen3-4B, fresh LoRA r8 q/v, alpha16/dropout0. Official TRL0.29 GRPO, LR1e-5 constant, beta0, group-scaled binary state reward, one iteration and no replay. Four trajectories/group; one fresh group/optimizer update via batch1 × accumulation4 and steps_per_generation4. Thinking, temperature1/top_p1/top_k0, protocol prompt from the calibration, retained completion cap4096 including feedback, eight tool rounds and twelve executable calls. Truncations are retained and charged. The failed developmental cap screen remains documented; 4K is a pilot candidate, not a configuration that passed that screen.

Training: seeds92000/92001 × send/new_contact/replace (six instances). Final evaluation: seed93000 × those three families (twelve episodes). Neither set was sampled in the prior calibration. All deterministic oracle rewards are checked before launch. Same simulator and same task families: this evaluates only new instances, not cross-tool or cross-family transfer. No evaluation feedback enters the allocator or changes configuration.

Both arms use initialization seed20261008. Allocation RNG is independent of model generation RNG; group step k resets generation seed to20261008+100*k. Final evaluation task index i uses seed20261008+100000+i. One seed only; no statistical efficacy decision.

## The intervention

`uniform`: equal probabilities after common warmup. `frontier`: recent16 binary training outcomes per task, Beta(1,1) mean p, score4p(1-p), probabilities20% uniform +80% normalized score. Both start with the same shuffled one-group-per-training-instance warmup; warmup updates the policy and is fully charged. Shared warmup means identical initial information by design; matching actual outcomes/parameter hashes is audited rather than presumed across devices.

Select one task **before** collecting its four fresh rollouts. Save the choice, probabilities, prior cost, rewards and actual raw sampled cost. Only past training rewards are available. No success filtering, loss reweighting, extra probes or altered rewards. Official loss and update implementation remain unchanged.

## Budget and stop rules

Per arm, target120,000 **all generated model tokens**, including failures and suffixes later discarded by tool truncation. Stop after the first completed group/update for which charged training tokens reach/exceed the target. Permit that entire crossing group; do not trim it or retrospectively match successes. Report actual cost and overshoot separately for each arm. No claim of exactly equal endpoint cost; a large overshoot can make an endpoint comparison unsuitable for effect estimation.

Safety limits:16 updates or a120-minute process timeout per arm, whichever comes first. If safety limit precedes token budget, explicitly report the smaller budget. If the budget is crossed during warmup, mark the allocation comparison unavailable. Final evaluation follows a completed training run and its generated tokens are reported separately, never charged as training acquisition. Evaluation has three four-trajectory groups with the same4096 horizon. A timeout/failure remains visible; no silent rerun.

Save per-update policy hashes, gradients, state/action/feedback, raw generation tokens, retained masks, task allocation and cumulative cost. The trainer checks exactly one group per optimizer step, and no new group after budget. Record peak memory, generation vs update timing, task proportions and effective reward groups. Save final adapters locally; push code, logs, summaries and evidence.

At most two GPUs, rechecked idle at launch; never use occupied0–3. No paid API, new model download or external messaging.

## Interpretation

Report final new-instance success alongside actual budgets and uncertainty. A single small pilot cannot decide whether the heuristic improves transferable ability; parameter movement on zero-gradient groups may be Adam momentum. Successful completion validates allocation/online-update/budget integration. Richer tasks, stronger held-out evaluation and multiple independent training seeds remain required for the research question.
