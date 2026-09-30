# 03 — Experiment Plan: Measuring Transfer Value of Agent Experience

> **历史记录提示（2026-09-30）：** 当前在线 RL 主线见 [PROJECT_STATE.md](PROJECT_STATE.md)。本文原始计划、预注册和结果保留；SFT 闸门不作为 RL 的停止条件，旧轨迹缺标签不作为在线环境奖励不可得的证据。


## 1. Core empirical target

For base learner \(\pi\) and experience subset \(S\):

\[
V_{\text{transfer}}(S;\pi)
=
M_{\text{OOD}}(\operatorname{Train}(\pi,S))
-
M_{\text{OOD}}(\pi)
\]

We ask which experience properties predict \(V_{\text{transfer}}\).

The paper should use existing public data/benchmarks; **no new benchmark is required**.

---

## 2. Experimental philosophy

The key unit is a **training intervention**, not a correlation on raw traces.

Bad:
> recovery score correlates with current task success.

Good:
> token-matched subsets with different recovery properties are fine-tuned under identical settings, then evaluated on held-out OOD tasks/tools/environments.

Every central claim about “value” must ultimately trace to intervention outcomes.

---

## 3. Primary data source

### OpenThoughts-Agent
Preferred starting point because it provides:
- public code,
- public training data,
- public checkpoints,
- diverse trajectories,
- existing training/eval infrastructure,
- strong comparability with prior data-recipe ablations.

Only add a second data source if the primary setting produces a clear phenomenon.

---

## 4. Experience hypotheses / selectors

Start with **6–8**, not dozens.

### H0 Random
Uniform token-matched sample.

### H1 Success / outcome
Prefer successful/high-reward trajectories.

### H2 Diversity
Prefer:
- rare tool families,
- uncommon tool compositions,
- high motif/domain diversity.

DIVE/TDScaling-style hypothesis.

### H3 Recovery / topology
Prefer traces with:
- explicit error,
- correction,
- rollback,
- alternate plan,
- independent verification.

TopoCurate-style hypothesis.

### H4 Information
Prefer high-information actions/trajectories:
- PMI,
- policy-change information gain,
- novelty/surprise.

AgentBrew/InfoGain-style hypothesis.

### H5 Learner compatibility
Prefer low current-model NLL / high student compatibility.

SmartAD-style hypothesis.

### H6 Learnability / frontier
Prefer experiences that are neither mastered nor impossible:
- moderate pass rate,
- moderate uncertainty,
- current capability frontier.

PLR/CRPS/SEC-style hypothesis.

### H7 Optional gradient influence
LESS-style influence/gradient similarity if computationally manageable.

Do not let H7 dominate engineering time.

---

## 5. Two complementary intervention designs

### 5.1 Property bins
For each signal create low/medium/high quantile bins, then sample equal-token subsets.

This tests monotonicity:

\[
P_j(S)\uparrow \stackrel{?}{\Longrightarrow} V_{\text{transfer}}(S)\uparrow
\]

### 5.2 Selection-method comparison
Each prior notion selects its preferred subset:
- Random,
- Success-only,
- Diversity,
- Recovery/Topology,
- Information,
- Student-compatible,
- Learnability/Frontier.

Train each under identical budgets.

This produces the main table.

---

## 6. Critical controls

### 6.1 Token-matched budget
Primary control:

\[
\boxed{\text{same training tokens}}
\]

not same number of trajectories.

Long/recovery traces are often longer; otherwise token count confounds the result.

Also report:
- number of trajectories,
- tool calls,
- mean horizon,
- unique tools,
- unique task sources.

### 6.2 Same optimization
Same optimizer, LR, steps/epochs, context limit, batch/token budget.

### 6.3 Same base checkpoint
No mixing base-model strength with experience quality.

### 6.4 Multiple training seeds
Headline conditions: target **3 independent training seeds**.

Pilot: 1–2 seeds only for kill/continue decisions.

### 6.5 Freeze final OOD evaluation
Do not tune selectors on final held-out benchmarks.

---

## 7. Benchmark plan

Use a small number of benchmarks that cover distinct transfer axes.

### Development / cheap diagnostics

#### BFCL
Purpose:
- function/tool schema generalization,
- fast iteration.

#### OT-TBLite or fixed Terminal-Bench subset
Purpose:
- long-horizon execution,
- realistic action-observation traces,
- cheap development feedback.

### Final held-out evaluation

#### Toolathlon
Purpose:
- unseen tools,
- unseen toolsets,
- stateful multi-tool interaction,
- strongest fit with “transfer” claim.

#### MedAgentBench
Purpose:
- domain shift,
- specialized tools,
- protocol/stateful-environment shift.

#### GAIA
Purpose:
- broad generalist-agent transfer.

#### One long-horizon coding/terminal benchmark
Candidate:
- Terminal-Bench 2.0,
- SWE-Bench Verified subset,
- Aider Polyglot.

Pick one based on infrastructure compatibility.

---

## 8. Evaluation taxonomy

Do not hide everything in one average.

Report:
- **ID** — same/nearby distribution,
- **Tool Transfer** — unseen tools/schemas/pools,
- **Composition Transfer** — new combinations/orderings,
- **Environment Transfer** — new state/protocol,
- **Domain Transfer** — new semantic domain,
- **Long-Horizon Transfer** — longer action chains/recovery/delayed feedback.

Also report:
- macro OOD average,
- delta over base,
- delta over random token-matched baseline.

---

## 9. Ground-truth intervention protocol

For every experience selector/subset \(S_i\):

1. sample token-matched data,
2. fine-tune the same base model,
3. run ID and frozen OOD evaluations,
4. compute:

\[
\Delta_{\text{ID},i}
\]

and

\[
\Delta_{\text{OOD},i}.
\]

These are the empirical ground-truth value targets.

---

## 10. Core analysis: do proxies predict transfer?

For every proxy \(P_j\):

Primary statistics:
- Spearman \(\rho\),
- Kendall \(\tau\),
- bootstrap confidence intervals.

Compare:

\[
\rho(P_j,\Delta_{\text{ID}})
\]

against:

\[
\boxed{\rho(P_j,\Delta_{\text{OOD}})}.
\]

A local-to-transfer gap is a central possible result.

---

## 11. Main Table

### Table 1 — Experience Selection vs Transfer

| Experience selection | Train tokens | ID | Tool Transfer | Composition Transfer | Env/Domain OOD | Long-Horizon OOD | OOD Avg | ΔOOD vs Random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | — | — | — | — | — | — | — |
| Random | 1× | | | | | | | |
| Success-only | 1× | | | | | | | |
| Diversity | 1× | | | | | | | |
| Recovery/Topology | 1× | | | | | | | |
| Information/PMI | 1× | | | | | | | |
| Student-Compatible | 1× | | | | | | | |
| Learnability/Frontier | 1× | | | | | | | |
| Transfer-aware method (optional) | 1× | | | | | | | |

Rules:
- same token budget,
- mean ± SE/CI,
- statistically supported bolding only.

---

## 12. Main Figures

### Figure 1 — Concept
Contrast:
- a highly successful but API-specific trace,
- a failure→inspect→repair→verify trace with reusable structure.

Question: which teaches more transferable capability?

### Figure 2 — Proxy vs actual transfer
Scatter/correlation plot for each value notion.
Show ID correlation next to OOD correlation.

### Figure 3 — Cross-scale stability

\[
V_{4B}(S_i) \text{ vs } V_{8B}(S_i)
\]

Report rank correlation.

### Figure 4 — Training-stage rank reversal
Heatmap across early/mid/mature learners.

Potentially the most memorable mechanism figure.

---

## 13. Cross-scale experiment

Do not start with 32B.

Recommended:
- broad sweep on a ~3B–4B model,
- validate only key conditions on ~8B.

Question:

\[
V_{\text{small}}(S)\stackrel{?}{\rightarrow}V_{\text{large}}(S)
\]

If rank correlation is high, small models can proxy experience allocation.
If low, experience value is scale-dependent.

A 32B result is bonus only.

---

## 14. Training-stage experiment

Use 2–3 learner checkpoints:
- base / weak,
- mid post-training,
- mature agent.

Train each on the same experience groups.

Ask whether:

\[
\text{rank}(V(S_i;\pi_{\text{early}}))
\]

matches:

\[
\text{rank}(V(S_i;\pi_{\text{late}})).
\]

Systematic reversal is a strong result.

---

## 15. Optional complementarity experiment

If time allows:

\[
I(A,B)=V(A+B)-V(A)-V(B)
\]

Test whether experience types are:
- additive,
- synergistic,
- redundant,
- interfering.

Example: diversity, recovery, diversity+recovery.

Not required for first submission.

---

## 16. Minimum viable pilot

Before writing a new method, run a four-bin causal pilot.

### Four bins
1. Easy successful.
2. Long successful.
3. Failure→recovery.
4. Structurally diverse / rare-tool-composition.

### Budget
Token-matched 1k–2k examples per bin.

### Model
Manageable 3B–4B class model.

### Evaluation
- BFCL,
- small fixed Terminal subset,
- one held-out domain/tool benchmark.

### Continue if
- between-bin OOD effects exceed seed/eval noise,
- and/or ID ranking differs from OOD ranking,
- and/or ranking changes with learner stage/scale.

Otherwise kill or pivot before building more infrastructure.

---

## 17. Statistical protocol

Headline experiments:
- 3 training seeds where feasible,
- multiple stochastic eval runs where benchmark requires,
- bootstrap over tasks,
- paired comparisons on shared test sets,
- Spearman/Kendall CIs.

Always report:
- mean,
- SE or 95% CI,
- number of tasks,
- number of independent training runs.

---

## 18. Compute-saving funnel

### Phase A — cheap pilot
4 bins, small model, limited benchmarks.

### Phase B — measurement sweep
6–8 value hypotheses, small model, final matched-budget protocol.

### Phase C — validation
Only the strongest/most diagnostic 4–6 conditions on medium model and final held-out benchmarks.

### Phase D — optional method
Only if measurement results justify it.

---

## 19. Strong-result patterns

1. **Local-to-transfer gap:** proxy predicts ID but not OOD.
2. **One universal-ish signal:** a prior value notion robustly predicts transfer across settings.
3. **Rank reversal:** value changes with learner scale/stage.
4. **Small-to-large prediction:** small-model ranking predicts medium-model optimal data.

Any one can carry a strong paper if reproduced cleanly.

---

## 20. Weak-result patterns

Warning signs:
- effects smaller than noise,
- one benchmark only,
- only ID gain,
- effect vanishes after token matching,
- selector wins only because traces are longer,
- final benchmarks are used during tuning,
- one model / one seed / one environment only.

These should trigger a rethink before paper writing.

---

## 21. Paper skeleton implied by the experiments

1. Introduction — multiple incompatible notions of valuable experience.
2. Related Work — diversity, curation, learner-aware curriculum, RL replay, data valuation.
3. Measuring Transfer Value — formal target + controlled intervention.
4. Do Existing Value Signals Predict Transfer? — main comparison.
5. Is Experience Value Stable? — scale/stage analysis.
6. Optional Transfer-Aware Selection — only if supported.
7. Limitations — environment coverage, compute scale, experience interaction, SFT/RL boundary.

---

## 22. Final experiment scope freeze

### Required
- one public experience pool,
- one primary model family,
- 6–8 value hypotheses,
- token-matched controlled SFT,
- multiple OOD transfer axes,
- small→medium scale check,
- early→late learner check.

### Optional
- 32B validation,
- RL extension,
- new selector,
- complementarity analysis.

Do not let optional experiments delay the core measurement.
