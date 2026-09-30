# 01 — Scope: What Makes Agent Experience Valuable?

> **历史记录提示（2026-09-30）：** 当前在线 RL 主线见 [PROJECT_STATE.md](PROJECT_STATE.md)。本文原始计划、预注册和结果保留；SFT 闸门不作为 RL 的停止条件，旧轨迹缺标签不作为在线环境奖励不可得的证据。


**Working title:** *From Experience to Capability: What Makes Agent Experience Valuable?*

## 1. One-sentence paper story

Current agent post-training methods implicitly disagree on what makes an experience “valuable” — success, diversity, information gain, recovery structure, learner compatibility, uncertainty, or learnability — but these signals are rarely compared against the same causal target: **how much transferable capability the agent gains after actually training on that experience**.

Our paper asks:

> **Which properties of interactive agent experience predict post-training transfer to unseen tasks, tools, and environments?**

The core contribution is not another agent framework and not another heuristic data selector. It is a **controlled measurement study** connecting experience properties to actual capability gain.

## 2. Big question vs. paper question

### Big question

> **What makes interactive experience valuable for a learning agent?**

Long-term problem:

\[
\text{Experience} \rightarrow \text{Learning} \rightarrow \text{Capability}
\]

As agent training becomes interaction-heavy, the bottleneck is no longer merely “how to collect more rollouts,” but which rollouts are worth learning from, which failures teach robust skills, which environments should be revisited, and how this value changes as the learner evolves.

### Operational paper question

\[
\boxed{\text{Which experience-value signals predict actual post-training OOD transfer?}}
\]

This is narrower than the big question, but still central.

## 3. Why this matters now

The community is converging on the same hidden problem from different directions:

- **DIVE:** structural diversity improves OOD tool-use generalization.
- **TopoCurate:** recovery/topology may be more useful than outcome-only filtering.
- **AgentBrew:** individual actions can be valued through information/PMI.
- **SmartAD:** trajectories should be compatible with the current student.
- **CoEvolve / curriculum work:** useful training tasks should track learner failures and uncertainty.
- **OpenThoughts-Agent:** source, teacher, trajectory length, and mixture matter non-monotonically.
- **EFC:** raw interaction count is a poor measure of useful feedback at inference time.

These works create **multiple competing definitions of value**. Our goal is to compare them against one common intervention target.

## 4. Core distinction

### 4.1 Local task utility
Does the experience help solve the current task?

Examples: reward, pass rate, current-task information gain.

### 4.2 Immediate learning utility
Does the experience produce a useful gradient for the current learner?

Examples: student NLL / compatibility, absolute advantage, TD error, uncertainty, capability-frontier signals.

### 4.3 Transfer utility
After training on the experience, does the model gain capability on **unseen** tasks/tools/environments?

\[
V_{\text{transfer}}(S;\pi)
=
M_{\text{OOD}}(\operatorname{Train}(\pi,S))
-
M_{\text{OOD}}(\pi)
\]

The paper asks whether:

\[
\text{Local Utility}\stackrel{?}{\approx}\text{Transfer Utility}
\]

and under what conditions.

## 5. Competing hypotheses

- **H1 Outcome value:** successful/high-reward trajectories are most useful.
- **H2 Diversity value:** structurally diverse experiences produce broader capability.
- **H3 Information value:** high-information feedback/actions produce better learning.
- **H4 Recovery/topology value:** failure, rollback, correction, and recovery are especially instructive.
- **H5 Learner-compatibility value:** experiences close to the student's capacity are best.
- **H6 Frontier/learnability value:** partially solvable/uncertain experiences produce the strongest learning signal.
- **H7 Influence/gradient value:** gradient/influence measures predict downstream usefulness.

These are not territories to avoid; they are hypotheses to test under one protocol.

## 6. Core claims we are allowed to make

### Claim A — Measurement
We operationalize agent experience value as **post-training transferable capability gain**, rather than only reward, difficulty, diversity, or current-task informativeness.

### Claim B — Comparative evidence
We compare existing notions of experience value under the **same model, experience pool, training budget, and OOD evaluation protocol**.

### Claim C — Stability
We test whether experience-value ranking is stable across:
- model scale,
- training stage,
- optionally environment/tool families.

These three claims are sufficient for the paper.

## 7. Claims we should not make unless the data earns them

Do not pre-claim:
- a universal scaling law,
- that nobody studied experience value before,
- that learner dependence is novel,
- that diversity is wrong,
- that failure trajectories are always superior,
- that our score is universal,
- or that this solves RSI.

## 8. Publishable scientific outcomes

### Outcome A — Stable transferable value exists
Ranking is consistent across scale/stage. Then we have a plausible transferable experience-value coordinate.

### Outcome B — One existing proxy wins
For example, diversity or learnability robustly predicts transfer. This is still valuable cross-setting validation.

### Outcome C — Local proxies fail
Success/information/compatibility predict ID gain but poorly predict OOD transfer. This reveals a **local-to-transfer gap**.

### Outcome D — Rank reversal
Useful experiences change with learner scale/stage. Then experience value is dynamic/relational rather than globally intrinsic.

## 9. Why stronger base models do not make this obsolete

Even stronger agents have:
- finite rollout budgets,
- finite post-training compute,
- larger tool/environment spaces,
- more expensive long-horizon trajectories.

Thus stronger models make the allocation problem more important:

> **Which experiences are worth paying to collect and train on?**

This is a capability-convex problem rather than a temporary capability patch.

## 10. Out of scope for the first paper

- new benchmark,
- new environment generator,
- giant-model training from scratch,
- code + GUI + browser + robotics all at once,
- full-scale SFT and RL simultaneously,
- recursive self-improvement system,
- universal theory,
- complicated new `AgentExperienceScore` before measurement.

## 11. Intended contribution structure

1. **Measurement target:** controlled measurement of post-training transfer value.
2. **Unified comparison:** outcome, diversity, information, topology/recovery, compatibility, learnability, optionally gradient influence.
3. **Stability analysis:** cross-scale and cross-training-stage behavior.

A new selector is optional and should be added only if the measurement results naturally motivate it.

## 12. Working thesis statement

> Agent post-training increasingly depends on choosing which interactive experiences to learn from. Existing methods operationalize “valuable experience” through different local signals — success, diversity, information, topology, compatibility, or learnability — but these signals are rarely calibrated against the same downstream quantity. We study experience value through controlled training interventions and ask which signals actually predict transferable capability on unseen tasks, tools, and environments, and whether these rankings remain stable across learner scale and training stage.

## 13. Scope freeze

- **Big question:** What makes agent experience valuable?
- **Paper question:** Which experience properties predict post-training OOD transfer?
- **Core experiment:** Controlled experience intervention + held-out transfer evaluation.
- **Generality test:** Cross-scale + cross-training-stage stability.
- **Optional method:** Only if existing value proxies fail.

This scope should change only when new **evidence**, not merely a similar paper title, forces a revision.
