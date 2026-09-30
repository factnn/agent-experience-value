# 02 — Literature Review: Agent Experience Value, Selection, and Transfer

**Survey question:** Which properties of interactive agent experience predict post-training transferable capability?

This document is organized around **competing definitions of experience value** and a **claim-occupancy test**: a prior paper only occupies territory if its experiments actually establish the relevant claim.

## 1. Threat-audit principle

For every close paper, ask:

1. **Claim:** what is actually claimed?
2. **Isolation:** is the core variable isolated by matched-budget controls/ablations?
3. **Statistics:** are effects larger than run-to-run noise?
4. **Generalization:** is there true OOD transfer rather than in-distribution improvement?
5. **Reproducibility:** are code/data/models/environments available?

Do not shrink scope simply because a paper exists.

---

## 2. Strongest direct priors

### 2.1 DIVE — *Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use*
Aili Chen et al., 2026  
https://arxiv.org/abs/2603.11076

**Core:** scales tool-pool coverage, per-task toolset variety, and tool-use structural diversity across 373 tools / 5 domains; reports strong OOD gains and that diversity scaling can outperform quantity scaling.

**What it occupies:**
- tool diversity matters,
- structural coverage can beat raw quantity,
- OOD tool-use benefits from diverse composition.

**What it does not settle:**
- per-experience transfer value,
- whether diversity beats competing value signals under one protocol,
- cross-scale/stage ranking stability,
- whether some diversity is just test-support matching rather than a universal value property.

**Threat:** **HIGH**. Must be a primary baseline/hypothesis.

---

### 2.2 OpenThoughts-Agent — *Data Recipes for Agentic Models*
Code: https://github.com/open-thoughts/OpenThoughts-Agent

**Core:** large controlled agent-data study with 100+ ablations over task source, teacher, filtering, trajectory properties, scale, and mixture.

Important conclusions from our review:
- source choice matters greatly,
- mixtures are non-monotonic,
- more sources can hurt,
- stronger teacher does not necessarily produce better student data,
- long traces can remain useful under token-matched controls,
- repeated trajectories plateau while task-description diversity can continue helping.

**What it occupies:** broad agent data-recipe ablations.

**What remains:** it mostly establishes

\[
\text{recipe} \rightarrow \text{downstream score}
\]

rather than a transferable theory/value model for

\[
\text{experience property} \rightarrow \text{transferable learning gain}.
\]

**Threat:** **VERY HIGH**. Strongest empirical neighbor and best motivation source.

---

### 2.3 TopoCurate — *Modeling Interaction Topology for Tool-Use Agent Training*
Jinluan Yang et al., 2026  
https://arxiv.org/abs/2603.01714

**Core:** outcome-only filtering misses interaction dynamics. Selects SFT trajectories using reflective recovery, semantic efficiency, strategic diversity; selects RL tasks using error-branch ratio and strategic heterogeneity.

**What it occupies:** recovery/topology has already been used for data/task selection.

**Why it does not close our problem:**
- narrower benchmark evidence than DIVE/OT-Agent,
- interaction-topology score is still a heuristic proxy,
- no direct calibration to true post-training cross-tool/environment transfer value.

**Threat:** **MEDIUM-HIGH**. Serious hypothesis to reproduce, not a reason to retreat.

---

### 2.4 AgentBrew — *Offline Tool-Use Agent Learning from Raw Real-World Trajectories*
2026.

**Core:** retrospective task inference + PMI-style per-action credit for raw trajectories in real tools such as GitHub/Notion/PostgreSQL.

**What it occupies:** different actions inside a trajectory can have different training value; information-based credit is feasible.

**Remaining question:** PMI/current-task information is not the same as cross-task/tool/environment transfer value.

**Threat:** **HIGH but complementary**. Information-value baseline.

---

### 2.5 *Scaling Laws for Agent Harnesses via Effective Feedback Compute*
Xuanliang Zhang et al., 2026  
https://arxiv.org/abs/2605.29682

**Core:** test-time agent scaling should measure effective feedback rather than raw tokens/tool calls. Feedback counts when informative, valid, non-redundant, and retained.

**Critical distinction:** EFC predicts **current-run success/failure**. Our target is:

\[
\text{training on experience}\rightarrow\Delta\text{future OOD capability}.
\]

A feedback event can be very useful locally but have almost zero training transfer.

**Threat:** **HIGH conceptually; target variable differs.**

---

## 3. Learner-aware / curriculum priors

### 3.1 SmartAD — *Capacity-Aligned Agent Distillation for Small Language Models*
Guokai Tang, Feng Zhao, ACL Findings 2026  
https://aclanthology.org/2026.findings-acl.1349/

**Core:** among multiple correct teacher trajectories, select the one with minimum student NLL; use segment-weighted supervision.

**What it establishes:** data usefulness can depend on learner capacity.

**Why it is narrower than our problem:**
- teacher distillation,
- correct trajectories only,
- small students,
- multi-hop QA/math rather than general stateful tool environments.

**Key baseline:** student compatibility / NLL.

**Question for us:** Is an experience that is easy for the student to learn also the one that produces the most transfer?

---

### 3.2 CoEvolve — *Training LLM Agents via Agent-Data Mutual Evolution*
Shidong Yang et al., ACL 2026  
https://aclanthology.org/2026.acl-long.1055/  
Code: https://github.com/StoneHanaMori/CoEvolve

**Core:** use forgetting and uncertainty from current rollouts to synthesize new failure-targeted tasks and update the training distribution.

**What it occupies:** learner-adaptive data generation/curriculum is not new.

**Remaining question:** Are forgetting/uncertainty actually good predictors of long-horizon transferable learning value, or merely useful heuristics in their setup?

---

### 3.3 Self-Evolving Curriculum for LLM Reasoning
Xiaoyin Chen et al., COLM 2026  
https://www.microsoft.com/en-us/research/publication/self-evolving-curriculum-for-llm-reasoning/

**Core:** non-stationary multi-armed bandit over problem categories; uses absolute policy-gradient advantage as immediate learning-gain proxy.

**What it occupies:** dynamic curriculum selection and immediate-learning-gain signals.

**Open question:** immediate learning gain may differ from long-horizon transfer.

---

## 4. Classical RL ancestors

### 4.1 Prioritized Experience Replay
Schaul et al., 2015  
https://arxiv.org/abs/1511.05952

Core lesson: experiences need not be replayed uniformly.

**Constraint:** never claim “first to realize different experiences have different value.”

### 4.2 Hindsight Experience Replay
Andrychowicz et al., NeurIPS 2017.

Core lesson: failed trajectories can become useful after relabeling.

### 4.3 Prioritized Level Replay
Jiang et al., ICML 2021.

Core lesson: environment/level selection can be based on future learning potential and improve unseen-level generalization.

**Constraint:** learner-dependent learning potential is not novel by itself.

### 4.4 No Regrets / Learnability
Rutherford et al., 2024.

Core lesson: a theoretically appealing proxy can fail to measure the intended quantity. Curriculum methods may accidentally prioritize already-mastered or otherwise unhelpful levels.

**Methodological lesson:** do not invent an AgentExperienceScore first. Measure:

\[
\text{proxy}\rightarrow\text{actual intervention gain}.
\]

---

## 5. Static LLM data valuation / mixture

### 5.1 LESS — *Selecting Influential Data for Targeted Instruction Tuning*
Mengzhou Xia et al., ICML 2024  
https://proceedings.mlr.press/v235/xia24c.html

**Core:** optimizer-aware gradient influence for targeted data selection.

**Key result relevant here:** selections from smaller models can transfer to larger models / different families.

**Template for us:**

\[
\text{small-agent experience ranking}\rightarrow\text{large-agent ranking?}
\]

### 5.2 RegMix — *Data Mixture as Regression for Language Model Pre-training*
Qian Liu et al., ICLR 2025  
https://proceedings.iclr.cc/paper_files/paper/2025/hash/5f67d864aae6115374fed7beddd119e0-Abstract-Conference.html

**Core:** train many tiny proxy models on different mixtures, regress mixture to performance, predict the best mixture for much larger training.

**Template for a follow-up:** Agent Experience Mixture optimization if cross-scale value structure exists.

---

## 6. Robotics trajectory-valuation neighbor

### CUPID — *Curating Data your Robot Loves with Influence Functions*
CoRL 2025.

**Core:** estimate demonstration influence on closed-loop expected return and curate trajectories accordingly.

**Constraint:** downstream trajectory influence is not novel by itself.

**LLM-agent-specific structure that must matter:**
- language-conditioned state/action semantics,
- tools/APIs,
- long horizons,
- compositional tool use,
- stateful feedback,
- unseen-tool/environment transfer,
- cross-model-scale behavior.

---

## 7. Claim occupancy table

| Prior | Value notion | Evidence strength | True OOD transfer target? | Cross-scale? | What remains |
|---|---|---:|---:|---:|---|
| DIVE | structural diversity | strong | strong OOD eval | no | compare against other notions; isolate transfer value |
| OpenThoughts-Agent | source/teacher/recipe | strong | held-out benchmarks | partial | predictive experience-value structure |
| TopoCurate | recovery/topology | moderate | limited transfer | partial | unified validation, stronger reproducibility |
| AgentBrew | PMI/information | good for current-task credit | limited | no | information value vs transfer value |
| EFC | effective feedback | strong for inference | no training-transfer target | multi-model inference | current-run utility vs learning transfer |
| SmartAD | student compatibility/NLL | good, narrow | OOD QA/math | small scales | compatibility vs transfer in real tool agents |
| CoEvolve | uncertainty/forgetting | strong method evidence | yes | multiple scales | proxy calibration vs true transfer |
| SEC | immediate learning gain | strong formulation | harder OOD | not central | immediate gain vs long-term transfer |
| PLR | learning potential | classical strong prior | unseen levels | standard RL | LLM-agent semantics and cross-tool transfer |
| LESS | gradient influence | strong static-data method | targeted downstream | yes | interactive trajectories |
| RegMix | mixture predictor | strong static-data method | downstream | yes | policy-dependent experience mixture |
| CUPID | demonstration influence | strong robotics neighbor | shift eval | no LLM scale | language/tool agent transfer |

---

## 8. Central gap after all rounds

The remaining gap is not:
- nobody studies data selection,
- nobody studies trajectory quality,
- nobody studies learner dependence,
- nobody studies diversity,
- nobody studies curriculum.

The gap is:

> **Existing methods operationalize experience value using different local quantities, but the field lacks a unified, controlled comparison against actual post-training transfer gain in a common agent setting.**

Formally, compare each proxy:

\[
P_j(S)
\]

against:

\[
V_{\text{transfer}}(S;\pi)
=
M_{\text{OOD}}(\operatorname{Train}(\pi,S))
-
M_{\text{OOD}}(\pi).
\]

The key measurement is:

\[
\rho(P_j,V_{\text{transfer}}).
\]

---

## 9. Key scientific tension

All of these sound plausible:
- success is clean,
- hard examples contain new information,
- easy examples are learnable,
- frontier examples maximize learning progress,
- diverse examples increase coverage,
- recovery examples teach robust policies,
- high-PMI actions contain task-relevant signal.

They cannot all be globally optimal for every learner and target distribution.

That tension is the conceptual opening.

---

## 10. Rule for future related papers

Classify every new close paper using the **Claim Occupancy Test**:

1. Same words, or same scientific target?
2. Same target variable?
3. Same controlled intervention?
4. Reproducible evidence?
5. Comparable OOD/generalization test?

Only if 2–5 are largely “yes” should it force a scope change.

Otherwise the paper becomes another baseline/hypothesis rather than a reason to retreat.

---

## 11. Priority reading order

### Tier A — must understand/reproduce deeply
1. OpenThoughts-Agent
2. DIVE
3. TopoCurate
4. AgentBrew
5. CoEvolve
6. SmartAD

### Tier B — conceptual/method baselines
7. Effective Feedback Compute
8. Self-Evolving Curriculum
9. Prioritized Level Replay
10. No Regrets / Learnability
11. LESS
12. RegMix

### Tier C — cross-domain ancestors
13. Prioritized Experience Replay
14. Hindsight Experience Replay
15. CUPID

---

## 12. Current literature verdict

The literature does **not** justify saying “agent experience value has not been studied.”

It **does** justify the stronger premise:

> The field has multiple partially validated, mutually non-equivalent notions of valuable agent experience, but lacks a common causal calibration target for transferable capability.
