# 04 — Decision Log, Kill Criteria, and Research Discipline

This file exists to stop the project from drifting every time a new paper appears.

## 1. Why we chose this problem

We are not choosing it because:
- “agent is hot,”
- nobody used the exact same phrase,
- or we can add another training trick.

We chose it because agent post-training is shifting from **collecting more interaction** to **deciding which interaction is worth learning from**.

The problem likely survives stronger models because rollout/post-training budgets remain finite while tool/environment spaces expand.

---

## 2. Research-taste principle

> Taking a small rigorous step on an important problem is often more valuable than taking a large step on a problem nobody cares about.

Therefore:
- do not retreat because a paper exists,
- do not claim novelty because wording differs,
- judge whether prior evidence actually occupies the scientific claim.

---

## 3. Claim Occupancy Test

For every new close paper:

1. What exact claim is made?
2. Is the core variable isolated?
3. Are statistics convincing?
4. Is there true OOD/generalization evidence?
5. Are code/data/environments reproducible?
6. Does it measure **post-training transferable capability**, or only a nearby proxy?

Only a strong yes across most items should force a scope change.

Otherwise the paper becomes a baseline/hypothesis.

---

## 4. Stable project scope

- **Big question:** What makes agent experience valuable?
- **Operational question:** Which experience properties predict post-training OOD transfer?
- **Target:**

\[
V_{\text{transfer}}(S;\pi)
=
M_{\text{OOD}}(\operatorname{Train}(\pi,S))
-
M_{\text{OOD}}(\pi)
\]

- **Generality:** cross-scale + cross-training stage.

---

## 5. Kill criteria

### K1 — No measurable signal
After token matching and repeated runs, experience types differ less than training/evaluation noise.

### K2 — One existing proxy already fully explains transfer
A strong prior method predicts OOD transfer robustly across benchmarks, model scales, and training stages, leaving little beyond replication.

### K3 — Transfer cannot be evaluated cleanly
Training and “OOD” benchmarks share too much support/contamination to make the main claim verifiable.

### K4 — Ground-truth interventions are unaffordable
Reliable transfer values require compute beyond available resources and cluster-level approximation is insufficient.

### K5 — A new paper truly occupies the exact claim
It must do all of:
- define the same transfer target,
- use matched-budget interventions,
- compare the same value hypotheses,
- test scale/stage stability,
- provide strong reproducible evidence.

A similar title is not enough.

---

## 6. Continue criteria

### C1 — Local-to-transfer gap
A proxy predicts ID gain but poorly predicts OOD transfer.

### C2 — Learnability-transfer gap
Student-friendly/easy examples train well but teach less general capability.

### C3 — Rank reversal
Experience ordering changes across learner stage or model scale.

### C4 — Stable transfer signal
One experience property robustly predicts OOD transfer.

### C5 — Small-to-large predictability
Small-model experience ranking predicts a larger model's best training data.

Any one can carry the paper if reproduced cleanly.

---

## 7. What not to chase

Avoid scope creep into:
- general-purpose agent framework,
- environment generator,
- giant benchmark,
- RSI demo,
- dozens of model families,
- every possible value proxy,
- full SFT + full RL + multimodal + robotics in one submission.

---

## 8. Research roadmap if Paper 1 succeeds

### Paper 1 — Measurement
What makes agent experience transferable?

### Paper 2 — Allocation
Can small-scale interventions predict an optimal experience mixture for larger agents?

### Paper 3 — Dynamic curriculum
How should experience allocation change as the learner evolves?

### Paper 4 — Self-directed experience acquisition
Agent diagnoses capability gaps and selects/generates the next experiences to train on.

This connects naturally to RSI without making the first paper overclaim.

---

## 9. Fast-field execution cadence

The goal is not a six-month infrastructure project.

- **Week 1:** environment/data/eval pipeline.
- **Week 2:** four-bin pilot; first kill/continue decision.
- **Week 3–4:** full small-model measurement sweep.
- **Week 5:** cross-scale / training-stage validation.
- **Week 6:** final held-out evaluation + statistics.
- **Week 7:** writing, figures, appendix, only essential missing experiments.

If the signal is weak early, pivot quickly instead of improving infrastructure indefinitely.

---

## 10. Future-paper classification rule

Every new related paper should be classified as:

### A — Scope killer
Directly solves and convincingly validates our scientific question.

### B — Strong baseline
Another notion of experience value to compare.

### C — Supporting evidence
Shows experience composition matters but does not calibrate transfer value.

### D — Nearby noise
Similar language, different scientific target.

Most new papers should be B/C, not automatically A.

---

## 11. Current status

**READY FOR PILOT, NOT YET COMMITTED TO FULL PAPER.**

The idea has survived:
- broad literature survey,
- classical RL prior check,
- static data-selection prior check,
- robotics trajectory-valuation prior check,
- idea-evaluator stress test,
- direct threat audit of the closest papers.

The next decision should come from data.

**Decisive next action:** four-bin token-matched controlled intervention pilot.
