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
