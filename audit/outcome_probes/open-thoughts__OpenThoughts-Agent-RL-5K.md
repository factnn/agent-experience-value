---
license: apache-2.0
language:
- en
tags:
- agents
- terminal
- code
- software-engineering
- reinforcement-learning
- rl
pretty_name: OpenThoughts-Agent-RL-5K
size_categories:
- 1K<n<10K
---
<p align="center">
    <img src="https://huggingface.co/datasets/open-thoughts/OpenThoughts1-Agent-SFT/resolve/main/ota-logo.png" width="50%">
</p>

<p align="center">
<a href="https://www.openthoughts.ai/blog/agent" style="margin-right: 24px;">Project</a> |
<a href="https://github.com/open-thoughts/OpenThoughts-Agent" style="margin-right: 24px; margin-left: 24px;">Code</a> |
<a href="https://huggingface.co/collections/open-thoughts/openthinker-agent" style="margin-left: 24px;">Collection</a>
</p>


# OpenThoughts-Agent-RL-5K

**OpenThoughts-Agent** is an open-source effort to curate the best datasets for training agents. Our release includes [datasets](https://huggingface.co/collections/open-thoughts/openthinker-agent), [models](https://huggingface.co/collections/open-thoughts/openthinker-agent) and our [research codebase](https://github.com/open-thoughts/OpenThoughts-Agent).

[OpenThoughts-Agent-RL-5K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-RL-5K) is a **5,000-task** reinforcement-learning task set used to RL-finetune the cold-start SFT model into the final agentic checkpoint. Unlike the SFT datasets, which hold full (task, trajectory) pairs, this dataset holds **executable agentic tasks** (the `pymethods2test-large` task pool) that the policy attempts on-line inside sandboxes; reward comes from the test verifier, not from a stored teacher trajectory.

This is the RL-task half of the OpenThoughts-Agent SFT-then-RL recipe:

1. [OpenThoughts-Agent-SFT-ColdStartForRL-10K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K) — cold-start SFT trajectories.
2. [OpenThinkerAgent-8B-ColdStartSFTForRL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL) — Qwen3-8B after cold-start SFT (the pre-RL base).
3. **[OpenThoughts-Agent-RL-5K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-RL-5K)** — the RL tasks in this repo.
4. [OpenThinkerAgent-8B-RL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-RL) — the final RL'd checkpoint (step 45).

- **Homepage:** https://www.openthoughts.ai/blog/agent
- **Repository:** https://github.com/open-thoughts/OpenThoughts-Agent

# Data

The dataset contains **5,000 rows**, one per agentic task drawn from the `pymethods2test-large` task pool. Each task is a self-contained, runnable software-engineering environment: the agent operates in a Daytona sandbox under the **terminus-2** harness and is rewarded by the task's own test verifier.

| Field | Description |
| --- | --- |
| `path` | task identifier / slug (e.g. `pymethods2test-0000`) |
| `task_binary` | the gzip-compressed task bundle (environment definition, source, and tests) consumed by the harness at rollout time |

- **Rows:** 5,000
- **Task source:** `pymethods2test-large` (Python method-to-test generation tasks)
- **Harness:** terminus-2 (agent: terminus-2) inside Daytona sandboxes
- **Reward:** oracle test-verifier outcome (no stored teacher trajectory)

# Intended use

This dataset is the **task distribution for on-policy RL**. It is consumed by an RL trainer (the OpenThoughts-Agent codebase uses SkyRL with the RLOO-n advantage estimator) that rolls out the policy against each task, scores it with the verifier, and updates the policy. It was used to train [OpenThinkerAgent-8B-RL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-RL) starting from the cold-start SFT base. It is **not** a supervised dataset: there are no reference completions to imitate.

# Links
- 🌐 [OpenThoughts-Agent project page](https://www.openthoughts.ai/blog/agent)
- 💻 [OpenThoughts-Agent GitHub repository](https://github.com/open-thoughts/OpenThoughts-Agent)
- 📚 [OpenThinker-Agent collection](https://huggingface.co/collections/open-thoughts/openthinker-agent)
- 🧠 [Cold-start SFT dataset: OpenThoughts-Agent-SFT-ColdStartForRL-10K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K)
- 🤖 [Pre-RL base model: OpenThinkerAgent-8B-ColdStartSFTForRL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL)
- 🤖 [Final RL model: OpenThinkerAgent-8B-RL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-RL)

# Citation
```
@misc{openthoughts-agent,
  author = {Team, OpenThoughts-Agent},
  title = {{OpenThoughts-Agent: Data Recipes for Agentic Models}},
  howpublished = {https://www.openthoughts.ai/blog/agent},
  year = {2026}
}
```
