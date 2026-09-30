---
license: apache-2.0
language:
- en
tags:
- agents
- terminal
- code
- software-engineering
- sft
- cold-start
pretty_name: OpenThoughts-Agent-SFT-ColdStartForRL-10K
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


# OpenThoughts-Agent-SFT-ColdStartForRL-10K

**OpenThoughts-Agent** is an open-source effort to curate the best datasets for training agents. Our release includes [datasets](https://huggingface.co/collections/open-thoughts/openthinker-agent), [models](https://huggingface.co/collections/open-thoughts/openthinker-agent) and our [research codebase](https://github.com/open-thoughts/OpenThoughts-Agent).

[OpenThoughts-Agent-SFT-ColdStartForRL-10K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K) is the **cold-start supervised-finetuning** dataset for the OpenThoughts-Agent SFT→RL recipe. It contains **9,437** (task, agent-trajectory) pairs that teach a base model the agentic format and behaviour before reinforcement learning. Fine-tuning [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on this set produces [OpenThinkerAgent-8B-ColdStartSFTForRL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL), the pre-RL base.

> **Note on the name.** The "10K" suffix is a round label for the cold-start tier; the dataset actually contains **9,437 rows**.

This is the cold-start half of the OpenThoughts-Agent SFT-then-RL recipe:

1. **[OpenThoughts-Agent-SFT-ColdStartForRL-10K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K)** — cold-start SFT trajectories (this repo).
2. [OpenThinkerAgent-8B-ColdStartSFTForRL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL) — Qwen3-8B after cold-start SFT (the pre-RL base).
3. [OpenThoughts-Agent-RL-5K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-RL-5K) — the on-policy RL tasks.
4. [OpenThinkerAgent-8B-RL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-RL) — the final RL'd checkpoint (step 45).

- **Homepage:** https://www.openthoughts.ai/blog/agent
- **Repository:** https://github.com/open-thoughts/OpenThoughts-Agent

# Data

Each row is a full multi-turn agentic trajectory: a software-engineering task solved by a teacher model acting in the **terminus-2** harness inside Daytona sandboxes. The tasks are SWE-Smith sandboxed-coding problems that ship with tests; trajectories are oracle-verified (verification timeout 120s) and run with a generous per-episode budget ("maxeps", 131k context regime). Trajectories are stored in the `conversations` chat format suitable for direct supervised finetuning.

| Field | Description |
| --- | --- |
| `conversations` | the multi-turn agent trajectory as a list of `{role, content}` messages (system / user / assistant) |
| `task` | the task identifier (e.g. `swesmith-00003`) |
| `agent` | rollout agent / harness (terminus-2) |
| `model`, `model_provider` | the teacher model and serving backend used to generate the trajectory |
| `date` | rollout timestamp |
| `episode` | episode index within the rollout |
| `run_id`, `trial_name` | rollout bookkeeping identifiers |
| `result`, `verifier_output` | verification outcome / verifier output for the trajectory |

- **Rows:** 9,437 (single `train` split)
- **Task source:** SWE-Smith sandboxed coding tasks with tests (oracle-verified, 120s verifier timeout)
- **Harness:** terminus-2 inside Daytona sandboxes
- **Format:** `conversations` (role/content multi-turn messages)

# Intended use

This is a **cold-start SFT** dataset: supervised finetuning on it gives a base model the agentic interaction format and tool-use behaviour needed to make subsequent reinforcement learning stable and sample-efficient. It was used to fine-tune [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) into [OpenThinkerAgent-8B-ColdStartSFTForRL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL), which is then RL-trained on [OpenThoughts-Agent-RL-5K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-RL-5K) to produce [OpenThinkerAgent-8B-RL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-RL).

# Links
- 🌐 [OpenThoughts-Agent project page](https://www.openthoughts.ai/blog/agent)
- 💻 [OpenThoughts-Agent GitHub repository](https://github.com/open-thoughts/OpenThoughts-Agent)
- 📚 [OpenThinker-Agent collection](https://huggingface.co/collections/open-thoughts/openthinker-agent)
- 🤖 [Cold-start model (SFT on this data): OpenThinkerAgent-8B-ColdStartSFTForRL](https://huggingface.co/open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL)
- 🧠 [RL tasks: OpenThoughts-Agent-RL-5K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-RL-5K)
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
