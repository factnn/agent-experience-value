---
base_model:
- Qwen/Qwen3-8B
datasets:
- OpenThoughts-Agent-v1-SFT
- OpenThoughts-Agent-v1-RL
library_name: transformers
license: apache-2.0
model-index:
- name: OpenThinker-Agent-v1
  results: []
pipeline_tag: text-generation
tags:
- agents
- terminal
- code
- software-engineering
---

<p align="center">
    <img src="https://huggingface.co/datasets/open-thoughts/OpenThoughts1-Agent-SFT/resolve/main/ota-logo.png" width="50%">
</p>

<p align="center">
<a href="https://www.openthoughts.ai/blog/agent" style="margin-right: 24px;">Project</a> |
<a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT" style="margin-right: 24px; margin-left: 24px;">SFT dataset</a> |
<a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL" style="margin-right: 24px; margin-left: 24px;">RL dataset</a> |
<a href="https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT" style="margin-right: 24px; margin-left: 24px;">SFT model</a> |
<a href="https://huggingface.co/open-thoughts/OpenThinker-Agent-v1" style="margin-left: 24px;">RL model</a>
</p>


# OpenThinker-Agent-v1-SFT

**OpenThoughts-Agent** is an open-source effort to curate the best datasets for training agents. Our first release includes [datasets](https://huggingface.co/collections/open-thoughts/openthinker-agent), [models](https://huggingface.co/collections/open-thoughts/openthinker-agent) and our research codebase.

[OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1) is a model trained for agentic tasks such as **Terminal-Bench 2.0** and **SWE-Bench**.

The [OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1) model is post-trained from [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
It is SFT-ed on the [OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT) dataset, then RL-ed on the [OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL) dataset. 

This [OpenThinker-Agent-v1-SFT](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT) model is the model after the SFT stage. For the model after both SFT and RL stages, see [OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1).

- **Homepage:** https://www.open-thoughts.ai/agent
- **Repository:** https://github.com/open-thoughts/OpenThoughts-Agent


# OpenThinker-Agent-v1 Model Performance

Our [OpenThinker-Agent-v1](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL) model is the state-of-the-art model at its scale on agent benchmarks.

| Model                                                                                           | Harness | Terminal-Bench 2.0 | SWE-Bench Verified | OpenThoughts-TB-Dev |
| ----------------------------------------------------------------------------------------------- | ------- | ------------------ | --------- | ------------------- |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)                                                | Terminus-2 | 0.0             | 0.7       | 5.7                 |
| **[OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1)**           | Terminus-2 | 4.9             | 15.7      | 17.3                |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)                                              | Terminus-2 | 1.9             | 5.7       | 10.2                |
| [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)   | OpenHands  | 10.1           | 49.2      |  24.5                |

# Data

We built [OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1) in two stages: **supervised fine-tuning**, followed by **reinforcement learning**. 
Each stage required its own data pipeline – RL tasks (instructions, environments, and verifiers) and SFT traces from strong teacher agents completing tasks.

[OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT) is an SFT trace dataset containing approximately **15,200 traces** drawn from two different data sources we curate:
- **nl2bash**: Simple synthetically generated tasks where the agent has to format shell commands effectively
- **InferredBugs**: A set of bugs in C# and Java collected by Microsoft that we turned into tasks

We generate the traces using QuantTrio/GLM-4.6-AWQ and Terminus-2 agentic harness with a maximum of 32 turns. We use the default sampling parameters from vLLM and a maximum context length of 64K. Please see our example [data generation script](https://github.com/open-thoughts/OpenThoughts-Agent/blob/main/notebook/datagen_sft_tutorial.ipynb) for more details. 


[OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL) is an RL dataset containing ~720 tasks drawn from the **nl2bash verified** dataset.

To stabilize training, we built a three-stage filtration pipeline that prunes tasks before they ever hit the learner:

1. Bad verifiers filter: drop tasks with flaky or excessively slow verifiers.
2. Environment stability: remove tasks whose containers take too long to build or tear down.
Optional difficulty filter: discard tasks that even a strong model (GPT-5 Codex) cannot solve in a single pass.


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 16
- total_train_batch_size: 16
- total_eval_batch_size: 128
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.98) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 7.0

### Framework versions

- Transformers 4.56.0
- Pytorch 2.9.0+cu128
- Datasets 4.4.1
- Tokenizers 0.22.1


# Links
- 🌐 [OpenThoughts-Agent project page](https://open-thoughts.ai/blog/agent)
- 💻 [OpenThoughts-Agent GitHub repository](https://github.com/open-thoughts/OpenThoughts-Agent)
- 🧠 [OpenThoughts-Agent-v1-SFT dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
- 🧠 [OpenThoughts-Agent-v1-RL dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL)
- 🧠 [OpenThoughts-TB-dev dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev)
- 🤖 [OpenThinker-Agent-v1 model](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1)
- 🤖 [OpenThinker-Agent-v1-SFT model](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT) --> this model


# Citation
```
@misc{openthoughts-agent,
  author = {Team, OpenThoughts-Agent},
  month = Dec,
  title = {{OpenThoughts-Agent}},
  howpublished = {https://www.open-thoughts.ai/blog/agent},
  year = {2025}
}
```