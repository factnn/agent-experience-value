---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- agent
- code
- agentic-traces
- reinforcement-learning
- terminus-2
- harbor
- agent-traces
size_categories:
- 1M<n<10M
---

# AgentTrove

**AgentTrove** is the largest open-source collection of agentic interaction traces to date, released by the [OpenThoughts-Agent](https://www.open-thoughts.ai/blog/agent) team. It contains **1,696,847 rows** drawn from 219 source datasets spanning code repair, shell scripting, mathematical problem-solving, competitive programming, and general computer-use tasks.

At 1.7 million rows, AgentTrove is **4× the size of the [Nemotron Terminal Corpus](https://huggingface.co/datasets/laion/nemotron-terminal-corpus-unified)** (430 K rows), the previous largest open-source agentic trace dataset.

---

## Data Format

Consistent with the Nemotron Terminal Corpus, all traces in AgentTrove are released in the **terminus-2 harness format** — a ShareGPT-style conversation layout where each row represents a complete agent trajectory, including tool calls, environment responses, and final reasoning.

All traces were generated using the open-source **[Harbor](https://github.com/harbor-framework/harbor)** agentic evaluation and data-generation framework.

---

## Schema

Each row contains the following columns. Note that source datasets have heterogeneous schemas; columns not present in a given source are filled with `null`.

| Column | Type | Description |
|--------|------|-------------|
| `messages` | `list[dict]` | The complete agent trajectory as a list of conversation turns. Each turn is a dict with `role` (`"user"`, `"assistant"`, or `"tool"`) and `content` (string). Follows the terminus-2 / ShareGPT format used by Harbor. |
| `original_source` | `string` | Label identifying the **task source** the row came from, extracted from the source dataset's repo name (e.g. `"swesmith"`, `"codeforces"`, `"nl2bash"`, `"r2egym"`, `"exp_rpt"`, `"exp_tas"`). |
| `original_teacher` | `string` | Label identifying the **teacher model** that generated the trace (e.g. `"GLM-4.7"`, `"GLM-4.6"`, `"GPT 5.1 Nano"`, `"Kimi K2.0 Thinking"`, `"MiniMax M2.0"`, `"Qwen3-8B"`). |
| `reward` | `float` | Pass/fail outcome of the agent trajectory, typically `1.0` (success) or `0.0` (failure). Present in most but not all source datasets. |
| `task_id` | `string` | Unique identifier for the task instance. Format varies by source dataset. |
| *(other columns)* | *varies* | Additional metadata columns (e.g. `trajectory_id`, `episode`, `model`, `sandbox_id`) preserved from source datasets where present. |

---

## Source Datasets

AgentTrove was assembled by concatenating training datasets generated throughought the course of the OpenThoughts-Agent project. The full list is provided below.

| repo_id | source | teacher |
|---------|--------|---------|
| DCAgent/code_contests_10k_OG_10k_New_Questions_GPT5-mini | code_contests | GPT-5-mini |
| mlfoundations-dev/staqc-sandboxes-traces-terminus-2 | staqc | GPT 5.1 Nano |
| DCAgent/freelancer-askllm-filtered-sandboxes-traces-terminus-2 | freelancer | GPT 5.1 Nano |
| DCAgent/nl2bash-8ep | nl2bash | GPT 5.1 Nano |
| penfever/GLM-4.6-codeforces-32ep-32k | codeforces | GLM-4.6 |
| DCAgent/freelancer-t2048s-32ep | freelancer | GPT 5.1 Nano |
| DCAgent/neulab-mind2web-sandboxes-traces-terminus-2 | neulab | GPT 5.1 Nano |
| DCAgent/freelancer-long-instruction-filter | freelancer | GPT 5.1 Nano |
| DCAgent/nl2bash-1ep | nl2bash | GPT 5.1 Nano |
| DCAgent/nl2bash-16ep | nl2bash | GPT 5.1 Nano |
| EtashGuha/nl2bash_verified_gpt-5-nano-traces | nl2bash | GPT-5-nano |
| penfever/nl2bash-verified-GLM-4.6-traces-32ep-32k | nl2bash | GLM-4.6 |
| penfever/GLM-4.6-stackexchange-overflow-sandboxes-32eps-32k | stack_exchange | GLM-4.6 |
| penfever/GLM-4.6-inferredbugs-32eps-65k | Inferred Bugs | GLM-4.6 |
| penfever/GLM-4.6-nl2bash-verified-32eps-32k | nl2bash | GLM-4.6 |
| DCAgent/freelancer-t512s-32ep | freelancer | GPT 5.1 Nano |
| penfever/glm46-qasper-maxeps-131k | Qasper | GLM-4.6 |
| penfever/glm46-neulab-synatra-32ep-131k | neulab | GLM-4.6 |
| DCAgent2/glm46-r2egym_sandboxes-maxeps-131k | r2egym | GLM-4.6 |
| penfever/GLM-4.6-swesmith-32ep-131k-nosumm-reasoning | swesmith | GLM-4.6 |
| DCAgent/exp_rle_detailed | exp_rle | GLM-4.7 |
| DCAgent/nl2bash-GLM-4.6-traces | nl2bash | GLM-4.6 |
| DCAgent/freelancer-t256s-32ep | freelancer | GPT 5.1 Nano |
| DCAgent/freelancer-t1024s-32ep | freelancer | GPT 5.1 Nano |
| penfever/GLM-4.6-taskmaster2-32eps-32k | taskmaster | GLM-4.6 |
| mlfoundations-dev/codeforces-gptoss120b-traces | codeforces | GPT-OSS-120B |
| penfever/kimi-k2t-freelancer-32ep-32k | freelancer | Kimi K2.0 Thinking |
| penfever/glm-4.6-nemo-prism | Nemotron Prism | GLM-4.6 |
| penfever/glm-4.6-all-puzzles-32ep-131k | puzzles | GLM-4.6 |
| penfever/glm46-swesmith-maxeps-131k | swesmith | GLM-4.6 |
| DCAgent2/glm46-swegym-tasks-maxeps-131k | SWEGym | GLM-4.6 |
| DCAgent/exp_tas_low_diversity_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_min_p_0.1_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_parser_xml_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_rpt_curriculum-medium | exp_rpt | GLM-4.7 |
| DCAgent/neulab-synatra-sandboxes-traces-terminus-2 | neulab | GPT 5.1 Nano |
| mlfoundations-dev/code-contests-sandboxes-traces-terminus-2 | code_contests | GPT 5.1 Nano |
| DCAgent/exp_rpt_crosscodeeval-csharp | exp_rpt | GLM-4.7 |
| DCAgent/taskmaster2-2ep | taskmaster | GPT 5.1 Nano |
| hbXNov/all-puzzles-sandboxes-traces-terminus-2-with-gpt-4o-mini-judgments-correct | puzzles | GPT 5.1 Nano |
| DCAgent/freelancer-projects-0-3k-traces | freelancer | GPT 5.1 Nano |
| DCAgent/freelancer-embedding-mean-instruction-filter | freelancer | GPT 5.1 Nano |
| DCAgent/nl2bash-2ep | nl2bash | GPT 5.1 Nano |
| DCAgent/exp_rle_error_report | exp_rle | GLM-4.7 |
| DCAgent/taskmaster2-1ep | taskmaster | GPT 5.1 Nano |
| DCAgent/wikitable_format_conversion-qwen3-coder-480b-a35b-instruct-awq-traces | WikiTable Format Conversion | Qwen3 |
| DCAgent/freelancer-projects-0-1k-traces | freelancer | GPT 5.1 Nano |
| DCAgent/freelancer-short-instruction-filter | freelancer | GPT 5.1 Nano |
| DCAgent/nl2bash-4ep | nl2bash | GPT 5.1 Nano |
| DCAgent/nl2bash-32ep | nl2bash | GPT 5.1 Nano |
| DCAgent2/codeforces-GLM-4.6-traces-32ep-32k-1-2-4-dv | codeforces | GLM-4.6 |
| DCAgent/taskmaster2-4ep | taskmaster | GPT 5.1 Nano |
| mlfoundations-dev/swesmith_with_plain_docker-sandboxes-traces-terminus-2 | swesmith | GPT 5.1 Nano |
| DCAgent/code_contests-Qwen3-Coder-480B-traces | code_contests | Qwen3 |
| DCAgent/freelancer-projects-1k-traces | freelancer | GPT 5.1 Nano |
| DCAgent/nl2bash-0-1k-traces | nl2bash | GPT 5.1 Nano |
| DCAgent/nl2bash_gpt-5-nano-traces | nl2bash | GPT-5-nano |
| penfever/inferredbugs-GLM-4.6-32ep-32k | Inferred Bugs | GLM-4.6 |
| DCAgent/bash_textbook_tasks_traces | bash textbook | GPT 5.1 Nano |
| DCAgent/gsm8k-qwen3-coder-30b-a3b-instruct-traces | gsm8k | Qwen3 |
| DCAgent/nl2bash-0-3k-traces | nl2bash | GPT 5.1 Nano |
| DCAgent/code_contests-GLM-4.6-traces | code_contests | GLM-4.6 |
| penfever/gpt-oss-120B-stack-overflow-32ep-131k-summtrc-fixthink1 | StackExchange Overflow | GPT-OSS-120B |
| penfever/minimax-m2-stack-overflow-32ep-131k-summtrc | StackExchange Overflow | MiniMax M2.0 |
| penfever/kimi-k2t-neulab-synatra-32ep-131k | neulab | Kimi K2.0 Thinking |
| DCAgent/taskmaster2-8ep | taskmaster | GPT 5.1 Nano |
| DCAgent/freelancer-projects-3k-traces | freelancer | GPT 5.1 Nano |
| DCAgent/nl2bash-1k-traces | nl2bash | GPT 5.1 Nano |
| DCAgent/All_Puzzles_5k_OG_5k_New_Context_GPT4o-mini_new_context | puzzles | GPT-4o |
| DCAgent/taskmaster2-banana | taskmaster | GPT 5.1 Nano |
| DCAgent/taskmaster2-1k-traces | taskmaster | GPT 5.1 Nano |
| penfever/glm46-defects4j-32ep-131k | defects 4j | GLM-4.6 |
| penfever/GLM-4.6-nl2bash-verified-32ep-32k-reasoning | nl2bash | GLM-4.6 |
| penfever/kimi-k2-swegym-tasks-maxeps-32k | SWEGym | Kimi K2.0 Thinking |
| penfever/kimi-k2-swesmith_with_plain_docker-sandboxes-maxeps-32k | swesmith | Kimi K2.0 Thinking |
| penfever/glm-4.6-freelancer-32ep-131k-torch | freelancer | GLM-4.6 |
| DCAgent/taskmaster2-0-1k-traces | taskmaster | GPT 5.1 Nano |
| DCAgent/taskmaster2-3k-traces | taskmaster | GPT 5.1 Nano |
| DCAgent/taskmaster2-gpt5mini | taskmaster | GPT-5-mini |
| DCAgent/taskmaster2-16ep | taskmaster | GPT 5.1 Nano |
| DCAgent/freelancer-projects-gpt5mini | freelancer | GPT-5-mini |
| DCAgent/neulab-codeactinstruct-sandboxes-traces-terminus-2 | neulab | GPT 5.1 Nano |
| DCAgent/taskmaster2-32ep | taskmaster | GPT 5.1 Nano |
| DCAgent/nl2bash-Qwen3-Coder-480B-traces | nl2bash | Qwen3 |
| DCAgent/freelancer-projects-10k-traces | freelancer | GPT 5.1 Nano |
| DCAgent/freelancer-projects-gpt5 | freelancer | GPT-5 |
| DCAgent/nl2bash-3k-traces | nl2bash | GPT 5.1 Nano |
| penfever/inferredbugs-GLM-4.6-32ep-65k | Inferred Bugs | GLM-4.6 |
| penfever/GLM-4.6-stackexchange-overflow-sandboxes-32eps-65k | stack_exchange | GLM-4.6 |
| DCAgent/taskmaster2-64ep | taskmaster | GPT 5.1 Nano |
| DCAgent/code_contests_new_questions_gpt-5-mini | code_contests | GPT-5-mini |
| DCAgent/freelancer-random-instruction-filter-traces-terminus-2 | freelancer | GPT 5.1 Nano |
| mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2 | freelancer | GPT 5.1 Nano |
| DCAgent2/freelancer-projects-31k-traces | freelancer | GPT 5.1 Nano |
| penfever/glm-4.6-r2egym-32ep-32k | r2egym | GLM-4.6 |
| DCAgent2/freelancer-projects-100k-traces | freelancer | GPT 5.1 Nano |
| DCAgent/exp_rle_github_issue | exp_rle | GLM-4.7 |
| DCAgent2/gemini25flash-stackexchange-overflow-32ep-512k-v3-traces | stack_exchange | Gemini-2.5-Flash |
| DCAgent2/glm-4.6-freelancer-traces | freelancer | GLM-4.6 |
| penfever/GLM-4.6-codeforces-32eps-32k | codeforces | GLM-4.6 |
| penfever/Qwen3-Coder-480B-nl2bash | nl2bash | Qwen3 |
| penfever/GPT-OSS-120B-codeforces | codeforces | GPT-OSS-120B |
| penfever/r2egym_gpt5_codex_solve_traces | r2egym | GPT-5 |
| penfever/GLM-4.6-stackexchange-superuser-32ep-32k | stack_exchange | GLM-4.6 |
| DCAgent/All_Puzzles_5k_new_context | puzzles | GPT 5.1 Nano |
| DCAgent/All_Puzzles_5k_New_Context_GPT4o-mini_new_context | puzzles | GPT-4o |
| mlfoundations-dev/stackexchange-tezos-sandboxes-traces-terminus-2 | stack_exchange | GPT 5.1 Nano |
| DCAgent/exp_rle_heavy_padding | exp_rle | GLM-4.7 |
| mlfoundations-dev/defects4j-sandboxes-traces-terminus-2 | defects 4j | GPT 5.1 Nano |
| penfever/Qwen3-Coder-480B-codeforces | codeforces | Qwen3 |
| mlfoundations-dev/nemo-prism-math-sandboxes-traces-terminus-2 | math | GPT 5.1 Nano |
| DCAgent/exp_tas_repetition_penalty_1.05_traces | exp_tas | GLM-4.7 |
| DCAgent/a1_multifile_composition | multifile composition | GPT 5.1 Nano |
| DCAgent/exp-gfi-staqc-embedding-mean-filtered-10K_glm_4.7_traces_jupiter | staqc | GLM-4.7 |
| DCAgent/exp_tas_max_episodes_512_traces | exp_tas | GLM-4.7 |
| DCAgent/a1_repo_scaffold | repo scaffold | GPT 5.1 Nano |
| DCAgent/swesmith-sandboxes-with_tests-gpt-5-mini-passed_glm_4.7_traces | swesmith | GLM-4.7 |
| penfever/Kimi-2.5-r2egym_sandboxes-maxeps-32k | r2egym | Kimi-2.5 |
| DCAgent/exp_rle_proportional | exp_rle | GLM-4.7 |
| DCAgent/exp_rpt_stack-junit | exp_rpt | GLM-4.7 |
| DCAgent/selfinstruct-naive-sandboxes-2-traces | Self-Instruct Naive | GPT 5.1 Nano |
| DCAgent/exp_tas_repetition_penalty_1.05_traces-3pct | exp_tas | GLM-4.7 |
| DCAgent/a1_multifile_composition-3pct | multifile composition | GPT 5.1 Nano |
| DCAgent/exp-gfi-staqc-embedding-mean-filtered-10K_glm_4.7_traces_jupiter-3pct | staqc | GLM-4.7 |
| DCAgent/exp_tas_max_episodes_512_traces-3pct | exp_tas | GLM-4.7 |
| DCAgent/a1_repo_scaffold-3pct | repo scaffold | GPT 5.1 Nano |
| DCAgent/swesmith-sandboxes-with_tests-gpt-5-mini-passed_glm_4.7_traces-3pct | swesmith | GLM-4.7 |
| penfever/Kimi-2.5-r2egym_sandboxes-maxeps-32k-3pct | r2egym | Kimi-2.5 |
| DCAgent/exp_rpt_stack-selfdoc-v2 | exp_rpt | GLM-4.7 |
| DCAgent/exp_rle_structural_debug | exp_rle | GLM-4.7 |
| penfever/glm46-neulab-agenttuning-alfworld-sandboxes-maxeps-131k | neulab | GLM-4.6 |
| DCAgent/stackexchange-tezos-sandboxes_glm_4.6_traces_locetash | stack_exchange | GLM-4.6 |
| DCAgent2/glm46-glaive-code-assistant-sandboxes-maxeps-131k | Glaive Code Assistant | GLM-4.6 |
| penfever/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k-reward1 | swesmith | Kimi-2.5 |
| penfever/Kimi-2.5-r2egym_sandboxes-maxeps-32k-reward1 | r2egym | Kimi-2.5 |
| DCAgent/stackexchange-tezos-sandboxes_glm_4.6_traces_together | stack_exchange | GLM-4.6 |
| penfever/kimi-k2-r2egym_sandboxes-maxeps-32k | r2egym | Kimi K2.0 Thinking |
| DCAgent/glm46-Magicoder-Evol-Instruct-110K-sandboxes-1-traces | MagiCoder Evol Instruct | GLM-4.6 |
| DCAgent/stackexchange-tezos-sandboxes_glm_4.6_traces_together_again | stack_exchange | GLM-4.6 |
| DCAgent2/glm46-stackexchange-tezos-maxeps-131k | stack_exchange | GLM-4.6 |
| DCAgent/exp_tas_linear_history_off_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_rpt_stack-bash-withtests | exp_rpt | GLM-4.7 |
| DCAgent/stackexchange-tezos-sandboxes_glm_4.7_traces_locetash | stack_exchange | GLM-4.7 |
| DCAgent/exp_tas_baseline_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_full_thinking_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_interleaved_thinking_on_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_frequency_penalty_0.5_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_frequency_penalty_0.25_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_presence_penalty_0.25_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_max_tokens_1024_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_max_tokens_8192_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_presence_penalty_1.0_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_max_episodes_32_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_raw_content_off_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_frequency_penalty_1.0_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_min_p_0.05_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_top_k_16_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_min_p_0.01_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_max_tokens_4096_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_top_k_64_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_summarize_threshold_2048_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_summarize_off_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_repetition_penalty_1.2_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_top_k_128_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_high_diversity_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_max_tokens_2048_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_top_p_0.95_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_temp_0.5_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_top_p_0.8_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_top_p_0.9_traces | exp_tas | GLM-4.7 |
| DCAgent/exp_tas_summarize_threshold_16384_traces | exp_tas | GLM-4.7 |
| DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k | r2egym | GLM-4.7 |
| DCAgent/stackexchange-tezos-sandboxes_glm_4.6_traces_locetash_again | stack_exchange | GLM-4.6 |
| DCAgent/perturbed-docker-exp-taskmaster2-tasks_glm_4.7_traces_locetash | taskmaster | GLM-4.7 |
| DCAgent/exp_tas_repetition_penalty_1.05_traces-10pct | exp_tas | GLM-4.7 |
| DCAgent/a1_multifile_composition-10pct | multifile composition | GPT 5.1 Nano |
| DCAgent/exp-gfi-staqc-embedding-mean-filtered-10K_glm_4.7_traces_jupiter-10pct | staqc | GLM-4.7 |
| DCAgent/exp_tas_max_episodes_512_traces-10pct | exp_tas | GLM-4.7 |
| DCAgent/a1_repo_scaffold-10pct | repo scaffold | GPT 5.1 Nano |
| DCAgent/swesmith-sandboxes-with_tests-gpt-5-mini-passed_glm_4.7_traces-10pct | swesmith | GLM-4.7 |
| penfever/Kimi-2.5-r2egym_sandboxes-maxeps-32k-10pct | r2egym | Kimi-2.5 |
| DCAgent/exp_rpt_nemotron-junit | exp_rpt | GLM-4.7 |
| DCAgent/exp_rpt_nemotron-cpp | exp_rpt | GLM-4.7 |
| DCAgent/exp_rpt_stack-cpp | exp_rpt | GLM-4.7 |
| DCAgent/exp_rle_2skill | exp_rle | GLM-4.7 |
| DCAgent/exp_rle_curated | exp_rle | GLM-4.7 |
| DCAgent/exp_rle_expert | exp_rle | GLM-4.7 |
| DCAgent/exp_flat25_baseline | unknown | GLM-4.7 |
| DCAgent/exp_rle_minimal_instructions | exp_rle | GLM-4.7 |
| DCAgent/exp_rpt_stack-pytest-large | exp_rpt | GLM-4.7 |
| DCAgent/exp_rpt_stack-csharp | exp_rpt | GLM-4.7 |
| DCAgent/exp_rpt_methods2test-v2 | exp_rpt | GLM-4.7 |
| DCAgent/exp_rpt_stack-pytest-v2 | exp_rpt | GLM-4.7 |
| DCAgent/g1_timeout_e1_gpt_long_d1_original_40k_glm47_traces | unknown | GLM-4.7 |
| DCAgent/g1_subagent_e1_gpt_long_d1_original_40k_glm47_traces | unknown | GLM-4.7 |
| DCAgent/g1_min_episodes_e1_gpt_long_d1_original_40k_glm47_traces | unknown | GLM-4.7 |
| DCAgent/swesmith-glm5-awq-traces-10k | swesmith | GLM-5.0 |
| DCAgent/exp-syh-r2egym-askllm-constrained_glm_4.7_traces_jupiter_cleaned | r2egym | GLM-4.7 |
| DCAgent/exp-gfi-swesmith-random-filtered-10K_glm_4.7_traces_jupiter | swesmith | GLM-4.7 |
| DCAgent/mix_v2_h2_language_balanced | unknown | GPT 5.1 Nano |
| DCAgent/stackexchange-unix-sandboxes_glm_4.7_traces_jupiter | stack_exchange | GLM-4.7 |
| DCAgent/g1_timeout_e1_gpt_long_sampled_swesmith_psu_d1_original_40k_glm47_traces | swesmith | GLM-4.7 |
| DCAgent/g1_min_episodes_e1_gpt_long_sampled_swesmith_psu_d1_original_40k_glm47_traces | swesmith | GLM-4.7 |
| DCAgent/g1_min_episodes_e1_gpt_long_d1_original_8x_glm47_traces | unknown | GLM-4.7 |
| DCAgent/taskmaster2-0-3k-traces | taskmaster | GPT 5.1 Nano |
| penfever/GLM-4.6-gemini25flash-stackexchange-overflow-32ep-512k | stack_exchange | GLM-4.6 |
| DCAgent/neulab-code-feedback-sandboxes-traces-terminus-2 | neulab | GPT 5.1 Nano |
| DCAgent/taskmaster2-10k-traces | taskmaster | GPT 5.1 Nano |
| mlfoundations-dev/taskmaster2-sandboxes-traces-terminus-2 | taskmaster | GPT 5.1 Nano |
| mlfoundations-dev/codeforces-sandboxes-traces-terminus-2 | codeforces | GPT 5.1 Nano |
| mlfoundations-dev/all-puzzles-sandboxes-traces-terminus-2 | puzzles | GPT 5.1 Nano |
| mlfoundations-dev/inferredbugs-sandboxes-traces-terminus-2 | Inferred Bugs | GPT 5.1 Nano |
| mlfoundations-dev/stackexchange-overflow-sandboxes-traces-terminus-2 | stack_exchange | GPT 5.1 Nano |
| DCAgent/exp_rle_partial_ambiguity | exp_rle | GLM-4.7 |
| mlfoundations-dev/qasper-sandboxes-traces-terminus-2 | Qasper | GPT 5.1 Nano |
| DCAgent/exp_rle_moderate_padding | exp_rle | GLM-4.7 |
| DCAgent/exp_rpt_exercism-python | exp_rpt | GLM-4.7 |
| mlfoundations-dev/stackexchange-codereview-sandboxes-traces-terminus-2 | stack_exchange | GPT 5.1 Nano |
| DCAgent/e1_gpt_long_d1_original_sandboxes_8x_glm47_traces | unknown | GLM-4.7 |


---

## Citation

If you use AgentTrove in your research, please cite:

```bibtex
@misc{openthoughts-agent,
  author = {Team, OpenThoughts-Agent},
  month = Dec,
  title = {{OpenThoughts-Agent}},
  howpublished = {https://www.open-thoughts.ai/blog/agent},
  year = {2025}
}
```