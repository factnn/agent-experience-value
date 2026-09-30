---
language:
- en
license: apache-2.0
task_categories:
- text-generation
size_categories:
- 1M<n<10M
tags:
- agent
- code
- agentic-tasks
- harbor
- reinforcement-learning
- swe-bench
configs:
- config_name: default
  data_files:
  - split: train
    path: "*/tasks.parquet"
- config_name: deprecated
  data_files:
  - split: train
    path: "deprecated/*/tasks.parquet"
---

# TaskTrove

> **v5.1 (current)** — independent-review source retirement — moves 15 sources with majority or unanimous REJECT verdicts out of the default config and into `deprecated/`. Three blinded reviewers each sampled 10 tasks per source from all 50 v5.0 source-drop candidates, read the instructions and packaged tests, and issued independent KEEP or REJECT verdicts. The 15 retired sources received at least two REJECT votes. The active catalog changes from 93 sources and 1,674,033 tasks to 78 sources and 1,604,128 tasks.
>
> The full synthesis, all three reviewer reports, the blinded source list, and a machine-readable release audit are stored under `audits/v5.1/` and `v5.1-source-retirement-audit.json`. The retired Parquets are byte-identical to their v5.0 versions.
>
> **v5.0** — audited row filtration — retains all 93 active v4.15 sources and removes 65,293 task-level defects: exact within-source instruction duplicates, tasks whose prompts expose all hidden stdio gold cases, null graders, independently reviewed task defects, graders that accept empty output, and graders whose expected answer leaks into the instruction. The active catalog changes from 1,739,326 to 1,674,033 tasks.
>
> Source-level drop recommendations are not applied; those sources remain intact pending individual audits. Rows labeled `unsupported_variant` also remain because that label describes conversion coverage, not task validity. `v5.0-filtration-audit.json` records the exact policy, counts, source hashes, and output hashes; `v5.0-filtration-ledger.parquet` preserves every audit finding.
>
> **v4.15** — SWE-Gym verifier batching — changes all 2,428 retained SWE-Gym tasks from one pytest process per configured node ID to one process per nonempty pass-to-pass or fail-to-pass group. This reduces the static process count from 146,435 to 4,628 (31.6x) without changing task membership, test targets, trusted-test restoration, setup reuse, or fail-closed reward handling. Batched pytest runs use fail-fast mode to preserve the old verifier's first-failure exit behavior.
>
> R2E-Gym, SWE-Smith, OpenSWE, and SWE-ReBench already make one task-level test invocation; this release leaves their archives byte-for-byte unchanged. Synthetic behavior tests cover passing, failing, and missing targets. The release audit records executable old/new oracle equivalence probes on selected real SWE-Gym tasks.
>
> `v4.15-swegym-verifier-batching.json` records the source audit, process counts, candidate hash, and executable evidence.
>

> **v4.14 (superseded)** — clone-verifier alignment — audits all 48,003 tasks in the five active sources that require the agent to clone a repository. The release hardens and retains 47,978 of those tasks and removes 25 tasks. TaskTrove now contains 1,739,326 active (non-deprecated) tasks and 1,966,088 packaged tasks, including deprecated sources.
>
> Every retained task requires its expected workspace path to be a Git worktree containing the task's pinned base commit; `HEAD` may contain the agent's solution. Before testing, the verifier replaces paths that can change test discovery or execution, including test directories and test-runner configuration. This overwrites uncommitted and committed agent changes plus untracked and ignored files. SWE-smith uses the task commit for existing tests and a pinned default-branch commit only for tests absent at the task commit. R2E-Gym replaces its packaged test directory instead of merging into an agent-created directory. SWE-Gym installs the verifier-test patch embedded in the packaged oracle archive and fails when a configured test is missing. SWE-ReBench and OpenSWE keep their packaged hidden-test patches; OpenSWE tasks without a patch run tests from the pinned base commit.
>
> R2E-Gym's verifier dependencies and SWE-smith's pytest dependency now come from their task images. OpenSWE and SWE-Gym skip repeated setup when the setup command reached its completion marker during the agent phase; absent markers trigger verifier-side setup. The marker does not skip the Git gate, trusted-test restoration, or test execution. OpenSWE activates its named Conda environment directly and never sources the agent-writable environment snapshot. SWE-ReBench performs its Git gate before package installation.
>
> Five SWE-Gym tasks had no configured tests. Five other SWE-Gym tasks and 15 SWE-smith tasks require the solution to modify a test-control path such as `conftest.py` or a test plugin. Restoring those paths would undo or supply part of the intended solution; leaving them mutable would permit verifier bypass. The release removes all 25 tasks.
>
> Static validation transformed all 48,003 source tasks without an unexpected archive or shell error. One executable probe per source scored 0 for a no-op, 0 after deleting tests and injecting test-runner configuration, and 1 for the packaged oracle. `v4.14-clone-verifier-audit.json` records the per-source transformations, purge IDs, probe output, and verifier timing.
>

> **v4.13 (superseded)** — corrected structural and no-op audit — restores 97,621 tasks that v4.12 incorrectly rejected for concise instructions or intentional empty fixture files. It removes 5,426 confirmed-invalid tasks and leaves 1,739,351 active tasks across 93 sources. The complete packaged dataset contains 1,966,113 tasks.
>
> The no-op audit covers 1,744,475 corrected active candidates: 207,028 heterogeneous tasks were executed individually, 461,305 were covered by byte-verified early empty-output gates, and 1,076,142 shared-template tasks were checked through 29 exact verifier/environment signatures. Every positive-reward task or signature was removed. Verifier and environment failures were removed unless a source-wide fail-closed repair was mechanically valid.
>
> These are separate outcomes relative to v4.9's active catalog: 5,426 rows were removed from packaged data, 20,000 intact Qasper/STaQC rows moved from active to deprecated, and 23,251 retained rows received fail-closed repairs. Source counts include only nonempty packaged Parquets; the CSV ledger retains zero-row source records for provenance. Machine-readable audit evidence is included as `v4.13-audit-summary.json`.
>
> `laion/exp_rpt_stack-pytest-large-v3` now maps ordinary pytest collection and zero-test failures to reward 0 while preserving dependency, malformed-test, and unexpected runner failures as infrastructure errors.
>
> `laion/nemotron-gym-qa-abstention-v4` now emits reward 0 before invoking its exact-match grader when `/app/answer.txt` is missing or empty.
>
> `DCAgent/swe_rebench_v2_patched_oracle-v2` now emits reward 0 when the agent has not cloned the assigned repository into `/testbed`, before the trusted hidden-test patch is installed.
>
> David Hall's review of the original 10,000-task `qasper-v3` source is quoted verbatim below. No public link or date was supplied with the review.
>
> > It’s supposed to be a reading comprehension task: the agent gets a paper and a question, answers it, and an LM grades the answer. But 49.51%(!) of the 10,000 tasks are missing the paper entirely, so they’re effectively exercises in hallucination.
>
> > Furthermore, 74.07% of the tasks are `_copyN` duplicates. Their agent-visible instructions and questions are byte-for-byte identical to another task; spot checks also found identical complete archives.
>
> > Finally, even when the paper is present, the GPT-4o-mini judge receives no gold answer, reference answer, or evidence spans. It gets essentially the same paper and question as the agent and must independently re-solve the task before grading the candidate. So the reward measures whether 4o-mini finds the answer plausible based on its own attempt. This is just distillation with extra steps.
>
> `qasper-v3` and `staqc-v4` move to the deprecated config with their complete 10,000-task populations. Our verifier review found that STaQC's short prompts feed a generic plausibility judge with no executable workspace contract or reference answer; restoring its rows does not make that verifier meaningful for agent-capability training.
>
> Purge tally by affected source:
>
> - `DCAgent__exp_rle_adversarial-v6`: 2,730 → 2,726 (verifier_execution_error=4)
> - `DCAgent__exp_rpt_curriculum-easy`: 514 → 509 (noop_positive_reward=5)
> - `DCAgent__exp_rpt_curriculum-medium-v2`: 495 → 492 (noop_positive_reward=3)
> - `DCAgent__exp_rpt_e2egit-large`: 5,000 → 4,998 (noop_positive_reward=2)
> - `DCAgent__exp_rpt_e2egit-v2`: 500 → 487 (noop_positive_reward=13)
> - `DCAgent__exp_rpt_multifile-v3`: 4,884 → 4,843 (noop_positive_reward=41)
> - `DCAgent__exp_rpt_nemotron-cpp`: 5,000 → 4,196 (noop_positive_reward=802, verifier_execution_error=2)
> - `DCAgent__inferredbugs-sandboxes-verifier`: 10,000 → 9,659 (missing_verifier=9, noop_positive_reward=22, verifier_execution_error=310)
> - `DCAgent__mix_h4_binary_easy`: 2,010 → 1,996 (noop_positive_reward=14)
> - `DCAgent__selfinstruct-naive-sandboxes-2-verified-v3`: 7,132 → 6,665 (noop_positive_reward=383, verifier_execution_error=84)
> - `DCAgent__swe_rebench_v2_patched_oracle-v2`: 18,334 → 18,319 (invalid_environment=15)
> - `laion__exp_rpt_methods2test-large-v4`: 1,203 → 1,194 (noop_positive_reward=9)
> - `laion__exp_rpt_nemotron-junit-v6`: 493 → 447 (noop_positive_reward=46)
> - `laion__exp_rpt_scaffold-v3`: 3,136 → 3,121 (noop_positive_reward=15)
> - `laion__exp_rpt_stack-cpp-v4`: 7,896 → 7,878 (noop_positive_reward=18)
> - `laion__exp_rpt_stack-dockerfile-gpt5mini-v7`: 592 → 587 (noop_positive_reward=5)
> - `laion__exp_rpt_stack-go-v5`: 2,313 → 2,275 (noop_positive_reward=38)
> - `laion__exp_rpt_stack-jest-v5`: 465 → 424 (noop_positive_reward=41)
> - `laion__exp_rpt_stack-junit-v6`: 872 → 843 (noop_positive_reward=29)
> - `laion__exp_rpt_stack-php-v2-v8`: 403 → 0 (invalid_environment=403)
> - `laion__exp_rpt_stack-pytest-large-v3`: 1,783 → 1,782 (noop_positive_reward=1)
> - `laion__exp_rpt_stack-rspec-v4`: 8,960 → 8,860 (noop_positive_reward=81, verifier_execution_error=19)
> - `laion__mix_h10_reward_proportional-v2`: 2,873 → 2,858 (noop_positive_reward=15)
> - `laion__mix_h11_single_skill_only-v2`: 2,873 → 2,859 (noop_positive_reward=14)
> - `laion__mix_h8_original_tests-v2`: 2,862 → 2,848 (noop_positive_reward=14)
> - `laion__openswe-tasks-patched-v7-oracle-success`: 13,945 → 11,730 (noop_positive_reward=534, verifier_execution_error=1681)
> - `laion__r2egym-patched-full-oracle-v3`: 3,328 → 2,574 (empty_issue_description=293, noop_positive_reward=461)
>

> **v4.12 (superseded)** — missing-component purge — removes 97,923 of 1,764,777 active tasks (5.549%) from 41 of 96 active sources. Deprecated sources are unchanged.
>
> An active task is removed when its `instruction.md` is absent or invalid UTF-8, or its task-specific problem text is shorter than 100 characters after trimming whitespace; when `tests/` or `environment/` is absent or has no regular files; or when either directory contains a zero-byte regular file.
>
> For `laion__r2egym-patched-full-oracle-v3`, task-specific problem text is the `<issue_description>` field. For other active sources, it is the instruction after removing exact lines found in at least 90% of a deterministic 100-task source sample. All deprecated sources are preserved without audit.
>
> **v4.11 (superseded)** — do not use. It removed hidden empty `r2egym` issue descriptions, but measured the complete instruction for other sources and therefore retained concise tasks padded by repeated boilerplate.
>
> **v4.10 (superseded)** — do not use. It audited deprecated sources and retained 293 `r2egym` tasks whose complete instructions exceeded 100 characters but whose embedded `<issue_description>` fields did not.
>
> - `DCAgent2__nl2bash-tasks-cleaned-oracle-v2`: 1,498 → 1,046 (452 removed; 30.174%)
> - `DCAgent__inferredbugs-sandboxes-verifier`: 10,000 → 9,991 (9 removed; 0.090%)
> - `DCAgent__selfinstruct-naive-sandboxes-2-verified-v3`: 7,132 → 7,131 (1 removed; 0.014%)
> - `SankalpKJ__nemotron-code-oracle-filtered`: 15,165 → 15,102 (63 removed; 0.415%)
> - `SankalpKJ__nemotron-math-oracle-filtered-v2`: 57,777 → 48,552 (9,225 removed; 15.967%)
> - `laion__all-puzzles-v2`: 6,926 → 6,860 (66 removed; 0.953%)
> - `laion__codeelo-v2`: 500 → 499 (1 removed; 0.200%)
> - `laion__codeforces-v3`: 10,000 → 9,953 (47 removed; 0.470%)
> - `laion__exp_rpt_crosscodeeval-csharp-v4`: 1,768 → 1,760 (8 removed; 0.452%)
> - `laion__exp_rpt_methods2test-large-v4`: 1,203 → 1,199 (4 removed; 0.333%)
> - `laion__exp_rpt_stack-pytest-large-v3`: 1,783 → 578 (1,205 removed; 67.583%)
> - `laion__exp_rpt_taco-v2`: 10,000 → 9,816 (184 removed; 1.840%)
> - `laion__glaive-code-assistant-sandboxes-verified-v2`: 10,000 → 6,023 (3,977 removed; 39.770%)
> - `laion__magicoder-v4`: 4,096 → 3,917 (179 removed; 4.370%)
> - `laion__nemo-prism-math-v3`: 2,404 → 2,240 (164 removed; 6.822%)
> - `laion__nemotron-gym-arc-agi-python-inductive-v2`: 10,000 → 9,999 (1 removed; 0.010%)
> - `laion__nemotron-gym-arc-agi-transductive-v3`: 10,000 → 9,999 (1 removed; 0.010%)
> - `laion__nemotron-gym-cfbench-v4`: 468 → 467 (1 removed; 0.214%)
> - `laion__nemotron-gym-competitive-coding-v2`: 15,713 → 15,636 (77 removed; 0.490%)
> - `laion__nemotron-gym-instruction-following-adversarial-v5`: 1,000 → 995 (5 removed; 0.500%)
> - `laion__nemotron-gym-instruction-following-v3`: 46,391 → 46,380 (11 removed; 0.024%)
> - `laion__nemotron-gym-inverse-ifeval-v4`: 1,000 → 995 (5 removed; 0.500%)
> - `laion__nemotron-gym-knowledge-openqa-v4`: 122,357 → 122,239 (118 removed; 0.096%)
> - `laion__nemotron-gym-math-advanced-calculations-v4`: 5,291 → 4,200 (1,091 removed; 20.620%)
> - `laion__nemotron-gym-math-openmathreasoning-v2`: 42,636 → 36,366 (6,270 removed; 14.706%)
> - `laion__nemotron-gym-math-stack-overflow-v3`: 110,730 → 82,595 (28,135 removed; 25.409%)
> - `laion__nemotron-gym-math-v5`: 4,096 → 4,026 (70 removed; 1.709%)
> - `laion__nemotron-gym-qa-abstention-v4`: 3,150 → 1,231 (1,919 removed; 60.921%)
> - `laion__nemotron-gym-reasoning-gym-v2`: 14,259 → 12,405 (1,854 removed; 13.002%)
> - `laion__nemotron-gym-safety-v3`: 89,066 → 56,731 (32,335 removed; 36.305%)
> - `laion__nemotron-gym-science-so-openq-v3`: 150,644 → 150,639 (5 removed; 0.003%)
> - `laion__openswe-tasks-patched-v7-oracle-success`: 13,945 → 13,944 (1 removed; 0.007%)
> - `laion__qasper-v3`: 10,000 → 9,905 (95 removed; 0.950%)
> - `laion__r2egym-patched-full-oracle-v3`: 3,328 → 3,035 (293 removed; 8.804%)
> - `laion__stackexchange-codereview-sandboxes-verified-v2`: 10,000 → 9,998 (2 removed; 0.020%)
> - `laion__stackexchange-overflow-sandboxes-verified-v2`: 10,000 → 9,890 (110 removed; 1.100%)
> - `laion__stackexchange-superuser-sandboxes-verified-v2`: 10,000 → 9,980 (20 removed; 0.200%)
> - `laion__stackexchange-tezos-sandboxes-verified-v2`: 10,000 → 9,960 (40 removed; 0.400%)
> - `laion__stackexchange-unix-sandboxes-verified-v2`: 10,000 → 9,967 (33 removed; 0.330%)
> - `laion__staqc-v4`: 10,000 → 167 (9,833 removed; 98.330%)
> - `laion__wizardlm-orca-v4`: 10,000 → 9,987 (13 removed; 0.130%)
>

**TaskTrove** is an open-source collection of agentic task datasets, released by the [OpenThoughts-Agent](https://www.open-thoughts.ai/blog/agent) team. It is the task complement to **[AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove)** — the agent traces in AgentTrove were generated by running models against these task datasets using the [Harbor](https://github.com/open-thoughts/OpenThoughts-Agent) framework.

> **v4.9** — trusted-test and calendar-verifier remediation — replaces three sources. The SWE-ReBench and OpenSWE verifiers restore hidden-test paths from the immutable base commit before applying the trusted patch, so agent edits cannot suppress or replace target tests; patch and setup failures now remain infrastructure failures rather than scoreable zeros. SWE-ReBench explicitly requests 4 GiB memory and 8 GiB storage; OpenSWE requests 4 GiB. The instruction-following calendar verifier now rejects pairwise overlaps using half-open intervals and reports both event IDs and intervals. The agent-calendar sibling was audited and already contained this check. The release uses 20 unique images, and versioned sources are hosted only inside TaskTrove. Superseded source versions remain available through earlier TaskTrove tags.
>
> - `DCAgent/swe_rebench_v2_patched_oracle` → `DCAgent/swe_rebench_v2_patched_oracle-v2`
> - `laion/openswe-tasks-patched-v6-oracle-success` → `laion/openswe-tasks-patched-v7-oracle-success`
> - `laion/nemotron-gym-instruction-following-calendar-v2` → `laion/nemotron-gym-instruction-following-calendar-v3` (5,673 retained; 2,714 without recoverable exact event names removed)
>
> **v4.8** — verifier and sandbox-memory remediation — replaces four sources. `DCAgent/exp_rle_adversarial-v6` records target exceptions and target import failures as scoreable zeros and removes the one statically invalid verifier task (2,731 → 2,730). `laion/exp_rpt_stack-php-large-v9` recognizes PHPUnit 10 successful-test output instead of treating it as a verifier crash. The C++, PHP-large, and PHP-v2 sources now explicitly request 4 GiB RAM, double the previous 2 GiB default. Every active source retains at least 300 tasks, the release uses 6 unique images, and versioned sources are hosted only inside TaskTrove. Superseded source versions remain available through earlier TaskTrove tags.
>
> - `DCAgent/exp_rle_adversarial-v5` → `DCAgent/exp_rle_adversarial-v6`
> - `laion/exp_rpt_stack-cpp-v3` → `laion/exp_rpt_stack-cpp-v4`
> - `laion/exp_rpt_stack-php-large-v8` → `laion/exp_rpt_stack-php-large-v9`
> - `laion/exp_rpt_stack-php-v2-v7` → `laion/exp_rpt_stack-php-v2-v8`
>
> **v4.7** — low-ceiling source remediation — replaces ten sources after 20-failure-trace audits. Math answers use bounded, typed, fail-closed equivalence; Stack tasks expose their exact executable test contract and use dependency-complete shared images; NL2Bash compares semantic command results; and ToolScale restores its isolated offline tool service. Every replacement retains at least 300 tasks and uses one image; the release adds 7 distinct images, within the 20-image limit. The 297-task `laion/nemotron-gym-agent-workplace-v2` source is removed from the default config, and its packaged Parquet is preserved under `deprecated/`. Superseded rewards are excluded, and versioned sources are hosted only inside TaskTrove.
>
> - `laion/nemo-prism-math-v2` → `laion/nemo-prism-math-v3` (10,000 → 2,404 tasks)
> - `SankalpKJ/nemotron-math-oracle-filtered` → `SankalpKJ/nemotron-math-oracle-filtered-v2` (114,280 → 57,777 tasks)
> - `laion/nemotron-gym-math-openmathreasoning` → `laion/nemotron-gym-math-openmathreasoning-v2` (112,867 → 42,636 tasks)
> - `laion/nemotron-gym-math-stack-overflow-v2` → `laion/nemotron-gym-math-stack-overflow-v3` (436,307 → 110,730 tasks)
> - `DCAgent2/nl2bash-tasks-cleaned-oracle` → `DCAgent2/nl2bash-tasks-cleaned-oracle-v2` (1,570 → 1,498 tasks)
> - `laion/exp_rpt_stack-cpp-v2` → `laion/exp_rpt_stack-cpp-v3` (9,943 → 7,896 tasks)
> - `laion/exp_rpt_stack-jest-v4` → `laion/exp_rpt_stack-jest-v5` (471 → 465 tasks)
> - `laion/exp_rpt_stack-php-v2-v6` → `laion/exp_rpt_stack-php-v2-v7` (438 → 403 tasks)
> - `laion/exp_rpt_stack-rspec-v3` → `laion/exp_rpt_stack-rspec-v4` (10,000 → 8,960 tasks)
> - `laion/toolscale-v3` → `laion/toolscale-v4` (4,048 → 4,048 tasks)
>
> - `laion/exp_rpt_stack-ruby-v3` is retired: the full population is an exact verifier-test subset of the repaired RSpec source.
> - `DCAgent/exp_rpt_stack-dockerfile-v3` is retired: the source has task-specific images and systematic verifier-contract and dependency defects.
>
> Retired active-source Parquets are preserved only in the `deprecated` config for provenance.
>
> **v4.6** — semantic-judge runtime repair — replaces the two remaining sources whose TaskTrove v4.4 evaluations exceeded the infrastructure-pass gate because Together returned empty or truncated structured judge output. Exact normalized reference matches now score deterministically; other answers retain semantic Harbor RewardKit grading with `together_ai/Qwen/Qwen3.5-9B`. RewardKit failures receive bounded backoff retries, and exhausted provider, parser, timeout, or transport failures still exit without a reward so Harbor can refire them. Every source has at least 300 tasks, the release retains the v4.5 eight-image set, and versions are hosted only inside TaskTrove.
>
> - `laion/nemotron-gym-knowledge-openqa-v3` → `laion/nemotron-gym-knowledge-openqa-v4` (122,357 tasks)
> - `laion/nemotron-gym-science-so-openq-v2` → `laion/nemotron-gym-science-so-openq-v3` (150,644 tasks)
>
> **v4.5** — GLM 5.2 low-ceiling remediation — replaces pure sources only after deterministic failure-trace audits and retires problematic mixed sources. Repaired verifiers fail closed: verifier, dependency, judge-provider, parsing, and timeout failures emit no reward, while genuine answer or test failures remain scoreable. LLM judging uses Harbor RewardKit with `together_ai/Qwen/Qwen3.5-9B`. Every retained source has at least 300 tasks, the release stays within the 20-image limit, and superseded rewards are excluded. New versions are hosted only inside TaskTrove.
>
> - `laion/exp_rpt_bugsinpy-v3` → `laion/exp_rpt_bugsinpy-v4` (500 → 479 tasks)
> - `laion/magicoder-v3` → `laion/magicoder-v4` (4,096 → 4,096 tasks)
> - `laion/nemotron-gym-instruction-following-adversarial-v4` → `laion/nemotron-gym-instruction-following-adversarial-v5` (1,000 → 1,000 tasks)
> - `laion/nemotron-gym-multichallenge-advanced-v3` → `laion/nemotron-gym-multichallenge-advanced-v4` (1,068 → 1,068 tasks)
> - `laion/nemotron-gym-qa-abstention-v3` → `laion/nemotron-gym-qa-abstention-v4` (3,150 → 3,150 tasks)
> - `laion/nemotron-gym-sysbench-v3` → `laion/nemotron-gym-sysbench-v4` (1,010 → 1,010 tasks)
> - `DCAgent/selfinstruct-naive-sandboxes-2-verified-v2` → `DCAgent/selfinstruct-naive-sandboxes-2-verified-v3` (9,638 → 7,132 tasks)
> - `laion/exp_rpt_stack-php-large-v7` → `laion/exp_rpt_stack-php-large-v8` (1,628 → 462 tasks)
> - `laion/nemotron-gym-inverse-ifeval-v3` → `laion/nemotron-gym-inverse-ifeval-v4` (1,000 → 1,000 tasks)
> - `laion/nemotron-gym-instruction-following-multiturnchat-v3` → `laion/nemotron-gym-instruction-following-multiturnchat-v4` (2,011 → 1,982 tasks)
> - `DCAgent/exp_rle_adversarial-v4` → `DCAgent/exp_rle_adversarial-v5` (4,093 → 2,731 tasks)
> - `laion/nemotron-gym-cfbench-v3` → `laion/nemotron-gym-cfbench-v4` (1,105 → 468 tasks)
> - `laion/nemotron-gym-agentic-swe-pivot-v3` → `laion/nemotron-gym-agentic-swe-pivot-v4` (3,978 → 1,541 tasks)
>
> - `laion/mix_h10_reward_binary-v3` is retired: problematic synthetic mix; standing policy is retirement.
> - `laion/mix_h11_compositional_gradient-v3` is retired: problematic synthetic mix; standing policy is retirement.
>
> **v4.4** — low-ceiling source remediation — repairs four sources identified by the GLM 5.2 failure-trace audit. Scaffold tasks now provide their actual starter code before the agent runs; scaffold, multifile, and curriculum tasks expose the generated verifier test as an explicit task contract and share one dependency-complete Python image. Math uses typed, fail-closed answer comparison and excludes free-form or multiple-valid-answer prompts that cannot be scored soundly by deterministic equality. Two shared images cover the release. Every retained source remains above the 300-task floor; superseded rewards are excluded.
>
> - `laion/exp_rpt_scaffold-v2` → `laion/exp_rpt_scaffold-v3` (4,861 → 3,136 tasks)
> - `DCAgent/exp_rpt_multifile-v2` → `DCAgent/exp_rpt_multifile-v3` (4,907 → 4,884 tasks)
> - `DCAgent/exp_rpt_curriculum-medium` → `DCAgent/exp_rpt_curriculum-medium-v2` (512 → 495 tasks)
> - `laion/nemotron-gym-math-v4` → `laion/nemotron-gym-math-v5` (6,534 → 4,096 tasks)
>
> - `DCAgent/exp_rpt_curriculum-hard` is retired: it was an independent rewrite of the same source pool as the retained medium curriculum, and its sample had more task/verifier defects and substantially less evaluation coverage.
>
> **v4.3** — task storage remediation — replaces 13 sources with explicit task-level storage requirements. All other packaged task files are preserved byte-for-byte, every source remains above the 300-task floor, and the new versions are hosted only inside TaskTrove. Historical rewards from the superseded versions are excluded.
>
> - `DCAgent/exp_rle_adversarial-v3` → `DCAgent/exp_rle_adversarial-v4`: 4 GiB storage (4,093 tasks)
> - `DCAgent/exp_rpt_multifile` → `DCAgent/exp_rpt_multifile-v2`: 4 GiB storage (4,907 tasks)
> - `DCAgent/selfinstruct-naive-sandboxes-2-verified` → `DCAgent/selfinstruct-naive-sandboxes-2-verified-v2`: 4 GiB storage (9,638 tasks)
> - `laion/exp_rpt_bugsinpy-v2` → `laion/exp_rpt_bugsinpy-v3`: 4 GiB storage (500 tasks)
> - `laion/exp_rpt_stack-dockerfile-gpt5mini-v6` → `laion/exp_rpt_stack-dockerfile-gpt5mini-v7`: 4 GiB storage (592 tasks)
> - `laion/mix_h10_reward_binary-v2` → `laion/mix_h10_reward_binary-v3`: 4 GiB storage (2,862 tasks)
> - `laion/mix_h11_compositional_gradient-v2` → `laion/mix_h11_compositional_gradient-v3`: 4 GiB storage (3,873 tasks)
> - `laion/nemotron-gym-agentic-indirect-prompt-injection-v2` → `laion/nemotron-gym-agentic-indirect-prompt-injection-v3`: 20 GiB storage (1,272 tasks)
> - `laion/nemotron-gym-instruction-following-citation` → `laion/nemotron-gym-instruction-following-citation-v2`: 20 GiB storage (9,033 tasks)
> - `laion/nemotron-gym-instruction-following-v2` → `laion/nemotron-gym-instruction-following-v3`: 20 GiB storage (46,391 tasks)
> - `laion/nemotron-gym-knowledge-web-search-mcqa` → `laion/nemotron-gym-knowledge-web-search-mcqa-v2`: 20 GiB storage (2,915 tasks)
> - `laion/nemotron-gym-litmus-bench` → `laion/nemotron-gym-litmus-bench-v2`: 20 GiB storage (5,232 tasks)
> - `laion/nemotron-gym-qa-abstention-v2` → `laion/nemotron-gym-qa-abstention-v3`: 20 GiB storage (3,150 tasks)
>
> **v4.2** — task storage remediation — replaces 1 source with explicit task-level storage requirements. All other packaged task files are preserved byte-for-byte, every source remains above the 300-task floor, and the new versions are hosted only inside TaskTrove. Historical rewards from the superseded versions are excluded.
>
> - `laion/codeforces-v2` → `laion/codeforces-v3`: 4 GiB storage (10,000 tasks)
>
> **v4.1** — data-quality survey remediation — validates 3,400 traces across 34 statistically flagged v3.42 sources. This release publishes 14 fail-closed or semantic-verifier replacements covering 718,930 retained tasks. Verifier code, data, runtime, dependency, collection, and result-parsing failures now leave no reward for Harbor to classify as infrastructure; ordinary wrong or missing agent answers remain scoreable zeroes. Every replacement remains above the 300-task floor.
>
> Replacements:
> - `laion/nemotron-gym-agent-calendar` → `laion/nemotron-gym-agent-calendar-v2` (3,358 → 2,699 tasks)
> - `laion/exp_rpt_stack-php-large-v6` → `laion/exp_rpt_stack-php-large-v7` (3,789 → 1,628 tasks)
> - `laion/exp_rpt_stack-pytest-large-v2` → `laion/exp_rpt_stack-pytest-large-v3` (2,552 → 1,783 tasks)
> - `laion/nemotron-gym-agentic-swe-pivot-v2` → `laion/nemotron-gym-agentic-swe-pivot-v3` (3,978 → 3,978 tasks)
> - `laion/nemotron-gym-arc-agi-python-inductive` → `laion/nemotron-gym-arc-agi-python-inductive-v2` (10,000 → 10,000 tasks)
> - `laion/nemotron-gym-arc-agi-transductive-v2` → `laion/nemotron-gym-arc-agi-transductive-v3` (10,000 → 10,000 tasks)
> - `laion/nemotron-gym-competitive-coding` → `laion/nemotron-gym-competitive-coding-v2` (15,713 → 15,713 tasks)
> - `laion/nemotron-gym-instruction-following-calendar` → `laion/nemotron-gym-instruction-following-calendar-v2` (8,387 → 8,387 tasks)
> - `laion/nemotron-gym-instruction-following-freeform` → `laion/nemotron-gym-instruction-following-freeform-v2` (8,869 → 8,869 tasks)
> - `laion/nemotron-gym-instruction-following-structured-v2` → `laion/nemotron-gym-instruction-following-structured-v3` (9,437 → 9,437 tasks)
> - `laion/nemotron-gym-knowledge-mcqa` → `laion/nemotron-gym-knowledge-mcqa-v2` (616,888 → 616,888 tasks)
> - `laion/nemotron-gym-math-advanced-calculations-v3` → `laion/nemotron-gym-math-advanced-calculations-v4` (5,291 → 5,291 tasks)
> - `laion/nemotron-gym-reasoning-gym` → `laion/nemotron-gym-reasoning-gym-v2` (14,259 → 14,259 tasks)
> - `laion/tulu3-sft-personas-math-sandboxes-verified-v2` → `laion/tulu3-sft-personas-math-sandboxes-verified-v3` (9,998 → 9,998 tasks)
>
> Removed mixed sources:
> - `laion/mix_baseline_uniform-v2`: empty verifier suites in 5/100 audited traces
> - `laion/mix_h1_struggle_zone-v2`: empty verifier suites in 14/100 audited traces
>
> Retained pure-source quarantines (exclude their current rewards):
> - `laion/exp_rpt_methods2test-large-v4`: 0/32 attempted repaired oracles passed the exact Java image
> - `laion/exp_rpt_stack-jest-v4`: project context and a trustworthy oracle are not recoverable
> - `laion/exp_rpt_stack-rspec-v3`: candidate repairs still passed with an empty workspace
> - `laion/nemotron-gym-agentic-indirect-prompt-injection-v2`: the package lacks an authoritative legitimate action
> - `laion/toolscale-v3`: a sound repair requires an agent-isolated tool service
> - `laion/nemotron-gym-agentic-swe-pivot-v3`: single-reference matching does not prove semantic completion
> - `laion/nemotron-gym-instruction-following-freeform-v2`: regex-only checking validates format rather than content
>
> Standalone repositories for superseded, removed, or quarantined broken versions are retired only after the main-repository commit and tag are verified. Any nonidentical standalone Parquet is preserved under `deprecated/` first.
>
> **v4.0** — RewardKit LLM-judge migration — replaces all 22 TaskTrove dataset versions that embed hand-written LiteLLM/OpenAI judges. The full migration covers 490,727 judge-backed tasks across 495,646 rows. Judgeable tasks now use Harbor RewardKit with root-level TOML rubrics and structured outputs. Missing or empty `/app/response.txt` and `/app/answer.txt` remains a scoreable zero; otherwise the verifier does not prewrite a reward, does not use `|| true`, and lets judge authentication, quota, rate-limit, timeout, transport, and server errors propagate without `reward.json`, so Harbor classifies them as infrastructure failures and retries them. CFBench and SysBench retain their deterministic gates and invoke RewardKit only after those gates pass. Every replacement preserves its source row count, remains above the 300-task floor, and is hosted only inside TaskTrove. Fifteen affected standalone repositories are retired after publication: nine were byte-identical to TaskTrove, while six nonidentical Glaive/StackExchange snapshots are preserved under the `deprecated` config before deletion. All rewards from the superseded judge implementations are excluded.
>
> **v3.42** — audited pure-source remediation — removes `laion/nemotron-gym-agentic-conversational-tool-use-pivot-v2`, `laion/bash-textbook-v2`, and `laion/code-feedback-v2`; their packaged protocols cannot support sound deterministic scoring. Replaces structured output with `laion/nemotron-gym-instruction-following-structured-v2` (9,437 tasks), whose contract explicitly asks for any schema-valid instance. Rebuilds ToolScale as `laion/toolscale-v3` (4,048), with an executable offline tool backend and response/evidence grading. Detailed, heavy-padding, CodeNet, Stack C#, Nemotron C#, and the full original PR population are rebuilt from self-contained contracts and tests: `laion/exp_rpt_codenet-python-v4` (6,975). Every generated source requires oracle success, empty-workspace failure, and an applicable mutant or multi-case negative gate; C# uses the exact packaged verifier in Daytona. The replacement set uses 3 unique Daytona image designs, within the 20-image release limit. Sources with fewer than 300 admitted tasks are not published. Historical rewards from all superseded or removed versions are excluded. Four nonidentical standalone provenance Parquets are preserved under the `deprecated` config before their repositories are retired; all new versions are hosted only inside TaskTrove. The following repairs fell below the 300-task floor and are removed without successors: `laion/exp_rpt_stack-bash-v3` (296 candidates after its repair gate), `laion/exp_rpt_stack-bash-withtests-v2` (189 candidates after its repair gate), `laion/exp_rpt_nemotron-bash-withtests-gpt5mini-v2` (199 candidates after its repair gate), `laion/exp_rle_detailed-v3` (0 candidates after its repair gate), `laion/exp_rle_heavy_padding-v4` (0 candidates after its repair gate), `DCAgent/exp_rpt_pr-v3` (133 candidates after its repair gate), `laion/exp_rpt_stack-csharp-v6` (86 candidates after its repair gate), `laion/exp_rpt_nemotron-csharp-v2` (124 candidates after its repair gate).
>
> **v3.41** — DCAgent verifier repair and source retirement — replaces `DCAgent/exp_rle_adversarial-v2` with `DCAgent/exp_rle_adversarial-v3` (4,093 tasks), installing only each packaged verifier's exact imported test dependencies instead of the historical global heavyweight whitelist. Replaces `DCAgent/exp_rpt_pr-v2` with `DCAgent/exp_rpt_pr-v3` (2,554), removing 77 verifier-invalid rows, and removes the superseded broken `laion/exp_rpt_pr-v2` mirror. Retires both issue mirrors because the population overlaps SWE-bench, and retires `DCAgent/llm-verifier-freelancer`. Removes `DCAgent/exp_rpt_stack-jest-large`: only 257 tasks survived the full environment/source-contract audit, below TaskTrove's 300-task floor. `DCAgent/inferredbugs-sandboxes-verifier` remains unchanged with a warning that its regex/source-shape tests are weak and its rewards are not strong behavioral-correctness labels. The two Stack Bash sources remain unchanged and warned; their original generators and verifier inputs are recoverable, but Qwen reconstruction is deferred until the requested Together inference endpoint is available. Historical rewards from every superseded or retired version are excluded.
>
> **v3.40** — targeted pure-source data-quality remediation — replaces five audited datasets. `laion/openswe-tasks-patched-v5-oracle-success` is superseded by in-repo `laion/openswe-tasks-patched-v6-oracle-success` (13,946 tasks): retained tasks use four shared images, require the oracle to pass against the exact published archive, and fail closed unless pytest executes the target suite and emits a valid nonempty session/JUnit record; source-matched Dockerfile reconstruction supplies verifier dependencies without task-specific images. `laion/nemotron-gym-identity-following-v2` becomes `laion/nemotron-gym-identity-following-v3` (21,660), replacing the source-specific `Nemotron 3 Super` identity with `Marin` and documenting that identity tasks must be customized for the evaluated model. `laion/nemotron-gym-math-stack-overflow` becomes `laion/nemotron-gym-math-stack-overflow-v2` (436,307), using typed parsers for intervals, sets, tuples, equations, scalars, and normalized text instead of the permissive untyped fallback. `laion/exp_rpt_stack-rspec-v2` becomes `laion/exp_rpt_stack-rspec-v3` (10,000), preserving scores while exposing runner output and exit state. `laion/exp_rpt_stack-ruby-v2` becomes `laion/exp_rpt_stack-ruby-v3` (2,310), retaining only packages valid under one dependency-complete shared image. Removes the explicitly rejected `laion/exp_rpt_stack-bash-withtests-gpt5mini-v2` (8,922), `laion/freelancer-projects-sandboxes-ta-rl-gpt-5-mini-v2` (9,999), and `laion/freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2` (10,000). Adds dataset-card warnings to `laion/nemotron-gym-agentic-swe-pivot-v2`, `laion/exp_rpt_stack-bash-v3`, and `laion/exp_rpt_stack-bash-withtests-v2`; the two Bash sources cannot be faithfully regenerated from the preserved synthesis artifacts and are retained without scoring claims. `laion/nemotron-gym-agentic-indirect-prompt-injection-v2` and `laion/nemotron-gym-instruction-following-freeform` are accepted unchanged. Historical rewards from superseded versions are excluded.
>
> **v3.39** — scaling-audit mix-source removal — removes seven mixed sources (24,949 tasks) whose TaskTrove v3.38 rollouts established systematic verifier, environment, or task-contract defects: `DCAgent/mix_h2_language_proportional-v2` (4,066), `DCAgent/mix_h6_test_quality_top25` (2,747), `laion/mix_h10_reward_staged-v2` (3,873), `laion/mix_h2_language_balanced-v2` (4,506), `laion/mix_h5_skill_diverse-v2` (3,166), `laion/mix_h7_raw_volume_5k-v2` (3,718), and `laion/mix_h8_adversarial_tests-v2` (2,873). Confirmed failure modes include false passes after compiler or command failures, stale selectors that collect zero tests, missing fixtures and dependencies, syntax-invalid tests, and tasks whose language or required repository state does not match the packaged verifier. All seven sources are removed rather than partially filtered because they mix unrelated task populations and harnesses. No pure single-source dataset is removed. Campaign rewards for these versions should be excluded.

> **v3.38** — invalid self-documenting verifier source removal — removes `laion/exp_rpt_stack-selfdoc-gpt5mini-v2` (6,547 tasks). A full bounded-memory audit found population-wide task-contract failures: 290 syntax-invalid test files, 696 missing relative-suite imports, 1,519 packages without an AST-visible test, 932 likely verifier-embedded requested symbols, and 5,070 packages importing unmentioned nonstandard modules; none includes source provenance needed for a faithful repair. An empty-workspace probe of 60 evenly spaced first-300 tasks produced 50 collection errors, seven ordinary failures, and three vacuous passes. Exact-byte evaluations were all-zero across four model campaigns. No certifiable repaired population exists, so the source and its byte-identical standalone repository are removed; all campaign rewards for v2 are excluded.

> **v3.37** — invalid Rust integration-test source removal — removes `laion/exp_rpt_stack-rust-v2` (9,987 tasks). A full package audit found no certifiable valid remainder. In 9,984 tasks, packaged tests embed upstream production or unit-test code and then execute it as a Cargo integration test: 4,748 contain invalid `crate::` references, 6,600 contain invalid `super::` references, and many pass using complete answers embedded in the verifier rather than agent-authored output. The other three tasks have unrelated prompt/test pairs. Missing baselines and file wiring make a semantics-preserving mechanical repair impossible, so the source is removed under the 300-task minimum rather than publishing a misleading repair. All Rust v2 campaign results are excluded. Its standalone repository was already retired in v3.35.

> **v3.36** — stack-C# verifier repair — replaces `laion/exp_rpt_stack-csharp-v5` with in-repo `laion/exp_rpt_stack-csharp-v6` (9,485 tasks). V5 unconditionally ran `dotnet new` whenever the project was nested under `TestProject/`, so the packaged project collided with template creation and the verifier exited before running tests. It also treated a successful runner exit as sufficient without proving any tests executed. V6 reuses an existing `TestProject`, creates one with `--force` only when absent, removes the template dummy test, always injects the packaged `TestSolution`, and reads fresh TRX counters. Reward 1 now requires at least one discovered test, every discovered test to execute, and all tests to pass. Full validation retained all 9,485 tasks and confirmed that only `tests/test.sh` changed; bounded .NET execution covered existing-project collision, fresh-project creation, an actual packaged task, failure, skip, and zero-test cases. All v5 campaign results are excluded. The v5 standalone repository was already retired in v3.35; v6 is hosted only in TaskTrove.

> **v3.35** — standalone-repository retirement + `deprecated` subset. TaskTrove is now the single source of truth for the v2-gate lineage. A new **`deprecated` subset** (config `deprecated`, served from `deprecated/`, non-overlapping with the `default` config) preserves the final standalone snapshots of 26 retired repositories, each under the existing `org__name/` naming scheme: **20 intermediate version rungs** that never appeared in any TaskTrove tree (`exp_rpt_stack-junit` v2–v5, `exp_rpt_stack-csharp` v2–v4, `exp_rpt_stack-php-large` v2–v5, `exp_rpt_crosscodeeval-csharp` v2–v3, `exp_rle_detailed-v2`, `exp_rpt_ghactions-v2`, `exp_rpt_pymethods2test` base, `nemotron-gym-instruction-following-adversarial` base+v2, `nemotron-gym-math-advanced-calculations` base+v2); **5 datasets removed from the main tree** by later policy (`exp_rpt_defects4j-v3-v4`, `exp_rle_error_report-v3`, `exp_rle_github_issue-v3`, `exp_rle_minimal_instructions-v3`, `exp_rpt_exercism-python-v2`); and **1 provenance bundle** (`nemotron-gym-agent-workplace-v2`, raw `data/`-parquet layout + its `convert_v2.py` conversion script; byte-identical to the main-tree copy, excluded from the `deprecated` config glob for schema reasons). Transfers were verified by sha256 (LFS) and byte size. 65 standalone repositories in the `laion`/`DCAgent` orgs were then deleted: the 26 transferred above plus 39 whose TaskTrove copies were verified byte-identical (LFS sha256 match) at deletion time. 13 further standalone repositories had already been deleted by earlier releases; their content remains resolvable at pinned TaskTrove revisions (e.g. `revision="66f5a3e1"` for the v2 tree). No main-tree dataset changed in this release.

> **v3.34** — minimum-source-size policy — removes every source with fewer than 300 tasks: `laion/exp_rle_error_report-v3` (261), `laion/exp_rle_github_issue-v3` (264), `laion/exp_rle_minimal_instructions-v3` (233), `laion/exp_rpt_bugswarm-v2` (2), `laion/exp_rpt_exercism-python-v2` (133), and `laion/exp_rpt_nemotron-rust-v2` (170). These sources are excluded from current campaign trackers. Going forward, a repair that leaves fewer than 300 tasks removes the source from TaskTrove instead of publishing the undersized result.
>
> **v3.33** — generated-Dockerfile environment repair — replaces `laion/exp_rpt_stack-dockerfile-gpt5mini-v5` with in-repo `laion/exp_rpt_stack-dockerfile-gpt5mini-v6` (592 tasks). V5 still contained deterministic image-build failures from undefined build variables, missing tools, dead package repositories, 32-bit images, and stale remote artifacts. V6 gives retained tasks one current multi-language environment and conservatively removes tasks whose prompt or verifier requires bundled services, hidden filesystem fixtures, accelerator/GUI stacks, container orchestration, or unsupported runtimes. A full 2,554-row audit verified the retained population and byte preservation outside `environment/Dockerfile`; a bounded GLM/Daytona smoke reached ordinary positive rewards with no build errors or retries. All v5 campaign results are excluded.
>
> **v3.32** — final Nemotron JUnit fixture repair — immediately supersedes v5 with in-repo `laion/exp_rpt_nemotron-junit-v6` (493 tasks). V6 removes one additional multi-public-type fixture and repairs legacy Mockito runner imports, missing parameterized-test imports, and JUnit-4-style message-first assertions. Targeted remote smokes confirmed the repaired Mockito and assertion-import paths pass and that parameterized tests reach ordinary assertions. All earlier Nemotron rewards remain excluded.
>
> **v3.31** — conservative Nemotron JUnit repair — replaces the short-lived v4 candidate with in-repo `laion/exp_rpt_nemotron-junit-v5` (494 tasks). A full 5,000-row audit removes 4,506 malformed, assertion-free, network-dependent, mixed-framework, externally dependent, or unresolved fixtures. V5 stages the real test class/package, compiles prompt-prescribed sources and resources from Maven or `/app` layouts, supplies only audited test dependencies, selects the exact hidden class, and fails closed on compilation errors, zero tests, failures, errors, or skips. Two bounded remote GLM smokes verified positive and ordinary failing paths; all v3/v4/smoke rewards are excluded.
>
> **v3.30** — Nemotron JUnit fallback-build repair — immediately supersedes the v3.29 Nemotron repair with in-repo `laion/exp_rpt_nemotron-junit-v4` (1,203 tasks). The v3 verifier still assumed every valid response contained a Maven `pom.xml`, although some retained prompts explicitly allow a single Java source file. V4 supplies a verifier-owned Java 17 fallback POM only when the response has no project POM; it retains the same fail-closed Surefire XML checks and the same conservatively audited task population. Seven provisional GLM v3 trials were discarded.
>
> **v3.29** — Nemotron JUnit verifier repair — replaced `laion/exp_rpt_nemotron-junit-v2` with in-repo `laion/exp_rpt_nemotron-junit-v3` (1,203 tasks). V2 copied hidden Java fixtures to `TestSolution.java` even when they declared a different public class, omitted required test dependencies/imports, accepted assertion-free and skip-gated tests, and included external-network fixtures. The full 5,000-row audit conservatively removed 3,797 invalid or ambiguous tasks. V3 stages the parsed class/package correctly, supplies only required JUnit/TestNG test dependencies, and scores fresh Surefire XML while failing closed on compilation errors, zero tests, failures, errors, and skips.
>
> **v3.28** — invalid-dataset removal — removed `laion/codeactinstruct-v2` (10,000 tasks). A full archive audit found that every task omitted its concrete user goal: the packages contained only one of three generic CodeAct system/tool preambles, and the LLM judge graded that same incomplete preamble. The original 7,139 source trajectories retain the missing user messages, but all 2,031 embodied/tool trajectories depend on simulator state that was never packaged and do not end with a final answer, so rebuilding them as a different static or LLM-judged benchmark would not be a faithful repair. GLM 5.2 and Qwen3.5 campaign rewards from v2 are invalid and excluded.
>
> **v3.27** — CrossCodeEval Java prompt-contract repair — replaced `laion/exp_rpt_crosscodeeval-java-v2` with in-repo `laion/exp_rpt_crosscodeeval-java-v3` (2,139 tasks). The v2 prompt asked for a complete solution and labeled Java context and metadata as Python, while its strict verifier expected only the short continuation fragment; valid full-file answers were therefore rejected. V3 explicitly requests only the Java line-completion fragment, fixes the code-fence and language metadata, and preserves every gold fragment, verifier, environment, task config, and other archive payload byte-for-byte.
>
> **v3.26** — full-dataset repair + repository consistency — replaced six broken datasets with full repaired versions: `DCAgent/exp_rpt_stack-dockerfile-v2` → in-repo `DCAgent/exp_rpt_stack-dockerfile-v3` (485 tasks; 21 malformed verifiers repaired, 12 additional malformed verifiers removed, stale environment/package fixes applied, and reward-file fallback added); `DCAgent/mix_h2_language_proportional` → in-repo `DCAgent/mix_h2_language_proportional-v2` (4,066 tasks; 69 shell-invalid verifiers removed and reward-file fallback added); `laion/exp_rle_heavy_padding-v3` → in-repo `laion/exp_rle_heavy_padding-v4` (784 tasks; removed the deprecated `sklearn` PyPI shim while retaining `scikit-learn`); `laion/exp_rpt_stack-dockerfile-gpt5mini-v3` → in-repo `laion/exp_rpt_stack-dockerfile-gpt5mini-v5` (2,554 tasks; adopts v4's full base-image repair, removes 100 shell-invalid verifiers, and adds reward-file fallback); `laion/exp_rpt_stack-go-v4` → `laion/exp_rpt_stack-go-v5` (2,313 tasks; adopts the existing full Debian-based repair); and `laion/exp_rpt_stack-jest-v3` → in-repo `laion/exp_rpt_stack-jest-v4` (471 tasks; repaired shared Node environment). A rollout-quality audit additionally replaced three semantically noisy synthesis datasets with conservative, versioned in-repo quarantines: `DCAgent/exp_rpt_pymethods2test-large` → `DCAgent/exp_rpt_pymethods2test-large-v2` (4,991 tasks), `DCAgent/exp_rpt_unitsyn-python-large` → `DCAgent/exp_rpt_unitsyn-python-large-v2` (4,991 tasks), and `DCAgent/exp_rpt_unitsyn-python-v3` → `DCAgent/exp_rpt_unitsyn-python-v4` (491 tasks). The replacements remove 23 confirmed and four likely prompt/oracle/test contract defects; all other task archives remain byte-identical. A verifier-integrity audit replaced `DCAgent/exp_rle_adversarial` → `DCAgent/exp_rle_adversarial-v2` (4,093 tasks), `DCAgent/exp_rpt_issue` → `DCAgent/exp_rpt_issue-v2` (3,883 tasks), and `DCAgent/exp_rpt_pr` → `DCAgent/exp_rpt_pr-v2` (2,631 tasks). Every retained pytest verifier now fails closed when zero non-skipped tests execute; 4,016 tasks with skip-capable tests, unparseable test code, or unresolved runtime prerequisites were conservatively removed. `laion/exp_rpt_methods2test-large-v2` and the already-removed v3 population were superseded by in-repo `laion/exp_rpt_methods2test-large-v4` (1,203 tasks): Surefire XML now requires at least one passing, non-skipped `TestSolution` case, mechanically repairable imports/assertion dependencies were fixed, and 3,269 structurally ambiguous or invalid tests were removed. Also repaired the v3.17 duplicate-verifier-environment regression in Glaive, Qasper, five StackExchange datasets, STaQC, and WizardLM-Orca (90,000 tasks): each `task.toml` declared `verifier.env` twice, causing Harbor to reject every task; the valid environment entries are consolidated into one table. STaQC and WizardLM-Orca now resolve through in-repo `laion/staqc-v3` and `laion/wizardlm-orca-v3` (10,000 tasks each), with every non-configuration archive payload preserved byte-for-byte. Freelancer was audited and required no change. The lingering superseded Jest v2 copy was removed. A v3.0–v3.25 tree audit confirmed the other advertised additions, replacements, and removals. Going forward, repaired dataset versions are hosted directly in TaskTrove; standalone repositories are not required for new versions.
>
> **v3.25** — removal — retired 6 datasets per campaign wind-down: `laion__exp_rpt_methods2test-large-v3`, `laion__exp_rpt_manybugs-v2`, `laion__exp_rpt_defects4j-v3-v4`, `DCAgent__swe_rebench_patched_oracle`, `laion__exp_rpt_softwareheritage-v2`, `laion__exp_rpt_quixbugs-v2`. Source HF repos remain available; only the TaskTrove copies are removed.
>
> **v3.24** — DEPRECATION — removed the remaining `exp_flat25_*` datasets (`laion/exp_flat25_pseudocode-v2`, 728 tasks; `laion/exp_flat25_subtle_debug-v3`, 289 tasks). Both show the same instruction/test shuffle corruption as the earlier flat25 removals (v3.22/v3.23): median instruction/test token overlap ~0.05–0.06 with ~70–85% of pairs under 0.10, and verified mismatches among the highest-overlap pairs. The entire flat25 generation pipeline is corrupted — regenerate from source; do not reuse these pairs.
>
> **v3.23** — verifier repair + deprecation — added `laion/exp_rpt_stack-jest-v3` while the superseded `DCAgent/exp_rpt_stack-jest-v2` copy remained until v3.26. The v3 repair baked the union of all test-required npm packages into the shared `node:20-slim` image and removed runtime npm installation from the verifier; it dropped 18 tasks requiring unresolvable repo-internal modules and 11 requiring binary-downloading packages, retaining 471. Deprecated `laion/exp_flat25_stackoverflow-v2` because its instruction/test pairs were also shuffled.
>
> **v3.22** — DEPRECATION — `laion/exp_flat25_speed_bonus-v2` removed. The r11-58 data-quality audit found the instruction/test pairs were SHUFFLED from different sources (e.g. an `indy_common/types.py` instruction paired with a pandas date_range test; a MODFLOW instruction paired with a sklearn test). Verified mismatches exist at every instruction/test overlap band, so no reliable filtered subset exists; 98/764 tests additionally import sibling repo modules that were never shipped. The dataset is unfixable at the task level — regenerate from source instead of reusing these pairs.
>
> **v3.21** — verifier repair — `laion/exp_rle_heavy_padding-v2`→`v3` (bake runtime pip-install whitelist into the shared Dockerfile, eliminating VerifierTimeoutError; 784 tasks) and `dcagent/exp_rpt_crosscodeeval-java`→`laion/exp_rpt_crosscodeeval-java-v2` (Java snippets were being imported as Python — oracle could never pass; switched to a normalized text-diff verifier on /app/solution.txt, added oracle solve.sh, kept the single shared Dockerfile; 2,139 tasks)
>
> **v3.20** — base-commit fix — `laion/r2egym-patched-full-oracle-v2`→`v3` (every task's git checkout pointed at the SOLUTION commit instead of its parent; replaced all 3328 base_commits with their parents so the agent starts at the pre-fix state)
>
> **v3.19** — Dockerfile base-image fix — repaired `laion/exp_rpt_stack-go-v4` in place in TaskTrove by switching `golang:1.21-alpine` to Debian-based `golang:1.21`; the standalone repair used the v5 label, and v3.26 adds that canonical v5 path. All 2,313 tasks were affected.
>
> **v3.18** — verifier-environment repair — repaired `laion/exp_flat25_speed_bonus-v2` in place in TaskTrove by baking 46 runtime pip dependencies into its Dockerfile; its standalone repair used the v3 label. Also repaired `laion/exp_rpt_stack-dockerfile-gpt5mini-v3` in place by dropping 1,483 tasks with unbuildable exotic base images and fixing EOL Ubuntu repositories, retaining 2,654; its standalone repair used the v4 label, and v3.26 adds a canonical v5 path.
>
> **v3.17** — LLM-judge API key fix: 14 datasets with LLM-judge verifiers (litellm/OpenAI) lacked `[verifier.env] OPENAI_API_KEY` in task.toml → verifier AuthenticationError at runtime. Fixed: added `[verifier.env]` to all 14; freelancer's hardcoded `sk-proj-...` credential replaced with `os.environ.get(...)`. Affected: freelancer (10000, also hardcoded cred removed), stackexchange×5 (50000), glaive (10000), wizardlm-orca/staqc/qasper/magicoder/code-feedback/codeactinstruct/bash-textbook (63074). TaskTrove contains the canonical repaired copies; standalone copies are not version pointers.
>
> *Correction to the v3.17 mechanism, from RL run evidence (2026-07-30).* The fix above is right, but the causal story conflates two distinct faults and overstates the blast radius. **freelancer did not fail for want of `[verifier.env]`:** all 10,000 of its tasks hard-coded the credential and none read `OPENAI_API_KEY`, so the change that mattered for it was the `os.environ.get(...)` replacement, not the env-var addition. An earlier diagnosis attributing the failure to the key not propagating into the Daytona verifier sandbox was **retracted** — harbor propagation tests pass 3/3 and Daytona 76/76. Separately, six of the fourteen listed datasets (glaive and stackexchange×5) produced graded, non-degenerate reward during agentic RL training in the same period, which is inconsistent with a blanket runtime authentication failure for those sources. The listed dataset counts total 133,174 tasks, not 133,074. A full v3.26 parse audit found that 20,000 tasks in STaQC and WizardLM-Orca were structurally broken by duplicate environment declarations; their v3 replacements repair that regression.
>
> **v3.16** — reverted v3.15 heredoc inlining in instruction.md back to original `/setup_files/` references (Harbor patch will mount setup_files/). Verifier-side changes kept: test.sh/solve.sh reference `/tests/setup_files/`. Both `setup_files/` and `tests/setup_files/` present in tar. R2E-Gym test-staging removal kept as-is. Affected: nl2bash (1570), swe_rebench (3787), mix_baseline (333), mix_h7 (333), openswe (17504).
>
> **v3.15** — setup_files propagation fix: Harbor does not mount `setup_files/` during ANY phase (dead code). 6 datasets referenced `/setup_files/` in instruction.md (agent), test.sh/solve.sh (verifier), or both. Fixed: agent-facing refs inlined as heredocs (agent creates files itself); verifier-facing refs moved to `/tests/` (uploaded before verifier); R2E-Gym test-staging removed from instruction (verifier handles it). Affected: nl2bash (1570), swe_rebench (3787), mix_baseline (333 of 3718), mix_h7 (333 of 3718), openswe (17504), r2egym-v2 (3328).
>
> **v3.14** — replacement: 2 live-arm verifier fixes. `crosscodeeval-java` (2,139): was Python import of Java code (SyntaxError, reward always 0); now language-aware text-diff verifier (gold→1, empty→0, Python junk→0). `r2egym-patched-full-oracle` (3,328): Harbor doesn't mount `setup_files/`; relocated `test_info.json` + `r2e_tests/` into `tests/` (which IS mounted). Docker-verified `test_state.py` loads expected tests.
>
> **v3.13** — replacement: `SankalpKJ/swesmith-oracle-filtered` (12,942 tasks) had a systematic verifier/task-package mismatch — SweSmith task branches are created at an older commit where test files added by the fix PR don't exist. The verifier's expected test paths (FAIL_TO_PASS / PASS_TO_PASS) referenced files absent from the checkout, scoring every trial 0.0 regardless of agent correctness. Fixed in `laion/swesmith-oracle-filtered-v2`: the verifier now restores missing test files from the repo's default branch via `git fetch origin && git checkout origin/main -- <test_files>` before running pytest. Docker two-sided validated on 4 diverse tasks (bottlepy/oauthlib/tenacity/pdfminer): buggy→0, gold→1.
>
> **v3.12** — additive: +8 RL-ready datasets completing the SFT→RL conversion (Stages 2+3 of issue #7418). LLM-judge verified: wizardlm-orca (10k), staqc (10k), qasper (10k). MIXED→judge: magicoder (4096), code-feedback (10k), codeactinstruct (10k), bash-textbook (9078). Deterministic: softwareheritage (7 of 500 — most source tasks embed reference in test). All 8 carry default-fail reward + test_state.py + pass_ratio output. Total: 63181 tasks.
>
> **v3.11** — additive: +10 RL-ready datasets from the paper's SFT gap set (Stage 1 complete). Each had a vacuous/broken/missing verifier; all now have default-fail verifiers with pass_ratio output. 45359 total tasks: codeforces (10000), nemo-prism-math (10000), stack-rspec (10000), all-puzzles (6926), toolscale (4035), crosscodeeval-ts (3356), codeelo (500), crosscodeeval-py (500), quixbugs (40), bugswarm (2).
>
> **v3.10** — additive: +9 RL-ready datasets from the paper's SFT gap set (issue #7418). Each had a broken/missing/gamed verifier; all now have default-fail verifiers with pass_ratio-parseable output (test_state.py). 46432 total tasks across stack-cpp (9943), nemotron-bash×3 (25000), nemotron-csharp (4108), nemotron-rust (170), stack-selfdoc (6547), bugsinpy (500), manybugs (164). See individual entries for per-source fix details.
>
> **v3.9** — replacement: `DCAgent/exp_rpt_nemotron-junit` (5,000 tasks) had a **vacuous verifier** — it returned reward 1.0 for an untouched workspace due to three compounding bugs: pipe masking (`javac | tee` without `set -o pipefail`), JUnit reporting success with 0 tests found, and an EXIT trap defaulting to reward 1. Replaced by `laion/exp_rpt_nemotron-junit-v2`: default-fail `test.sh` (reward starts at 0, only a validated Maven pass overwrites to 1), requires `pom.xml`, verifies `Tests run: N>0`, added `test_state.py` for pass_ratio-parseable output. Docker-validated: empty /app → reward 0 (was 1 under the old verifier). No other dataset changed.
>
> **v3.8** — replacement: `laion/tulu3-sft-personas-math-sandboxes-verified` (9,998 tasks) had a verifier format mismatch — its `tests/test.sh` emitted `Correct answer: N` / `Incorrect answer: expected N, got M`, which the `pass_ratio` reward shaper cannot parse. The shaper silently fell back to binary reward, providing no graded signal for RLOO (an arm trained on it collapsed to a one-token policy). Replaced by `laion/tulu3-sft-personas-math-sandboxes-verified-v2`: `test.sh` now invokes `python3 -m pytest /tests/test_state.py` (2 tests: file-exists + contents-match), emitting parseable pytest output with a graded signal (no answer→0.0, wrong answer→0.5, correct→1.0). Also fixes a latent bug: the original `tr -d ' '` stripped spaces from the answer but not the expected, making 961/9998 tasks silently impossible. No other dataset changed.
>
> **v3.7** — replacement: DCAgent/exp_rpt_stack-pytest-large (5,000 tasks) was removed because its verifier installed only pytest and omitted task dependencies, causing widespread collection failures and zero-reward RL rollouts. laion/exp_rpt_stack-pytest-large-v2 replaces it with 2,552 dependency-complete, deterministic-pytest tasks. It has one shared snapshot, a private-reference oracle rerun of 40/40 reward=1.0, and a public-form 200-task smoke with 0 exceptions and 62.2% positive rewards. Public artifacts contain no reference solutions. No other TaskTrove dataset changed.
>
> **v3.6** — additive: +5 verifier-equipped SFT-source datasets (124 subdirs total), built by reconciling the paper's SFT task-gen strategies against TaskTrove. Four LLM-judge computer-use/Q&A sets — `laion/stackexchange-unix-sandboxes-verified` (10,000), `laion/stackexchange-overflow-sandboxes-verified` (10,000), `laion/stackexchange-codereview-sandboxes-verified` (10,000), `laion/glaive-code-assistant-sandboxes-verified` (10,000) — carry the `data.nemotron_gym` LLM-judge verifier (litellm/gpt-4o-mini, per-dataset adapted rubric in `verifier_data.json`; **needs `OPENAI_API_KEY` at trial time**). One deterministic set, `laion/tulu3-sft-personas-math-sandboxes-verified` (9,998), keeps its exact-match numeric verifier (no API cost).
>
> **v3.5** — additive: +3 verifier-equipped datasets (119 subdirs total). `laion/stackexchange-superuser-sandboxes-verified` (10,000) and `laion/stackexchange-tezos-sandboxes-verified` (10,000) are computer-use task sets (Super User Linux/sysadmin; Tezos blockchain node-op) whose Skywork reward-model verifier was replaced with the `data.nemotron_gym` **LLM-judge** verifier (litellm/gpt-4o-mini, per-dataset adapted rubric in `verifier_data.json`, `\boxed{score}`→reward; **needs `OPENAI_API_KEY` at trial time**). `laion/exp_rpt_issue-verified` (4,830) is a `laion` mirror of `DCAgent/exp_rpt_issue`, keeping its original deterministic **pytest** verifier unchanged.
>
> **v3.2** — replaced the old swegym task dataset (`laion__swegym-tasks-patched-validated-v2`, 989 tasks) with `laion/swegym-tasks-patched-validated-v5` (2,438 tasks, patched + validated). No other dataset changed.
>
> **v3.1** — verifier fix for `DCAgent__code-contests-noblock` (8,728 tasks): the v1/v2/v3 `tests/test.sh` ran pytest under `set -euo pipefail`, so a failing solution aborted the script before `reward.txt` was written — failed solutions raised `RewardFileNotFoundError` and were dropped instead of recorded as `reward=0`, silently biasing the reward distribution toward `1.0`. The verifier now wraps pytest (`set +e` / capture `PYTEST_EXIT` / `set -e`) so a reward is always recorded. Only this one dataset's `test.sh` changed; all other task content is byte-identical to v3.
>
> **v3** — additive expansion of v2: all 96 v2 datasets are retained, plus **20 new Nemotron-Gym RLVR task datasets** converted from the [nvidia/Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection (instruction-following, math, science, knowledge, reasoning, multi-turn chat, and single-step agentic pivots). Prior releases remain resolvable at the [`v1`](https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/v1) and [`v2`](https://huggingface.co/datasets/open-thoughts/TaskTrove/tree/v2) tags.

---

## Versioning Policy

Starting with v3.26, repaired dataset versions are published directly in TaskTrove. A versioned TaskTrove subdirectory is the canonical artifact even when no matching standalone Hugging Face dataset repository exists. Broken standalone repositories are removed after their canonical replacements are verified here — and, since v3.35, any standalone whose content is not already byte-identical in TaskTrove is first transferred into the `deprecated` subset (`deprecated/org__name/`) before deletion. Historical release notes identify artifact versions; they do not imply that a standalone repository remains available.

TaskTrove only retains sources with at least 300 packaged tasks. If a repair reduces a source below that floor, the source is removed rather than published as a smaller version. Campaign inventories and trackers omit undersized sources.

## Repository Structure

Each source dataset is stored as a subdirectory named `org__name/`, where the original HuggingFace repo `org/name` has its `/` replaced with `__`. Each contains the task binaries as `tasks.parquet` (columns `path`: str, `task_binary`: gzip tar).

Retired standalone repositories live under `deprecated/org__name/` (same naming scheme), served only through the `deprecated` config; they never appear in the `default` config. Most carry `tasks.parquet` in the standard layout. The Agent Workplace bundle also preserves its raw `data/` Parquet and conversion script alongside the v4.7 deprecated task archive.

---

## Task Format

All tasks in the default config are [Harbor](https://github.com/open-thoughts/OpenThoughts-Agent) task binaries with an instruction, a populated `tests/` verifier payload, and a populated `environment/` payload. The v4.12 audit checked this archive structure for active sources. It did not audit deprecated sources, execute verifiers, or prove that their test runners discover or score runtime tests correctly.

---

## New in v3: Nemotron-Gym RLVR conversions (19 retained)

> **Known limitation — `laion/nemotron-gym-instruction-following-calendar`:** The deterministic verifier checks required event IDs, durations, global time windows, and only a small set of recognized constraint phrases. It does not verify event names, reject overlapping or extra events, or enforce unrecognized natural-language constraints (which soft-pass). Treat its reward as partial calendar-format/time-window compliance, not proof of complete schedule correctness.

Converted with the OpenThoughts-Agent `data.nemotron_gym` framework. Grading is self-contained where the source carries a deterministic gold (string/regex/numeric/JSON-schema/grid), and LLM-judge where grading is inherently subjective (equivalence, rubric, hybrid IFEval+judge). Each task instruction explicitly directs the agent to write its answer to the grader's file path.

- Instruction-following: citation-formatting, free-form-formatting, structured-outputs (json/yaml/xml/toml/csv), multi-turn chat, adversarial inverse-IFEval, CFBench, SysBench
- Math / science / knowledge: math (sympy/latex boxed, with external-pointer hydration), litmus-bench (chemistry), science (Physics/Bio/Chem equivalence judge), QA-abstention, ReasoningGym, ARC-AGI (transductive + python-inductive), Multichallenge (advanced + vanilla)
- Single-step agentic pivots: function-calling, SWE, indirect-prompt-injection (graded on the agent's next action)

---

## Usage

```bash
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet open-thoughts/TaskTrove \
  --output_dir $SCRATCH/tasks/tasktrove --on_exist overwrite
```

See the [OpenThoughts-Agent repository](https://github.com/open-thoughts/OpenThoughts-Agent) for the datagen/RL pipeline.

---

## Resolving prior versions

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="open-thoughts/TaskTrove", repo_type="dataset", revision="v2", local_dir="./tasktrove_v2")  # or "v1"
```

---

## Citation

```bibtex
@misc{openthoughts-agent,
  author = {Team, OpenThoughts-Agent},
  month = Dec,
  title = {{OpenThoughts-Agent}},
  howpublished = {https://www.open-thoughts.ai/blog/agent},
  year = {2025}
}
```
