# 05 — 数据池实查（2026-09-29）

## 结论

保留原大 scope。OpenThoughts-Agent 是可用的交互过程研究起点，但不能把现有发布数据直接视为支持「随机／成功／多样性／恢复」四组实验的成品池。

建议以 **SFT-100K 的可验证来源主轨迹**作为首选工程起点，以 AgentTrove 作为补充来源目录；成功标签补齐前，不启动成功优先组，不把空 result 当成功，不把异常当任务失败。当前状态：**数据审计完成第一轮，四组因果 pilot 尚未达到启动条件**。

## 证据与复现

本轮读取四个公开数据集的 repo metadata、固定 SHA 的数据卡，以及 Hugging Face Dataset Viewer 的实际 schema/行内容。每个数据集在全表跨度上取 10 个等距窗口，每窗口连续 10 行，共 400 行。窗口不是 IID 随机样本，下面计数只描述样本，不能外推总体占比。

- [审计脚本](audit/audit_pool.py)：`python audit/audit_pool.py`（依赖 requests，联网读取；已有窗口使用缓存）。
- [全部摘要](audit/summaries.json)：字段覆盖率、行数、分布、窗口位置、内容 SHA256。
- 每个 `audit/<dataset>/` 保存 `metadata.json`、`README.upstream.md`、原始 `rows_*.json`、`records.json`、`summary.json` 和候选片段。
- Viewer 返回的 400 行均无 truncated_cells 标记。这不证明上游轨迹从未被压缩或裁切。
- **版本限制**：数据卡固定在所记录 SHA；Viewer 使用默认分支缓存，不能保证与该 SHA 一致。训练前必须固定原始 Parquet 修订，并重新核对 row count/schema/hash。这里的行数仅为 Viewer 报告值。
- 原始轨迹只作为数据读取，未执行其中任何命令。

## 四个候选池

| 数据集 | Viewer 总行数 | 100 行样本中的有效对话 | 结果标签实查 | 助手轮数 min/median/max | 初步用途 |
|---|---:|---:|---|---|---|
| SFT-100K | 94,334 | 100 | result：57 空、43 AgentTimeoutError；无 reward/verifier_output 列 | 5 / 12 / 46 | 同教师、较集中来源的工程起点 |
| AgentTrove | 1,696,847 | 90 | result：79 空、21 AgentTimeoutError；无 reward 列；verifier_output/judgment 样本全空 | 0 / 5 / 31 | 来源发现；需严格清洗与元数据核查 |
| v1-SFT | 15,209 | 100 | 无 result/reward/verifier_output 列 | 2 / 7 / 17 | 较简单的格式、训练管线验证 |
| ColdStartForRL-10K | 9,437 | 100 | result：81 空、19 TypeError；无直接 reward/verifier_output 列 | 7 / 18 / 79 | 不能仅凭数据卡称 oracle-verified 就按行判成功 |

来源：
- https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K
- https://huggingface.co/datasets/open-thoughts/AgentTrove
- https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT
- https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K

### 数据卡与实际数据的差异

1. SFT-100K 卡片标称 100,000 行，Viewer 报告 94,334。可能存在发布版本/缓存差异，本轮不推断原因。
2. SFT-100K 卡片把 trace_source 描述成任务源，但样本实际值为 main（96）、summarization-1-summary（3）、summarization-1-answers（1）。它不能直接作为领域标签。
3. AgentTrove 卡片写 messages/reward/task_id，实际 schema 使用 conversations/task，且没有 reward 列。不能按文档字段名直接写训练筛选器。
4. AgentTrove 样本 original_teacher 与 model 存在不一致：最前窗口前者是 GPT-5-mini，后者是 gpt-5-nano-2025-08-07。两列均保留并记录冲突，不能盲选其一视为已验证教师。
5. AgentTrove 样本中 10 行没有 conversations，但有 path/task_binary。不能把全部行数视为可直接 SFT 的轨迹数；本轮未解包 task_binary。
6. SFT-100K 卡片注明至少 5 model turns 的预筛选；样本最小助手轮数为 5。它不适合无条件代表包含短简单经验的完整原始池。

## 四类信号是否可实现

### Random：可以定义，但需先确定共同候选池

过滤空对话、压缩衍生分支、不可解析记录及重复；按任务家族分 train/dev/test。随机组与其他组共享同一池、长度可用范围和 teacher/source 配额。不能让 random 只抽干净成功轨迹，而其他组来自杂合池。

### Success：当前直接标签不足

result 中的 AgentTimeoutError/TypeError 描述运行异常，空值仅代表没有该异常字段值，不证明通过 verifier。助手的 task_complete=true 是自报完成，也不构成外部成功标签。

上游当前代码有可追溯的恢复途径：`scripts/harbor/analyze_sft_export_gaps.py` 读取 trial result.json 的 `verifier_result.rewards` 或 `verifier_result.reward`；导出脚本支持 success/failure 过滤。已保存代码快照，对应 Git SHA `3bd1917e62c9d03d73063b433f5c442c279c0563`。这说明正确标签应追溯到 trial artifacts，不说明这些 artifacts 已公开、已获得，也不证明当前发布数据使用何种历史导出参数。

通过原始 repo 查找标签的第一轮探测：三个 penfever 候选在未认证 Viewer 访问下返回 401，原因未确定，不能径直断言私有；`DCAgent/exp_tas_baseline_traces` 可访问，前三行 result 为空且无 reward 列。探测响应保存在 audit 根目录。

**后续优先级**：查找公开 trial-level verifier artifacts，按原始 repo + run_id + trial_name 精确关联；若无法获得，再选带可执行 verifier 的任务集生成受控 rollout。人工/模型根据文字判断不能替代 oracle success，若采用必须改名并另作误差评估。

### Recovery：过程存在，但必须定义标签

反馈中搜索 Traceback、command not found、测试失败等，再要求有后续助手动作，本轮得到候选：SFT-100K 71/100、AgentTrove 37/100、v1-SFT 19/100、ColdStart 66/100。**这些是宽松词法候选，不是恢复率，不保证纠正成功**；命令回显、代码字符串也可能误触发。

人工查看片段已确认两种不同情况：
- SFT-100K row 1：运行 qrcode 触发 UnboundLocalError，随后分析题目中的缺陷。这是「复现被修复程序原有 bug」，不能自动算 agent 自我纠错。
- AgentTrove row 188540：git command not found，随后提出安装 git 并重新初始化。这是明确环境失败与恢复尝试；要查看后续执行证据才可标成成功恢复。
- AgentTrove row 4：反馈后助手承认其生成程序 main guard 出错并准备改写。属于 agent 自身错误的纠正候选，不根据助手陈述认定已成功。

采用四级字段：`error_origin = task_bug / agent_action / environment / unknown`；`recovery_status = none / attempted / locally_verified / unknown`；`terminal_outcome = pass / fail / unknown`；每个判断保存反馈、纠正动作、验证反馈的 turn index。终局成功与局部恢复分开。

### Diversity：可先做终端操作结构，不冒充多 API 工具多样性

样本采用 terminus-2：助手文本内 JSON 的 commands/keystrokes，终端反馈多以 user role 返回。因此仅统计 tool role 会漏掉真实交互。

建议解析助手 JSON 的命令批次，提取 read/search/edit/execute/test/install 等操作类型及相邻 motif，先抽样校验解析率。不要执行或 eval 字符串。以集合覆盖/边际覆盖做多样性选择，而非简单选最长轨迹。

这能检验「终端操作结构多样性」；无法等同 DIVE 的工具 schema / API 池覆盖。本阶段不据此声称 unseen-tool 泛化。

## 污染、重复与可比性

- task 名存在 `_copy...` 后缀，说明需要审计同源任务家族；不能仅凭后缀断言任务内容相同。结合源 repo、基础 task id、初始指令、代码仓库与 commit 检查。
- main 与 summarization-* 可能是同一 trial 的不同训练片段。按 trial 关联，首轮优先主轨迹，排除无法判断来源的片段。
- source 不能直接用 trace_source；可将 task 前缀作为临时 source_hint，核对上游映射后再用于配额。
- teacher/source/horizon 与选择信号可能相关：记录协变量分布与重叠；做策略效果比较时显式说明，做属性因果归因时进一步匹配。
- 字符数不是 tokenizer token 数。本轮未声称完成 token matching。

## 当前机器与执行边界

只读检查发现 8 张 A100-SXM4-40GB；检查时前 4 张各约 13 GB 占用，后 4 张接近空闲。torch/transformers/datasets/pyarrow/requests 可导入定位。此信息不是独占资源承诺，也不是完整训练兼容性验证。未占用 GPU、未启动付费服务或训练。

完整训练前还需：核实资源分配、确定模型与上下文、模板和 loss mask、精确 token 预算、跑通评测环境并测量耗时。当前没有任何训练后能力结果。
