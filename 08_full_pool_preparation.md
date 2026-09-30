# 08 — 固定版本全池准备（2026-09-29）

## 本轮完成内容

已下载 SFT-100K 全部 10 个原始 Parquet，固定 revision `45fb28fcc38d352133cb28a1c8a43a2f14fea97b`，逐文件验证与 HF LFS SHA256 一致。实际总行数 **94,334**；因此这个版本与数据卡标称 100K 的差异并非仅由 Viewer 缓存造成，但尚未确定上游发布原因。

全量 metadata：[full_metadata.json](prepared/full_metadata.json)。

- main：75,879 行。
- summarization-*：6,560 行。
- trace_source 为空：11,895 行。首轮只取显式 main；不能假定空值也是 main。
- model 全部是 hosted_vllm/glm，这只是后端别名，GLM-4.7-AWQ 身份仍依据数据卡，未独立核对服务端模型。
- result 为空 60,296；AgentTimeoutError 32,764；ContextLengthExceededError 916；其余为 DaytonaError、DaytonaNotFoundError、RuntimeError、AgentEnvironmentTimeoutError、CancelledError。
- 全表 schema 没有 reward / verifier_output，result 的全量取值不是任务成败。这一版本不能直接构造 verifier-success 筛选。

## 标签来源追查

[11 个公开候选的探测结果](audit/outcome_probes/index.json)已保存：8 个可访问，3 个返回 401（未确定原因）。旧 SWE-Smith 数据源解析到 `DCAgent/a1_swesmith`，现有行同样只有空值或 runtime result，没有独立成功列。本次可访问候选的公开文件清单未发现按 trial 暴露的 result.json / reward.json / reward.txt；不证明整个互联网不存在。

TaskTrove 与 RL-5K 暴露 path/task_binary：它们是可执行任务，不是已有轨迹结果。已只读查看一个 RL-5K 任务压缩包，确认包含 test.sh 与 reward.txt 写出逻辑，未解压执行。该任务的 verifier 可用于未来新 rollout，但不能凭任务存在推断旧轨迹成功。

公开检索也出现 Terminal-Bench 评测轨迹和成功轨迹切片；没有将它们混入训练池，以免污染计划中的评测，或用全成功数据假装成功/失败对照。

## 恢复标注改进

新增 [事件级标注规范](07_recovery_annotation_guide.md) 和 [5 个初审事件](prepared/reviewed_events.json)。初审由 assistant 完成，不冒充人工金标准。对 SFT row 1 已核实 Viewer 原文与固定 Parquet row 1 完全一致。

关键变化：一条轨迹可同时包含任务 bug、agent 自己制造的错误、局部恢复和后续再次破坏。用 events 数组记录，不能只给整条轨迹一个 error_origin。

## 代码与可复现性

见 [pipeline/README.md](pipeline/README.md)。

- 固定版本下载、hash 校验。
- 单 CPU 进程全池扫描：对话/指令 hash、任务 family hint、主分支、命令 JSON、安全操作分类、错误候选。
- 均匀 reservoir 抽取 200 条唯一主对话，再对错误候选抽取 100 条。保留重叠与分组信息。
- 临时 train/dev 通过 task-family hint、精确指令/对话 hash、run+trial 的连通分量分组，防止这些已知关联跨分区。仍需仓库/近重复核验，不能称已去污染。
- 审阅 HTML 只渲染转义后的文本，不执行轨迹。
- 5 个回归检查通过，覆盖思考内 JSON、畸形命令、shell 包装/重定向、unknown outcome 和终端标识去除。

## 资源

本轮 GPU 使用 **0 张**。原始文件约 1.75 GB；下载 2 个并发、扫描 1 个 CPU 进程。tokenizer-only 测量不下载模型权重。

原环境 transformers 4.37.2/tokenizers 0.15.1 无法读取该 Qwen3 tokenizer。已在 `.venv-audit` 内安装隔离的 tokenizer stack；未修改系统包。精确版本保存在 [runtime_versions.json](pipeline/runtime_versions.json)。用户资源限制记录在 [resource_policy.json](pipeline/resource_policy.json)：未来最多 4 张当时空闲 GPU，够用时取更少，不干扰他人任务。

## 还不能宣称完成的部分

没有 SFT 训练或 OOD 评测结果；没有将 300 个样本标成已人工审核；没有完成 oracle outcome 关联或最终 loss-mask/token-matched 训练集。局部程序测试成功和 task_complete 都不会自动生成终局成功标签。

## 全池扫描与候选清单结果

- 全量 94,334 条对话无完全相同的 conversation hash；但 task family hint 只有 37,186 个，初始指令 hash 有 40,968 个，不能按轨迹独立随机切 train/dev。
- 1,187,572 个助手轮次中，1,171,838 个能解析出合法 commands JSON：**98.675% 格式解析率**。不是操作语义准确率。
- 主轨迹 75,879 条；其中 50,145 条触发宽松错误关键词候选。不是 50,145 条成功恢复。
- 排除 18,455 条非明确 main，以及 main 中 4,624 条含未解析助手轮次后，候选清单共 **71,255 条**。
- 已生成临时 train 64,062 / dev 7,193；按连接关系成组划分，全部记录保留 outcome=unknown、training_ready=false。并非完成四组 token 匹配训练集。
- 200 个随机主轨迹 + 100 个错误候选有 1 个重叠，共 **299 条不同轨迹**待标注。

产物：
- [全池摘要](prepared/summary.json)
- [临时切分摘要](prepared/manifest_summary.json)
- [候选清单](prepared/candidate_manifest.jsonl)
- [可直接打开的轨迹审阅页](prepared/review/index.html)
- [200 条随机标注包](prepared/annotation_random200.jsonl)
- [100 条错误候选标注包](prepared/annotation_error100.jsonl)

注意：严格格式过滤可能引入长度/来源偏差；目前是工程候选清单，正式实验应报告过滤前后分布。任务 family_hint 和 source_hint 来自名称，仍需上游 lineage 核对。

## Chat template 检查

实测 Qwen3 默认聊天模板会删除较早助手轮次的 `<think>` 内容。因此仅用默认模板统计 token 会低估「保留完整原始推理」的 SFT 输入。脚本改为同时输出：

- chat_tokens：显式 ChatML，保留所有原始 content（含历史 think）。
- official_template_tokens：上游默认模板的结果。
- assistant_content_tokens_diagnostic：单独内容 token 数，不等于已实现的训练 loss mask。

未来训练必须显式确定是否保留历史推理，不能靠默认推理模板隐式决定。此次 tokenizer-only 测量未加载模型权重。

## Token 实测与 pilot 上下文建议

随机 200 条主轨迹，采用保留全部原始 content 的 ChatML：最短 3,797 tokens，中位数 **13,713**，P90 **24,401**，最大 **30,356**。其中 **159/200 超过 8,192**，**67/200 超过 16,384**，本样本没有超过 32,768 的轨迹。

因此第一轮优先测试 32K 上下文；直接裁成 8K/16K 会丢失很多轨迹后半段，尤其可能删掉恢复后的验证。这个判断仅基于随机样本，不宣称全池都能放入 32K。正式分组前需对实际候选逐条 tokenize；可完整保留地筛除超长轨迹，并报告筛除偏差，不能默默截断。

[token_lengths.json](prepared/token_lengths.json) 同时保存 299 条的原始保留长度和官方模板长度；[token_summary.json](prepared/token_summary.json) 是随机 200 条摘要。

临时切分验证通过：所有纳入记录的 task-family hint、精确指令 hash、精确对话 hash 均不跨 train/dev；这不涵盖语义近重复或仓库级污染。299 个 HTML 页面及索引已生成，仍保留“未标注”状态。
