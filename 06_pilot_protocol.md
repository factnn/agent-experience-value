# 06 — 第一轮 pilot 协议（依据实际数据修订）

## 原则

保留「什么使 agent 经验有价值」的大 scope。先验证数据与测量路径，支持在已有工作基础上扩充。不会预设排名反转、恢复胜出或统一 proxy 必然存在。

## Phase 0：立即可执行的数据准备

1. 固定 SFT-100K 原始 Parquet 的 revision/hash，核对 94,334 与卡片 100K 的差异，生成全量字段/分支分布。AgentTrove 仅针对能追溯的 source×teacher 分区扩展，避免直接混用全池。
2. 过滤空对话；以 trial 为单位处理 main/summary 衍生记录；提取 task family 和初始指令 hash，检查近重复及 train/eval 重叠。
3. 从主轨迹随机取 200 条、错误候选过采样取 100 条做人工标注，两组分别统计，不能混起来当总体比例。记录抽样概率/来源、error_origin、局部恢复证据与 terminal_outcome。双人复核部分样本；单人阶段不能报告标注一致率。
4. 构建 commands JSON 安全解析器和操作 motif；统计无法解析样本并人工核对，不以工具 role 是否出现作为工具调用标准。
5. 查找公开 verifier 结果并关联 trial。成功标签未解决前，四组训练不启动。

本轮已提供 400 条审计样本及原始证据，并非已完成上述 300 条人工标注或全池清洗。

## 标签记录结构

每条经验至少包括：dataset_revision、source_repo、row/trial identity、task_family、teacher_raw、teacher_resolved、branch、instruction_hash、trajectory_hash、assistant_turns、total_input_tokens、supervised_tokens、terminal_outcome、outcome_evidence、error_origin、recovery_status、recovery_evidence_turns、operation_sequence、parser_status。

unknown 独立编码。result=None 不转 pass；异常不自动转 fail；task_complete 不转 oracle pass；未见错误不转无错误。

## Pilot A：完整四策略实验（标签具备后）

共同候选池：同教师、可追溯任务来源、可用主轨迹、任务结果已知，确保每个纳入来源有必要的信号变异。统一上下文上限，不把裁切后的恢复轨迹当完整恢复。

| 策略 | 定义 | 核心注意事项 |
|---|---|---|
| Random | 共同池内按预设 source/长度分层随机抽样 | 基线不能换池 |
| Success | 相同分层内优先 verifier pass | 检查分层内成功数据是否足够 |
| Diversity | 相同分层配额内增加操作 motif 覆盖 | 避免只增加长度与领域数量 |
| Recovery | 相同分层内优先有证据的恢复过程 | 局部恢复与终局成功分别报告 |

这是筛选策略整体效果比较；不自动等于单一属性的因果效果。后续 recovery 机制实验可限定终局成功、匹配长度/来源/教师后比较有无恢复。

预算：首先用最终模型 tokenizer/chat template 计算全部 token 和参与 loss 的 assistant token。主预算固定实际处理 token 数与 optimizer updates，并报告 supervised token、轨迹数、唯一任务数、packing/padding。监督 token 占比差异过大时另做匹配敏感性实验。不能同时假定同 token 数、同样本数、同 epoch 都成立。

训练方式：先固定一种 SFT 配方（LoRA 或全参择一，所有组一致），固定原始 assistant loss mask，包括动作与推理是否监督。若研究失败动作 masking，则单列为另一因子。模型暂以同家族 4B 作为候选，需格式执行 smoke test 和长上下文显存实测；没有证据就不强行选择小模型。

重复：先四策略 × 2 个独立子集种子 × 1 个训练种子 = 8 次筛查运行；这会混合子集与训练变异，不能声称已分离二者。出现现象后，在关键策略上交叉至少 2 子集 × 3 训练种子。只重跑同一个训练集不能检验子集选择稳定性。

## Pilot B：成功标签暂时不可恢复时

先做 Random / Diversity / Recovery 三组的解析与管线验证，并将终局 outcome 标为 unknown；只有恢复标签已经人工验证时才训练相应组。该实验只能讨论未验证结果数据上的选择，不能声称完成 outcome proxy 比较。是否转向重新生成可验证数据，由公开结果可恢复性和实测成本决定。

## 评测与统计

- 先统一终端 harness，做格式有效率与基本任务 smoke test，避免所有小模型都卡在 0 分附近。
- ID：同任务来源、独立任务家族，排除 copy/近重复。
- 第一条迁移轴：预先留出任务来源或代码仓库家族；称为来源/仓库迁移。先审核初始指令与仓库重叠，再冻结清单。不要把换 benchmark 自动叫 unseen tools。
- Terminal-Bench 固定开发子集只用于调试。最终独立评测清单在选择器冻结前封存；暂不同时铺开 Toolathlon、GAIA、MedAgentBench。
- 所有策略使用相同评测任务、harness、token/时间限制、重试策略；保留逐任务结果。任务 bootstrap 与训练运行差异分开呈现，不将同一任务多次采样伪装成独立任务。
- pilot 用于估计方差与可行性，不用 1–2 个种子认定「没有效果」。根据观测噪声与预先约定的有意义效果大小决定正式样本量。
- 只有 4 个策略均值时不把相关系数当可靠预测证据；后续再增加独立干预子集并测试跨阶段或跨规模外推。

## 启动条件与目前状态

| 条件 | 当前状态 |
|---|---|
| 公开数据可读、实际 schema 已查 | 完成初审 |
| 原始 Parquet 固定版本且全池核查 | 待做 |
| 任务 family/source/teacher 可信映射 | 待做，已发现字段冲突 |
| 成功标签可验证 | 未通过 |
| recovery 标注可靠且区别错误来源 | 已发现候选，待系统标注 |
| tokenizer/模板/loss mask 与 token 预算确定 | 待做 |
| 训练和评测 smoke test、有实际成本测量 | 待做 |
| 资源分配与最终评测清单明确 | 待确定 |

本轮交付是数据可行性证据与下一阶段协议，没有伪造四组数据集或训练结果。最先解决的是 verifier 结果关联与恢复标签定义，而非新选择算法。
