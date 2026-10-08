# 主基准候选与短程干预矩阵

2026-10-08。**研究设计草案，尚未冻结新训练协议。** 已生成可核对的题目清单；开发/训练池 runtime、真实多用户 token/mask 接入与 Qwen3 完整状态恢复已有工程证据；固定策略开发校准已出现两组真实终局奖励差异，见 [21](21_bfcl_fixed_policy_calibration_result.md)。文本重叠筛查发现共享首轮子任务，不能声称无模板共享；科学分支尚未训练。不据此声称正式实验已就绪。

## 主测试场候选：本地 pinned BFCL 多轮 base

选用现有 `bfcl_eval==2026.3.23` 中的 `BFCL_v4_multi_turn_base.json`，共 200 题，保留现有问题、工具和初始状态。数据卡标注 Apache-2.0；文件与任务记录哈希见 [audit.json](rl/bfcl_research_split_candidate/audit.json)。[官方多轮说明](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)描述了多轮、状态式 API 与评测机制；本地官方 checker 同时检查每轮状态及必要函数响应，不能只比较最终状态。

这是基于已有 benchmark 的**自定义研究划分**，不是官方 train/test 划分，也不报告官方全榜成绩。BFCL 原本是评测集；将其部分题用于训练时必须在所有报告中明确研究用途与排除规则。本切分不能认证模型预训练未接触过 BFCL。

| 部分 | 题数 | 用途/隔离规则 |
|---|---:|---|
| 训练 | 87 | 仅此部分的历史交互可供分配器使用 |
| 开发 | 22 | 接口、奖励、长度与配置验收；不作最终确认性成绩 |
| ID 新题 | 22 | 与训练存在相同可见工具类组合，题目隔离 |
| 未见工具组合 | 53 | 四种完整工具类组合在训练中全部留出；组成它们的单个工具类在训练中出现 |
| 历史隔离 | 16 | 历史本地 BFCL 生成记录涉及的题目索引；对应增强版本也按 base 索引隔离 |

迁移组合：GorillaFileSystem+MathAPI（11）；VehicleControlAPI+TwitterAPI（14）；TradingBot+MessageAPI（13）；TravelAPI+TicketAPI（15）。它检验的是**已见模拟工具的未见可用组合**，不能直接叫未见工具、执行路径组合或跨真实环境迁移。

[manifest.json](rl/bfcl_research_split_candidate/manifest.json)保存每题 ID、可见工具类、问题/初始状态哈希和用户轮数。划分仅使用可见工具类、题目哈希和既有曝光记录，不使用 gold `path` 或答案作为特征。相同问题及问题+状态的跨划分精确重叠检查已通过；文本/模板筛查与一对共享首轮样例核对已完成，完整语义独立性未认证；见 [筛查](rl/bfcl_research_split_candidate/text_overlap_screen.json)与[核对](rl/bfcl_research_split_candidate/text_overlap_review.json)。开发/ID 在非留出组合内按固定哈希分配；少于三题的稀有组合只入训练，评测并不覆盖全部训练组合。

复现：`python pipeline/audit_bfcl_research_split.py`。当前四种增强类别全部不进入新训练或评测，避免同题变体跨集合。

## 采样前信号与候选规则

- Uniform：共同任务池上的参照分配。
- Frontier：过去训练交互的成功率/可学习性。E0 的逐题 Beta 平滑与混合探索是已有候选；大任务池需要另行定义估计单位及历史窗口，不能把 87 题逐题 warmup 免费提供给它。
- Coverage：过去训练交互对**可见工具类组合**的覆盖不足。拟以组合层级的采样次数定义一个简单覆盖 proxy；名称数量和任务 ID 数量不是结构覆盖。具体公式及探索比例在新协议中冻结，当前尚未实现或训练。

所有起点信号在分支收益产生之前记录。分支也保存在线分配概率和实际暴露；更新后的历史只能影响之后的采样。oracle 路径、最终 ID/迁移分数不进入规则。

## 短程分叉矩阵

| 维度 | 设计 | 当前状态 |
|---|---|---|
| 共同起点 z_early | 在训练部分构建共同学习状态，保存权重+优化器+调度器+RNG+训练/分配历史；从同一文件分叉 | 快照组件、随机 AdamW 续训及真实 Qwen3/TRL 更新边界恢复验收已通过，见 [19](19_qwen3_resume_acceptance.md)。不能复用 E0 两臂分别结束的状态冒充共同起点 |
| 任务组/配置 | 按四个 primary 工具类组织预先定义的训练任务组；组内比较三种规则，并检查各组存在实际可改变的工具组合暴露 | 待多轮成本/信号验收后冻结任务组和有效干预长度 |
| 规则 | 每个任务组下 Uniform / Frontier / Coverage，算法、奖励和 rollout 配置相同 | 目标为多个干预条件，而非三个总体均值 |
| 起点参照 | 未继续训练的同一 z_early 在共同 ID/迁移集的分数 | 必须测，E0 未测 base 不能声称 RL 净增益 |
| 获取预算 | 全部生成 token；起点构建/探测、分支获取和评测分账；实际超额及输入/GPU 开销单列 | E0 与 BFCL 固定策略成本已有；本轮 65,500 生成 token / 约 0.507 单卡 main 小时，不把生成 token 当全部计算成本 |
| 确认重复 | 独立训练/采样重复，共享起点与对照按配对设计分析 | 数量/种子须在采样前冻结；评测题数不替代训练重复 |
| 信号校准 | 起点信号对各分支 ID/迁移 G、相对 Uniform 的 V 的预测；留出部分任务组/干预条件验证 | 不用三个规则均值拟合普适关系；不以事后相关性作轨迹属性因果证据 |
| 第二阶段 | 按训练预算预先选择 z_later，复验最有信息量的相同对照 | 阶段选择不看最终迁移排名；阶段学习状态差异需明确 |

E0 实际分配阶段只有 4/5 组，前四次选题完全相同，且 warmup 后状态不相同。这说明后续须保证有可测的分配暴露差异，并从**同一个完整状态**分叉；重复当前小 pilot 不会自动补齐证据链。

## 启动新采样之前剩余工作

1. **开发集已验收：** 22 题共 83 个用户轮次的 oracle 检查通过；no-op、重置、实例隔离及非字面量/未文档化调用反例通过。新执行器与官方执行器的标准参考动作状态/响应一致。证据见 [runtime_acceptance.json](rl/bfcl_research_split_candidate/runtime_acceptance.json)，复现 `.venv-rl/bin/python pipeline/bfcl_safe_runtime.py`。后续已补真实开发模型采样与全部 87 训练题 / 317 轮 runtime 自一致验收，见 [training_runtime_acceptance.json](rl/bfcl_research_split_candidate/training_runtime_acceptance.json)，没有 ID/迁移得分。更广的奖励语义仍有局限；oracle 自一致不证明完整奖励有效性。
2. 真实 user 轮次控制器在开发集 22 题的完整 oracle 对话验收已通过，见 [18 组件验收](18_learning_state_and_conversation_acceptance.md)。实际 Qwen3/GRPO token/mask 接口已通过，开发校准已出现真实混合终局奖励；下一步进入共同起点的科学干预；不能把后续用户消息换成普通工具反馈后仍称完整 BFCL 任务。若使用第一轮子任务，必须另名、单列奖励和结论范围。
3. 本地官方调用执行器使用 Python `eval` 和全局缓存。新模型调用应通过方法白名单、AST/literal 参数或结构化工具分发执行，并显式重置每个 episode；保留标准 checker 语义。八个模拟 API 的实际主机副作用须检查，不能从类名推断。
4. 审查语义/模板重叠；如需调整清单，保存新版本，不追认当前候选为已冻结划分。
5. 沿用已验收的完整 checkpoint/optimizer/history 恢复路径，冻结具体分支预算、起点成本、重复、评测采样、信号可见性与停止规则。

更强的跨工具/跨环境测试保留为扩展。若 BFCL 只能支持组合层级的可靠测量，第一篇相应限制迁移结论，长期 scope 不取消。
