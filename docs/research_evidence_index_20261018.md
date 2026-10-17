# 研究证据、版本与复现入口

本页是已有证据的取用索引，按完成范围区分已验证结果、前瞻预测和仍未产生的标签。机器可读引用及 SHA256 见 [证据索引收据](../rl/research_evidence_index_20261018.json)。执行 agent 整理，不构成独立人类复核，也不宣称整个研究完成。

## 从研究问题读到证据

研究对象是在线获取经验后，完整学习系统带来的迁移变化。G 相对同阶段共同起点；V 相对同阶段对应训练重复的 Uniform。局部混合奖励机会、终端成功、优化器步骤、新梯度、权重变化和迁移收益分别读取。当前不能从局部校准改善、正 V 或最低预测 MSE 推出稳定的高价值采样规则。

原问题和假说从 [01 scope](../01_scope.md)、[15 证据层级](../15_scope_and_evidence_ladder.md) 进入，当前覆盖与缺口以 [SCOPE_REVIEW](../experiments/SCOPE_REVIEW.md) 为准。以下均为带时间的完成证据；实时运行须查看服务器上的原队列和对应 PID。

## 已完成的 Qwen3-4B 证据

Stage0 原 51 条在线 RL 训练、80 个拟合面板、预测封存与远端核验、24 个留出面板全部完成。完整 [科学审阅](../rl/bfcl_source_stage0_transfer_prediction_20261015/SCIENTIFIC_REVIEW.md)、[原始重放与评分](../rl/bfcl_source_stage0_transfer_prediction_20261015/audit_status.json)、[CPU 复现命令](../rl/bfcl_source_stage0_transfer_prediction_20261015/REPRODUCE.md) 和 [检查点取得](../68_stage0_artifact_retrieval.md) 是本阶段入口。报告的 evidence 字段绑定原执行协议、拟合和留出清单、实际封存收据及每个面板文件；implementation_sha256 绑定当时分析实现。

Stage1 使用事前指定的 Stage0 Uniform repeat0 完整 7 步终点，按用户决定收窄为 15 条训练／32 个评测，全部完成，原晚 51／104 及阶段内重新拟合延后。[完整报告与 claim–evidence](../rl/bfcl_source_stage1_forward_report_20261016/README.md)、[原始重放与评分](../rl/bfcl_source_stage1_forward_report_20261016/audit_status.json)、[原预测及标准化逐行文件](../rl/bfcl_source_stage1_forward_report_20261016/frozen_predictions_and_observations.csv) 和 [复现／取回说明](stage1_forward15_reproduction.md) 共同界定完成范围。父学习状态的优化器、历史和 RNG 必须保留，不能只用相同政策权重代替。

[4B 结果正文草稿](4b_results_draft_20261017.md) 汇集已完成的有限发现。主张与证据对应如下：

- “局部预测更准”对应 Stage0 原探测结果和 Stage1 E303，不自动成为真实迁移结论。
- “任务级预测改善不能由局部优势保证”对应原 Stage0 留出预测误差；其相对均值不利的结果保留。Stage1 的小幅误差改善及整体偏差／中心化误差分解另行报告。
- “V 为正不推出自身正 G”由 Stage1 的来源 G、配对 Uniform 和 V 逐重复值支持；相关排序不算两次独立验证。
- “有些零新梯度分支仍改变参数”对应 Stage1 E313／E316／E359 和固定父优化器状态，与继承动量解释一致，尚无数值反事实识别。
- “已有稳定的经验选择规律、某机制普遍失效或普遍胜出”当前没有证据支持。四留出来源、共享任务、祖先与重复的依赖进入所有解释。

## 原生 Qwen3-8B：已封存与待验证必须分开

当前 [E369 原科学设计](../rl/QWEN3_8B_STAGE0_FORWARD15_SCIENTIFIC_DESIGN_20261017.json) 是同四留出来源各三次真实在线 RL 加三次 Uniform，再做共同起点与 15 个终点各两次评测。原生 8B 使用自身共同起点和空 AdamW，不能与 4B 晚阶段的暖优化器状态混为纯参数量对照。

8B 的共同与 16 来源采样前测量已经完整审计。原五规则、独立 H1、两个结构规则的 [实际 128 点封存收据](../rl/qwen3_8b_cross_size_prediction_seal_20261017/receipt.json) 及 [发布证明](../rl/qwen3_8b_cross_size_prediction_seal_20261017/publication.json)，两个模型条件代理的 [实际 32 点收据](../rl/bfcl_model_conditioned_prediction_seal_20261017/receipt.json) 及 [发布证明](../rl/bfcl_model_conditioned_prediction_seal_20261017/publication.json)，连同设计内 16 个零收益预测，共 176 个实际事前点，均先于当前源训练。原斜率、4B 标准化、极端外推和负值保留。只有四个来源会在本 cohort 获得 G 标签，另外十二个来源的预测明确未验证。

截至 [E380 的完整八分支审阅](../rl/qwen3_8b_stage0_eight_complete_reward_review_20261018.json)，原 15 条训练尚未全齐，32 个目标面板尚未登记和运行，也没有 8B G/V 或最终预测误差。该记录包含各完成分支原始审计引用、奖励组成及新分支 raw／checkpoint 的 SHA、大小和授权服务器绝对取回路径。局部 Brier、权重簇、部分训练完成数都不是迁移收益。

实际执行版本由 [15 训练绑定](../rl/QWEN3_8B_STAGE0_FORWARD15_EXECUTION_20261017.json) 的 runtime_code_sha256、runtime_versions、原 job SHA 和 [源预测依赖](../rl/qwen3_8b_stage0_forward15_binding_20261017/source_prediction_binding.json) 确定。冻结绑定所在 Git 提交为 a745b6bb476f2d7ea0215b441eac3cd45313c011；[原发布凭据](../rl/qwen3_8b_stage0_forward15_binding_20261017/publication_verified.json) 与真实进程开始时间另存。后来增加的文档提交不代表运行时替换了冻结代码。

全 15 条原始审计、执行审阅和 owned 进程终止后，才按当前独立评测归属生成原 32 面板清单并核验发布。已有最终 CPU 报告入口为 pipeline/report_bfcl_qwen8_forward15_science.py；目前没有实际 target manifest SHA，故本页不提供可误执行的占位命令。完整结果后补入实际 manifest、完整证书、报告和复现命令。

## 环境、原始文件与取回

实际 RL 环境为 .venv-rl；[RL 依赖锁](../pipeline/requirements-rl-lock.txt) 与 [入口依赖](../pipeline/requirements-rl.txt) 单独保留。当前 8B 绑定记录 Python 3.10.12、torch 2.5.1+cu121、transformers 5.2.0、trl 0.29.0、peft 0.21.1、accelerate 1.15.0、datasets 4.5.0、tokenizers 0.22.1。旧 .venv-train 或通用 runtime_versions 文件不能替代本批绑定；Stage1 复现元数据是有日期的环境快照，不能补写为历史安装连续性证明。

Git 保存代码、协议、元数据及已登记证据；模型二进制仍在授权服务器。4B 按上述两个取回说明取得共同状态、全部终点 adapter 和完整 learning_state.pt，并以 Git 原 manifest 校验。基础模型另按原 revision／文件哈希取得，8B 缓存不替代 4B 模型。

8B 只取已完成且有原始审阅引用的目录；正在写入的日志不能充作完整归档。以 E380 新完成分支为例，在独立副本中用自己获授权的 SSH 别名复制：

```bash
RL_SSH=your_authorized_ssh_alias
RL_REMOTE=/share/project/zpy/RL/agent_experience_project_20260928
mkdir -p restored-native8b/rl/qwen3_8b_stage0_forward15_train_20261017
scp -P 2222 -r "$RL_SSH:$RL_REMOTE/rl/qwen3_8b_stage0_forward15_train_20261017/q8_s0_group_08_r0" restored-native8b/rl/qwen3_8b_stage0_forward15_train_20261017/
```

复制后从本地 Git 中的 E380 记录读取 raw_and_checkpoint_refs，核对 bytes 和 SHA256；不要用复制来的 manifest 替换 Git 校验依据。完整重放还需取得原共同状态、相应原始输入与基础模型；仅复制一个 adapter 不足以复现。此例不启动实验，仓库不提供凭据或公开二进制下载服务。

## 成本、失败与仍未覆盖范围

4B Stage0 成本在原报告 costs 和 training_cost_and_exposure.json，Stage1 的逐分支／逐面板／共享探测／超额和故障在 [成本失败账](../rl/bfcl_source_stage1_forward_report_20261016/failure_and_cost_ledger.json)。8B 共同与探测、有限 forward、当前源训练和未来目标分账；共享内容只计一次。E361 容量例单列，不重采。部分前缀不是整批最终成本，上限不是 ETA 或实际 GPU 时间，G/token 也不能自动消除剂量混杂。

科学子进程失败、准入／发布／SSH 查询故障和 CPU 后处理问题分别保留；相关 8B 记录包括 E365、E375、E380，不把它们统称训练负结果，也不因发布或分析故障重跑科学任务。本索引制作新增科学 token、更新和 GPU 作业均为零。

ID、非 Qwen 家族、适用的梯度影响与更强机制识别、环境外推及另行协议下的 21 道保留目标仍未完成。后续批次根据完整 8B 结果选择关键解释，独立登记政策、目标统计量、实际事前预测、资源上限与停止条件；本页不批准新批次，不恢复晚全矩阵，也不缩减整个 Goal。
