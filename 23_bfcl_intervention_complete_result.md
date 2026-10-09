# 第一批共同起点经验价值干预：完整结果

2026-10-09 核查：六分支训练、七个完整 16 题评测全部完成，原始 token/mask、共同恢复、预算与面板审计通过；失败清单为空。GPU 4/5/6 已释放。机器结果见 [comparison.json](rl/bfcl_intervention_20261008/comparison.json)。

共同起点 ID 3/8、留出组合 2/8。G 为相对共同起点收益；V 为相对同条件 Uniform 的描述性差异。百分点均基于八题，每题相当于 12.5pp。

| 条件 | 规则 | 新生成 token | 更新 / 新非零梯度组 | ID | 留出组合 | G_ID / G_transfer | V_ID / V_transfer |
|---|---|---:|---:|---:|---:|---:|---:|
| GorillaFileSystem | uniform | 34,245 | 2 / 1 | 3/8 | 3/8 | +0.0 / +12.5pp | +0.0 / +0.0pp |
| GorillaFileSystem | frontier | 38,818 | 2 / 0 | 3/8 | 2/8 | +0.0 / +0.0pp | +0.0 / -12.5pp |
| GorillaFileSystem | coverage | 38,818 | 2 / 0 | 3/8 | 2/8 | +0.0 / +0.0pp | +0.0 / -12.5pp |
| TradingBot | uniform | 40,610 | 4 / 3 | 2/8 | 3/8 | -12.5 / +12.5pp | +0.0 / +0.0pp |
| TradingBot | frontier | 41,125 | 4 / 3 | 2/8 | 3/8 | -12.5 / +12.5pp | +0.0 / +0.0pp |
| TradingBot | coverage | 43,815 | 5 / 0 | 3/8 | 2/8 | +0.0 / +0.0pp | +12.5 / -12.5pp |

GFS Frontier/Coverage 与 Trading Coverage 的最终参数均与起点相同，完整评测逐条 token/奖励/停止行为精确相同。Uniform/Frontier 的 Trading 四题顺序相同，第一组生成 token 也相同；训练后的梯度数值和后续轨迹不同。相同任务暴露不等于相同完整经验，也不保证不同 GPU 的训练逐位一致。不能将这种差异直接归因分配规则。

所有三个改变模型的分支都在 task 190 成功，均对应用户指定 description 末尾句号的修正。Trading Uniform/Frontier 同时在 ID task 124 退化：虽然工具返回价格 227.16，两者下单时填入不同错误价格。Frontier 另外在 165 完成正确订票与预算设置，却在 78 漏做要求的轮胎店查询；它与 Uniform 留出均值相同，但逐题不相同。见 [案例诊断](rl/bfcl_intervention_20261008/final_case_diagnostics.json)。奖励和指标不改，这些案例不足以证明稳定组合迁移能力。

本轮完成了“信号→实际经验→真实参数更新→共同面板收益”的首批测量，但实际干预弱：GFS 两次、Trading 四至五次获取；多个规则暴露相同；初始 Coverage 标量恒为 0.5。单训练 seed、每题单轨迹和八题轴不足以检验预测稳定性。没有宣称 Uniform/Frontier/Coverage 优劣，也没有否定原经验价值问题。

获取成本：共同探测 90,122 token，六分支新增 237,431 token，共 327,553；七面板评测 328,746 token，总计 656,299 模型生成 token。重复输入、外部反馈和时间成本另见机器结果。预算阈值相同不等于实际成本相同。

下一批协议见 [24](24_bfcl_followup_design.md)：全 87 训练池、两个新的条件采样 seed、三条原规则、128K 生成阈值；新的 38 题确认面板与已查看八题 ID 分开报告。复用共同状态的历史依赖和成本明确保留。原始大 scope 与阶段/规模研究仍按 [15](15_scope_and_evidence_ladder.md) 保留，结论按实际证据收窄。
