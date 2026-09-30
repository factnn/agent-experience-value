# 07 — Recovery 事件标注规范 v0.1

## 为什么用事件而非整条轨迹单标签

一条轨迹可能同时包含：题目已有缺陷 → agent 错误修改 → 环境问题 → 局部恢复 → 再次破坏 → 最终自报完成。单一「有错误」「有反思」标签会混淆这些过程。

以事件为单位保存 `error_turns / correction_turns / verification_turns`（从 0 开始），再汇总到轨迹层。使用 `events: []` 允许多个来源共存；初始包中的 annotation.error_origin 只是待标注占位字段，不能强迫整条轨迹只能选一种。

## 字段

- error_origin：task_bug / agent_action / environment / unknown。可多事件，单事件也允许说明归因不确定。
- recovery_status：attempted / locally_verified / unresolved / unknown。没有事件时明确 `events=[]`，不要用 unknown 冒充无错误。
- terminal_outcome：pass / fail / unknown；只有关联到可信的独立 verifier 才允许 pass/fail。局部测试、助手 task_complete、runtime_result 均不能替代。
- evidence_turns：每个错误、纠正、验证对应消息索引，保存关键片段并保留完整原文链接。
- annotator / human_verified / confidence / ambiguity_notes：区分 assistant 初审、人工复核与独立标注。

## 判定步骤

1. 读初始任务：错误是否就是题目要求修复的原有缺陷？
2. 看实际反馈：区分真实 shell 输出与命令回显、代码字符串、模型推理。
3. 对齐后续动作：动作是否针对该错误改变策略/命令，还是重复同一失败？
4. 找后续反馈：仅提出安装/修改只能标 attempted；修正后相关命令或测试实际执行成功才标 locally_verified。
5. 检查再次破坏：曾经修好又坏不能概括成「最终成功恢复」。保留多事件或 regression_after_recovery 标记。
6. 独立查看 terminal_outcome。若 verifier 不可得，保持 unknown。

## 本轮已初审示例

[reviewed_events.json](prepared/reviewed_events.json) 保存 3 条轨迹中的 5 个示例事件，由 assistant 阅读证据后标注，**未经人类复核，不是金标准或训练标签**。

- SFT-100K viewer row 1：turn 4 是任务已有的 QR bug；turn 10/16/24/30 出现 agent 错误编辑/代码损坏；turn 36/38 有局部运行成功证据。两类事件同时存在。
- 同条轨迹 turn 20：bash history expansion 引起测试命令失败；turn 21 修改字符串、turn 22 实际执行成功。是局部命令恢复，不能证明代码整体正确。
- AgentTrove row 188540：git 缺失，最后助手提出安装，但已存对话没有后续执行反馈，只能标 attempted。
- AgentTrove row 4：反复修复 heredoc / 换行问题，多次收到相同失败反馈。反复说「我会修复」不等于恢复成功。

这修正了 05 文档仅观察首个错误片段的简化：SFT-100K row 1 既包含题目 bug 修复，也包含后续自我纠错，不能整体排除出 recovery 候选。

## 评估标注质量

200 条随机主轨迹用于估计候选规则漏检与误检；100 条候选过采样用于覆盖错误类型。两组可能重叠，按 row/hash 去重并保留分组；不能把合并集当均匀样本。采样以唯一对话为单位，不是任务家族均匀采样，重复任务家族仍可能出现。

人工标注完成后分别计算事件检出精度、不同 error_origin 混淆、attempted 与 locally_verified 混淆。在至少一部分数据上独立双标再仲裁，才能报告一致性。当前不报告这些未测指标。

## 使用边界

首轮 recovery 策略只在标注定义经过校验后扩展。命令解析成功率衡量格式可读性，不衡量操作类型的语义准确率；第一命令启发式不能充分解析管道、heredoc、多行 Python 编辑和复合 shell。
