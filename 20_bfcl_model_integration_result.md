# BFCL 多用户轮次模型接入结果

2026-10-08。继续已有在线 RL 主线，为共同学习状态分叉实验接入候选 BFCL 基准。本轮完成生成与 GRPO 接口验收，**尚未取得 BFCL 非零梯度更新**。

## 固定配置与范围

协议：[BFCL_INTEGRATION_PROTOCOL.md](rl/BFCL_INTEGRATION_PROTOCOL.md)。只用开发集 `multi_turn_base_26`（文件系统，3 个用户轮次）和 `multi_turn_base_70`（车辆工具，2 个用户轮次），各 4 条新交互、最多 2 次 optimizer step。Qwen3-4B base、LoRA r8 q/v、官方 TRL 0.29 GRPO、beta=0、group reward scaling、temperature=1、seed 20261008。

本轮关闭 thinking，仅用于接口验收。完整 completion 上限 4096 token，包含外部消息；每个 assistant segment 上限 1024，最多 16 个 segment/工具调用。只有整个多用户任务的逐轮官方 state/response 检查都通过，才得终局奖励 1；未完成或任一轮失败为 0。未修改奖励、延长预算或根据结果挑选更容易的题。

## 接入路径及检查

`pipeline/bfcl_token_rollout.py` 保留所有原始 assistant token，包括每段生成的 EOS；仅把新工具反馈、下一轮 user 消息与 generation header 编码后追加。外部 token 的 mask=0，原始模型 token 的 mask=1；旧回复不重新 token 化。开始只显示第一轮请求，后续请求按原始 `user` 消息推进，oracle/checker 只在服务端运行。

Qwen 原生模板将语义 role=tool 的消息序列化为 user 分隔符中的 `<tool_response>`；这与下一轮真正的用户请求分开记录。当前实现使用 Qwen 模板，不能当成任意模型的通用 serializer。

本地 Transformers 采样 hook 返回完整 token 序列和 `env_mask`，其后的优势计算与 GRPO loss 仍使用官方实现。运行时核对 mask 完整传入 scoring/loss 输入，loss mask 的有效模型 token 数等于实际生成数；用户和工具反馈不计策略损失。

4 项 CPU token 测试与 4 项 controller 测试通过。实际模型证据审计核对全部生成前缀、轨迹片段、外部桥接、用户顺序、组内优势公式、策略参数版本和获取成本。

## 实测结果

[成功运行审计](rl/bfcl_integration_20261008_002/audit.json)：

- 8/8 条交互完成全部用户轮次，共完成 20 个轮次，无长度截断。
- 官方终局成功 0/8；逐轮检查通过 2/20。
- 新采样 2,179 token；损失排除 1,808 个外部 token。
- 两组奖励均全零；优势、梯度为零，两次 optimizer step 均未改变参数。**不能说已经在 BFCL 学起来，更不能解释为经验分配无效。**
- 从程序 main 开始计时约 66.6 秒，约 0.0185 单卡小时；使用 GPU 4，运行后已释放。

首次尝试在保存官方诊断时遇到 JSON 序列化错误：检查器结果含模拟文件系统 `Directory` 对象。该次已生成 1,016 token，没有 scoring/update；[失败记录](rl/bfcl_integration_20261008_001/failure.json)和原始采样已保留。修正诊断序列化后使用相同配置重新运行。两次新采样合计 3,195 token；首次失败未保存完整 wall time，不能把 66.6 秒当成包含失败尝试的总耗时。

## 失败诊断

[动作重放诊断](rl/bfcl_integration_20261008_002/diagnosis.json)在 CPU 重放已生成动作，没有新的模型调用；终局奖励与原运行一致。18 次失败边界中，11 次状态不匹配，7 次响应不匹配。

模型曾在目录名错误后没有修复，也曾在引擎启动返回条件错误后直接结束当前请求；车辆加油的单位处理也没有符合工具接口。这些属于真实执行/反馈利用失败。

另有一条文件系统交互每轮状态均与参考相同，但读文件使用 `tail` 返回 `last_lines`，官方必要响应要求包含 `file_content`，因此失败。这提示官方 BFCL 奖励约束了执行响应口径，不能无条件外推为任意语义等价的任务能力。**仅检查状态也不能替代奖励**：只读任务可能不做任何动作就保持同一状态。本轮没有为凑梯度修改官方 checker。

重放同时发现诊断对象可能引用会继续变动的模拟状态；现在 controller 在每个 user 边界 deepcopy 检查结果，并补测试验证后续动作不改变先前诊断。该修正不改变奖励结果；原运行证据保留原样，重放诊断采用修正后的快照。

## 对主研究的意义与下一步

共同 checkpoint 恢复和真实多用户生成/mask 接口已具备工程证据。下一项障碍是：固定 learner 是否能在适当开发配置下产生有效终局奖励差异，以及奖励是否足以操作化我们要测的可迁移能力。下一轮先做有界、forward-only 的开发校准和奖励诊断，避免继续在全零组上空更新；仍需审查语义重叠，再冻结科学分叉协议。

本轮未访问 ID/组合迁移评测题，未比较价值信号，也未运行正式训练。开发集上的接口更新不作为科学实验起点；正式训练应从共同 base/预声明 checkpoint 开始。最初的经验价值、迁移及 learner 阶段问题保持不变，本结果只解决其中一个具体接口障碍。
