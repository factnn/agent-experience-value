# 共同状态恢复与多用户轮次：组件验收

2026-10-08。继续复用当前在线 RL 管线，补 E0 暴露的共同分叉起点问题，并为既有 BFCL 候选准备真实多用户轮次。**本轮是 CPU 组件验收，没有新增模型训练或经验价值结果。**

## 共同学习状态

`pipeline/rl_learning_state.py` 保存可训练权重、所有 named buffers、模块 train/eval 状态、AdamW/其他优化器状态、参数组顺序、调度器、Python/NumPy/Torch RNG、可选 CUDA RNG、训练进度和分配历史。冻结基座不重复落盘，但逐参数哈希核对；task manifest 和算法配置哈希、base revision 必须由调用者提供并在恢复时匹配。

保存只允许在 optimizer update 和 `zero_grad(set_to_none=True)` 之后；分配器也必须已经观察完整组。保存 pending rollout 或在共同 warmup 完成前切换分配规则会被拒绝。局部可信二进制快照不上传 git，JSON manifest 与验收证据可审阅。每个实际训练分支应在独立进程恢复；同进程多模型共享全局 RNG，不能当成互不干扰的在线分支。

分配器现在支持 checkpoint：保留任务注册、最近奖励、共同 warmup 顺序、独立 RNG、已完成 batch、全部累计获取成本及 episode/selection 计数。分叉可以保留全部历史与成本，仅更换下一步的 uniform/frontier 规则。规则更换不意味着此前信息免费获得，也不保证两个规则实际选到的任务不同。

CPU 验收：[learning_state_cpu_acceptance.json](rl/learning_state_cpu_acceptance.json)，5 项检查通过。

- 带 dropout、Python/NumPy/Torch 随机输入、AdamW 和 StepLR 的不中断更新，对比从快照恢复的下一步：输入、损失、各参数、调度器与分配历史逐项相同。
- 权重/RNG/LR 相同但清空 Adam 动量：下一步参数不同，说明 weights-only 不能替代完整学习状态。
- Uniform/Frontier 从同一保存状态恢复：权重、优化器动量、训练历史与已付成本相同，规则概率不同；一分支新增历史不修改另一分支。
- provenance、冻结基座、optimizer 参数组顺序及 payload 校验不匹配被拒绝。
- 组边界与 JSON roundtrip 的分配 RNG/下一次选择保持一致。

复现：`.venv-rl/bin/python pipeline/test_rl_learning_state.py`。这是小型 CPU learner 的组件结果，**还没有验证 Qwen3 LoRA / Accelerate / HF trainer 的整套恢复**。后续须在真实 optimizer 边界恢复 global step、microstep、callback 与历史，并清空上游 buffered inputs，确认下一组是当前策略的新交互。数据采样进度也必须显式处理，不能把通过这五项检查直接叫完整在线分叉验收。

## BFCL 多用户轮次控制器

`pipeline/bfcl_conversation.py` 调用已验收的 episode-local simulator 和官方 state/response checker。开始只显示当前请求；assistant/工具响应结束后才加入下一批原始 `user` 消息。oracle 动作和逐轮评分留在服务端，不进入策略观察。工具调用错误可返回反馈，不自行推进用户轮次。

终局奖励要求完成所有用户轮次且每轮检查通过。未完成的任务得零；已失败轮次的结果被锁定，后续恢复状态不能追认之前的轮次通过。这与 BFCL 逐用户轮次检查和 force-terminated 失败的要求一致。

CPU 验收：[conversation_acceptance.json](rl/bfcl_research_split_candidate/conversation_acceptance.json)，3 项检查通过，覆盖开发集全部 22 题（83 个用户轮次）的完整参考交互、原始 user 消息顺序、未完成任务、延迟修复和工具错误。没有进行 ID/迁移评测、模型采样或 GPU 训练。复现：`.venv-rl/bin/python pipeline/test_bfcl_conversation.py`。

## 本地 TRL 接入的实际约束

核对已安装 TRL0.29 源码：`rollout_func` 传入 `VLLMGeneration`；本地 Transformers `_generate_single_turn` 直接调用 `model.generate`，没有执行该 callback。当前机器的旧 vLLM/Transformer Engine 与隔离 RL 环境存在既有版本冲突，因此不为这个组件验收重装系统依赖。

可复用的路径是本地采样适配：把控制器的完整交互送入官方 GRPO scoring/loss，并提供真实模型 token 与外部 user/tool token 的 mask。上游 `_generate` 在无 tools 时支持 `extra_fields['env_mask']`；可以利用这一接口，但本地采样 hook、原始 token lineage、user/tool 桥接、结束标志、全部采样成本与 loss 仍需实际验收。

不能仅把后续用户请求包装成工具返回而称为完整 BFCL；也不能用重新 token 化的“看起来相同”对话替代原始生成 token。此接口工作只为已有主实验解决具体障碍，不另搭通用 agent 框架。

## 下一项有界验收

先在原有小工具环境验证 Qwen3/TRL 完整 checkpoint 恢复与一组 fresh rollout/update，不改已完成 E0 协议。随后在 BFCL 开发任务上接通真正的多用户轮次，验证原始生成、mask、奖励和有效更新。成本与配置确认后再冻结科学分叉实验；当前候选切分与研究范围保留，不因接口方便而重新定义 goal。

### Qwen3 工程实测进行中

单卡 4、同一 send fixture、seed 20260930、2048 thinking tokens/trajectory、4 rollouts/group、GRPO 两步；每个进程硬限 1200 秒。先不中断两步，再由 step-1 的 HF checkpoint 和组件快照恢复第二步。检查新采样、优势、梯度、完整状态和累计成本；不测价值信号优劣。结果尚未产生。

兼容性：现有 Torch 2.5 下，Transformers 5 拒绝原生 optimizer/RNG 的 `torch.load`。恢复流程使用 HF 的 safetensors adapter 与 trainer progress，并由本项目 callback 恢复本地生成、checksum/provenance 校验后的组件快照；跳过原生 optimizer/RNG loader。没有关闭全局安全检查或升级共享环境。此路径与未经修改的原生 HF resume 有区别，需单独验收。
