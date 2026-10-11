# 非零 learner 状态分叉：实现与待运行验收

2026-10-11。当前状态：独立 runner 与审计实现完成，CPU 边界检查和历史数据审计夹具通过；真实 GPU 重放尚未运行。四卡仍执行 31 的冻结剂量重复，保持原队列。

## 为什么现在做

原目标是测量经验价值与 learner 阶段的关系。已有组件快照验证只保证保存/同一实例恢复，不足以证明一个新 trainer 能从训练后的学习状态正确分叉。E003–E005 同时显示 fresh 梯度为零时参数仍可受 AdamW 历史动量影响；只加载 LoRA 权重会改变干预起点。

## 冻结的工程对照

[协议](rl/BFCL_STAGE_FORK_ACCEPTANCE_PROTOCOL_20261011.json) SHA256：`5f60c862c82a1571ee2e95c11b84f03ec00fd11dea99da910c84201de8c97233`。

从旧发现轨迹 `train/doses/step_006` 的完整组件状态，在新 Qwen3-4B/TRL trainer 中恢复并重放原第 7–9 步。选择依据是可检查的优化器状态：第 7、8 步 fresh 梯度为零但参数改变，第 9 步有非零 fresh 梯度。没有根据本次重复开发分数或迁移测试选点。原后续记录作为冻结参考，是工程回放，不增加独立科学重复数。

- 祖先累计成本 165,799 token = 有偿探测 90,122 + 轨迹前六步 75,677。
- 新生成上限三组 × 四轨迹 × 8,192 = 98,304 token；原参考三个组实际 31,078 token，重放仍重新计费/计成本。
- 恢复真实 raw AdamW 非零动量、常数 scheduler、参数/缓冲区/训练模式、RNG、allocator 完整历史和私有 RNG。不得重新 seed allocator、清历史或换任务池。
- HF optimizer step 是本分支局部 0–3；累计 learner step 是 6–9。TRL micro step 显式恢复 24，后续 28/32/36。现有 num_iterations=1 且一组四 rollout 恰好对应一个 optimizer step，边界后必定新生成，不带未消费 buffer。
- rollout seeds 延续旧轨迹：20266810/20266910/20267010；原始生成、外部反馈 mask、奖励、优势、每步参数哈希与梯度范数逐项比较。包括两个动量更新及一个非零梯度更新。

这是已完成更新边界的新分支接入，不声称支持任意 HF dataloader resume、任意 GRPO 配置或嵌套分叉。当前已有 dataloader 行被 allocator 选定的任务整组替换；新旧流程每组按累计步骤重新设采样 seed，因此本工程对照的种子与真正选题来自继承状态和显式映射。

## 验收与后续决定

GPU runner：`pipeline/accept_bfcl_stage_fork.py --protocol rl/BFCL_STAGE_FORK_ACCEPTANCE_PROTOCOL_20261011.json`，使用 `.venv-rl/bin/python` 和一张可用 GPU。当前四卡队列完全退出后，先审阅其结果与资源，确认 Goal active，再启动；不抢占任何已运行任务。软时限 2,700 秒、外部硬时限 3,600 秒。

完成后运行 `pipeline/report_bfcl_stage_fork.py --protocol ...`。只有新实例恢复检查和冻结参考精确重放都通过，才允许据此准备科学阶段分叉。若不一致，保存差异与消耗，先定位随机性、进度/微批或优化器恢复原因；不静默重跑、不放宽标准冒充通过。

CPU 验收：边界契约拒绝未完成微批、复用 buffer、多次迭代及进度混淆；真实旧轨迹记录构成的模拟重放文件通过报告器，人为改一个采样 seed 会报告 mismatch。该夹具仅测试报告器，绝不算 GPU 复现。runner 已在禁用 CUDA 的进程完成导入检查。

通过这一工程门槛后，下一科学设计仍需根据完整重复结果确定可识别的阶段/任务组预测检验；不能把阶段恢复成功解释为经验价值假说成立。
