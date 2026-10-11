# 非零 learner 状态分叉：实现与待运行验收

2026-10-11。当前状态：31 的冻结剂量重复已全部完成并通过 E028 审阅；12:57 UTC 在实际空闲 GPU 4 启动一次真实重放。新 trainer 完整状态恢复检查已通过，三步生成/更新重放仍运行，尚不能宣布验收通过。

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

正式执行入口改为 `/usr/bin/python -u pipeline/supervise_bfcl_stage_fork.py --gpu N`（以 detached 进程启动，继承本线程 CODEX_HOME/CODEX_THREAD_ID）。监督器校验冻结协议 SHA、active Goal、原调度器退出、整批审计通过且审阅已登记，以及所选允许 GPU 无计算进程且显存不超过 64 MiB；独占创建启动记录，拒绝覆盖已有尝试。它调用上述原 runner，不改变训练或重放判据。硬时限到达时终止其自建进程组，最多再给 30 秒退出宽限；非零退出、缺少完成 summary 或软时限 incomplete 均写 `scheduler_failure.json`，由现有观察器登记事件。保留原始失败/部分记录；中断成本只报告已持久化生成 token 下界，明确可能存在未写入的在途生成，不伪装成精确总量。runner 完成仍需独立 replay audit，监督器不会宣布科学或恢复验收通过。

监督器新增四个 CPU 子进程测试：硬超时保留部分日志及成本下界、零退出但 incomplete 进入事件收件箱、非零退出进入收件箱、正常完成仍标为待重放审计；与已有两项分叉测试合计六项通过。首次超时测试因测试子进程字符串换行转义错误而提前退出，修正测试字符串后重跑通过；真实 GPU 尚未启动，冻结协议未修改。

完成后运行 `pipeline/report_bfcl_stage_fork.py --protocol ...`。只有新实例恢复检查和冻结参考精确重放都通过，才允许据此准备科学阶段分叉。若不一致，保存差异与消耗，先定位随机性、进度/微批或优化器恢复原因；不静默重跑、不放宽标准冒充通过。

CPU 验收：边界契约拒绝未完成微批、复用 buffer、多次迭代及进度混淆；真实旧轨迹记录构成的模拟重放文件通过报告器，人为改一个采样 seed 会报告 mismatch。该夹具仅测试报告器，绝不算 GPU 复现。runner 已在禁用 CUDA 的进程完成导入检查。

通过这一工程门槛后，下一科学设计仍需根据完整重复结果确定可识别的阶段/任务组预测检验；不能把阶段恢复成功解释为经验价值假说成立。

2026-10-11 12:50 UTC 后：剂量重复整批审计已通过（E028）。原调度器留下 `/proc` zombie 状态 Z；该状态表示进程已退出、父进程未回收，并非仍在执行。监督器启动检查已修正为明确区分运行/停止进程与 Z/X/已消失进程，不杀进程或改协议；新增回归测试覆盖 R/S/T/D 阻止启动、Z/X/缺失视为已退出，合计七项分叉相关测试通过。真实 GPU 重放仍待下述启动记录与验收结果。

启动证据：[supervisor_launch.json](rl/bfcl_stage_fork_acceptance_20261011/supervisor_launch.json)、[进程记录](rl/bfcl_stage_fork_acceptance_20261011/replay.console.process.json)、[runner 版本与源码哈希](rl/bfcl_stage_fork_acceptance_20261011/replay/run.json)。监督器 PID 1802115，runner PID 1802120；原始日志在同目录。GPU 4 启动前显存 3 MiB、无计算进程；未动 GPU 0–3 的他人作业。

[restored.json](rl/bfcl_stage_fork_acceptance_20261011/replay/restored.json) 确认在新 trainer 中精确恢复参数、raw AdamW 非零动量、scheduler、allocator、buffers、训练模式和 RNG；来源第 6 步、TRL micro step 24，allocator 未重置，继承成本 165,799 token。GPU 已开始新生成。此文件只证明起点恢复，三步逐项重放和终点哈希仍待独立审计。
