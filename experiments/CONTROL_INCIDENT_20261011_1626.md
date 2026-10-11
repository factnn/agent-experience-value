# 16:26 UTC 第二次 Goal 控制服务 I/O 故障

2026-10-11。任务级机会实验协议与代码已冻结/push（f0e64fca），监督器 PID1869880 在初始 Goal 守卫连续三次 unknown 后退出。尚未创建任何GPU子进程，新生成0 token。随后原生 get_goal 返回 `failed to read goal: ... (code: 522) disk I/O error`；同一 daemon 的独立 RPC snapshot 同样确认 SQLite522。

只读快照显示两个已加载会话、一个 active（当前工作回合）、零 unknown，因此现在不能重启。四张允许GPU均为3MiB，没有我们的活动GPU作业。文件系统仍可读写，项目卷剩余约523T；不据此认定根因或数据库正常。

使用本次独立 incident `20261011_1626`，复用此前验证的 idle-only 恢复逻辑。给恢复器增加显式 incident 参数，为运行状态和收据分开命名，保留09:21那次恢复及“已尝试一次”记录，不清空或覆盖它。三个CPU测试通过，包括默认路径兼容、不同incident隔离、两次idle/最多一次重启、active/unknown禁止重启。

恢复器只有在所有加载会话连续两次idle（间隔30秒）且执行前再次确认时，才允许本事件唯一一次 daemon restart；最久等6小时。不会改Goal、不会直接访问SQLite，也不会停止其他会话或GPU进程。恢复后从服务读取状态，仅当原本active才通知同一线程。新科学采样仍需重新get_goal并走同一守卫；原失败根目录和协议保留，新的执行尝试另行登记。

官方 App Server 文档支持用 `thread/goal/get` 读取 Goal、`thread/loaded/list` 与 `thread/read` 读取运行会话状态。[OpenAI App Server 文档](https://learn.chatgpt.com/docs/app-server)。这里的SQLite522判断和一次性空闲重启策略依据本机证据与此前恢复实践，不是官方文档保证的522修复方案。

本次失败证据：`rl/bfcl_task_probe_stage0_20261011/failures.json` 与 `preflight_failure.log`。恢复收据预定为 `experiments/control_recovery_20261011_1626_receipt.json`，运行状态在 `experiments/.runtime/control_recovery_20261011_1626.json`。不是 Goal paused/blocked/complete 的状态变更。

恢复器已以独立session启动，PID1871355；首个真实snapshot仍为两个会话、一个active、零unknown，decision=wait，尚未尝试重启。
