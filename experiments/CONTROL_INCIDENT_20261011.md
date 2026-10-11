# Goal 控制接口 I/O 故障

2026-10-11 09:02 UTC 左右，在处理已经审阅的 E007 自动通知时，原生 `get_goal` 连续返回 `SQLite code 522: disk I/O error`。同一 daemon 的 `thread/goal/get` WebSocket RPC 也实测返回相同错误。没有证据表明 GPU 实验失败；原四个进程和队列仍存活。

只读检查显示磁盘空间充足，本地 SQLite quick_check 为 ok，但侧读得到的 Goal 记录更新时间约 08:38、tokens_used 61,761，明显早于原生接口此前报告的状态。因此不能根据这个较旧视图断言当前状态或绕过暂停。故障根因尚未确认；不将其归因于硬件或观察器，也不把数据库完整性检查当作服务端连接正常的证明。

已采取的修复：

- 观察器不再直接打开 goals SQLite/WAL 文件，改用本机 Codex 0.162.0 生成的 `thread/goal/get` 协议，通过同一个 daemon 查询当前线程。
- 查询错误、超时、线程不匹配均返回 unknown，不登记/通知新的研究事件；每 30 秒自动重试，恢复 active 后补扫所有完成结果。原始实验结果仍由实验调度器正常保存/发布。
- 只重启本项目的观察器，保留编号/已通知列表；不重启 Codex daemon、不改数据库、不修改 Goal 状态，也不终止或重跑任何 GPU 实验。
- 6 项观察器测试通过，覆盖服务端错误不通知、paused/limited 不通知、错误线程不采信、事件去重与保留人工审阅。

当前边界：观察器的查询路径已修复，但服务端自身的 I/O 错误尚未确认恢复。自动研究决策应在原生 Goal 状态可确认后继续；当前冻结实验不受影响。该记录不是 Goal paused/blocked/complete 的状态变更。

## 后续：仅在全部会话空闲时恢复服务

再次排查确认同一 daemon 加载了三个会话，除了本研究线程，还有另一条活跃会话。因此不立即重启。已准备 `pipeline/recover_codex_goal_when_idle.py`：

- 只对已确认的 SQLite 522 I/O 错误考虑重启；其他错误只等待。
- 连续两次（间隔 30 秒）确认所有已加载会话 idle，包括本线程；分页不全、未知状态、读取失败或任何活跃会话都会阻止重启。执行前再检查一次。
- 已核对本项目调度器及四个 GPU worker 各自有独立 session/process group。执行前再查，恢复器不向它们发信号。
- 最多尝试一次 `codex app-server daemon restart`，先持久化尝试标记；超时也不重复重启。最多等待六小时，失败/超时保留记录。
- 恢复后从服务端查询 Goal；仅当它本来就是 active 时通知当前线程，不执行 set/resume、不改 SQLite。若服务自行恢复，则跳过重启。

恢复器的两个状态决策测试通过，真实 daemon 恢复尚待空闲窗口。状态保存在 `.runtime/control_recovery.json`，完成/到期会生成并尝试发布 `control_recovery_receipt.json`。这是已授权自动化的维护，不新增研究实验或模型会话。
