# Goal可读，但自动事件入队失败

2026-10-12 02:17 UTC状态检查：原生Goal为active，观察器心跳正常，E055–E071已登记但未审阅，四卡采样批次已于00:05:46完成。实际存在约2小时11分的批次结束到本次检查间隔，期间没有新科学作业。

02:20同线程 `codex queue` 真实投递失败：`thread/queue/add failed`，`queue storage failed`，SQLite `(code: 522) disk I/O error`。失败发生在消息存储，不是GPU队列或实验错误。Goal读取成功不能认证消息存储可用；目前不足以把原生自动Goal没有续跑的全部原因唯一归到此错误。

旧观察器 `notify()` 只返回bool，失败后主循环无条件写last_error=null，因此界面掩盖了通知错误。旧state只记录E001–E013通知成功、最后成功时间09:47:47；不能把后续每个事件都认定有实际投递。这次已补审E055–E071，不重跑完成作业。

修复：`research_queue.py` 返回时间、queued、返回码和脱敏错误类别；观察器保留last_notification，只有queued=true才标已通知，明确失败时写last_error。不保存命令原始输出或凭证。5项CPU检查通过，包括旧Goal恢复逻辑兼容、522错误分类、所有会话空闲门槛、非active禁止重启、每事件最多一次重启。

独立恢复器 `recover_research_queue_when_idle.py --incident 20261012_0220` 只在本Goal为active且所有已加载会话连续两次idle（间隔30秒）时检查队列；只有明确queue_storage_sqlite_522、再次即时核对全idle，才允许该事件唯一一次daemon restart。重启前持久化attempted标记，6小时有界等待，不访问数据库、不恢复/修改Goal、不杀GPU进程；未知投递超时不重复投递测试。使用与已有恢复器相同的全局锁，不能并行重启。当前工作回合active时不会执行重启。

实际恢复需要真实投递成功，最终收据为 `queue_recovery_20261012_0220_receipt.json`，运行记录在 `.runtime/queue_recovery_20261012_0220.json`。启动脚本不等于修复成功。诊断收据见 `queue_delivery_diagnostic_20261012.json`。

本机方案沿用既有空闲恢复实践；官方文档并未承诺重启能修复SQLite522。OpenAI Docs的[App Server文档](https://learn.chatgpt.com/docs/app-server)支持读取已加载会话状态；[Goals文档](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)说明续跑依赖空闲边界、无其他待处理工作及Goal状态，不能只看active就宣称持续推进。
