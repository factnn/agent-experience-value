# 自主研究循环

用户于 2026-10-11 授权按完成事件继续分析和调整实验，不需要逐次 go。当前线程已建立原生 active Goal。目标仍是经验价值对真实在线 RL 迁移收益的预测，以及其 learner 阶段依赖；规模、家族、环境外推保留。负结果同样可以完成证据链，不以阳性为目标。

## 三个层次

1. 现有四卡调度器完成已冻结的训练/评测对照。
2. `pipeline/watch_research_events.py` 每 30 秒观察登记的输出；完成、失败和批次审计分别写 `E001.md` 起的记录，生成索引并 push。观察器不自行选实验。
3. 原生 Goal 推进研究；观察器还用 `codex queue` 向同一线程投递新事件。agent 读取实际证据，写分析与下一决策，然后修改下一轮实验计划、验收并执行。

## 每次自动继续

先 `get_goal`。若非 active，不因自动消息恢复任务。检查 `experiments/README.md` 待分析事件、`NEXT.md`、`rl/STATUS.json`、GPU/进程和 Git。机器事实完整与否必须单独判断；失败也登记，不默认重采样。

读取原始 summary/outcomes/protocol，必要时执行对应审计。在对应 E 文档写实际观察、解释、限制和决定；`reviews/E###.json` 至少含 `status: reviewed`、审阅时间、证据、结论、下一动作。不得只创建模板就声称已分析。每批完成另写整批统计和审计，不将每个执行作业误当独立实验样本。

更新 NEXT，明确下一步假设、判别标准、依赖。与现有工作独立的实现/CPU 验收可并行；GPU 实验在依赖满足时启动。新实验先冻结协议并将输出根目录加入 watch_roots；已运行对照不能根据部分测试分数删臂或改奖励。新批次事件页需在审阅时补充其具体设计与协议链接。

每个实质进展显式 commit/push；不用 git add -A、不覆盖别人未提交文件、不删除 Git lock、不强推。后台观察器只发布自己生成的记录；agent 自己发布审阅与代码。观察器通知成功代表入队，实际阅读以 reviews 确认。失败会保留并重试；异常与心跳在本机 `.runtime/state.json`。

## 定期回顾原始scope

用户于2026-10-12要求定期回忆与收束。每批科学实验完成审阅、每次新科学协议/范围调整之前，以及距上次回顾发生三次实质研究决策时，读取 [SCOPE_REVIEW](SCOPE_REVIEW.md)、[01](../01_scope.md) 与 [15](../15_scope_and_evidence_ladder.md)，更新原目标—证据—缺口—下一决策。用户提供方向审阅文件也触发回顾。记录有依据的结论收窄，不能只重复“方向不变”。

当前候选主要集中于可学习性，尚非机制不同假说的全面比较；任务级Beta、组合级Beta、Frontier不能冒充三种独立价值机制，Coverage计数不等于结构多样性。局部混合奖励预测、代码验收和单个ridge结果不能替代真实迁移收益结论。阶段/规模/家族/环境及比较广度的未覆盖部分必须持续可见。

回顾是agent决策步骤，不新增后台定时器，不阻断已冻结实验，也不修改Goal。第一篇可形成有限结论，但不能因此自动把长期Goal标完成。最新完整回顾见 [53](../53_gpt12_scope_review.md)。

## 运行与停止

在继承本会话 CODEX_HOME/CODEX_THREAD_ID 的环境启动 `python pipeline/watch_research_events.py`，用 detached 进程和 `.runtime/watch.log` 保存输出。重启通过已有 index 去重，不重复编号或重跑实验。单实例 flock。Goal 状态通过同一 Codex daemon 的 `thread/goal/get` 查询，不直接打开 SQLite；接口不可用或返回其他线程时不投递，并每 30 秒重试。Goal 暂停/限额/结束时停止自动研究通知；已启动的独立 GPU 作业不会因此自动杀掉，用户说停作业时另行处理。

机器、Codex daemon 和账户额度必须可用；这不是不受约束的永动机。只最多四张实际空闲 GPU，不碰他人进程，不新增付费 API。原生 Goal 在暂停/额度/生命周期边界下可能停止；观察器也不能跨机器断电运行。需要暂停可用 `/goal pause`，恢复 `/goal resume`。

官方机制说明：[Using goals in Codex](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)。本机 CLI 0.162.0，已确认提供 queue 命令及正在运行的 daemon。

## 本机验收记录（2026-10-11）

原生 Goal 已 active；实际 `codex queue` 向当前线程提交 E001–E005，08:38:08 UTC 返回成功。此处确认入队，自动新回合的处理需由后续事件记录证明。E001–E005 已在当前设置回合实际审阅并写入 reviews。观察器不会将通知成功冒充科学审阅。

08:40 UTC 后真实监控消息已触发下一回合；agent 首先确认 Goal active，核对 E001–E005 已审阅，随后开始非零学习状态分叉验收实现。这验证了入队 → 新回合 → 读记录 → 推进下一工作的路径。

09:02 UTC 左右 Goal 接口出现 SQLite code 522 I/O 错误；侧读数据库能返回更旧的 active，不能替代服务端状态。已移除观察器直接读库路径，改用本机生成协议确认并实测的 daemon WebSocket `thread/goal/get`。故障及恢复跟踪见 [CONTROL_INCIDENT_20261011](CONTROL_INCIDENT_20261011.md)。
