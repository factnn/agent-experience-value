# V2 接续因 Goal blocked 停止准入

2026-10-12 14:02 UTC 的只读检查：原生 `get_goal` 和观察器均返回 `blocked`；原生记录的最后更新时间为 12:04:12 UTC。本回合未调用 `update_goal`，没有恢复 Goal、重启监督器、修改数据库或新增科学采样。状态变化的原因尚未确定，不能由旧消息队列 SQLite522 错误推断。

## 实际终态

- 原执行保留24条完整训练，V2执行完成12条，总计36/51；当前0运行、15未启动。
- V2在12:25:42 UTC记录 `s0_group_03_r2` 准入失败：`Goal explicitly unavailable for work: blocked`。该job没有输出目录或训练进程收据，属于控制准入失败，不是科学训练失败。
- 既有四条继续按冻结预算完成；13:00:19 UTC V2监督器写入 `incomplete_requires_review`。PID2421606已为僵尸终态，四个训练PID均不存在；GPU4–7各3MiB/0%利用率，GPU0–3各13,106MiB，仍不能占用他人作业。
- 四个新终点的保存审计均为 `passed`：Uniform r2为78,332新token/7步；TravelAPI r1为78,112/8步；GorillaFileSystem r2为84,190/6步；MessageAPI+VehicleControlAPI r1为72,325/4步。合计312,959新token，祖先成本不重复计费。
- 上述四条仅归档机器终态和保存审计，本回合没有重新原始重放或完成科学人工审阅；没有迁移G/V。事件索引目前止于E115，观察器在非active时不扫描登记新事件，不能将其空待审列表理解为四条已经审阅。

证据与文件SHA见 [控制快照](control_blocked_v2_receipt_20261012.json)。原V2的queue/failures/supervisor及新四条summary/audit作为终态证据发布，保留原文件字节；没有把这次停机包装成实验完成。

## 恢复入口与后续依赖

[RESEARCH_LOOP](RESEARCH_LOOP.md)规定“若非 active，不因自动消息恢复任务”。需要通过原生Goal控制恢复状态并再次确认 `active`；用户本回合的“继续”已表达继续研究的意愿，但服务端状态仍为blocked，agent不能自行改Goal或绕过准入守卫。[OpenAI官方Goals说明](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)提供原生 `/goal resume` 生命周期入口；本机是否实际恢复，仍以 `get_goal` 为准。

恢复后先登记和重新原始审阅四个新终点及这次控制失败，再登记独立V3控制根，仅接续原15个从未启动job。原51个科学job、seed、预算、GPU lane顺序及全部对照保持不变，不重跑已完成36条；系统Python运行CPU监督器，原RL解释器运行科学子进程。

[62全矩阵合并](../62_complete_training_matrix_finalization.md)的当前实现要求“原24 + V2完成27”。本次V2实际只完成12，不能运行该实现生成51证书；恢复后必须把合并入口扩展到原24 + V2的12 + V3的15，保留所有控制失败血缘并对原51个job各重放一次。没有完整训练证书和人工整批审阅前，不注册目标评测。

科学顺序仍是完整51训练 → 整批审阅/scope回顾 → 80 fit面板 → 实际预测封存并发布 → 24 heldout面板 → G/V报告 → 晚阶段。当前停机及单条/批次结束均不构成Goal完成。
