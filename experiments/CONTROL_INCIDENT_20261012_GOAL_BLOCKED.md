# Goal blocked导致新作业准入停止

记录时间：2026-10-12T08:21:08.250974+00:00。当前回合开始原生get_goal返回active，后来训练队列报告Goal blocked；再次原生get_goal确认blocked。本回合没有调用update_goal改变Goal，也没有恢复Goal或修改数据库。

## 已确认的状态

- 21条训练已完成；3条既有作业仍实际存活：GPU6 group06 r1 PID2353038、GPU5 group03 r1 PID2360583、GPU7 group07 r1 PID2370788。后续以实际PID为准。
- 待运行27条；GPU4的下一case group09 r1在准入时被拒绝，queue记录RuntimeError：`Goal explicitly unavailable for work: blocked`。这是控制准入失败，不能当作该来源的科学训练负结果。
- 固定监督器收到failure后停止后续准入；既有作业按原界限完成。未重启任务、未重采或增加预算。
- E098已经完成完整原始审计与人工审阅；同步这份已完成证据。[NEXT](NEXT.md)记录控制条件，原scope与科学协议仍保留。

## 恢复要求与未完成工作

自动研究遵循非active不推进的规则。需要用户通过原生`/goal resume`恢复Goal；active重新确认之后，还须检查监督器终态/实际进程、保留这次失败证据并登记控制恢复方案，才能接续剩余队列。不能把恢复Goal误写成旧监督器必然继续。

新完成的group02 r1尚待人工原始复审；其他已运行作业未来终点也待审。整体51分支与真实迁移G/V尚未齐全，当前不具备Goal完成证据。
