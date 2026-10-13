# 六卡资源接续：同一51-job科学矩阵

用户在当前SSH会话明确要求额外两卡，随后确认对GPU0–1旧OQPD Ray/vLLM服务有管理权并批准停止。实际仅SIGTERM已列出的对应进程；worker随EngineCore退出，0/1各降至3MiB。清理收据记录PID身份、GPU绑定、时间、信号及原研究进程存活。最初条件性清理被自动审批拒绝且未执行；新增明确管理权授权后才获准执行。没有宣称整个历史期间无人调用或这些旧服务原属本研究。

## 执行归属与科学边界

原V3仅停止新准入，已运行子进程继续自然完成。实际case05/gpu4的RuntimeError“SSH owner stopped further job admission”证明内部stop latch已生效；三条原job仍pending，且无原输出/子进程凭据。独立冻结queue/failures快照和handoff收据保留该行政失败。它不构成RL负结果，也不取消历史失败记录。

V4使用新GPU0/1及已释放原GPU4：0→group08_r2，1→group13_r2，4→group05_r2。3个job的原JSON/hash、seed、任务池、科学runner、软12,600秒/硬14,400秒界、128步、65,536名义新token及完整组停止均不变。三条名义新token合计196,608，原完整组上界合计294,912；整个科学矩阵仍51条，不增加采样预算。跨旧保留子进程与V4最多6卡；实际活动数随任务结束减少，不以卡数宣称严格等计算量。

首份未执行登记继承了旧15-job kind及总预算字段，子代理在发布/启动前发现。首份文件和收据完整保留，最终FINAL文件引用supersedes证据，纠正为3-job预算。没有因此执行、失败或重跑科学任务。

## 控制与发布

新的authority绑定精确FINAL协议、当前原生Goal记录和后续用户资源授权，旧Goal不修改/不查询。独立V4停止新增准入标记为experiments/.runtime/ssh_stage0_v4_stop_20261013；全局为experiments/.runtime/ssh_stop_20261013。行政交接临时全局标记只有在真实stop latch后、当前Goal active且内容hash仍为本次请求时才能由执行端撤销。真实用户停止标记不能撤销。停止新增准入与终止在运行作业分别处理。

新控制器每条检查精确job、原输出不存在、真实空卡（<=64MiB且无compute PID）、当前phase/global stop和跨段存活PID及预约集合<=6；无自动重试，保留原科学审计与时间界。新增四段矩阵合并保留原24/V2的12、动态V3保留终点及V4转交终点，必须互斥覆盖51并逐条完整raw audit、manifest/hash和执行agent审阅；不能把部分V3伪造为15-job通过。

11控制CPU测试与15初始血缘测试通过；真实receipt字段集成修复后4项针对性测试通过。血缘测试的科学auditor被mock，只证明控制路径；完整51实际证书仍须所有终点完成后执行。旧运行冻结控制与科学输入未改。

## 下一科学依赖

51完整原始审计/整批scope回顾→80 fit面板→按47冻结实际点预测并push核验→24 heldout→G/V、误差方向、重复与成本报告。21保留目标题保持关闭。新增资源只补原主链，不新增校准、假说或任务；晚阶段固定Uniform r0祖先，机制广度、规模/家族/环境原scope保留。
