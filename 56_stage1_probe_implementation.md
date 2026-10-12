# 晚阶段探测入口与原始证据审计

2026-10-12。落实[55的固定祖先探测设计](55_late_stage_probe_inputs.md)。本轮仅CPU实现和验收，未启动晚阶段采样，未产生迁移G/V。当前49训练和47主分析保持冻结。

## 实现与证据

新增 `pipeline/bfcl_stage1_probe_contract.py`、`run_bfcl_stage1_probe.py`、`report_bfcl_stage1_probe.py`，保留旧stage0文件原样。运行入口要求完整新执行协议和其SHA，不接受55设计输入直接运行。

祖先必须是49预指定的stage0 Uniform repeat0完整终点。逐一验证summary、audit、adapter、学习状态与manifest的哈希、政策指纹和修正算法来源；读取CPU payload核对非空AdamW、7个优化器步/28微步、参数哈希及160,662继承获取token。从该payload的allocator重新计算全部16组历史，拒绝沿用旧阶段缓存；要求87题全覆盖及174个互不重复seed。探测不恢复或推进祖先的优化器/RNG，独立采样seed由新协议固定，采样后重新核对父文件未变。

runner沿用4条feature+4条独立validation的采样和预测方法，显式记录stage1、父状态SHA、阶段位置与开始时间。特征预测先封存再采validation。审计逐条重放生成token/mask、奖励、题目/seed/阶段/政策身份、预算、预测文件及时间顺序。父训练成本仅作为继承沿革保留，不再次计为探测新增成本。

完整stage0迁移报告（104面板、对应49协议）及其已审阅证据是加载stage1执行协议的前置条件。入口检查本地不可变引用；本轮随后补齐的独立监督器负责远端发布、四卡调度、每作业Goal/空卡检查，详见下节。仍无最终科学执行协议或实际GPU验收。

## 验收与边界

`test_bfcl_stage1_probe.py`的3项CPU检查通过：实际读取固定祖先；拒绝stage0历史冒充、缺失AdamW、重复seed、缺失全矩阵/审阅；合成原始轨迹能够重放，并拒绝错误阶段/父状态/政策、mask损坏和封存时间倒置。结果保存在 `rl/bfcl_stage1_probe_cpu_20261012/audit_status.json`。当前49所冻结的31个pipeline文件SHA均未改变。

这些检查包含真实checkpoint读取和合成轨迹故障注入，不是实际晚阶段GPU推理的验收，也不是阶段依赖结果。3项通过不能替代后续真实完整原始轨迹审计。

## 四卡调度与发布核验补齐

新增 `pipeline/supervise_bfcl_stage1_probe.py`。监督器要求完整新协议及SHA、真实远端main包含的发布commit。协议、设计输入、父绑定/训练job/摘要/审计/manifest、完整stage0迁移报告/审阅以及全部pipeline依赖必须与该commit中的字节一致。模型和学习状态二进制继续只验证本地哈希；Git中存在manifest不代表二进制已上传。

16个来源组必须在GPU4–7的四条队列中各出现一次。继承49的全部运行依赖哈希，并新增stage1程序依赖。每作业启动前检查Goal与卡的实际空闲状态；输出根目录须已登记到观察器。独占创建输出保留单次尝试；即使启动前检查失败，也写0子进程的failure收据。一次runner、硬超时或原始审计失败即停止后续启动，已入场的独立作业仍完成原有有界尝试。只有16条全部完成且整批原始审计通过才写audited_complete；整批审计失败不会误报完成。没有自动重试或跨入训练/目标评测。

6项新增CPU测试通过（`rl/bfcl_stage1_supervisor_cpu_20261012/audit_status.json`）：真实四线程并发上限/16条各一次；runner/超时返回/审计失败后停止后续启动并保留其他活动作业；preflight与批审计失败处理；预检查失败收据与拒绝覆盖；缺组/重复/依赖替换拒绝；临时本地Git仓库实际验证未push拒绝、push后接受以及错误字节/未发布commit拒绝。线程worker使用合成回调，超时测试是返回状态注入，不是新GPU硬超时试验。Git测试使用本地bare remote，不是科学协议已经发布到GitHub。

## 下一步

四卡继续当前51条科学训练；完成后按52/54运行并审阅stage0迁移预测矩阵。晚阶段监督器与发布检查已实现，尚缺在stage0完整审阅后绑定实际报告、生成最终运行依赖哈希与完整执行协议；以后才启动55新采样。早期模型与标准化仍须冻结，获得晚feature后封存早→晚预测并push，再打开任何晚G标签。候选机制、规模和环境缺口继续由53台账追踪，本次实现未新增科学设计决定。
