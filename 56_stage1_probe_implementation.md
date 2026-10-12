# 晚阶段探测入口与原始证据审计

2026-10-12。落实[55的固定祖先探测设计](55_late_stage_probe_inputs.md)。本轮仅CPU实现和验收，未启动晚阶段采样，未产生迁移G/V。当前49训练和47主分析保持冻结。

## 实现与证据

新增 `pipeline/bfcl_stage1_probe_contract.py`、`run_bfcl_stage1_probe.py`、`report_bfcl_stage1_probe.py`，保留旧stage0文件原样。运行入口要求完整新执行协议和其SHA，不接受55设计输入直接运行。

祖先必须是49预指定的stage0 Uniform repeat0完整终点。逐一验证summary、audit、adapter、学习状态与manifest的哈希、政策指纹和修正算法来源；读取CPU payload核对非空AdamW、7个优化器步/28微步、参数哈希及160,662继承获取token。从该payload的allocator重新计算全部16组历史，拒绝沿用旧阶段缓存；要求87题全覆盖及174个互不重复seed。探测不恢复或推进祖先的优化器/RNG，独立采样seed由新协议固定，采样后重新核对父文件未变。

runner沿用4条feature+4条独立validation的采样和预测方法，显式记录stage1、父状态SHA、阶段位置与开始时间。特征预测先封存再采validation。审计逐条重放生成token/mask、奖励、题目/seed/阶段/政策身份、预算、预测文件及时间顺序。父训练成本仅作为继承沿革保留，不再次计为探测新增成本。

完整stage0迁移报告（104面板、对应49协议）及其已审阅证据是加载stage1执行协议的前置条件。这里检查本地不可变引用；远端发布、四卡调度、每作业Goal/空卡检查仍需独立监督器实现，不能把本入口当完整执行闭环。

## 验收与边界

`test_bfcl_stage1_probe.py`的3项CPU检查通过：实际读取固定祖先；拒绝stage0历史冒充、缺失AdamW、重复seed、缺失全矩阵/审阅；合成原始轨迹能够重放，并拒绝错误阶段/父状态/政策、mask损坏和封存时间倒置。结果保存在 `rl/bfcl_stage1_probe_cpu_20261012/audit_status.json`。当前49所冻结的31个pipeline文件SHA均未改变。

这些检查包含真实checkpoint读取和合成轨迹故障注入，不是实际晚阶段GPU推理的验收，也不是阶段依赖结果。3项通过不能替代后续真实完整原始轨迹审计。

## 下一步

四卡继续当前51条科学训练；完成后按52/54运行并审阅stage0迁移预测矩阵。晚阶段尚缺监督器/远端发布检查、最终运行依赖哈希与完整执行协议；以后才启动55新采样。早期模型与标准化仍须冻结，获得晚feature后封存早→晚预测并push，再打开任何晚G标签。候选机制、规模和环境缺口继续由53台账追踪，本次实现未新增科学设计决定。
