# Stage0控制恢复：仅接续原未启动的27条训练

登记：2026-10-12T09:10:05.286441+00:00。用户已将原生Goal恢复active。本次是[49](49_source_group_stage0_execution.md)的执行恢复，原科学矩阵、[47](47_source_group_prediction_contract.md)分析契约和目标封存顺序保持。

## 原始证据与恢复条件

旧监督器2230422已终止，原queue为24完整终点/0活动/27待运行；supervisor保留incomplete_requires_review，failures保留Goal blocked准入错误。[E099](experiments/E099.md)确认group09 r1没有输出、console/process/preflight，新增采样0，不是失败训练或负收益样本。后来的完整终点E100–E103已从原始生成、mask、allocator与完整状态复审，与保存audit一致。

本轮准备只读核验全部24终点的19文件manifest/hash/大小、对应人工review、51个原job SHA及全部冻结依赖。24终点和27待运行严格不相交且并集为原51；所有待运行输出及旧启动收据均不存在，旧监督器无活动进程。证据见[登记收据](rl/bfcl_source_stage0_continuation_registration_20261012_0844/audit_status.json)，GPU作业0/新token0。

## 已登记的执行接续

协议：[BFCL_SOURCE_STAGE0_CONTINUATION_20261012_0844.json](rl/BFCL_SOURCE_STAGE0_CONTINUATION_20261012_0844.json)，SHA `51873c3dfe2de8149933cc4bb1910f4db117874f18222b506c47519ee28ee6c2`。它引用原master SHA `ea7c8a3f883d30a4358b05d74101dbf8653e379356ae4bf996c0075d511a98dd`，只包含原未启动27条job的原样引用。各GPU lane是原lane删除已完成元素后的子序列：4/5/6各7条、7为6条。首批分别group09/10/11/12 r1。

使用原冻结监督器、trainer、job及全部科学依赖，不修改任何采样代码或已完成输出。唯一新输出根为控制收据目录`rl/bfcl_source_stage0_resume_20261012_0844`；政策输出仍是每条原job登记路径。原控制queue/failure/supervisor不覆盖。新监督器每次启动仍检查Goal active和实际空卡，最多四卡，无自动重试。

先发布本协议、复审及登记收据并核验远端包含，再启动：

```bash
.venv-rl/bin/python -u pipeline/supervise_bfcl_source_training.py \
  --protocol rl/BFCL_SOURCE_STAGE0_CONTINUATION_20261012_0844.json \
  --protocol-sha256 51873c3dfe2de8149933cc4bb1910f4db117874f18222b506c47519ee28ee6c2
```

执行启动与实际PID核验另追加；当前本段不冒称GPU已启动。

## 全批合并和下一科学动作

这27条接续完成只形成局部执行审计。必须按原master重新审计全部51条原始训练，与各保存audit匹配；合并24+27的实际成本且不重复计算任何共享成本，记录原失败与两段执行协议。原master输出根中的完整51条audit_status此前不存在；届时新增完整审计，不改旧incomplete supervisor/failure。

监督器不会执行目标评测。完整51条审阅与scope回顾之后才按52登记80个fit面板、拟合并发布真实四留出预测，再执行24个heldout面板。当前没有新G/V、点预测或阶段收益，仍不能推出跨规模/家族/环境结论。

此项为控制恢复，没有新增科学候选、调整预算、选seed或删臂；scope的实质设计计数仍是55/58两项。

## 首次准入失败与v2控制修订

2026-10-12T09:35:51.480623+00:00：首次CPU监督器2406157以RL虚拟环境运行，初始Goal查询unknown三次后退出，0 GPU子作业/0新token；旧根与失败保留，E105已复审。实际异常为该解释器缺websockets，系统Python同时间可读active，未证明SQLite故障。

v2协议：[BFCL_SOURCE_STAGE0_CONTINUATION_V2_20261012.json](rl/BFCL_SOURCE_STAGE0_CONTINUATION_V2_20261012.json)，SHA `be849ae8ab8d4878cd574f6d96a7fe51d6102272e401e4e59955a79abd0fa26b`，独立控制根`rl/bfcl_source_stage0_resume_v2_20261012`。仅CPU监督器改用系统Python（路径/版本/依赖SHA已记录），其代码和全部27job不变；训练与CPU原始审计仍由原.venv-rl子进程执行。发布后实际启动并核对四卡，不把Popen成功冒充训练已经启动。

## 实际v2启动核验

2026-10-12T09:48:27.082628+00:00：发布commit81f777e核验后启动系统Python监督器2421606；GPU4/5/6/7实际PID分别2421701/2421699/2421700/2421698，对应group09/10/11/12 r1。四个子进程使用原RL虚拟环境、共同step0政策与90122历史，完整状态恢复verified。见[startup_verified](rl/bfcl_source_stage0_resume_v2_20261012/startup_verified.json)和发布收据。当前24全局完成/4活动/23待运行；旧控制失败保留。运行状态以v2 queue和实际PID为准，原queue已是历史终态。
