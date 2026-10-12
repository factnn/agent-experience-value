# 控制接续后的完整51训练审计入口

2026-10-12T11:20:52.567093+00:00。实现59规定的收尾：原执行保留24终点，v2接续27任务后，必须以49原科学master形成完整51条审计，才能满足52目标评测登记要求。当前训练仍在运行，没有执行科学全批合并。

`pipeline/report_bfcl_source_matrix.py`只接受明确SHA绑定的原master和v2执行协议。验证24/27不相交且覆盖原51、job原样对应、原GPU lane子序列、原控制终态和完整接续终态。原blocked失败与第一次零GPU子进程准入失败分别保留；接续批次audit须与原任务逐一对应，不能用旧任务审计替换接续任务。

原/接续监督器都必须实际终止，全部冻结依赖SHA保持。逐条重新调用原训练审计器，核对原始生成、mask、allocator、归一化、参数链及完整AdamW状态，与各保存audit一致；终点manifest文件hash/大小及每个终点人工review也检查。接续成本与27审计逐项相符；完整新增训练成本只累加51分支一次，不重复叠加24与27或共同获取成本。

通过后只新建原master输出根的`audit_status.json`，字段仍兼容52：passed/complete_jobs51/new_sampled_tokens/jobs，并增加三段控制执行血缘与失败原SHA。旧queue/failure/supervisor不改写。已有完整审计存在则拒绝覆盖；部分完成、原始复核不一致或身份替换都不形成完整证书。

四项CPU检查通过：[收据](rl/bfcl_source_matrix_cpu_20261012/audit_status.json)，工具3ac4c8：合成24+27完整集调用51次审计且成本只计51次、失败文件不变；未完队列不做原始重放或写证书；更换job/GPU顺序、重复/重叠、把科学训练失败当控制失败均拒绝；新鲜audit不符或用旧审计替换接续审计不产生输出，既有证书不可覆盖。首次测试曾因fixture共享对象导致替换同时改变“原master”，修正为独立JSON对象后通过；不是研究训练故障。

测试中原始训练审计调用是mock，不能认证真实51条完成。实际CLI在RL环境的导入/help通过，原31冻结pipeline SHA一致，没有新GPU或采样。实际全矩阵仍待后续CPU重放，不能把这份测试收据作为目标评测门槛。

届时执行：

```bash
.venv-rl/bin/python pipeline/report_bfcl_source_matrix.py \
  --protocol rl/BFCL_SOURCE_STAGE0_EXECUTION_PROTOCOL_20261012.json \
  --protocol-sha256 ea7c8a3f883d30a4358b05d74101dbf8653e379356ae4bf996c0075d511a98dd \
  --continuation rl/BFCL_SOURCE_STAGE0_CONTINUATION_V2_20261012.json \
  --continuation-sha256 be849ae8ab8d4878cd574f6d96a7fe51d6102272e401e4e59955a79abd0fa26b
```

全批审计事件人工审阅与scope回顾之后，再按52登记80 fit、发布预测、24 heldout。CPU训练审计/报告用RL环境；CPU监督器用59已确认支持Goal查询的系统Python，其GPU训练/评测子进程仍用原RL环境，避免重复E105的控制解释器问题。原分析/候选/预算/目标均保持；完整训练仍不是Goal完成。
