# 单轨迹剂量诊断：冻结执行设计

2026-10-10。延续 [26](26_dose_experiment_preparation.md)，冻结 [剂量协议](rl/BFCL_DOSE_PROTOCOL_20261010.json)，使用已有 Qwen3-4B 与原共同起点，仅一张实际空闲 GPU 4，无付费 API 或新下载。

## 问题与流程

在相同 4B/LoRA/GRPO 配方下，沿一条新的全 87 题 Uniform 轨迹增加生成剂量，开发集能力如何变化？该诊断为后续阶段实验提供状态和观察，不把规则排名、梯度或参数位移当作经验价值规律。

初始化 seed 仍为 20261018，恢复原共同 payload 与 90,122 token 探测历史；分配器 seed 20265210，训练 seed `20266210 + 100 × optimizer step`，评测 seed `20267210 + 100 × panel index`。无新探测。保存阈值为 32,768 / 65,536 / 131,072 新生成 token；完整四轨迹组和 optimizer 更新后保存首次到达阈值的状态。同组跨多个阈值时多个标签引用同一检查点，同一状态只评一次，报告实际剂量及超额。

顺序固定：训练 → 起点开发评测 → 各唯一检查点开发评测。开发面板是既有全部 22 题，按 manifest 顺序，每题一条轨迹。既有 ID、组合留出题及新增确认面板不参与此轮评测，也不用于选剂量。全部预声明检查点照常报告，不选最佳，未来阶段实验另行冻结。

训练最多 32 更新，软上限每进程 10,800 秒，在完整组/评测题之间停止；硬上限每进程 14,400 秒，失败及未达阈值原样保留，无重采样或追阳性加预算。基于上一批吞吐估计整轮约 4–6 小时；这是估计，最多五个进程的硬上限合计 20 小时。仅一张卡，等待其他用户释放显存不挤占已有任务。

## 检查点及验证

新入口 [run_bfcl_dose.py](pipeline/run_bfcl_dose.py) 复用原训练/多轮执行器，不修改历史协议。回调顺序为完整边界检查、更新证据、剂量保存。组件记录原始 AdamW backend、scheduler、RNG、LoRA、buffers、training flags、分配器历史和累计成本，同时记录 Accelerate wrapper。

每个保存点先核对保存本身不改变 RNG、参数、buffers、训练模式、优化器、scheduler、分配器，再在当前实例恢复刚保存的完整组件并逐项核对。检验耗时单独计入；随后沿原轨迹继续。CPU 验证带非零 AdamW moments 的随机延续一致性及同组多阈值规则。

这验证组件往返及当前进程延续，不代表非零 global_step 的 HuggingFace dataloader/GRPO 新分支恢复已完成。检查点明确存累计 learner 更新和本地计数，后续从学习阶段作正式分叉时仍需实现 trainer 计数/新 rollout buffer 集成并真实验收，不能套用旧 step-zero 分叉断言。

## 审计与同步

源代码、协议及 CPU 检查先 commit/push，再启动。独立会话 CPU 调度器负责固定队列、硬停止、原始 token/模型 mask、成本、剂量标签与全开发面板审计；每个训练组、保存点和评测题的进展自动 commit/push，保留日志、PID、启动命令和失败。权重与本地完整二进制状态不上传 Git，manifest/checksum 与原始文字证据上传。

一条轨迹的三个相关剂量和一个开发面板不能支撑稳定排序、最优剂量、独立阶段效应或跨模型规律。后续范围保持 [25](25_gpt9_research_roadmap.md)。

## 实际启动验收

七项 CPU 检查和数据/祖先协议/源 payload checksum 预检通过。GPU 4 已启动单轨迹训练，现场恢复权重 fingerprint、原始 AdamW step 0、90,122 token 历史和新分配 seed 全部核对通过，见 [启动验收](rl/bfcl_intervention_dose_20261010/launch_acceptance.json)。新的真实模型保存/组件往返尚待首个剂量检查点，不提前宣称通过。
