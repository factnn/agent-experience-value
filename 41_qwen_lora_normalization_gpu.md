# Qwen/LoRA 单次归一化：固定轨迹 GPU 验证

2026-10-11，执行前冻结。依据 [40](40_grpo_loss_normalization_contract.md) 的独立 CPU 目标及完整 Trainer.train 检查，验证候选在实际 Qwen3-4B/LoRA/bf16/梯度检查点路径中的数值行为。与 [39](39_deterministic_full_state_pair.md) 并行，后者所有文件/协议不变。

## 固定设计

协议：`rl/BFCL_NORMALIZATION_GPU_PROTOCOL_20261011.json`。源为历史第六步完整状态，按原算法执行两次零 fresh 梯度的 AdamW 更新，精确重建旧第八步权重/动量；再固定 RNG 和四条旧轨迹。禁止生成或调用环境，不接触保留迁移题。

在同一空闲 GPU 5 的三个独立进程中依次执行 legacy、single_1、single_2。所有情况开启同一确定性配置；只在两例 candidate 中把 Accelerator backward 除数由 4 改为 1，保留 GRPO 除数 4 和四微批。历史动量不重缩放。每例执行四个真实 training_step、一次裁剪和 AdamW 更新，保存每微批累计梯度、初末参数和优化器供 CPU 审计。

判据事前固定：候选每微批累计梯度精确等于 legacy 的四倍；两例候选输入、起始参数/AdamW、微批 loss/梯度、末参数/AdamW 精确一致，梯度非零且有限。四倍梯度不意味着四倍 AdamW 更新。任何偏差记录实际距离并审阅，不放宽判据。

每例最多 450 秒，总 GPU 批次最多 1350 秒，结束宽限 30 秒、CPU 审计最多 120 秒。新生成 0 token，报告处理输入位置和耗时。每个子进程前重查 active Goal 和实际空闲 GPU；不自动重试。现有 GPU 4 作业继续，总使用两张卡。

## 限制与下一决策

本次是实际模型的固定输入 backward/优化器接口验证，非完整 GPU Trainer.train、采样或完整学习状态分叉。来源第八步只重建权重/动量，没有重建当时 allocator/RNG。通过后结合 CPU 全循环结果，再验收修正配置的完整分叉；不能把历史参数与 AdamW 当作修正算法从头重训所得，也不由工程通过推断迁移收益。

代码：`pipeline/diagnose_bfcl_normalization_gpu.py`、`pipeline/report_bfcl_normalization_gpu.py`、`pipeline/supervise_bfcl_normalization_gpu.py`。使用独立文件以保证正在运行的确定性历史对照不变。
