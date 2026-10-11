# 修正归一化接入完整 Qwen/LoRA Trainer.train

2026-10-11，执行前冻结。39 的确定性采样/完整状态分叉已通过，40 的真实目标与完整 CPU Trainer.train 已通过，41 的 Qwen/LoRA GPU 固定输入四倍缩放与重复更新已通过。本检查只补此前尚未实测的交集：修正配置在实际 Qwen/LoRA 的完整 GPU Trainer.train 中是否保持，并与已验收的手动优化器循环得到同一结果。

协议 `rl/BFCL_NORMALIZATION_GPU_LOOP_PROTOCOL_20261011.json`。在新进程加载相同 Qwen3-4B、LoRA、bf16、确定性配置；启用 Accelerator backward divisor=1、HF/GRPO 累积仍为4。on_train_begin 在实际框架准备好的模型/优化器上恢复历史第六步并做两次零 fresh 梯度的历史动量更新，精确到达 41 的既定起点。固定同一 RNG、旧四条轨迹与 mask，禁止 generate/reward/environment 调用。

正式 `Trainer.train()` 运行 max_steps=1。每个微批的累计梯度、裁剪后优化器前梯度、最终参数/完整 AdamW 必须与事前冻结的 `bfcl_normalization_gpu_20261011/single_1` 原始张量精确一致。初始权重和 AdamW 也必须精确相同；验证微批 0/1/2/3、micro32→36、HF一步更新，以及归一化在循环前后保持为1。不放宽误差、不自动重试。

单 GPU 实际空闲检查和 active Goal 守卫；最多450秒+30秒结束宽限，新生成预算0。历史动量不重缩放，结果是接入检查，不当作修正配方从头训练或经验价值标签。

通过后收尾这一组实现验收，科学 runner 采用显式单次归一化契约与已验收的完整状态恢复机制，在每次作业中记录并检查实际 divisor。下一个新采样目标是任务级探测与来源组迁移 G/V；不重复已通过的历史分叉或继续无目的剂量实验。若本检查失败，则仅根据实际失败证据修复接入问题并单独记录，不修改旧结果。
