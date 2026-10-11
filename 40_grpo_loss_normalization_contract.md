# GRPO 梯度归一化：CPU 目标函数核验

2026-10-11。与 39 的冻结 GPU 分叉验收独立；本检查没有模型生成、GPU 工作或优化器更新，未修改任何正在运行的依赖文件。

当前安装组合确实重复缩放了微批梯度。四条不等长有效 mask 的固定样本、二元奖励对应的 group advantages，在 dropout=0 的小型 GPT2 上：历史路径梯度乘以 4 后才匹配整组目标梯度；候选单次归一化直接匹配。两者相对整组梯度的残余误差约 9.1371e-8，最大元素误差 2.9802e-8。历史原始梯度相对目标误差约 0.75，与整体缩小到四分之一相符。

## 独立目标与实际调用

整组目标直接计算 `-mean_i[A_i * mean_model_token(log pi_i)]`，排除外部反馈和 padding。四个样本有效 loss token 数分别 2/2/2/3。对齐 GRPO 的 beta=0、一次迭代、采样政策等于当前政策时，importance ratio=1 的策略梯度应与该直接目标一致。

对照实际调用安装的 BFCL/TRL/Transformers `training_step` 和 Accelerate.backward，逐个微批累积。历史配置中 GRPO 除以四，Accelerate 再除以四；候选仅将 Accelerator 的 backward 除数设为一，HF 实际累积微批数和 GRPO 自身除数仍为四。没有改 reward、advantages、生成分组、loss 公式或优化器状态。

[原始结果](rl/bfcl_loss_normalization_cpu_20261011/analysis/summary.json)、[来源哈希及核验](rl/bfcl_loss_normalization_cpu_20261011/audit_status.json)、[复现脚本](pipeline/check_bfcl_loss_normalization.py)、[未接入正式 runner 的候选修正](pipeline/bfcl_loss_scaling_contract.py)。运行命令 `CUDA_VISIBLE_DEVICES='' .venv-rl/bin/python pipeline/check_bfcl_loss_normalization.py --output <新目录>/analysis/summary.json`，拒绝覆盖旧结果。

CPU 比较事先使用 rtol=2e-5、atol=2e-7，容许 B=4 与四次 B=1 的浮点求和差异；它不是放宽 39 的 GPU 逐位重放判据。检查确认没有参数更新。

## 结论边界与下一步

这建立的是这个 CPU fixture 下、单进程四微批 GRPO 的梯度契约，不是 Qwen/LoRA GPU 更新或完整 Trainer.train 循环的验收。候选函数限制现有 num_generations=4、num_iterations=1、loss_type=grpo；不泛化到 DAPO、多卡、FSDP 或 DeepSpeed。

梯度缩小四倍不等于 AdamW 参数更新缩小四倍；动量、epsilon、weight decay 和裁剪需要分别考虑。不能由此直接解释历史剂量负收益，不能回写或缩放历史参数/动量，也不能悄悄把旧实验更名为修正后的算法。

39 继续使用历史缩放直至冻结配对完成。随后先验证候选在完整 Trainer 循环中的分组/累积不变，再对新科学运行明确冻结使用的算法及数值设置；若改用单次归一化，必须保留旧证据并把它作为新配置。阶段内各干预臂和 Uniform 参照使用同一契约。此 CPU 检查没有消耗保留迁移题，也不产生经验价值收益标签。
