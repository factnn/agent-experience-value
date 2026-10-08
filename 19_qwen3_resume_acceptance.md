# Qwen3 / GRPO 检查点恢复实测

2026-10-08。沿用已有在线 RL 管线，验证完成 optimizer update 的边界能否保存共同学习状态，再独立恢复并获取新经验。不是分配效果实验。

## 配置与路径

Qwen3-4B cached base，LoRA r8 q/v，官方 TRL 0.29 GRPO，AdamW、constant LR 1e-5、beta=0、num_iterations=1。一个独立 send 工程 fixture，seed 20260930；每组 4 个 rollout，thinking、2048 completion tokens、最多 8 个工具轮次。连续运行两步，在第一步保存 HF checkpoint 与组件快照；第二个独立进程从 checkpoint-1 恢复，只运行第二步。先后使用同一张物理 GPU 4，每个进程硬限 1200 秒，无付费服务。

现有 Torch 2.5 与 Transformers 5 的原生 optimizer/RNG 加载存在版本限制。当前恢复路径保留 HF 的 safetensors adapter、trainer progress 和数据进度，callback 恢复本地生成且 checksum/provenance 校验后的权重、buffers、模块模式、optimizer、scheduler、RNG、分配历史和成本；跳过原生 optimizer/RNG loader，没有关闭全局安全检查。此结论只覆盖这条实际路径。

## 结果

- 连续第一步奖励均值 0.25，grad_norm 约 0.04154，参数改变，保存完整共同状态。
- 恢复入口 global_step=1，参数指纹等于连续第一步后的指纹；继承 8,493 个已付采样 token，buffer 为空。
- 恢复进程实际重新采样 4 条交互；其全部原始生成 token、奖励和优势与连续第二步逐项相同。
- 两条路径第二步后的参数指纹完全相同；新增获取成本均为 5,612 token，累计成本均为 14,105 token。
- 第二组全部成功，advantages 和新梯度为零。参数仍改变，来自已有 Adam 状态，**不能算恢复后新增非零学习信号**。

完整验收新采样成本为 19,717 token（连续两组加重新生成第二组），两个进程 wall time 合计约 501 秒，即约 0.139 单卡小时。恢复路径继承的成本不重复计算为新 GPU 采样，但不能从科学分支预算中抹掉。作业已结束，GPU 4 已释放。

## 证据与复现

[审计 JSON](rl/resume_acceptance_20261008_audit.json)、[连续运行](rl/resume_acceptance_20261008_continuous/summary.json)、[恢复运行](rl/resume_acceptance_20261008_restored/summary.json)。各目录保存原始生成、rollout、mask、奖励/优势、参数指纹、日志、配置及 checkpoint manifest；checkpoint 二进制仅保留本机，不进入 git。

运行入口是 `pipeline/accept_rl_resume.py --out <新目录>`；恢复增加 `--resume <连续运行目录>/checkpoint-1`。须使用隔离的 `.venv-rl`，与固定的一张实际空闲 GPU。输出目录必须不存在。

```bash
.venv-rl/bin/python pipeline/report_rl_resume.py \
  rl/resume_acceptance_20261008_continuous \
  rl/resume_acceptance_20261008_restored \
  --out rl/resume_acceptance_20261008_audit.json
```

这是单任务、单 GPU、完成更新边界的工程验收，不保证任意数据采样器、并行训练、任意 microstep 或不同硬件上的逐位可复现。Uniform/Frontier 规则切换与历史继承另有 CPU 验收；本次没有运行真实不同分配规则的 GPU 分叉。

## 对研究主线的作用

E0 的两条独立 warmup 曾出现状态漂移；现在已有在共同 checkpoint 继承完整状态的可用路径。后续仍须接通 BFCL 真实多用户轮次生成和 token mask，验证非零梯度更新，并审查语义重叠，之后才能冻结经验价值分叉实验。本结果不改变最初的经验价值与迁移问题，也不支持 Frontier 优于 Uniform 的结论。
