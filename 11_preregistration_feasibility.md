# 11 — 预注册：可行性判据实验（2026-09-30）

> **历史记录提示（2026-09-30）：** 当前在线 RL 主线见 [PROJECT_STATE.md](PROJECT_STATE.md)。本文原始计划、预注册和结果保留；SFT 闸门不作为 RL 的停止条件，旧轨迹缺标签不作为在线环境奖励不可得的证据。


**在跑之前写定。** 目的：回答 10 文档 §0 的第三条——训练这批经验后，模型是否在**任何一个**
本地可评的轴上有可测的变化。这是整个测量研究成立的前提：如果 SFT 搬不动任何轴，
那 $\Delta_{binA} - \Delta_{binB}$ 恒等于 0，bin 造得再干净也无从比较。

## 实验

| 项 | 设定 |
|---|---|
| 训练数据 | `candidate_manifest` train split 中**随机**抽 1,500 条（seed 20260930），即 H0/Random |
| 目标函数 | 与既有 smoke 一致：显式 ChatML + assistant-only loss mask，超长丢弃不截断 |
| 训练配置 | LoRA r=16（q/k/v/o/gate/up/down）、gradient checkpointing、AdamW lr 2e-4、32K、micro-batch 1 |
| 训练量 | 1 个 epoch（约 1,500 步量级），不调参、不 early stop |
| 评测 | 同一 base 检查点、同一 harness、同一题序；budget 3072（与冻结中的 java/js 配置一致） |
| 主指标 | `simple_java` + `simple_javascript`（150 题，base ≈0.65/0.64，有头部空间） |
| 次指标 | `simple_python`（400 题，base 0.932，饱和；预期 0 或负） |
| 拆分报告 | 每个轴都拆成 **decodability** 与 **P(correct \| decodable)**，不只看 accuracy |

## 判据

全部与 base 做**同题配对**比较，逐题结果保留。

- **GO（研究可行）**：至少一条轴出现**明确为正**的变化——
  配对 95% CI 不含 0，或点估计 ≥ +8pp（n=150 时的 MDE）。
  含义：训练这批经验确实产生可测的能力变化，测量研究有信号可测。
- **NO-GO（不可行）**：所有轴的点估计 ≤ 0，或为正但落在噪声内。
  含义：本池 + 本 harness 的 SFT 不产生可测能力变化，四组 bin 的比较无意义。

## 已声明的局限（不做事后补救）

- 单次训练、单个种子：这次只回答"有没有信号"，不回答"信号多大"。
- 1,500 条 / 1 epoch 是**一个**训练量。效应为负不能排除"训练量不足"；
  所以 NO-GO 的结论只在"这个训练量下"成立，需要时再测更大训练量。
- BFCL java/js 在 budget 3072 下的 cap hit 数需报告；若不为 0，结论按"未冻结"标注。
- 若 `simple_python` 变负而 java/js 变正，则漂移主要集中在已被 SFT 见过的格式上，
  这本身是 09 §2.8 的可检验推论，不是失败。
