# Smoke 报告：训练 + BFCL 评测闭环（2026-09-29）

目的：验证「数据 → 训练 → 评测」闭环在这台机器上可跑通并测量真实成本。
**本轮不产出任何能力结论**，也不是四组 token-matched 实验。

## 1. 环境

`.venv-train` 原本只有 torch 2.5.1+cu121，没有能读 Qwen3 的训练栈。补齐方式（**未修改系统包**）：

| 组件 | 版本 | 来源 |
|---|---|---|
| transformers | 4.57.1 | 从 `.venv-audit` 离线复制（该组合已被 tokenizer 测量验证过） |
| tokenizers | 0.22.1 | 同上 |
| huggingface_hub | 0.36.2 | 同上 |
| safetensors | 0.8.0 | 同上 |
| torchvision | 0.20.1+cu121 | `download.pytorch.org/whl/cu121`，`--no-deps` |
| peft | 0.21.1 | 清华 PyPI 镜像 |
| accelerate | 1.15.0 | 清华 PyPI 镜像 |

- 系统 torchvision 是 0.17.0a0（配系统 torch 2.2.0a0），与 venv 的 torch 2.5.1 ABI 不匹配，会让 `Qwen3ForCausalLM` 导入直接抛 `operator torchvision::nms does not exist`。必须覆盖。
- pip 明确拒绝卸载 venv 外的 torchvision，系统包保持原样。
- 直连 PyPI 多次 `Read timed out`；`pypi.ngc.nvidia.com`（pip.conf 里配的 extra index）在本机根本不解析。清华镜像 0.24s 响应，改用之。
- 已知遗留冲突：venv 的 fsspec 2026.7.0 与系统 datasets 2.16.1 要求的 `<=2023.10.0` 不兼容（torch 安装带入）；本流程未使用 datasets。

## 2. 数据

新增 [pipeline/build_sft.py](../pipeline/build_sft.py)：显式 ChatML（保留全部历史 content 含 think）+ **assistant-only loss mask**；超长轨迹**丢弃不截断**。

```
.venv-train/bin/python pipeline/build_sft.py --split train --limit 32 --out smoke/sft_smoke
```

产物 `smoke/sft_smoke/train.pt` + `train_report.json`：

- 32 条 train 候选，总 676,722 tokens，其中监督 232,628（34.4%）
- 长度 min 7,284 / 中位 22,605 / max 30,481；无超长丢弃
- framing 分词不一致 0（`encode(framing+content) == encode(framing)+encode(content)` 全部成立）
- loss mask = assistant 正文 + `<|im_end|>`

**注意**：这 32 条是 row_id 最小的主轨迹，全部来自 shard 0（swesmith），中位长度 22.6K 明显高于随机 200 条的 13.7K。它不是全池长度分布，正式分组必须重新采样并报告筛除偏差。

## 3. 显存实测（1×A100-40GB，fwd+bwd，**不含优化器状态**）

新增 [pipeline/train_sft.py](../pipeline/train_sft.py)，`--probe` 模式逐长度实测：

| seq_len | 全参不分块 loss | 全参 + 分块 loss(512) | LoRA + 分块 loss(512) |
|---:|---|---|---|
| 8,192 | 25.35 GB / 3.06s | 16.64 GB / 3.17s | 14.38 GB / 3.46s |
| 16,384 | **OOM** | 21.64 GB / 5.31s | 20.53 GB / 6.35s |
| 32,768 | **OOM** | 34.13 GB / 14.34s | 32.94 GB / 16.30s |

三条结论：

1. **必须分块计算 loss。** lm_head 的 logits 在 32K 是 `32768×151936×2B ≈ 9.96 GB`，`cross_entropy` 内部再 fp32 上采样翻倍到 ~20 GB。不分块时 16K 就 OOM，分块后 32K 可跑（8K 处省 8.7 GB）。
2. **单卡 40GB 上全参 SFT 不可行**，任何长度都不行：32K 时 fwd+bwd 已占 34.13 GB，AdamW 的 fp32 状态还需约 32 GB（4B 参数 × 4B × 2）。8K 也一样（16.64 + 32 > 40）。全参路线只能在 LoRA、8-bit Adam、FSDP/ZeRO 多卡、或显著降低上下文之间选。
3. **LoRA 32K 峰值 32.94 GB / reserved 37.74 GB**，余量不足 3 GB；16K（20.53 GB）明显更稳。正式 pilot 若坚持 32K 需盯碎片。

## 4. LoRA 训练

```
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py \
    --data smoke/sft_smoke/train.pt --out smoke/run_lora --mode lora --lora-r 16 --lr 2e-4 --max-steps 16
```

- LoRA r=16 作用于 q/k/v/o/gate/up/down；gradient checkpointing；AdamW lr 2e-4；无 packing，micro-batch 1
- 可训练参数 33,030,144 / 4,055,498,240 = **0.81%**
- 16 步 / 153 秒 / 120,159 监督 tokens → **约 787 监督 tokens/s**
- loss 0.904 → 0.468（数据高度模板化，loss 起点本来就低）
- 峰值 25.9 GB allocated / 30.9 GB reserved（含优化器状态，实际 seq 长度 9.6K–30.5K）

**成本外推**（单卡，1 个 epoch）：随机 200 条的中位监督长度 5,243 tokens → 1,500 条/bin ≈ 7.9M 监督 tokens ≈ **2.8 GPU-hours/bin**；四组约 11 GPU-hours/epoch，用 4 张空闲卡约 3 小时墙钟。这是 micro-batch 1、无 packing 的数字，packing 后应更好。

## 5. BFCL 评测

评测器已从「本地近似」升级为**官方实现**，详见 [09_evaluation_sensitivity.md](../09_evaluation_sensitivity.md)。

- `bfcl_eval-2026.3.23` 官方 wheel 解包到 `.third_party/bfcl_eval_pkg/`，只补了 `tree_sitter`、`tenacity`、`overrides` 三个依赖。
- [pipeline/eval_bfcl_local.py](../pipeline/eval_bfcl_local.py)：官方完整推理循环（单轮 + 多轮，多轮会本地执行官方 mock API），只把 vLLM 调用点换成本地生成。
- [pipeline/eval_bfcl.py](../pipeline/eval_bfcl.py)：同样走官方 prompt/decoder/checker，另加 `--batch-size` 与 `--offset`（分片并行）。
- [pipeline/test_eval_bfcl.py](../pipeline/test_eval_bfcl.py)：7 个自检，含 20 条 ground-truth 走官方 checker 的通过性检查。

20 题 `simple_python`，官方路径，greedy，budget 640：

| 配置 | accuracy | decodable |
|---|---:|---:|
| base（无 adapter） | 0.95 | 1.00 |
| LoRA 16 步 adapter | 0.80 | 0.85 |

- adapter 与 base 在 16 步 / 32 条规模上差 3 题，**不是能力结论**，只证明闭环能产出可比数字；配对不一致率 0.15。
- **base 全量 400 题是 0.932**：Qwen3-4B base 在 simple_python 上接近天花板，这条轴测不出迁移增益（见 09 文档第 3 节）。
- 主要失败模式是「在 think 里绕圈、预算内没吐出 JSON」：budget 256→640→1024 让同一批 120 题的准确率从 0.758 → 0.917 → 0.950；
  java 轴上 640→1536 是 0.510 → 0.650。
- 单题成本：约 7–9 秒（batch-1）；batch-8 提速 3.5–4.2 倍；多轮约 242 秒/题。

## 6. 复现命令

```bash
cd /share/project/zpy/RL/agent_experience_project_20260928
.venv-train/bin/python pipeline/build_sft.py --split train --limit 32 --out smoke/sft_smoke
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py --probe --mode lora --probe-lengths 8192,16384,32768 --out smoke/probe_lora
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/train_sft.py --data smoke/sft_smoke/train.pt --out smoke/run_lora --mode lora --max-steps 16
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl.py --limit 20 --tag base20_repeat
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl.py --limit 20 --tag lora16_official --adapter smoke/run_lora/adapter
CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl_local.py --category multi_turn_base --limit 16 --max-new-tokens 2048 --tag base_mt16
cd pipeline && ../.venv-train/bin/python -m unittest test_eval_bfcl -v
```

## 7. 不能宣称的部分

- 没有 OOD 迁移结论，没有 token-matched 四组，没有 selector 实现，没有训练种子重复。
- 训练集 32 条、1 个训练种子、1 个模型规模。
- 打分器为本地近似，未与官方 runner 对齐。
- 成功标签仍不可用；本轮完全没有用到 outcome 标签。
