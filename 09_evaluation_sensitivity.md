# 09 — 评测灵敏度研究（2026-09-29）

目的：在跑任何四组干预之前，先回答「用现在的评测，能不能测出 bin 之间的差异」。
本文件只报告**测量性质**（噪声、偏差、可检测效应、成本），不报告任何能力结论。

## 0. 结论摘要

1. **评测本身是确定的**：同一配置换物理卡重跑，20 题输出逐字节相同，逐题不一致率 0。固定配置下评测噪声不是瓶颈。
2. **但评测配置是最大的偏差源**：同一批 120 题只改生成预算，准确率从 0.758（256 token）到 0.950（1024 token）——**19.2 个百分点**；
   java/js 上 640→1536 差 **14pp**。批大小差 1.7pp，换 prompt 组装路径差 5pp。这些都比 pilot 想检测的效应大。
3. **BFCL 的 Python 轴饱和**：simple_python 0.932（n=400）、multiple 0.950（n=200）。base 没有提升空间，**不能**作为主要 OOD 指标。
4. **有空间且最贴近 scope 的轴是 multi_turn_base**：base 0.250（n=16，预算尚未冻结），
   decodable 100%，失败模式以「卡住」和「改错状态」为主。已本地跑通官方多轮推理 + mock API 执行 + 官方 state 检查，无需 Docker 或外部服务。
5. **配对 MDE**（不一致率 0.15）：n=100 → 10.9pp，n=200 → 7.7pp，n=400 → 5.4pp，n=1390 → 2.9pp。
6. **还没测的关键噪声是训练种子方差**——需要用 ≥2 个独立训练运行来估计，目前为 0 个。
7. 小样本会给出关于头部空间的**错误结论**：irrelevance 在 n=6 时是 1.000，扩到 n=240 后是 0.887。

## 1. 评测器状态：已接入官方实现

原来 `pipeline/eval_bfcl.py` 用的是自写 JSON 格式 prompt + 本地重实现的打分器。现在改为走官方 `bfcl_eval` 包：

- 官方 wheel `bfcl_eval-2026.3.23` 解包到 `.third_party/bfcl_eval_pkg/`（纯 Python，无需 pip 安装整套 API 客户端）。
- 只装了官方打分真正需要的少量依赖：`tree_sitter`(+java/javascript 语法)、`tenacity`、`overrides`。
- `bfcl_eval.constants.model_config` 会 import anthropic/cohere/mistralai/boto3 等全部厂商客户端；用一个只带 `underscore_to_dot=False`（该值已对照 wheel 源码第 936 行的 `qwen3-4b` prompt 变体核实）的最小替身模块替代，报告中记录 `model_config_source`。
- `pipeline/eval_bfcl_local.py`：**官方完整推理循环**（`handler.inference` → 单轮/多轮自动分派）、官方数据加载器 `load_dataset_entry`、官方 `decode_ast`、官方 `ast_checker` / `_evaluate_single_multi_turn_entry`；只把 vLLM 的 `_query_prompting` 换成 transformers 本地生成。
- 支持类别：simple_python / simple_java / simple_javascript / multiple / parallel / parallel_multiple / irrelevance（单轮）+ multi_turn_base / multi_turn_long_context / multi_turn_miss_func / multi_turn_miss_param（多轮，本地执行官方 mock API，无需 Docker，无需外部服务）。
- `live_*`、`memory`、`web_search` 类别需要真实 API 或向量库，**不纳入**。

交叉验证：官方循环 runner 与旧 runner 在同一 20 题上给出一致的 0.95（用相同数据加载路径时）。

## 2. 噪声与偏差

### 2.1 评测噪声 = 0（固定配置）

同配置（GPU 4 与 GPU 7 各跑一次，20 题）：

| 指标 | 结果 |
|---|---|
| 判定不同 | 0 |
| 原始输出不同 | 0 —— **20/20 逐字节相同** |

greedy 解码在本机完全确定。含义：跨卡分片跑同一次评测不会引入噪声，评测噪声不是瓶颈。

### 2.2 生成预算：最大偏差源

**轴一：simple_python（400 题）。** 同一批 120 题，唯一变量是 `max_new_tokens`：

| budget | accuracy | decodable | 撞上限题数 | 平均输出 token |
|---:|---:|---:|---:|---:|
| 256 | 0.758 | 0.800 | 25 | 179 |
| 640 | 0.917 | 0.967 | 4 | 202 |
| 1024 | 0.950 | 1.000 | 0 | 211 |

两两差异：256↔640 = 15.8pp，256↔1024 = 19.2pp，**640↔1024 仍有 3.3pp**。

**轴二：simple_java / simple_javascript（全量，640 vs 1536）。**

| 轴 | n | budget 640 | budget 1536 | 差异 | 不一致题数 |
|---|---:|---:|---:|---:|---:|
| simple_java | 100 | 0.510（dec 0.750） | **0.650**（dec 0.920） | **+14.0pp** | 16 |
| simple_javascript | 50 | 0.640（dec 0.780） | 0.640（dec 0.880） | 0.0pp | 8 |

三个结论：

1. 预算效应**不是某个轴的特例**：java 上同样有 14pp。任何轴的预算都必须设到撞上限数为 0。
2. 效应大小**因轴而异**：java +14pp、simple_python +3.3pp、js 净 0pp。
3. js 的净变化为 0 **但仍有 8/50 题翻转**——总量稳定会掩盖逐题churn。只看聚合数字会漏掉这类风险，
   必须保留逐题结果做配对分析。
4. 平均输出只有 200 token 左右，差异全部来自长尾：预算被少数「在 think 里绕圈」的题吃掉。

**结论：预算必须设到撞上限数为 0，并在所有组之间固定。** 官方 runner 对多轮默认给到 4096。

### 2.3 批大小

`--batch-size 8` 相对 `--batch-size 1`：吞吐提升 **3.5–4.2 倍**（20 题 112s → 26.5s），
但 60 题里 **1 题判定翻转（1.7pp）**，原始文本 58/60 有浮点级差异。
批大小是一个确定性但配置相关的变化：所有组用同一批大小即可保持配对公平，但**不能跨配置比较数字**。

### 2.4 Prompt 组装路径

裸读 JSONL 对比官方 `load_dataset_entry()`（后者会追加语言相关提示）：同样 20 题 **0.95 → 0.90**。
正式数字必须用官方 loader。

### 2.5 配置灵敏度总表

同一批题、只改一个配置项，base 模型自己跟自己的差异（`pipeline/eval_sensitivity.py` 输出）：

| 改动 | 题数 | 准确率变化 | 逐题不一致率 |
|---|---:|---:|---:|
| 换物理卡重跑 | 20 | 0.000 | **0.000** |
| 批大小 1 → 8 | 60 | +0.017 | 0.017 |
| 预算 640 → 1024 | 120 | +0.033 | 0.033 |
| 预算 640 → 1536（java/js） | 150 | +0.093 | 0.160 |
| 预算 256 → 640 | 120 | +0.158 | 0.158 |
| 预算 256 → 1024 | 120 | +0.192 | 0.192 |
| 换检查点（base → 16 步 adapter） | 20 | −0.150 | 0.150 |

只有「换卡」是 0。**其余每一项都与 pilot 想检测的效应同量级或更大。**

### 2.6 检查点不一致率（配对统计的输入）

base 与「16 步 LoRA adapter」在同样 20 题上：不一致 **3/20 = 0.15**（3 题全部是 adapter 由对变错）。
这个 0.15 是经验观测值，与 2.7 的 MDE 计算所用假设一致。

### 2.7 最小可检测效应（MDE）

给定 n 道题、α=0.05 双侧、80% power：

| 题数 n | 配对（不一致率 0.15） | 非配对 |
|---:|---:|---:|
| 20 | 24.3pp | 37.9pp |
| 50 | 15.3pp | 24.0pp |
| 100 | 10.9pp | 17.0pp |
| 200 | 7.7pp | 12.0pp |
| 400 | 5.4pp | 8.5pp |
| 1000 | 3.4pp | 5.4pp |

配对公式 `MDE = (z_{α/2}+z_β)·sqrt(d/n)`，d 为不一致率。**题目集在所有组之间共享，所以配对列才是 pilot 该看的。**
注意 d 本身取决于两个模型有多不同：当两个 bin 训练出的模型几乎一样时 d 更小、MDE 更小；d 必须用 pilot 自己的数据重新估计。

**把 2.2 和 2.7 放在一起看**：预算差一个设置就能造成 19pp 的移动，而 400 题只能分辨 5.4pp。
没有固定评测配置，pilot 测到的东西会被配置偏差淹没。

### 2.8 格式漂移：SFT 会改变输出方言（影响 BFCL 的可解释性）

16 步 LoRA adapter（在 terminus-2 终端轨迹上训练）在 simple_python 上从 0.95 掉到 0.80。
逐条看这 4 个失败，**大部分不是能力下降，而是输出方言漂移**：

| 题 | 失败类型 | 模型实际输出（尾部） |
|---|---|---|
| `simple_python_6` | decode 失败（语法） | `[func_name=solve_quadratic, params={"a": 2, "b": 5, "c": 3}]` |
| `simple_python_18` | decode 失败（语法） | `[func_name=number_analysis.prime_factors, params={"number": 123456}]` |
| `simple_python_17` | decode 失败（字符串未闭合） | （同类的 `key=value, params=` 结构） |
| `simple_python_13` | `type_error:nested` | `[calculate_area_under_curve(function="x**2", interval=[1, 3])]` |

前三条里模型输出的是 **`name=…, params={…}`** —— 这是 terminus-2 harness 的命令 JSON 方言
（`{"commands": [...], "task_complete": ...}`）泄漏进了 BFCL 要求的 `[func_name(param=value)]` 格式。
**这不是「SFT 让 agent 变笨」，而是「终端 agent SFT → 函数调用方言漂移」。**

**对实验设计的直接后果**：BFCL 主结果**不能只报 accuracy**，必须拆成

```
Decodability           # 输出能不能被官方 parser 解成调用
P(Correct | Decodable)  # 在可解码的前提下答对的比例
```

否则如果 Recovery bin 比 Random bin 在 BFCL 上低 5pp，而差距全部来自方言漂移，
就不能解释成「recovery 经验的迁移价值更低」。

**这不意味着 BFCL 不能用**，反而它应该被明确定义成一条 **far-transfer / cross-interface 轴**：
训练是 terminal 风格轨迹，测试是结构化函数调用 + stateful API。这是很强的迁移。
但论文不能只有这一条——否则 reviewer 完全可以说测到的是 output-format interference。
最终至少要有 **near OOD（同为 terminal/executable agent benchmark）+ far OOD（BFCL）** 两条：
若某个 experience bin 在两个方向上同时为正，general-transfer 的故事才成立。

## 3. 各评测轴的头部空间（全量扫描完成）

base 模型（Qwen3-4B，无 adapter），官方 prompt/parser/checker：

| 轴 | n | base 准确率 | decodable | 头部空间 |
|---|---:|---:|---:|---|
| simple_python | 400 | 0.932 | 0.968 | 几乎没有 |
| multiple | 200 | 0.950 | 0.975 | 几乎没有 |
| irrelevance | 240 | 0.887 | 1.000 | 小 |
| parallel_multiple | 200 | 0.865 | 0.905 | 中 |
| parallel | 200 | 0.860 | 0.910 | 中 |
| **simple_java** | 100 | **0.650** | 0.930 | **大** |
| **simple_javascript** | 50 | **0.700** | 0.940 | **大** |
| **multi_turn_base** | 16 | **0.250** | 1.000 | **最大** |

（java/js 为冻结配置 budget 3072；多轮 2048 待冻结；其余 640。见 2.2：预算会显著改变这些数字。）

**预算冻结状态（2026-09-30 更新）**：本节原先的标准是「cap hits 必须为 0」。实测后这个标准要改——
**它是不可达的**：`multi_turn_base` 里同样三道题（`_0/_12/_13`）在 2048 和 4096 下都撞上限，
加预算只让它们打转更久，cap hits 反而从 3 升到 5。真正相关的标准是
**「预算翻倍后逐题判定零翻转」**。按此标准：

| 轴 | 稳定预算 | 依据 |
|---|---|---|
| simple_java | **1536** | 1536→3072 零翻转，acc 均为 0.650 |
| simple_javascript | **3072** | 3072→6144 零翻转，acc 均为 0.700 |
| multi_turn_base | **2048** | 2048→4096 零翻转，acc 均为 0.250 |
| simple_python / multiple / irrelevance / parallel / parallel_multiple | 640 | 未做翻倍检验，但 base 已饱和，风险低 |

**冻结基线**：`simple_java` = **0.650**（dec 0.930，budget 3072）、
`simple_javascript` = **0.700**（dec 0.940，budget 3072）、
`multi_turn_base` = **0.250**（dec 1.000，budget 2048）。这三条是 pilot 的主配对轴。

三点解读：

1. **两条轴饱和**（simple_python 0.932、multiple 0.950）：base 已经没有提升空间，
   bin 之间的真实差异会被天花板吃掉。**不能作为主要 OOD 指标。**
2. java/js 的失败有相当比例是**格式失败**（decodable 0.88–0.92，即 8–12% 的输出根本解析不出调用），
   不是语义错误。这两条轴同时测量「格式泛化」和「语义正确」，分析时必须分开报。
3. multi_turn_base 的 decodable 是 1.000，但**失败并不是单一原因**。逐条统计 16 题的失败分类：

   | 结果 | 数量 |
   |---|---:|
   | correct | 4 |
   | `instance_state_mismatch`（调用成功但环境状态改错） | 4 |
   | `force_terminated` | 5 |
   | `empty_turn_model_response` | 2 |
   | `execution_response_mismatch` | 1 |

   12 个失败里只有 4 个是「状态被改错」。**`force_terminated` 占 5 个**——那是模型在单轮内反复重试直到被强制终止，
   属于「卡住」而不是「做错」。这两种失败对 experience value 的含义完全不同，不能混成一句结论。

   **而且这个 0.250 还不算冻结**：该次评测用 budget 2048，221 次 completion 中有 **3 次撞上限**，涉及 3 道题
   （`multi_turn_base_0/12/13`）。按本文件 §2.2 自己立的规则（cap hits 必须为 0），这个数字需要更高预算重跑后才能进表。
   （重跑 `mt16_b4096` 已启动。）

顺带修正一个早前的误读：irrelevance 在 n=6 时是 1.000，看起来饱和；
扩到 n=240 后是 0.887。**小样本会给出关于头部空间的错误结论**，这是本文件存在的另一半理由。

**对 pilot 的含义**：饱和轴（base ≈0.9+）即使训练真的带来能力提升也测不出来——
四组的差异会被天花板吃掉。判据 K1（效果小于噪声）在这类轴上会**假阳性触发**。
中段轴（simple_java ≈0.51、multi_turn_base ≈0.40）既有提升空间，又不会因为地板效应把差异压平，
是 pilot 应该优先用的测量面。

## 4. 成本

单轮（batch-1）：约 **7–9 秒/题**（budget 640）→ 1,390 道本地可评单轮题约 3.1 小时/卡；
batch-8 提速 3.5–4.2 倍，约 2 秒/题；budget 1536 时 java/js 约 14–19 秒/题。
多轮：**242 秒/题**（`multi_turn_base` 16 题 3,880 秒，budget 2048；budget 640 时约 218 秒/题），
即 800 道多轮题单卡约 54 小时。

因此 6.3 建议的 n≈200 多轮题 ≈ **13.4 GPU-hours/次评测**，4 卡分片约 3.4 小时。
这是 pilot 的主要成本项：8–12 个训练好的模型 × 每次评测 3.4 小时 ≈ 一整天的纯评测时间。

## 5. 还没做的

- **训练种子方差**：完全未测。这是与评测噪声并列的另一半噪声来源，需要 ≥2 个独立训练运行。
  pilot 的第一要务就是把它测出来，否则无法解释 bin 间差异。注意它与 MDE 里的不一致率 d 直接相关：
  如果同 bin 不同种子的 d 就有 0.15，那么 MDE 就永远卡在 5pp 量级。
- 打分器与官方榜单的一致性只做了内部交叉验证（同一 20 题两条路径一致），未与公开榜单数字对齐
  （那需要 vLLM 后端与官方生成参数）。
- multi_turn 的另外三个类别（long_context / miss_func / miss_param）只验证了可跑通，未做 headroom 采样。

## 6. 对 pilot 的具体建议

1. **冻结评测配置并写进协议**：官方 loader、预算设到撞上限数为 0（单轮 ≥1536、多轮用官方默认 4096 量级）、
   批大小固定、同一题序、同一 harness。2.2–2.4 说明这些偏差都比 pilot 想检测的效应大。
2. **不要**用 simple_python / multiple 作为主要 OOD 指标（饱和）。
3. 主指标选 **multi_turn_base**（+ miss_func / miss_param / long_context），理由：头部空间最大（base 0.250，
   预算待冻结）、decodable 100% 说明它测的是能力而不是格式。**注意失败分类要分开报**：
   `force_terminated`（卡住）与 `instance_state_mismatch`（改错状态）对 experience value 的含义不同。
   代价是 242 秒/题，必须限制题量。
4. 单轮侧用 **simple_java + simple_javascript + plain parallel/parallel_multiple**（0.64–0.87）作为次级轴，
   并按「格式有效率 / 条件正确率」两个指标分开报。
5. 题量按可检测效应定：想要 7–8pp 分辨率需要 ≥200 题；想要 5pp 需要 ≥400 题。
6. **顺序**：先花一轮预算测训练种子方差（2 个种子 × 1 个 bin），再决定 bin 数 × 种子数的分配。
   在种子方差未知的情况下扩大 bin 数量是本末倒置。
7. 保留逐题结果做配对分析——js 上「净变化 0 但有 8/50 题翻转」说明聚合数字会掩盖真实变动。

## 7. 证据文件对照

每条结论都能追到 `smoke/eval_bfcl/` 下的逐题结果（`.jsonl` 为逐题记录，`_summary.json` 为汇总）：

| 结论 | 结果文件 |
|---|---|
| 2.1 跨卡确定（20 题逐字节相同） | `base20_repeat_gpu7.jsonl` 对比 `base_sweep_a.jsonl` 前 20 题 |
| 2.2 预算 256 / 640 / 1024 | `base_budget256.jsonl`、`base_sweep_a.jsonl` 前 120 题、`base_budget1024.jsonl` |
| 2.2 预算 640 / 1536（java/js） | `base_sweep_a.jsonl`、`base_java_js_b1536.jsonl` |
| 2.3 批大小 1 / 8 | `base_sweep_a.jsonl` 前 60 题、`base60_batch8.jsonl` |
| 2.4 数据加载路径 | `official_loop_sp20.jsonl`（裸 JSONL）对比 `official_loop_sp20b.jsonl`（官方 loader） |
| 2.5 检查点不一致率 | `base20_repeat_gpu7.jsonl` 对比 `lora16_official.jsonl` |
| 3 各轴头部空间 | `base_sweep_a.jsonl`、`base_sweep_b.jsonl`、`base_java_js_b1536.jsonl`、`base_mt16.jsonl` |
| 1 官方路径自检 | `official_probe.jsonl`、`pipeline/test_eval_bfcl.py` |

汇总与 MDE 由 `pipeline/eval_sensitivity.py` 生成，输出在 `smoke/sensitivity/report.json`。
