# 12 — 可行性闸门实验结果（2026-09-30）

执行 [11_preregistration_feasibility.md](11_preregistration_feasibility.md) 里跑之前写死的实验。
**本文件不给新判据，只报告结果与字面裁决，并把判据本身的缺陷挂出来。**

## 0. 字面裁决：NO-GO

预注册写的是：

> **GO**：至少一条轴出现**明确为正**的变化（配对 95% CI 不含 0，或点估计 ≥ +8pp）。
> **NO-GO**：所有轴的点估计 ≤ 0，或为正但落在噪声内。

三条轴全部明确为负 → **按字面判据，NO-GO。**

但判据本身有缺陷（见 §5），字面裁决与判据写明的目的不一致。**这个歧义不由我裁定。**

## 1. 结果

训练：1 epoch / 1,499 条随机 token-matched 轨迹 / LoRA r=16 / 32K，2.49 小时。
评测：与 base 完全相同的题目与预算，逐题配对。

| 轴 | n | accuracy | Δ | 95% CI | decodable | 翻转（上/下） |
|---|---:|---|---:|---|---|---|
| simple_java | 100 | 0.650 → 0.220 | **−0.430** | [−0.530, −0.340] | 0.930 → 0.270 | **0 / 43** |
| simple_javascript | 50 | 0.700 → 0.260 | **−0.440** | [−0.600, −0.260] | 0.940 → 0.300 | 3 / 25 |
| simple_python | 400 | 0.932 → 0.675 | **−0.257** | [−0.305, −0.212] | 0.968 → 0.740 | 7 / 110 |

三条轴方向一致、效应远大于 MDE（n=400 时 5.4pp）、java 上完全单向（0 上 43 下）。

**这回答了这个闸门要问的问题：训练这批经验确实产生可测变化。**「SFT 搬不动任何轴、
Δ_binA − Δ_binB 恒等于 0」这个致命可能性被排除了。

## 2. 机制：输出方言漂移（确认，非推测）

simple_java 100 题里 **73 题**报 `SyntaxError: Error parsing java the source code`。
原始输出显示三种方言，逐个用官方解码器验证：

| 方言 | 真实输出样例 | 官方解码器 |
|---|---|---|
| markdown 围栏 | ` ```json\n[GeometryPresentation.createPresentation(controller="mapController", parent="mapArea")]\n``` ` | **拒绝**（注意：围栏内的调用本身是合法 Java 语法） |
| terminus-2 信封 | `{"function_calls": ["SQLCompletionAnalyzer.makeProposalsFromObject(object='Customers', ...)"]}` | **拒绝** |
| func_name/params | `{"func_name": "DB2Tablespace.resolveTablespaceReference", "params": {...}}` | **拒绝** |
| 裸调用（期望格式） | `[X(a=1)]` | **正常解出** |

围栏那条的失败原因很具体：官方 `default_decode_ast_prompting` 先做 `strip("`\n ")`，会把首尾反引号剥掉，
**但留下 `json` 这个语言标签**，于是变成 `[json\n[...]]` → 语法错误。

## 3. 其中多少是"纯序列化约定"

用 [pipeline/dialect_diagnostic.py](../pipeline/dialect_diagnostic.py) 对已保存输出做方言归一化后**重新送 checker**：

| 轴 | 官方 gap | 归一化后 gap | 序列化占比 | 挽回题数 |
|---|---:|---:|---:|---:|
| simple_java | 0.430 | 0.110 | **74.4%** | 32 |
| simple_javascript | 0.440 | 0.180 | **59.1%** | 13 |
| simple_python | 0.258 | 0.182 | **29.1%** | 30 |

**归一化是诊断，不是官方指标；官方数字仍是官方数字，两个都要报。**
而且这个归一化器覆盖的形式**不完备**，所以上表是**下界**——真实序列化占比只会更高。
（一个证据：把归一化器从三种形式扩展到覆盖嵌套 dict 形式后，java 的占比从 58.1% 升到 74.4%。）

## 4. 条件正确率：有小的真实退化

`P(correct | decodable)`：

| 轴 | base | trained | 判断 |
|---|---|---|---|
| simple_java | 0.699（65/93） | 0.815（22/27） | **不成立**：CI [0.60,0.78] vs [0.63,0.92] 重叠，trained 侧样本太小 |
| simple_python | 0.964（373/387） | 0.912（270/296） | **边缘成立**：CI [0.940,0.978] vs [0.874,0.939]，恰好不重叠（约 −5pp） |

所以归一化后残留的 −11 到 −18pp 里，**至少有一部分不是格式问题**——simple_python 上
即使只看能解析的输出，答对率也下降了。

## 5. 预注册的缺陷（我的，必须写明）

GO 分支我写成「至少一条轴明确为**正**」，这把两件事混为一谈：

- 「训练**产生可测变化**」（判据写明的目的）
- 「变化**是改善**」（判据的字面要求）

实测给出的是**大而可靠、方向为负**的变化。按字面 → NO-GO；按目的 → 是，信号充足。

正确的写法应该是 `|Δ| > MDE`（任意方向），而不是 `Δ > 0`。我假设了「如果有效果，那效果应该是好的」，
而这个假设没有依据。

**看到结果之后再改判据是预注册要防的事，所以我不改。** 字面 NO-GO 成立，歧义上交给人和外部评审。

## 6. 对 pilot 的含义

1. **仪器可用，但这是一条敌意轴。** 所有 bin 训完在 BFCL 上都会更差。V_transfer 可以是负的，
   比较「哪个 bin 掉得少」仍是有效测量，但叙事要从「哪种经验更有价值」改成
   「哪种经验破坏更小」，且必须解释清楚这是跨接口格式干扰。
2. **主指标不能是官方 accuracy。** 在 java/js 上，官方 accuracy 的 59–74% 是输出约定差异；
   拿它做主表等于测方言漂移。必须在 pilot 前定死用什么（decodability / 归一化 / 条件正确率），
   并同时报官方数字。
3. **共模风险。** 所有 bin 来自同一 pool、同一 agent、同一 teacher，格式高度一致，
   所以漂移很可能在各 bin 间**幅度相近**。那样官方 accuracy 上的 bin 间差异会很小。
   **这是 pilot 最大的未知数，也是唯一值得先花预算测的东西。**
4. **near-OOD 从"可选"变成"必需"。** 本轮结果正好证实了「只有 far-OOD 会被质疑是格式干扰」
   这个担心的有据性。

## 7. 下一步（建议，未执行）

**低学习率对照**：本次用 LoRA r=16 / lr 2e-4，loss 在 ~100 步内掉到 0.42 后完全平掉——
这个形状既像"学得快"，也像"只拟合了模板"。换 lr 5e-5（或更小 rank）重跑一轮（约 2.5 小时），
看方言漂移是否显著减小：

- 若显著减小 → 格式是配方可控的，pilot 能在干净条件下比较 bin，且格式漂移不再是 confound。
- 若照样崩 → 必须把归一化/条件指标定为主轴，并接受 BFCL 只能是辅助轴。

在拿到这个对照之前，**不应把"终端 agent SFT 会破坏 BFCL 方言"当成结论**——它目前只是
"这一组超参下会破坏"。

## 8. 证据

| 内容 | 文件 |
|---|---|
| 配对结果（含 CI、翻转数） | `smoke/compare/feasibility_gate.json`、`compare.json` |
| 序列化占比诊断 | `smoke/compare/dialect_*.json`，脚本 `pipeline/dialect_diagnostic.py` |
| 逐题结果（base / trained，三轴） | `smoke/eval_bfcl/{java_js_b3072,trained_java_js_b3072,trained_js_b3072,base_sweep_a,trained_sp_b640}.jsonl` |
| 训练配置与日志 | `smoke/run_random1500/{train_summary.json,train_log.jsonl}` |
| 本实验的预注册 | `11_preregistration_feasibility.md` |
