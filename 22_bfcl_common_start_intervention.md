# 第一批 BFCL 共同起点经验价值干预

2026-10-08。原始大目标见 [scope](15_scope_and_evidence_ladder.md)；本批将已有工程组件用于第一批受控短程干预，不新增泛化校准关卡。参考 `docs/ref/gpt7.md` 的反馈：真实奖励方差不等于实际学习；两个模式的成功不能拼成一种配置；历史探测必须计费；结果要进入采样前信号—暴露—收益的表。

## 采样前固定的设计

机器可读协议：[BFCL_INTERVENTION_PROTOCOL_20261008.json](rl/BFCL_INTERVENTION_PROTOCOL_20261008.json)。训练/评测划分沿用已保存清单及哈希；这是自定义研究划分，不是官方榜单。

第一批训练条件为包含 GorillaFileSystem 的全部 15 训练题，以及包含 TradingBot 的全部 23 训练题，每组各四种可见工具组合。不使用开发成功例作为训练题，也不根据奖励删题。Vehicle/Travel 条件、更多学习阶段、独立训练重复与更强环境迁移仍保留在后续 scope。

共同起点是 **base 等价的零初始化 LoRA + 初始化的空 AdamW/constant scheduler + 完整 RNG + 有偿训练探测历史**，不是两条独立 warmup 后的模型。对以上八种工具组合，各选一题，每题四条完整多用户轨迹，仅获取历史、不更新参数。按固定 salted ID hash 选题，未查看 gold 路径/结果决定选择。共同成本只记一次；各分支同时报告继承成本和新获取成本。其余训练组合无历史，按 Beta(1,1) 先验处理，绝不提供免费 87 题成功率。

所有分支统一 Qwen3-4B / official TRL GRPO，LoRA r8 q/v，AdamW 1e-5 constant，beta0，group 标准化，四条轨迹一更新，thinking=True，完整 completion 8192（包含外部反馈），单段 4096，32 calls / 32 segments。选择 thinking 是本批共同可运行配置，**不声称效率最优**；之前完整配对校准中，thinking 输出成本约为非推理的 9.6 倍。

| 规则 | 定义 |
|---|---|
| Uniform | 条件内任务均匀 |
| Frontier | 组合层最近 16 个终局奖励，Beta(1,1) 成功率 p；每个任务取所在组合的 4p(1-p)，归一化后混入 20% 任务均匀探索 |
| Coverage | 组合获取组数 n 的 deficit=1/(1+n)，按组合 deficit 归一化后在组合内均匀选题，混入 20% 任务均匀探索 |

Coverage 衡量可见可用工具组合的历史获取不足，不是 oracle 执行路径结构覆盖。共同历史和起点概率在首个分支 rollout 前保存；每次选择保存此前历史、信号与概率。在线历史只能影响之后的采样。

每个分支新生成 token 目标 32,768，失败/截断全部计入，在完整 optimizer 更新后停止。单组最大 32,768 超额；记录实际成本，**不把预算阈值相同写成实际成本完全相同**。最多 12 更新或 3,600 秒，在完整更新后检查。时间/更新数先到会标记为非 token-matched 分支。共同探测最多 3,600 秒、要求八组全部完成，若不完整则不分叉。

## 同一面板测 G、V

提前按相同 salted ID hash 固定 16 题：四个 primary 类各两道 ID 新题，共八题；四种留出组合各两道，共八题。每题一个固定采样 seed，各政策使用相同 seed/顺序/限制；评测不会进入分配器。起点与各分支分别评测。最多 3,600 秒、题间停止；不完整面板不计算完整 G/V，也不把尚未采样的题当零。

- G_ID / G_transfer = 分支表现减共同起点表现。
- V_ID / V_transfer = 分支表现减同条件 Uniform 表现。
- 主报告为全部 ID / 组合迁移面板均值；附 primary 对应子面板与逐题成对结果。
- 同时列实际任务/组合暴露、起点信号、输出/重复输入/时间成本、优势、梯度、参数变化。优化器 momentum 更新与新非零梯度区分。

第一批一套训练 seed、两个任务条件、每政策十六条评测轨迹，是探索性测量，无法支持普遍信号—收益预测关系。评测题数不是独立训练重复。零梯度、同样暴露或零 G/V 均是应保留的结果，不通过事后换题/改奖励追求阳性。

## 执行与发布

`pipeline/run_bfcl_intervention.py` 构建共同状态、训练与独立评测；`bfcl_allocation.py` 实现组合规则。恢复复用已验收的 checksum/provenance/model/optimizer/scheduler/RNG 校验路径；起点 global_step=0，没有虚构暖启动参数更新。BFCL 奖励、真实 user 轮次和原始 token 的外部 mask 沿用既有组件。新组合分配与完整恢复有针对性 CPU 检查。

先在一张实际空闲 GPU 构建共同状态，再最多三张空闲 GPU 执行分支/评测，低于用户四卡上限。保存协议、源文件哈希、PID、命令、日志、每组进度；CPU 调度器发布完成组并 push，最终汇总 G/V 或明确未完成项。二进制学习状态/adapter 本地保留，GitHub 发布可审计清单与轨迹。

## 已完成的共同起点（2026-10-08 09:48 UTC）

八组全部完成，32 条轨迹中 11 次官方终局成功；31 条对话完整结束，1 条触及生成段长度上限。生成 90,122 个模型 token、1,361,916 个非 padding 输入位置、9,738 个外部反馈 token；生成耗时 2,580.8 秒，main wall 2,607.9 秒。所有输出，包括失败和截断，均计入共同获取成本。原始 token/mask 审计通过，见 [common_audit.json](rl/bfcl_intervention_20261008/common_audit.json)。

探测没有 optimizer 更新，trainable 参数 fingerprint 前后相同；完整组件状态已保存。首两个 GFS 分支从同一 payload 恢复成功，权重校验值、空 AdamW/constant scheduler、RNG 和有偿历史一致。Trainer 的 AcceleratedOptimizer 包装内恢复原始 AdamW，修正在首个分支启动前完成并留痕；原始协议、探测数据与算法未改。

GPU 4 执行共同起点评测、GPU 5/6 执行 GFS Uniform/Frontier。后续 Coverage 与 Trading 分支仍在固定队列中。**当前尚未审计实际 fresh gradient，也没有 G/V 结果；11/32 是固定共同策略的探测成功率。**

解释边界补充（仅元数据核对，不改协议）：组合在整个 87 题研究训练划分中留出，但本批定向分支只在 GFS / Trading 子池更新。GFS 子池没有 MathAPI，Trading 子池没有 MessageAPI，Vehicle/Travel 也未进入这两个子池。因此本批迁移是“整体划分留出的组合面板上的收益”，不能自动解释成“每个组成工具都经过该分支训练后的纯组合泛化”。报告会逐组合列出该分支更新任务池中缺少的组成类；更多 primary 条件与更完整共同学习阶段仍保留在后续 scope。

## 首次真实更新已审计（2026-10-08 09:58 UTC）

同一个完整起点出发，GFS Uniform 首组 task 21 奖励 `[1,0,1,1]`，标准化优势非零，梯度范数 0.0335635，trainable fingerprint 从 `8684bab2…` 变为 `db714a65…`；7,685 个模型 token 进入 loss，895 个外部 token 排除。GFS Frontier 首组 task 22 为 `[0,0,0,0]`，优势/梯度为零、参数未变；10,831 个模型 token 全计入获取成本，1,540 外部 token 排除。两组各四条对话全部完成、无截断，且属于同一可见 GFS+Twitter 组合的不同题。

见 [first_update_audit.json](rl/bfcl_intervention_20261008/first_update_audit.json)：共同 payload、起点权重、成本、原始 token/mask 和优势公式逐项核对通过。**这是本批 actual fresh gradient/parameter update 的证据，不是能力收益，也不是 Uniform 胜出的证据。** 剩余固定预算训练和共同面板评测正在执行；G/V 尚待完整结果。

## gpt8 分析补充：保留策略，检查可识别性（2026-10-08）

`docs/ref/gpt8.md` 对已发布数据的审查指出：本批共同探测每组合恰好一组，候选组合的初始 coverage deficit 全为 0.5。故任意起点分配概率下 E[d] 都为 0.5，无法解释收益差异。候选组合的 observed episodes 也恒为 4。报告现在逐条件/总体标记这些常量为“无预测对比”，不拟合、不事后更换 coverage 定义。原始任务概率按既有可见类组合求和并保留，策略与预算不改；这些分布/暴露记录是诊断数据，不是根据成绩新挑选的预测假说。见 [初始信号核对](rl/bfcl_intervention_20261008/initial_signal_identifiability.json)。

GFS 两条已完成分支各只有两个获取决定。Uniform 新生成 34,245 token、Frontier 38,818 token；实际成本差约 13.4%，第二组长任务占各自输出获取成本 77.6% / 72.1%。Uniform 一组有新非零梯度，第二组虽然优势全零，Adam momentum 仍改变参数；Frontier 两组全零，最终模型参数与起点相同。报告增加逐组 prior 信号—实际奖励/优势—参数变化以及长组成本占比，区分信号估计噪声、组合估计单位与具体题目差异、更新效率和最终收益，不将两组全失败解释为难度假说失败。

Frontier 构成现成的未改变模型对照。用保存的原始 prompt/completion/mask/segments/bridges、奖励、停止原因和轮数逐条对照共同起点；当前已完成前缀 11 条逐项精确一致，未新增模型调用，见 [前缀核对](rl/bfcl_intervention_20261008/unchanged_frontier_control_prefix.json)。最终报告在完整面板上重新核对；如出现差异，标为采样/运行差异待查，不解释为学习收益。Uniform 已观察到 token/停止行为变化，但不能将其当二元任务收益；也不比较不同长度的未完成面板。

本轮继续完成六个声明分支与同一面板，不给单臂临时追加预算、不增加校准任务。更多获取决定、独立训练重复和后续 coverage 的有变差信号，应在下一批采样前定义；原始大 scope 保留。

## 第一份完整面板 G（GFS / Uniform，后续分支仍在执行）

GFS Uniform 的 16 题评测全部完成、同题/同 seed/原始 token/mask 审计通过。ID 从共同起点 3/8 到 3/8，G_ID=0；整体划分留出组合从 2/8 到 3/8，G=+0.125。唯一二元结果变化是 `multi_turn_base_190`（TravelAPI+TicketAPI）从失败变成功。见 [first_complete_uniform_gain.json](rl/bfcl_intervention_20261008/first_complete_uniform_gain.json)。

这是一训练 seed、两个获取决定、每题一条采样轨迹下的一题翻转，不是稳定迁移优势；没有定义或拟合新 predictor，也未改变策略/预算。Uniform 作为自身参照 V=0，其他规则的 V 待各自完整面板；整批六分支矩阵尚未完成。

## 第二份完整面板与 Coverage 暴露核对（2026-10-08）

GFS Frontier 的完整 16 题评测已结束，保存的 prompt/completion/mask/segments/bridges、奖励、停止原因和用户轮数全部与共同起点精确一致。ID=3/8、留出组合=2/8，两个 G 均为 0；相对同条件 Uniform，V_ID=0、V_transfer=-0.125。这是未发生参数更新的推理对照，核对复用已有轨迹、没有额外模型调用。见 [first_complete_frontier_control.json](rl/bfcl_intervention_20261008/first_complete_frontier_control.json)。一题差异、单训练 seed 和实际成本不等的限制仍成立，不给方法排名。

GFS Coverage 训练完成两组、38,818 新生成 token，任务顺序也是 22→16，两组奖励全零、无新梯度、参数与共同起点相同。规则的概率分布虽不同，此次实际任务暴露却与 Frontier 一致；不能把两个方法名当作两个独立经验干预。Coverage 的原定完整评测及 Trading 三分支仍继续，不按结果跳过或追加预算。

## 唯一 Uniform 成功翻转的具体含义

复用保存轨迹核查 task 190：共同起点与 Uniform 的前四轮均通过官方检查，最后一轮创建工单的 title/priority 相同。起点的 description 少了用户明确指定字符串末尾的句号，Uniform 保留该句号；起点最终 state 与 reference 的递归对比只有这一个字段差异。因此 +12.5pp 对应的是一次精确字符串遵从修正，不能单凭它宣称工具组合能力提高。早期调用路径也有差异，但两边此前四轮均已通过。见 [uniform_reward_flip_case_audit.json](rl/bfcl_intervention_20261008/uniform_reward_flip_case_audit.json)。

这不是判定 checker 错误：用户确实指定了包含句号的描述。官方奖励和 G/V 保持原样，案例分析没有新增模型调用、替换指标或事后定义 predictor。更强的能力收益仍需完成整批，并在后续预先定义的面板与独立训练重复中验证。
