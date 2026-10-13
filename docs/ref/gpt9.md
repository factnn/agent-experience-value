在线 Agent RL：核心参考文献与实验规模对齐
日期：2026-10-09
用途：为 Agent Experience Value 项目规划模型、训练对照、迁移评测与重复实验。
研究问题：哪些采样前可获得的经验价值信号，能预测预算受控在线 RL 干预的未见任务收益？
本文区分【论文报告】与【本项目建议】。依据论文正文、附录和表格核对，不代表独立复现。旧文献清单包含 SFT、静态数据选择及机器人工作，本清单优先服务在线 RL。文献存在不意味着问题被解决；模型数量也不是论文质量的充分条件。

1. 结论先行【本项目建议】
主矩阵继续用当前固定版本的 Qwen3-4B；在核心效应可靠后，增加同代 Qwen3-8B 验证规模，再用一个不同模型家族复验关键对照。第二、第三个模型不必复制全部实验。闭源 API 只做推理参照，不是本项目必要条件；32B 及以上也不是起步要求。
优先顺序：可识别的干预和学习剂量 → 可靠评测和训练重复 → 同族规模验证 → 不同家族或外部环境 → 更大模型／闭源参照。
这不是顶会统一门槛，更不是“做满三个模型就能发表”。证据范围由实际对照、效应、不确定性和适用条件决定。
2. 六篇核心实验参考【论文报告】
R1. CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution
- 论文：https://arxiv.org/abs/2604.15840
- 正文：https://arxiv.org/html/2604.15840v1
- 会议页面：https://aclanthology.org/2026.acl-long.1055/
- 实际后训练骨干：Qwen2.5-7B-Instruct、Qwen3-4B、Qwen3-30B-A3B。三个检查点都属于 Qwen 系列，不是三个独立家族。
- 主实验：AppWorld TestN/TestC、BFCL-v3；表中闭源大模型是推理参照。主要消融集中在 4B。
- 优先读 Table 1、Table 6、§4.4 与 Appendix A.2。Table 6 专门交叉比较训练域与测试域，值得借鉴；不能将“每个环境分别训练、分别测”混称跨环境迁移。
- 对本项目的价值：4B 可以是主要研究平台；更大模型只需验证关键结论。30B-A3B 是 MoE，不能当成密集 30B，也不能按 3B 活跃参数直接估计权重显存。
R2. Self-Evolving Curriculum for LLM Reasoning（SEC）
- 论文：https://arxiv.org/abs/2505.14970
- PDF：https://arxiv.org/pdf/2505.14970
- 主骨干：Qwen2.5-3B、7B；当前 PDF Appendix B 另有一个经过专门 mid-training 的 Llama-3.2-1B 变体，在 Countdown 上补跨家族验证。
- 主实验包括 Countdown、Zebra、ARC、数学；GRPO 为主，另有小范围 PPO/RLOO 对照。Appendix C 报告 puzzle 训练 3B 240 步、7B 120 步；数学均 240 步；batch 256、每题 8 rollout。
- 优先读 §3、Appendix B–C。最可借鉴的是“主要家族做系统实验，另一家族做针对性复验”，不是所有模型跑同样大的笛卡尔积。
- 边界：Llama 附录不证明全部任务跨家族复现；也不能忽略其特殊初始化。
R3. RAGEN-2: Reasoning Collapse in Agentic RL
- 论文：https://arxiv.org/abs/2604.06268
- PDF：https://arxiv.org/pdf/2604.06268
- 主骨干 Qwen2.5-3B。Table 4 另测 0.5B/1.5B/7B、3B-Instruct、Llama-3.2-3B，以及 Qwen2.5-VL-3B 的部分设置。
- 主算法比较 PPO/DAPO/GRPO/Dr.GRPO；规模扩展表的 scale block 使用 PPO。核心任务有 Sokoban、FrozenLake、MetaMathQA、Countdown，另有更广任务验证，不是所有轴全交叉。
- §4.1/Appendix B.2：至多 400 次 rollout–update，每次 8 个 prompt ×16 条轨迹；固定 512 个验证 prompt/环境。
- 优先读 Table 4、§4、Appendix B。它提供“现象—测量—机制—干预—外推”的研究结构。奖励方差过滤发生在采样后，不能当成我们的采样前分配器直接复现。
- Table 4 报 peak 表现；本项目必须预先说明终点/验证选点规则，不能按最终测试最优点挑结论。
R4. TopoCurate: Modeling Interaction Topology for Tool-Use Agent Training
- 论文：https://arxiv.org/abs/2603.01714
- PDF：https://arxiv.org/pdf/2603.01714
- 骨干 Qwen3-8B/32B；同时研究 SFT 初始化和 RL 任务选择。ID 为 τ²-bench 三域，OOD 为 BFCL-v3 multi-turn。
- 优先读 Table 2、§5.4、Appendix B.2/B.4：更换 SFT 起点与从相同 SFT 起点更换 RL 任务池，是两个分开的对照。
- 文中 8 次 repeated tests 用于 pass@1/4/8，是评测重复，不能算成 8 个独立 RL 训练种子。
- Appendix B.2 写 RL 80 步，60 题/轮、每题16条；同段又写 rollout batch 1024。60×16=960，与1024的口径不完全相同，精确复现前需核对实现，不能擅自补齐。
- 对本项目：学习基准结构和控制变量，不直接接受其机制解释为已被普适证明。
R5. An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents
- 论文：https://arxiv.org/abs/2505.15117
- PDF：https://arxiv.org/pdf/2505.15117
- 主受控比较含 Qwen2.5-3B/7B；§5 规模轴扩展到 14B/32B，并比较普通 Qwen 起点与 DeepSeek-R1-Distill-Qwen 起点。
- 七个 QA 评测数据集区分 ID/OOD；研究奖励、初始化、规模及检索环境，不是所有条件完全交叉。
- 优先读 §5、Figure 2、Tables 2–3、Appendix G。G.1–G.4 报告相关训练至多600步、batch512、GRPO group5，存在按训练崩溃早停的规则。
- 对本项目：参数规模与是否经过推理蒸馏应分开讨论；DeepSeek-R1-Distill-Qwen 不是一份与 Qwen 毫无联系的独立骨干家族证据。
R6. DIVE: Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use
- 论文：https://arxiv.org/abs/2603.11076
- PDF：https://arxiv.org/pdf/2603.11076
- 会议页面：https://proceedings.mlr.press/v306/chen26gj.html
- 主方法骨干是 Qwen3-8B；同时研究数据生成、SFT 和 RL。规模化对象主要是训练数据/工具多样性，不是模型参数量。
- 重点读 §4–5、Table 1/2、Figure 3、Appendix F。24 条数据 scaling paths 与多层 OOD 评测说明：一个主要骨干，也可以承担深入研究，但绝不意味着实验量很小。
- RL 设置为100步，正文说明512-task batch，附录 rollout group size8。
- Toolathlon 3次、其他benchmark4次是评测运行，不应读成训练seed数。
- 对本项目：参考迁移维度与数据结构控制；不能把“数据生成+SFT+RL”整体增益全归因于在线经验分配。
3. 补充参考：不必都成为新 baseline
文献	作用	主链接
Prioritized Level Replay	环境/任务学习潜力与 replay 的经典前身	https://proceedings.mlr.press/v139/jiang21b.html
No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery	检验 proxy 实际测量了什么；learnability 前身	https://arxiv.org/abs/2408.15099
Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?	用户早先关注的 RLVR 边界工作；版本v4中需区分现成RL checkpoint与作者新训练	https://arxiv.org/abs/2504.13837
Deep Reinforcement Learning at the Edge of the Statistical Precipice	小样本RL不确定性、汇总及评估纪律	https://arxiv.org/abs/2108.13264


LESS/RegMix 可保留为价值预测与小模型代理研究的邻域参考；不将静态数据结果冒充在线RL证据。OpenThoughts-Agent 仍对数据审计有用，但不应再主导当前在线RL实验配方。
4. 阅读别人实验表时，分开数五类东西
1. 实际训练的骨干：哪些模型真正接受本文的 RL 更新？
2. 方法基线：在同一骨干上比较 Uniform、Frontier、Coverage 或其他算法。
3. 固定模型参照：API、大模型、其他团队现成 checkpoint，只回答性能位置。
4. 训练重复：重新采样和更新的独立运行；不能用同一模型反复评测代替。
5. 评测重复：固定参数，对任务多次采样；不能把 pass@k 的 k 当作训练种子数。
一个 table 有十几行，并不意味着作者训练了十几个模型；反过来，一个骨干上的数十个受控条件也可能是很大的实验工作量。对本轮未明确核实的独立训练种子数，记录“未明确报告/本轮未确认”，不擅自填写1或3。
5. 本项目建议的模型分工
模型	分工	需要做什么	不需要做什么
当前 pinned Qwen3-4B	主研究平台	多条件干预、剂量检查、主要重复、ID/迁移、阶段复验	在没有信号时用更多表格掩盖问题
同代 Qwen3-8B	同族规模验证	预先选定的关键对照、可靠评测与重复	复制全部探索性消融
另一骨干家族，如 Llama-3.1-8B-Instruct	跨家族针对性验证	同一核心假说的真实 RL 干预	只跑零样本分数就声称跨模型成立
更大密集模型或MoE	后续强化	只有当结论明确涉及规模或资源充足时增加	为凑大参数标签而优先投入
闭源API	可选推理参照	如需展示任务难度，可放独立参考区	冒充同样参数更新干预的外推证据


官方候选模型卡：
- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
不同家族的 tokenizer/chat template/tool serialization 需各自正确适配，再保证任务、奖励与预算语义一致。选择另一模型应依据开发集上的接口正确性和可测学习区间，而不是哪一个恰好支持我们的结论。
“Qwen3-4B的base”必须写完整仓库ID和revision：相对本次RL未训练，不等于从未经过post-training。thinking模式也必须记录。4B→8B 仅支持这两个检查点范围的规模复验，不构成scaling law。跨检查点还有预训练/后训练和架构细节差异，不能把全部变化因果归给参数数量。
6. 先有足够的学习剂量，再谈模型名单
文献中的step不是统一单位。例如RAGEN-2一轮包含128条轨迹，SEC正文batch256且每题8条；本地一轮可能只有一个任务的4条轨迹。不能仅拿“他们80/100/240step、我们若干step”作可比预算。
本项目每个条件必须记录：全部生成token、输入/上下文处理量、不同任务组数、轨迹数、发生奖励差异的组数、有效更新与参数位移、评测曲线、时间/显存，以及探测和失败成本。
推荐先在4B跑一个预声明的训练剂量诊断，在同一条运行中按预算保留B/2B/4B等检查点，并在开发集观察可测学习范围。最终测试不参与剂量选择。检查点不是独立训练重复；不同分支继续保持同样的停止规则。目标不是一定看到正向收益，而是避免只由一两次随机任务选择支配全部结论。
7. 一个分阶段、可核算的实验计划【建议，不是文献门槛】
第一步：最小确认批次
同一个4B、一个预定义任务条件和起点，3条分配规则×3个独立训练/采样seed，共9条分支。共同历史与优化器状态配对；各seed三条规则都运行。先判断干预是否实际分离、学习剂量和评测是否有分辨率。
这9条不是完整论文，也不能凭3个方法均值拟合普适价值规律。
第二步：4B主要矩阵
3规则×2任务条件×2训练阶段×3独立分支seed = 36条RL分支。
注意：前一步可以纳入该矩阵，不必重复花费。阶段必须是实际不同学习状态，并按训练预算或开发规则提前选择；不能从最终测试中挑“最有故事”的阶段。仅在同一共同checkpoint上改变续训seed，反映的是条件于该起点的随机性；更广的结论还需考虑共同起点/历史本身的变化。
第三步：选择性外部验证
- Qwen3-8B：2个最关键规则×1任务条件×1阶段×3seed = 6条分支。
- 另一家族：同样6条分支。
- 合计示例 48条分支。每个外部模型增加第二个条件时各再加6条，共可到60条。
这些数字没有包含共同起点构建、探测、每个checkpoint评测、剂量预实验、失败运行与额外机制控制，不是总GPU预算。外部对照应由4B发现与预定规则决定，不能看外部结果后再换比较对象。
如果资源不足，可以先完成4B与8B，明确只在Qwen范围验证。不要用“更多零样本模型行”替代真正的跨模型学习干预。
与信号预测claim的边界
36个运行结果不等于36个独立任务机制。两个任务条件、三个规则、若干seed，主要支持规则比较和阶段交互。要声称“采样前信号可以预测新干预收益”，需要多个有实际信号变化的任务组/分配条件，并按任务组或干预配置留出验证。可重分配主要矩阵的预算，而不是一味加模型。不能随机打散共享任务/起点的seed造成预测泄漏。
8. 评测的优先级
- 先保证每个模型有共同起点M0、Uniform与候选规则，在相同完整面板上报告G和V。
- 主模型正式结果优先完成已声明的ID/迁移面板，不以8题面板的12.5pp刻度冒充稳定效应。存在模板或工具类相关时，统计分组也应承认这种依赖。
- 独立训练重复与每题重复采样分开报告。3个训练seed只是建议起点，区间仍宽时增加重复；不是“3必然足够”。
- 若声称跨环境，需要真正train-on-A/test-on-B。BFCL内部留出工具菜单组合，不能直接叫跨真实环境。
- 当前先把一个现成benchmark内的受控结论做牢。跨工具/环境的核心claim需要第二个独立执行环境的针对性验证，不能仅靠增加模型。
- 策略和配置选择只看训练/开发；最终held-out数据不反复用于挑prompt、算法、步数或最佳checkpoint。
9. LoRA与算法边界
本项目目前研究的是固定LoRA更新子空间、固定GRPO配置下的经验价值。发现出现后，用一个替代LoRA rank或target配置复验关键对照，比先加一个闭源推理基线更能检验结论是否依赖更新方式。不是要求立刻全量微调多个大模型。
若要声称超出GRPO，增加一种不同更新/归一化算法的针对性验证；只测GRPO，就写清楚GRPO条件。不要扩成“所有模型×所有算法×所有环境×所有seed”。
10. 建议的主表
Learner/阶段	任务条件	分配规则	获取预算/实际成本	有效奖励组/更新	ID score/G	Transfer score/G	V vs Uniform	独立训练重复
Qwen3-4B early	预定义组A/B	Uniform/Frontier/Coverage	待测	待测	待测	待测	待测	3，报告不确定性
Qwen3-4B later	相同对照	同上	待测	待测	待测	待测	待测	同上
Qwen3-8B	关键确认条件	两条关键规则	待测	待测	待测	待测	待测	3
另一模型家族	关键确认条件	两条关键规则	待测	待测	待测	待测	待测	3


闭源参照若加入，另放“未训练的推理表现”小表；没有对应干预的API模型，不填写本项目的G/V。
11. 本轮对旧清单的修正
1. 不再把SFT/data-recipe实验量当作在线RL的直接规范。
2. CoEvolve已经有明确跨域评测；不能概括所有邻近工作“只测局部收益”。
3. SEC当前PDF有Llama针对性附录，不能只凭旧版本说其完全没有跨家族验证。
4. TopoCurate明确有8次评测重复；不能说完全没有重复，也不能把它计为8次训练。
5. RLVR边界论文包含既有RL模型checkpoint评估；长模型名单不等于作者从头训练全部模型。
6. 对未明确报告的seed、成本或冲突口径保持缺失标记，不替作者猜。
12. 最终执行取向
用一个小模型把问题测清，再用更大同族和另一家族检验关键结论；不是先下载一批模型，再想每个模型该测什么。
核心参考阅读顺序：CoEvolve → SEC → RAGEN-2 → TopoCurate → Search-Agent Empirical Study → DIVE。读取时依次检查真实训练骨干、初始化、干预对象、预算、训练/评测重复、迁移划分和checkpoint选择。继续保护原来的测量研究scope，不为凑模型数量把主线改成排行榜。