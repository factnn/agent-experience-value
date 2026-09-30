# 10 — 投资判断：这个 idea 到底能不能跑通（2026-09-30）

这份文档回答一个问题，而且只回答这一个问题：

> **在已经投入更多训练、更多评测、更多基础设施之前，这个研究值不值得做下去？**

它不是实验报告。没有实验结果，因为**决定性的实验还没能跑**。这份文档记录的是：哪些风险已经被排除、
哪些风险仍然致命、以及让判断收敛的那一个有界实验。

## 0. 结论

| 问题 | 判断 | 依据 |
|---|---|---|
| 技术上做得出来吗？ | **能，已证明** | 全池扫描、切分、32K LoRA 训练、官方 BFCL 单轮+多轮评测全部跑通，成本实测 |
| 测量上测得准吗？ | **能，但有纪律要求** | 评测跨卡逐字节确定；但配置（预算/批大小/加载路径）造成的偏差大于待测效应，必须冻结 |
| 科学上测得到吗？ | **目前测不到** | 缺 verifier → H1 与 ID 增益都不可测；池子的主导变异是来源而非经验属性 |
| 值得投吗？ | **取决于一个有界实验** | 见 §6。通了就投，不通就按 K3/K4 收缩或转向 |

**一句话**：瓶颈从头到尾都不在算力、管线或评测，而在**标签**。所有其他工作都是在为一个还无法测量的靶子做准备。

## 1. 已排除的风险：技术执行不是瓶颈

这一轮（09-29/09-30）的工程量证明了下述事情都可做，且成本已知：

- **数据**：94,334 行全池扫描；71,255 条工程候选；四个键（task family / 指令 hash / 对话 hash / run+trial）
  跨临时 train/dev 的泄漏均为 **0**（已实测复核）。
- **训练**：32K 上下文 Qwen3-4B LoRA 在单张 40GB A100 上可跑。全参不可行（32K 时 fwd+bwd 已 34.1GB，
  AdamW 状态还需约 32GB），lm_head 必须分块否则 16K 就 OOM。成本约 **2.8 GPU-hours/bin/epoch**。
- **评测**：官方 `bfcl_eval` 的 prompt/parser/decoder/checker 全部接入，单轮与多轮推理循环都在本地跑通，
  多轮会真实执行官方 mock API 并做状态轨迹检查。**不需要 Docker，不需要外部服务，不花钱。**
- **可复现性**：同配置换物理卡重跑，20 题输出**逐字节相同**——评测噪声为 0。

这些都是任何分支都要用的东西，但它们确实排在了错误的位置：先做了仪器，没先问靶子在哪。

## 2. 仍然致命的风险：测量靶子

### 2.1 没有 verifier，核心 claim 就算不出来

这不只是「H1 成功假设测不了」。更根本的是：

- 没有 verifier → **没有 ID 增益**
- 而 C1「proxy 预测 ID 但不预测 OOD」是 04 文档列的头号可能结果，它**要求 ID 和 OOD 两侧都有数**

证据：
- 发布数据没有 `reward` / `verifier_output` 列。`result` 的全量取值是运行异常
  （`AgentTimeoutError` 32,764、`ContextLengthExceededError` 916、空 60,296），空值不代表 verifier 通过。
- 11 个公开候选探测无果；3 个返回 401，原因未确定。
- 词法错误候选在保留池里基率 **67%**（50,145 / 75,879）——作为 recovery 打分几乎没有方差。

在这个状态下能做的只有 Pilot B：random / diversity / recovery 三臂、outcome 全部 `unknown`。
按 04 文档自己的判据标准，那大概率不够一篇。

### 2.2 池子的主导变异是「来源」，不是「经验属性」

71,255 条候选的来源分布：

| 来源 | 候选数 | 占比 |
|---|---:|---:|
| issue | 24,375 | 34.2% |
| swesmith | 19,061 | 26.8% |
| superuser | 18,951 | 26.6% |
| tezos | 8,868 | 12.4% |

一个 teacher（GLM-4.7）、一个 harness（terminus-2）、四个 source。而 source 之间的行为差异极大——
实测操作分布：swesmith 分片 `other` 占 3.1%，superuser 分片占 **59.8%**。

OpenThoughts-Agent 已经证明 source 是最大影响因素。所以 **token-matched 的 bin 只要跨 source 抽样，
测到的就是「superuser 数据比 swesmith 好」，而不是「恢复结构有没有价值」。** 必须做 within-source 设计，
但那会把每个 bin 的可用样本压到四分之一（最大来源 issue 24,375 条，而且它还没有对应的可执行任务）。

### 2.3 评测轴与训练分布不匹配

训练是终端修 bug 轨迹，可用的评测是 BFCL 的 API 工具调用。这两者的关系很弱，训练完全不搬动
BFCL 是完全可能的。而且实测 base 模型在 BFCL 主要轴上已经**饱和**：

| 轴 | n | base 准确率 |
|---|---:|---:|
| multiple | 200 | 0.950 |
| simple_python | 400 | 0.932 |
| parallel_multiple | 200 | 0.865 |
| parallel | 200 | 0.860 |
| simple_java | 100 | 0.650 |
| simple_javascript | 50 | 0.640 |
| multi_turn_base | 16 | 0.250 |

饱和轴上即使有真效应也显示不出来。唯一有头部空间的 `multi_turn_base` 也不是这个训练分布的同族任务。
08/03 文档本来就把 BFCL 定位成 cheap diagnostic，真正的靶子是 terminal/stateful 环境——**而那需要 verifier**。

**§2.1、§2.2、§2.3 指向同一个东西：一个能对训练池自身任务给出 oracle 结果的 verifier。**

### 2.4 §2.2 有一个比我原先写法更强的解法：within-task 设计

我原来的缓解方案是「within-source 抽样」，但那只控制到来源层级。更干净的是 **within-task**：

> 同一个任务有多个 rollout——straightforward success / success-after-recovery / failure / 高-低多样度——
> 先在**任务内**比较和选择，再跨任务组成 token-matched 训练集。

这样才能把 **task value** 和 **experience value** 分开：否则 Recovery 组天然可能来自更难的任务，
Recovery 赢了到底是「recovery 经验好」还是「这批任务本身更有学习价值」，无法区分。

代价：这要求每个任务有多个 rollout。现有池子里 71,255 条候选只覆盖约 37,186 个 task family
（≈1.9 rollout/family），**密度太低，做不了 within-task 配对**。要真正做这件事，就必须
在可执行任务上**自己生成一批多-rollout 的、带 oracle 标签的经验池**——
这正好和 §4 的标签路径是同一件事。所以那个有界实验的收益不止是「有标签」，而是「能做出干净的设计」。

## 3. 平台约束（今天新发现，影响整个项目）

### 3.1 这台机器没有容器能力

在「能不能自己跑出标签」的思路下，先探测了执行能力，结论如下：

| 通道 | 状态 |
|---|---|
| docker / podman / nerdctl / ctr / runc | 未安装，无 socket |
| singularity / apptainer / enroot | 未安装 |
| Kubernetes 提交 pod | **无权限**——SA `system:serviceaccount:airs:default` 连 `list pods` 都被拒 |
| bubblewrap | 失败：`Creating new namespace failed: Operation not permitted` |
| `unshare --mount/--pid/--net` | 全部 DENIED |
| CAP_SYS_ADMIN | **不存在**（CapEff `0x00000000a80465fb`），mount 系统调用被禁 |
| `unshare --user` | 可用 |
| `chroot` | **可用**（CAP_SYS_CHROOT 在） |

我们跑在一个 K8s pod 里（namespace `airs`），但那个 SA 没有任何 pod 权限。

### 3.2 但 chroot 后备路径实测可用

任务验证器要求固定的绝对路径布局（`/tests`、`/logs/verifier/reward.txt`、`/app` 或 `/workspace`）。
用「硬链接 rootfs + chroot + 非 root 用户」实测：

- 构建 rootfs：`cp -al /usr` 硬链接，秒级完成、几乎不占空间（NVIDIA 驱动文件因跨设备失败，与任务无关）
- 验证器布局：`chroot $R /bin/bash /tests/test.sh` → 脚本正常运行，`reward.txt = 1` ✓
- 非 root 执行：`chroot $R /usr/sbin/runuser -u sandbox -- …` → `running as sandbox uid=1000`，能写 reward ✓
- **隔离边界成立**：沙箱内 `ls /share` → `No such file or directory`，项目目录不可见 ✓

**所以后备路径存在。** 但必须诚实标注三个残余风险：

1. **没有 PID 与网络隔离**。只有文件系统被约束；任务代码仍能看到宿主进程并访问网络。
   对公开研究数据集风险有界，但不是零。
2. **环境保真度不确定**。rootfs 是本容器的 Ubuntu 22.04 + Python 3.10，而任务声明的镜像是
   `ubuntu:24.04`、`python:3.10-slim` 等。验证器可能因此误判。
   **缓解办法**：每个任务先用它自带的参考解（`solution/`）跑一遍，要求 reward=1，作为环境标定。
3. 硬链接 rootfs 下，任务若**原地修改**既有文件会影响到宿主。只读使用是安全的。

### 3.3 网络

Hugging Face 今天不可达（`huggingface.co`、`datasets-server.huggingface.co` 均无响应），
而 PyPI 与 GitHub 正常。这直接阻断了任务数据的获取。

## 4. 标签路径：已知与未知

**已知（好）**：任务包结构完整且自洽。离线解码一个 TaskTrove 任务包得到：

```
instruction.md          # 任务描述
task.toml               # 超时、分类、标签
tests/test.sh           # 验证器入口 → 写 /logs/verifier/reward.txt (0/1)
tests/verifier.py       # 实际判定逻辑
environment/Dockerfile  # 环境定义
solution/               # 参考解（可用于环境标定）
```

验证器契约干净：跑 `tests/test.sh`，读 `/logs/verifier/reward.txt` 得 oracle 标签。
TaskTrove（161 万任务）中存在训练池四源中的三个，且名字直接标注验证状态：

| TaskTrove 分区 | 对应候选数 |
|---|---:|
| `laion__swesmith-oracle-filtered-v2` | 19,061 |
| `laion__stackexchange-superuser-sandboxes-verified-v2` | 18,951 |
| `laion__stackexchange-tezos-sandboxes-verified-v2` | 8,868 |
| **合计** | **46,880（占候选池 65.8%）** |
| issue 源 | 无对应分区（24,375 条，最大来源） |

**未知（关键）**：轨迹能不能 join 到任务。

- `task.toml` 里**没有可 join 的任务名或 ID**；`path` 是不透明 ID（`task_0`、`task_1007`）。
- 所以 join 键只能是**指令/正文文本**。但两侧包装不同：轨迹侧的描述在 terminus-2 prompt 的
  `Task Description:\n…\nCurrent terminal state:` 之间，任务侧是独立 `instruction.md`
  （例如 nl2bash 任务带 `## Environment Setup (run before starting)` 前缀）。
- 一次离线测试取了 6 个任务包做精确 hash join，全部不匹配、模糊相似度仅 0.05–0.12——
  **但该测试无效**：这 6 个取自各数据集的默认分区起始位置（nl2bash、pymethods2test），
  并不是 swesmith/superuser/tezos 分区。不能据此判断 join 不可行。
- 结论：**join 可行性未验证，需要网络恢复后拉取目标分区实测。**

## 5. 判断

- **技术可行性已证明**，不是瓶颈。
- **科学可行性未证明**，卡在 verifier 上。在拿到标签前，H1 与 local-to-transfer gap 都无法测量；
  只做 Pilot B 大概不够一篇。
- **平台把「自己跑出标签」这条路变窄了，但没堵死**：没有容器，但 chroot 后备路径实测可用。
- **所以值不值得投，取决于两个未知数**：join 是否可行、chroot 环境能否让验证器给出可信标签。
  这两个都可以用一个小实验回答。

## 6. 下一步：一个有界实验（建议 1 天，纯 CPU）

**目标**：回答「我们能不能为训练池的轨迹拿到 oracle 成功标签」。

**步骤**

1. 网络恢复后，下载 `laion__swesmith-oracle-filtered-v2` 分区（若过大则只取前若干行）。
2. **环境标定**：取 5 个任务，用 chroot harness 跑其自带 `solution/`，要求 `reward.txt = 1`。
   若参考解都过不了，说明环境保真度不足 —— 这条路的成本立刻上升一个量级。
3. **Join 测试**：把任务的 `instruction.md` 与 94,334 条的 `instruction_sha256`/正文做归一化匹配，
   报告命中率。归一化要处理包装前缀与空白差异。
4. **端到端**：取 5 条能 join 上的轨迹，用 chroot harness **重放其命令**，再跑 `tests/test.sh`，
   看是否产出 0/1 混合的真实标签。

**成败判据（事先写定）**

| 结果 | 判断 |
|---|---|
| 参考解全过 + join 命中率 ≥30% + 重放能产出非退化标签（非全 0/全 1） | **投**。核心 claim 变得可测，且后续成本低（replay 纯 CPU，可 256 核并行） |
| 参考解全过，但 join 命中率极低 | 改走 §2.4 的路：不 join 旧轨迹，直接在可执行任务上生成多-rollout 经验池（贵，但设计更干净） |
| 参考解过不了（环境保真度不足） | 需要真正的容器运行时；先解决平台，否则按 K4 判断 |
| 以上都失败 | 只剩 Pilot B。对照 K3（迁移无法干净评测）与 K4（ground-truth 干预不可负担）决定收缩或转向 |

**另外两条必须在正式实验里落实的要求**（来自 09 §2.8 与外部评审）：

1. **评测轴必须是 near OOD + far OOD 两条。** Near OOD 用同为 terminal/executable 的 agent benchmark，
   far OOD 用 BFCL。只有两个方向同时为正，general-transfer 的故事才成立；只有 BFCL 的话，
   reviewer 完全可以说测到的是 output-format interference。
2. **BFCL 结果必须拆成 decodability 与 P(correct | decodable)。** 16 步 LoRA 的实验已经证明
   终端 agent SFT 会让模型输出 `[func_name=X, params={...}]` 这种 harness 方言——不拆开就无法区分
   「能力下降」和「方言漂移」。

**这个实验的成本与收益**：一天、纯 CPU、不动 GPU、不训练。它决定的是整个项目能不能测——
相比之下，再加一轮训练或再加一条评测轴的信息量接近零。

## 7. 证据与复现

- 平台能力：`/proc/self/status` 的 `CapEff`、`unshare` 逐项测试、`bwrap` 报错、
  K8s SelfSubjectAccessReview、chroot 实测（本轮，未落盘为脚本）
- 池与标签：`05_data_pool_audit.md`、`08_full_pool_preparation.md`、`audit/outcome_probes/index.json`
- 训练与评测能力：`smoke/README.md`
- 评测灵敏度与 MDE：`09_evaluation_sensitivity.md`
- 任务包结构：`audit/outcome_probes/open-thoughts__TaskTrove.json` 与
  `open-thoughts__OpenThoughts-Agent-RL-5K.json` 中的 `task_binary`（base64+gzip+tar），
  可用 `python` 离线解码复核

## 8. 本轮的过程教训

这一轮把顺序做反了：先造仪器，后才问靶子在哪。项目文档里本来就有 K1–K5 判据和 claim occupancy test，
应该在最开始就拿来评估「这个靶子可不可测」，而不是在跑通训练和评测之后。

**下一轮开始，任何新的基础设施投入之前先问：它是否让某个 claim 变得可测？如果答案是否，就不做。**
