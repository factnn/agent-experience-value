我看完了。先说结论：**项目现在是活的，而且你已经做出了一些真正影响论文设计的实验，不只是 engineering smoke test。**尤其 09_evaluation_sensitivity.md 很值钱——它直接证明如果不先校准 harness，后面几 pp 的“experience effect”完全可能是假的。factnn/agent-experience-value
但我也抓到了一个需要马上修的实质问题，而且挺重要。
目前最好的结果其实是“测量系统本身有多危险”
你测到同一 Qwen3-4B、同一批 BFCL 题，只改 max_new_tokens：
\[
75.8\%\rightarrow91.7\%\rightarrow95.0\%
\]
能漂 19.2pp。
java 也有 14pp 的变化。
这个结果对我们整个 paper 都有价值，因为我们最终想讨论：
\[
\Delta_{\text{experience}}
\]
很可能就是 3–10pp 级别。如果 harness 自己能漂 19pp，那么很多 agent data paper 的几 pp ablation 都值得怀疑。
而且你还测出来：
\[
\text{simple\_python}=93.2\%,\qquad
\text{multiple}=95.0\%
\]
已经 saturation，拿它们测 transfer 基本没意义。
这一步做得非常对。
但 multi_turn_base 当前的解释有一个错误
README 和 09 现在写的是：
multi_turn_base 的失败全部是 state-trajectory error / instance_state_mismatch。

我把真正的 base_mt16.jsonl 逐条统计了一下，实际是：
outcome	数量
correct	4
instance_state_mismatch	4
force_terminated	5
execution_response_mismatch	1
empty_turn_model_response	2


所以 12 个失败里，真正 instance_state_mismatch 只有 4 个。
更关键的是，这 16 个问题里总共发生了 221 次 completion，其中 3 个 completion 正好撞到 2048-token cap，涉及 3 个不同问题。
而你自己在 09 里刚刚建立的规则就是：
generation budget 必须提高到 cap hit = 0 才能冻结评测。

所以现在：
\[
\boxed{\text{multi\_turn\_base}=25\%}
\]
还不能当成干净的 stateful capability baseline。
先用 4096 重跑，甚至必要时更高，直到 0 cap hits，然后重新统计 failure taxonomy。
这个不是小文字问题，是会影响我们选择 primary benchmark 的。
java/js 其实也还没有完全 freeze
你现在把 java/js 提到 1536 后：
\[
\text{java}=65\%,\qquad
\text{js}=64\%.
\]
我又检查了逐条输出。
1536 下仍然有：
- java：5/100 个 completion 正好撞 cap；
- javascript：5/50 个 completion 正好撞 cap。
所以 1536 其实也没有满足你自己制定的：
\[
\text{cap hits}=0
\]
标准。
这两条应该再提一次，比如 2048/3072，看数字还动不动。
另一个特别重要的发现：LoRA smoke 已经暴露了 format interference
这个我觉得甚至比 0.80 这个数本身重要。
16-step LoRA：
\[
0.95\rightarrow0.80
\]
20 道 simple_python。
具体四个错误是：
- 3 个 decode syntax failure
- 1 个 nested type error
其中模型已经开始输出：
[func_name=solve_quadratic, params={...}]

这种形式，而 BFCL 官方 parser 要的不是这个。
所以这不是简单：
“SFT 以后 agent 能力下降了。”

至少很大一部分是：
\[
\boxed{
\text{terminal-agent SFT}
\rightarrow
\text{function-call dialect drift}
}
\]
这是一个必须提前处理的 confound。
因此以后 BFCL 主结果千万不能只报：
\[
Accuracy.
\]
至少拆成：
\[
\text{Decodability}
\]
和：
\[
P(\text{Correct}\mid\text{Decodable}).
\]
因为如果 Recovery bin 比 Random bin BFCL 低 5pp，但全部来自格式输出漂移，那不能解释成：
Recovery experience transfer value 更低。

不过这并不意味着 BFCL 不能用了
反而我觉得还能用，而且挺有意思。
只是它应该被定义成一个：
\[
\boxed{\text{far-transfer / cross-interface axis}}
\]
训练：
terminal-style agent trajectories

测试：
structured function-calling + stateful APIs

这其实是一个很强的 transfer。
但论文不能只有这个。
否则 reviewer 完全可以说：
你测到的是 output-format interference，不是 capability transfer。

所以我现在会要求最终至少有：
\[
\text{Near OOD}
+
\text{Far OOD}.
\]
Near OOD 用一个同样是 terminal / executable agent 的现成 benchmark。
Far OOD 用 BFCL。
这样如果某种 experience：
\[
\Delta_{\rm terminal}>0
\]
同时：
\[
\Delta_{\rm BFCL}>0,
\]
那 general-transfer story 才很漂亮。
BFCL 多轮其实规模够，不需要只用 16 个
当前 BFCL multi-turn 类别每个大约 200 个任务；完整 multi-turn suite 是 5×200，其中你当前本地已经支持 base / long-context / miss-function / miss-parameter 四类，所以理论上就是约 800 个可本地跑的多轮任务。GitHub
因此我们不缺 statistical power。
真正的问题是成本：
你测到约：
\[
242s/\text{task}.
\]
800 个单卡大约：
\[
54\text{ GPU-hours}.
\]
4 卡平行大约 13–14 小时/model。
所以 final table 可以跑 full multi-turn；pilot 不应该。
Data pool 部分我也觉得做得挺扎实
这里最重要的不是 94,334 这个数字，而是你没有自欺欺人。
你发现：
- 数据集叫 SFT-100K，但 pinned release 实际 94,334；
- result=None 根本不能当 success；
- runtime exception 不能当 task fail；
- task_complete=true 只是 agent 自报；
- 真正 verifier reward 应该在 trial artifact 里，但公开 release 没给；
- 71,255 条最终只是 engineering candidate，不是假装“training-ready”。
这个研究习惯非常重要。
而且：
\[
1,187,572
\]
assistant turns 中：
\[
98.675\%
\]
能解析出 commands JSON，这说明这批数据做 structural analysis 是靠谱的。
随机 200 条的 median 又是：
\[
13.7K\text{ tokens},
\]
80% 左右超过 8K，1/3 左右超过 16K。
所以你坚持不能粗暴截成 8K，我认同。否则最容易把：
\[
\text{failure}\rightarrow\text{recovery}\rightarrow\text{verification}
\]
后半截刚好裁掉。
现在真正的 blocker 是 success label
这个不应该无限找下去。
我建议给它设一个明确止损点。
你已经证明公开的 SFT-100K release 没有 oracle outcome。继续在互联网上刨 trial artifacts，边际收益开始很低了。
而你自己已经发现 RL-5K / TaskTrove 的 task binary 里面有 executable test.sh / reward.txt。
所以如果很快仍找不到旧 rollout 的 verifier result，我会直接换策略：
不用重新造 benchmark，但在现有 executable benchmark/tasks 上自己产生一批小而干净的 oracle-labeled experience pool。

这反而科学上更好：
同一个 task：
\[
\tau_1,\tau_2,\tau_3,\tau_4
\]
有的成功、有的失败、有的 recovery。
这样你可以控制 task difficulty，而不是：
success group 恰好来自简单 task，failure group恰好来自困难 task。

这对我们做“experience value”非常重要。
我现在会稍微改一下实验设计
不是推翻，而是升级。
原来：
\[
\text{Random / Success / Diversity / Recovery}
\]
各自从巨大 pool 抽一批。
这里有个天然 confound：
\[
\text{trajectory property}
\leftrightarrow
\text{task identity}.
\]
例如 Recovery trajectories 天然可能来自更难 task。
于是 Recovery 赢了，到底是：
recovery 好，

还是：
这批 tasks 更有学习价值？

更干净的设计应该尽量利用：
\[
\boxed{\text{within-task variation}}
\]
即同一个任务有多个 rollout。
例如同一任务：
- straightforward success；
- success after recovery；
- failure；
- high/low diversity trajectory。
然后先在 task 内比较/选，再 across tasks 组成 token-matched training sets。
这样你把：
\[
\text{task value}
\]
和：
\[
\text{experience value}
\]
分得更开。
如果能做到，这篇 paper 的可信度会明显上一个档次。
我现在对项目状态的判断
目前不是：
“pilot 还没开始，所以还啥都没有。”

而应该说已经完成了 Stage -1：Measurement validation。
而且它已经发现了两个很重要的事实：
\[
\boxed{
\text{Harness configuration effect}
>
\text{expected training effect}
}
\]
以及：
\[
\boxed{
\text{Cross-interface evaluation is vulnerable to format drift}.
}
\]
这两个发现会直接改变正式 experiment。
但目前还没有：
\[
\boxed{
\text{任何关于 experience value 本身的 evidence}.
}
\]
repo 里这一点写得很诚实，这是对的。
我建议你现在按这个顺序跑
1. 先修 BFCL calibration：multi-turn 用 4096+ 重跑那 16 个；java/js 提高 budget 到 cap hit=0；修正文档里“全部 state mismatch”的错误。
2. 马上测 training-seed variance：同一个 random subset，至少 3 个 seed。这个数字决定你后面所有主表到底要几 seed。
3. 做 Random baseline 的正式规模 SFT，先搞清楚“纯 agent SFT”本身对 near-OOD 和 BFCL far-OOD 的平均影响与 format drift。
4. 不要无限等 success label。短期找不到就转 executable task pool，自生成一小批带 oracle 的多-rollout experience。
5. 第一轮真正的科学 pilot 先只比 Random vs Diversity vs Recovery，最好做 within-task matched；只要看到稳定的 OOD separation，再扩 Success / NLL / Learnability / PMI。
我现在最想看的已经不是再多几个 smoke test，而是：
\[
\boxed{
\text{同一批 tasks，两个不同 experience-selection rule，}
\\
\text{在完全相同 token budget 下训练后，}
\\
\text{OOD 到底差不差。}
}
\]
一旦这张最小的两三行表出来，我们才第一次真正触碰到这篇 paper 的核心假设。
还有一点我挺想强调：**你现在这个 repo 的研究纪律比很多 paper 强。**但不是因为“代码多”，而是你已经在主动找会把自己结论搞假的东西——缺失 success label、generation budget、ceiling、format drift、split leakage。这个方向如果最后出结果，这些前置 work 会让结果可信很多。
不过先把我抓到的那个 multi_turn_base 解释 bug 修掉，这个确实是当前最明显的一处。