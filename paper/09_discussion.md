# 09 — Discussion and Limitations

> 状态：**已成稿**。四个讨论结论各有实验支撑；限制清单已按最新结果更新
> （L6 已撤回、L4/L9 已被后续实验取代、新增 L13–L16）。

---

## 9.1 The chain the results support

$$
\underbrace{O_t\text{ insufficient}}_{\text{H1}}
\;+\;
\underbrace{M_t\text{ alone insufficient}}_{\text{H2}}
\;+\;
\underbrace{\text{geometric } V \neq \text{physical } V}_{\text{H3}}
\;\Rightarrow\;
\underbrace{P(V_{t+H}\mid O_t,M_t,a)}_{\text{E4}}
$$

而且第三项**不是感知产生的，是闭环产生的**（控制器跟踪、横向漂移、动作时长）：
把 oracle 放到**实际执行到的位姿**上，它与物理结果一致 **97.9 %**。
**所以换更好的传感器不能消除这条差距。** 这是本文与常规 active-perception 叙事的分界。

---

## 9.2 Four discussion points

### D1 — Why are the classic baselines indistinguishable?

不是"它们算法差"。在 24 个地形上，`Reactive` / `HeadingLock` / `HistoryOnly` /
`LostTargetSearch` / `ObjectPermanence` 五者的**逐地形成败模式完全相同**
（相对 `HistoryOnly` 的配对差 **0.0，CI [0, 0]**）。
这不是"性能接近"，而是**它们做出了同一个决策**。

原因是结构性的：这五种机制**都在处理同一个量**——"我该朝哪走/转到哪个朝向"，
即 $O_t, M_t \to$ **朝向或方向**。它们中**没有一个**显式建模

$$O_t, M_t, a \;\to\; \text{future physical consequence}$$

既然缺的是同一个东西，那么无论把"朝向/方向"这条规则写得多讲究，**决策结果都一样**。
配对 bootstrap 在这里没有方差可重采样，所以**不报 $p$**——报了没有意义。

这也解释了为什么 `NBV` 是**唯一**在普通地形上略有差异的一个（0.292 vs 0.250）：
它是唯一一个**碰巧**用了动作后果代理（运动学位移）的方法，而运动学位移恰好是
真实后果的一个**粗糙近似**。但如 D2 所示，这个近似**不够好**。

### D2 — Why does NBV not clearly beat HistoryOnly?

因为**几何可见性与物理后果之间存在系统差距**，而 NBV 只拿到了前者。

- 几何认可的动作里 **43.1 % 物理失败**；
- `NBV vs HistoryOnly` 差 **+0.042**，95 % CI **[0.00, 0.125]**，$p = 0.356$ —— **不显著**；
- 在 16 个成对历史世界上 `NBV` **0/16**，与其余五个经典方法一样。

而且这条差距**不是可见性模型错了**：同一个 oracle 放到实际执行位姿上准确率 **97.9 %**。
错的是 $a \to S$ 这一环：标称位移与实际执行位姿平均差 **0.89 m**
（前向 0.52、横向 0.34、高度 0.40、偏航 5.5°）。

> **结论**：`NBV` 的失败**不是实现问题，是 formulation 问题**。
> 它问的是"哪个视角能看到"，而任务需要的是"执行哪个动作之后能看到"。

### D3 — Why does an action-conditioned World Model have value?

**不是因为 "World Model 很强"，而是因为一个具体的、被测量的事实：**

$$\boxed{\text{The action itself changes what can be observed in the future.}}$$

三条直接证据，逐层加强：

| 证据 | 数字 |
| --- | --- |
| 加动作带来的信息，远大于加观测 | $O{+}M$ **0.854** → $O{+}M{+}a$ **0.934**（**+0.080**）；而 $M$ **0.852** → $O{+}M$ **0.854**（**+0.002**） |
| 后果预测可以学习，且明显优于解析模型 | 位姿 MAE **0.469 → 0.148**（**3.2×**） |
| 预测能力能转化为任务收益 | $RSR@10$ **0.250 → 0.792**（**3.2×**），非特权；$RSR@1$ **0.167 → 0.708** |

第二行有一个**必须一起写的限定**：学习模型在前向（5.5×）、横向（2.9×）、偏航（3.7×）
上大幅胜出，但**高度上输给解析模型**（0.090 vs 0.149）。
所以正确表述是"学习模型在**决定成败的分量**上胜出"，**不是**"学习全面更好"。

**并且——这也是本文最重要的方法论结论之一——收益并不是无条件的。**
E4-5 把安全性也纳入后果后：

$$Oracle\text{-}S\ (0.411) > Oracle\text{-}V\ (0.388) \quad\text{但}\quad
Learned\text{-}V{+}S\ (0.312) < Learned\text{-}V\ (0.366)$$

**安全信息在原理上有价值（oracle +0.023），但学习到的安全头在每一个惩罚权重下都
让规划器更差**（含软惩罚 $\lambda \in \{0.5,1,2,5\}$：0.34–0.36）。
安全头 AUC 0.900 说明它**排序好**，但**排序好不等于可以当决策项用**——
瓶颈是它在**决策边界上的校准**。

> 因此 World Model 的价值是**有条件**的：信息必须在**决策边界上可信**，
> 而不只是"平均指标高"。这是本文能给后续工作的一个可检验的告诫。

### D4 — Why the remaining loss is a modelling target, not a new modality

`LearnedConsequence` 的失败**换了类型**：5 次失败里 **4 次是 `F_physical_termination`**
（越界 2、摔倒 2），只有 1 次是 `B_action_wrong`。
而五个经典 baseline 的失败**全是 `B`**。

模型学会了"**动作把我带到哪里、从那里能否看到目标**"，
但**不会预测失稳/越界**。这是**后果预测的另一面**，是**同一个缺口的延续**，
而不是"候选动作不够多"或"需要理解语义"。

> **因此目前没有证据支持引入 VLM 式的候选生成**：瓶颈在**后果模型的覆盖范围**
> （终止/稳定性），不在**候选的多样性**。而且 WoMAP 一类工作已经覆盖
> "VLM + World Model 做具身探索"，把它加进来不会增加本文的科学内容。

---

## 9.3 Limitations

### L1 — All results are from simulation
MuJoCo 加一份**第三方（MIT）盲步态策略权重**，本仓库未重训练、未微调、
**未在真机验证**。`.gitignore` 忽略 `*.pt`，新克隆需按资产说明下载并核对 SHA256。
**没有任何 sim-to-real 结论。**

### L2 — `Oracle` is a privileged upper bound, not a method
每个决策点对候选各跑一次物理 rollout。它证明「该量携带决策信息」，
**不**证明「已经能预测它」。`OracleSearch` 同理，是**参考规划器**且**不完备**
（F9/F10）。

### L3 — The learned predictor does not generalise across mechanisms
OOD **0.50**，朴素基线 **0.75**。必须与 IID（1.00）和基线 OOD 同时给出。
**本文的 E4 用的是留一地形划分，即机制内的泛化**；跨机制仍是待做项（`E_ood`）。

### L4 — The safety head is learnable but not decision-ready, and calibration is not why
AUC 0.900、整体 ECE 0.042，却在**每一个**惩罚权重下降低规划成功率（0.366 → 0.31–0.36）。
**E4-5b 进一步排除了"校准不够"这个解释**：整体 ECE 0.042 掩盖了
**决策带内 ECE 0.197**（差 4.7×）；温度缩放把决策带 ECE **减半到 0.099**，
规划收益 **0.000**——三个变体在任意 $\tau$、任意 coverage 下都不超过不用安全的 0.3661，
且最优点都在 coverage ≈ 0。**限制在可见性与安全性的联合决策边界上，
不是"概率准不准"。**

### L5 — A binary fall label carries no signal
720 次单动作 rollout 里 **0 次摔倒**。安全标签因此改用
`out_of_bounds`（正例率 0.083）与 `upright_min`（连续）。
**这意味着真正的"摔倒"事件在单动作粒度上不可学**，需要多步或不同的观测粒度。

### L6 — ~~Part of the terrain is unsolvable even for a greedy oracle~~ **WITHDRAWN**
审计后结论相反：**24/24 全部可解，0 个不可解**。原先判定"4 个地形不可解"是
**我们的参考规划器不完备**（贪心在第一步无成功候选就退回 `forward`），不是地形的性质。
beam search 用 2–3 步序列就解决了那 4 个地形（`forward,forward` / `back,back` /
`right,right` / `toward,left_hold,back`）。

两个后果：`RSR on solvable ≡ RSR`，分母无需修正；
**上界是 1.00 而不是 0.88**。

这条撤回本身值得写进论文：**把 planner 的缺陷误当成 benchmark 的性质**是这类工作
里很容易犯的错，而它差点让一个人为的上限进入结果表。

### L7 — Controller capability bounds the terrain complexity
盲策略只能走 ≤0.15 m 竖板 / 约 30° 以下。**可见性翻转次数与可走性直接冲突**：
`waves` 幅度从 0.35 m 提到 0.45 m 并细分到 120 段时翻转从 3 升到 6，
但机器人 6 s 只走 0.2 m 卡死。

### L8 — Perlin terrain is not arbitrary 2D
为了可走性必须把 2D 噪声朝中轴线压平（$\beta = 0.35$）。
**完整 2D 起伏 + 有意义的幅度 + 可走的量化台阶，对这具机器人目前不可同时满足。**

### L9 — The paired-world construction is restricted to directional occluders, and that is characterised, not unaddressed
H1 的成对世界**只能**在能表达**方向性隐藏状态**的机制上构造。
Mechanism Support Audit 证明：`stairs_up` / `trench` / `pit` / `waves` / `basin` 的
高度场**严格 $y$ 对称**（$\max|h(x,y)-h(x,-y)|$ = **0.0000 m**），
**在原理上不可能**产生单侧致胜集；`barrier` 支持（单侧率 0.095）；
`perlin` 系虽有 0.2–0.3 m 的 $y$ 不对称，但**随机不对称不足以**产生单侧集。

> 所以正确表述是：H1 的成对世界构造**只在能够支持方向性隐藏状态的机制上评估**。
> 把 $y$ 对称机制放进去会把"构造不出来"误写成"现象不普适"。

### L10 — failure type A (wrong belief) cannot occur in this benchmark
目标重建是精确的（$\hat T = \text{eye} + r\,u(\beta,\varepsilon)$，无噪声），
所以"belief 估错"被 benchmark 设计**排除**了。要研究它需要注入检测噪声。
实测 `A_belief_absent` 与 `A_belief_wrong` **均为 0**。

### L11 — The time budget is part of the conclusion
`pit` / `barrier` 的最佳动作依赖预算。预算加到 12 s 后 `barrier` 不再强制侧移。
换预算会改变成功率，**因此预算必须与结果一起报**。

### L12 — The geometric $V(x)$ profile includes unreachable poses
profile 图里的可见性是几何的，包含不可达位姿；**物理 rollout 才是真值**。

### L13 — `C` and `D` failures have one example each
`C_local_viewpoint` 与 `D_inefficient` 在 protocol B 下各只有 **1** 例。
**单例不足以支撑机制性结论**，只能写成"存在该类失败"，不能写成发生率。

### L14 — Target memory is a structured representation, not open-world memory
`M_t` 是「目标位置 + 年龄」的**结构化**表示，由相机检测精确重建。
它**不是**开放世界的语义记忆（"那是一个红色的阀门"）。
本文关于 $M_t$ 的结论**只在这个表示上成立**。

### L15 — The action set is a fixed primitive set
10 个固定原语。动作空间的离散化是否影响 H1/H2 的结论，
**尚未做 sanity check**（计划中的实验 6）。目前不能排除这一可能性。

### L16 — The learned planner is evaluated on one controller
全部结果基于同一份 `go2rl` 权重。**更换控制器会改变 $a \to S$ 的映射**，
因此姿态后果模型的数字**不能跨控制器直接引用**。

---

## 9.4 What must not be claimed

- ❌ "首次研究目标丢失后的重新搜索"
- ❌ "首次用主动感知解决重新观察"
- ❌ "Agent + World Model + 候选动作"是创新（WoMAP / NavWAM 已做）
- ❌ 学习式预测器可以替代 oracle
- ❌ "World Model 全面优于解析模型"（高度上解析更好）
- ❌ "安全头可以用了"（E4-5 显示每个权重下都更差）
- ❌ History Horizon 找到了"最优 horizon"
- ❌ 跨地形族泛化成立
- ❌ RSR@10 = 0.792 是"解决了 79 % 的任务"——它是在**本 benchmark 的
  预算、控制器、动作集、地形分布**下的数字
- ❌ 任何真机结论

写法纪律：用 **a lack of unified analysis**，不用 *nobody has studied*。
