# 06 — Failure Analysis

> 状态：**已成稿**。五类失败标签逐类实测；`F_physical_termination` 是**本轮新增**，
> 因为把摔倒/越界归入 `B` 会**错误归因**。
> 仍待补：`E_ood`（跨机制留一，P1）。

---

## 6.1 Three assumptions under test

本文主张：在这个问题设定下，有三个看起来很合理的假设会失效，而且**失效方式不同**——
它们不是同一个失败的不同说法。

| # | 假设 | 若成立应观察到 | 实测 | 证据 |
| --- | --- | --- | --- | --- |
| **A1** | 当前观测足以决定下一步动作 | 存在只依赖 $O_t$ 的规则在两个世界里都好 | **否**：$O_A=O_B$ **逐像素相同**（max diff 0），最优动作相反，且**无公共可成功动作** 8/8 | H1 |
| **A2** | 知道目标在哪里就足以行动 | 朝记忆目标走能重新看到 | **否**：16 个成对世界 **0 %**，给到 10 步预算仍 **0 %** | H2 / E2 |
| **A3** | 几何视角质量足以衡量动作价值 | 几何认可的动作物理执行也成功 | **否**：几何认可的动作里 **43.1 % 物理失败**；`NBV` 在成对世界 **0/16** | H3 |

**三者的可分性是关键**，不是"它们都失败"。§6.4 给出可分性证据。

---

## 6.2 Failure taxonomy

`episode.py` 实现六类标签。前五类来自研究计划，第六类是本轮**由数据逼出来**的。

| 标签 | 定义 | protocol B（24 地形） |
| --- | --- | --- |
| `A_belief_absent` | 决策时没有任何目标 belief | **0**（进场段回放保证每个策略都有 belief） |
| `A_belief_wrong` | belief 存在但错 | **0**（重建精确，本 benchmark 上不可产生） |
| `B_action_wrong` | belief 正确、几何不是解释、仍然失败 | `HeadingLock` / `HistoryOnly` / `LostTargetSearch` / `ObjectPermanence` **各 18**；`Reactive` 17；`NBV` 16；`Oracle` 3 |
| `C_local_viewpoint` | 策略**由几何视角代理驱动**，其首动作被该代理认可而被物理否定 | `NBV` **1** |
| `D_inefficient` | 成功，但耗时超过 oracle 参考 2 倍 | `NBV` **1** |
| `E_ood` | 训练/测试机制外 | 需跨机制留一，**待做（P1）** |
| **`F_physical_termination`** | **动作被合理地选出，但物理后果没被预测到：摔倒或越界** | `Reactive` 1、`NBV` 1、**`LearnedConsequence` 4**（其 5 次失败中的 4 次） |

**一个 `A` 都没有，`B` 占满经典 baseline**——这就是「知道目标在哪 ≠ 知道该做什么动作」
在统一 benchmark 上的复现。

### `C` 的两个把关，都必须写进论文

1. **`C` 由 `uses_geometry_proxy` 把关。** 第一版把 `C` 也判给了 `Reactive` 等策略，
   只因为它们的盲选动作**恰好**落在几何预测认可的集合里。但 `C` 的语义是
   「失败的原因是几何代理」，而这些策略**没有这个代理**。
   **把巧合归因于不存在的机制**是 failure taxonomy 最容易犯的错。
2. **`C` 需要同时存在几何认可与物理否定的**可行动作。若不存在任何可成功动作，
   失败就不能归因于视角选择。

### `F` 为什么必须独立于 `B`

这是本轮**由数据逼出来的**修正，也是 taxonomy 里最容易被写错的一处。

`LearnedConsequence` 的 5 次失败，终止原因是
`out_of_bounds 2`、`fallen 2`、`budget_exhausted 1`。第一版分类器把它们**全部**记为
`B_action_wrong`。这是错的：

> 该规划器的动作是**基于一个合理的预测**选出来的——模型认为这个动作会重新看到目标。
> 它对**可见性**的判断可能是对的，对**稳定性**的判断是错的。

这既不是「belief 错」（`A`），也不是「有好动作但没选」（`B`），更不是「几何代理误导」（`C`）。
它是**物理后果预测不完整**。修正后的分布：

| 规划器 | `B_action_wrong` | `F_physical_termination` | 其他 |
| --- | ---: | ---: | --- |
| `HeadingLock` / `HistoryOnly` / `LostTargetSearch` / `ObjectPermanence` | **18** | 0 | — |
| `Reactive` | 17 | **1** | — |
| `NBV` | 16 | **1** | `C` 1、`D` 1 |
| `LearnedConsequence` (E4-3) | **1** | **4** | — |
| `Oracle` | 3 | 0 | — |

**读法**：经典 baseline 的失败**几乎全是动作选择**（`B`）；
学习式后果规划器的失败**主要是物理终止**（`F`，4/5）。
**失败类型随方法改变而改变**——这本身就是「后果预测是缺失环节」的直接证据，
而且它说明**下一步该补的是什么**（稳定性/边界后果），而不是"再多训一点"。

一个反过来的观察也值得写：经典 baseline 里 **`Reactive` 和 `NBV` 各有 1 次 `F`**
（都是摔倒）。所以 `F` 不是学习式方法独有的，但**只有学习式方法的失败以 `F` 为主**。

> 实现：`episode.classify_failure`；`episode.PHYSICAL_TERMINATION = "F_physical_termination"`。

---

## 6.3 Findings F1 → F5 in detail

### F1 — Visibility is non-monotonic
8 个机制全部非单调，$N_{\mathrm{flip}} \in \{2,3,4\}$。
`basin` 的**方向与** `stairs_up` **相反**：`stairs_up` 是"远处可见、靠近反而被顶峰挡住"，
`basin` 是"远处看不见、走近才露出"。→ §4.3

### F2 — The current observation is insufficient
8 对成对世界，观测**逐像素相同**（mean/max pixel diff = **0**），
最优动作相反，**8/8 无公共可成功动作**。
所以这不是"哪个 baseline 差"，而是 **partial observability 是任务的结构性困难**。→ §4.4

### F3 — Target memory alone is insufficient
16 个成对历史世界上，朝向记忆目标的规则 **0 %**，
**10 步预算不改变这一点**（仍 0/16）。→ §8 E2

### F4 — Heading preservation ≠ target-relative spatial preservation
IMU 层面的朝向锁定（`HeadingLock`）在两个世界都**保持住了丢失瞬间的朝向**，
仍然 **0/16**，且与 `Reactive` 的逐地形成败模式**完全相同**（配对差 0.0，CI [0,0]）。
**朝向在惯性系里被保持，但目标相对于机器人的几何已经变了。**

### F5 — Geometric viewpoint ≠ physical re-observability
- `NBV` 拿到了**精确高度场**，仍 **0/16**；
- 几何认可的动作用物理执行 **43.1 % 失败**；
- 反过来把 oracle 放到**实际执行到的位姿**上，与物理结果一致 **97.9 %**；
- 独立证据（Experiment 0）：几何 oracle 说 `climb` 能看到，物理 8 秒闭环执行后
  横漂 **1.54 m**、目标出视场；改用朝记忆目标闭环的 `climb_toward` 才成功
  （横漂仅 **0.08 m**）。

> **这条差距不是感知问题，而是闭环问题，所以换更好的传感器不能解决。**

---

## 6.4 Separability: A1, A2 and A3 are not the same failure

论文必须证明三者可分，否则会读成"同一个失败被说了三遍"。

| 对照 | 可分性证据 |
| --- | --- |
| **A1 vs A2** | A1 的世界里 $O_A=O_B$ 而 $M_A \neq M_B$（历史不同）；A2 的世界里 $O_A=O_B$ **且** $M_A=M_B$。构造层面就不同 |
| **A2 vs A3** | A2 只需要"目标位置"，A3 需要"动作后果"。E4-1 直接给出：只用 $M$ 的 AUC 0.852，加上动作后 0.934（**+0.080**），而加上观测 $O$ 只到 0.854（**+0.002**）。所以**补观测补不上 A3** |
| **A1 vs A3** | H1 的两个世界**几何都很清楚**，只是"哪边能看到"不同；H3 的世界几何**预测得很自信但错了**。前者是信息不足，后者是后果预测错误 |
| **A2 vs A3（行为层）** | `LostTargetSearch` 与 `ObjectPermanence` 是"更用力地做 A2"——它们与 `Reactive` 逐地形成败**完全相同**。**加强 A2 的机制没有换来任何 A3 的收益** |

### 交叉表：失败类型 × 机制

| 机制 | 经典 baseline 主要失败 | 学习式规划器主要失败 |
| --- | --- | --- |
| `barrier` | `B`（横向绕行不在其动作空间内） | 成功 |
| `trench` | `B` | `F`（越界） |
| `stairs_up` | `B` | `F`（越界） |
| `pit` | `B`（需要 `back`,`back` 这类序列） | `B` + 预算耗尽 |
| `basin` | `B` | 成功 |
| `waves` | `B` | 成功 |
| `perlin` / `perlin_rough` | `B` | 成功 |
| 16 个成对历史世界 | `B`（**全部**） | **成功（含动作切换）** |

**结论**：失败类型**随机制和方法一起变化**，不是常数。
`LearnedConsequence` 把成对世界从 `B` 变成了成功，同时把自己的失败从 `B` 变成了 `F`。

---

## 6.5 What is still missing

- [ ] `E_ood` 的定量：IID vs leave-one-mechanism-out 的下降幅度（P1）；
- [ ] `F` 的细分：`fallen` 与 `out_of_bounds` 是否需要分开建模（当前合成一类）；
- [ ] `D_inefficient` 目前只有 `NBV` 的 1 例，样本太少，**不宜单独下结论**。

> 诚实标注：`C` 只有 1 例、`D` 只有 1 例。**单例不足以支撑机制性结论**，
> 论文里应写成"存在该类失败"，而不是"平均每 N 个 episode 出现 1 次"。
