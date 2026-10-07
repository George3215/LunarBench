# 08 — Experiments

> 状态：**E1 / E2 有完整结果可直接写；E3 单步版可写但必须标注协议；E4 / E5 待补。**

顺序按用户给的研究线：**问题成立 → 历史需要多少 → 现有方法为什么失败 → 学到的模型是否泛化 → 改进是否解决失败**。
论文里的呈现顺序是 E1 → E2 → E3 → E4 → E5。

---

## E1 — Does the current observation contain sufficient information?

**Setup.** 8 对成对世界（§4.4）。每对共享地形、位姿、朝向与动作历史，只在
「目标藏在窗口哪一侧、因而最后一次从哪里看到它」上不同。成对世界由 7 条**预注册的纯几何判据**
自动搜索接受（49 候选 → 8 通过；三个种子分别接受 8/8/8）。

**Result.**

| 量 | 值 |
| --- | ---: |
| 观测逐像素相同的对数 | **8 / 8**（mean diff 0.0, max 0, 100% identical, target pixels 0/0） |
| 最优动作相反的对数 | **8 / 8** |
| **没有任何动作在两个世界都成功的对数** | **8 / 8** |

决策规则在 16 个世界上：reactive **0%**、history-only **0%**、heading-lock **0%**、oracle **100%**。

**Conclusion.** $O_A = O_B$ 逐像素相同，最优动作相反，且可行集不交。
任何 $O_t$ 的函数在两世界取值相同，因此**不存在无记忆策略能同时解决两者**。
历史不是「提高精度的补充特征」，而是**当前观测可证明不包含的决策信息**。

> Evidence: `history_necessity.json`, `history_pairs.json`, `history_camera/`, `figure_history.py`

---

## E2 — Can existing re-observation mechanisms solve the problem?

**Setup.** 冻结接口（§5），24 个地形（8 多地形场景 + 16 成对世界），8 个策略，
protocol B（闭环），预算 $N=10$，每步 $\Delta = 3.0$ s。全部策略共享同一观测流、
同一动作集、同一预算与同一终止条件（由 `test_api.py` 断言）。

**Result (protocol B).**

| 策略 | RSR@1 | RSR@3 | RSR@5 | RSR@10 | 成功时步数 | $T_{\text{reobs}}$ | 路径代价 | failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Reactive` | 0.17 | 0.25 | 0.25 | 0.25 | 1.33 | 4.0 | 2.29 | B:18 |
| `HeadingLock` | 0.17 | 0.25 | 0.25 | 0.25 | 1.33 | 4.0 | 2.59 | B:18 |
| `HistoryOnly` | 0.17 | 0.25 | 0.25 | 0.25 | 1.33 | 4.0 | 2.60 | B:18 |
| `LostTargetSearch` | 0.17 | 0.25 | 0.25 | 0.25 | 1.33 | 4.0 | 2.63 | B:18 |
| `ObjectPermanence` | 0.17 | 0.25 | 0.25 | 0.25 | 1.33 | 4.0 | 2.58 | B:18 |
| `NBV` | 0.17 | 0.25 | 0.25 | **0.29** | 2.29 | 6.86 | 3.33 | B:17, **D:2** |
| `Oracle` (greedy) | **0.83** | **0.88** | 0.88 | 0.88 | 1.05 | 3.14 | 1.94 | B:3 |
| **`OracleSearch` (beam, correct bound)** | 0.75 | **1.00** | **1.00** | **1.00** | — | — | — | — |

**The correct upper bound is 1.00, not 0.88** (see E2b). All 24 terrains are solvable; the
greedy oracle is an *incomplete reference*, not a tight bound. Any statement about
remaining headroom has to use 1.00.

分组（区分力全在右列）：

| 策略 | 多地形场景（8） | **成对历史世界（16）** |
| --- | ---: | ---: |
| 六个经典 baseline | 6 / 8 | **0 / 16** |
| `NBV` | 5 / 8 | **0 / 16** |
| `Oracle`（贪心） | 8 / 8 | **13 / 16** |
| `OracleSearch` | 8 / 8 | **16 / 16** |

#### E2b — Solvability audit (P0；任何「修正分母」的前置条件)

在拿 `RSR on solvable` 修正任何数字之前，必须先确定「solvable」是什么意思——
**特别是：贪心参考失败的地形，到底是地形不可解，还是参考规划器太弱。**

| 口径 | 定义 | 是否穷尽 |
| --- | --- | --- |
| `geometric` | 运动学原语模型下**是否存在**目标可见的可达位姿（量化位姿 BFS） | 是（到离散化为止） |
| `greedy` | 现有参考：每步对所有候选做一次物理 rollout 再选 | 否 |
| `search` | 对**动作序列**做 beam search，物理执行，启发式**刻意不看可见性几何** | 较强 |

beam 的启发式只用「走过多少个不同格子」加距离 tie-break，**故意不看几何可见性**：
否则它会继承 H3 要研究的几何/物理差距，就不是好参考了。

**结果：24 / 24 可解，0 个不可解。**

| 类别 | 数量 | 地形 |
| --- | ---: | --- |
| `solvable_greedy` | 20 | 其余全部 |
| `solvable_by_search` | **4** | `trench`、`pit`、`hist026_left`、`hist026_right` |
| `unsolvable_within_budget` | **0** | — |

四个需要 beam 的地形由很短的序列解决：`forward,forward`；`back,back`；`right,right`；
`toward,left_hold,back`。贪心在第一步看不到任何成功候选就退回 `forward`，再也出不来。

**三条结论：**

1. **`RSR on solvable` ≡ `RSR`**，分母不需要修正；审计的结论正是「没有不可解子集」。
2. **我们之前「4/24 连 oracle 都解不了」的判断是错的**——那是**参考规划器的缺陷**，
   不是地形的性质。**把 planner 的失败误当成 benchmark 的性质**，这件事本身值得写进论文。
3. **上界是 1.00 而不是 0.88。** 另外注意 `OracleSearch` 的 `RSR@1`（0.75）**低于**贪心（0.83）：
   贪心在第一步取最快的成功动作，而 beam 只保证**可解**、不保证**步数最优**。
   所以两者都不是紧的**步数**上界，论文里不能拿任何一个冒充最优。

**Conclusion — 三个要点：**

1. **协议截断是真的但不足以解释失败。** 多步把 `RSR@1 0.17` 抬到 `RSR@3 0.25`；
   然而在真正有区分力的 16 个成对世界上，**给到 10 步预算仍是 0/16**。
2. **闭环相对开环几乎没有增益（protocol A vs B）**：只有 `NBV` 从 0.25 升到 0.29
   （代价 2.2× 步数、2.2× 时间、1.7× 路径）。其余五个策略**每一步重看与只看一次结果完全一样**
   —— 它们实际上是开环规则：**给更多观测没用，因为它们不会用**。
3. **失败全是 B 类**（belief 正确、动作错），**一个 A 类都没有**。

> Evidence: `benchmark_multistep_B.json`, `benchmark_multistep_A.json`, `episode.py`

---

## E3 — How much history is necessary?

> ⚠️ **当前结果是单步协议下测的**，因此它回答的是「历史能决定**第一步**吗」，
> 而不是「历史如何支撑**整个 re-observation 过程**」。多步版待重做（P0）。

**Setup.** $H_k = \{o_{t-k+1}, \dots, o_t\}$，观测 10 Hz。机器人**不被告知目标世界坐标**，
只能从窗口内某次「确实看到目标」的观测里反推。**特征维度对每个 horizon 恒定（36 维）**，
只有窗口回溯多远在变，因此比较是公平的。只保留「此刻目标不可见」的样本（避免标签泄漏），
共 2481 样本 / 16 组 / 24 个决策状态。

**Result.**

| horizon | H0 | H1 | H2 | **H4** | H8 | H16 | H32 | H64 | Hfull |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 决策 RSR | 0.500 | 0.500 | 0.458 | **0.833** | 0.792 | 0.833 | 0.750 | 0.792 | 0.792 |
| 预测 AUC | 0.816 | 0.816 | 0.832 | 0.835 | 0.847 | 0.846 | 0.896 | **0.925** | 0.927 |

天花板（真实目标位置 + 零历史）= **0.833**。目击间隔：中位数 48 步（4.8 s）、最大 84 步（8.4 s）。

**Conclusion.** 决策在 **H4（0.4 s）就达到天花板**并再也不涨；预测 AUC 在 **H64（6.4 s）饱和**，
完整轨迹只多 **0.002**。所需 horizon 由「目标最后一次被看到有多久」决定。

**一个反直觉的分解**：目标记忆单独要 H64 才有用，而「最近运动摘要」在 H4 就到 0.79–0.83
—— 低 horizon 起作用的是**机器人从哪个方向走过来**，不是显式的目标位置。

**限制（必须写）**：决策指标只有 24 个世界，分辨率 1/24；H4 之后的排序在噪声内。
可靠的只有 H0→H4 的跳变与 AUC 的饱和点。

> Evidence: `history_horizon.json`, `history_horizon.py`, `figure_horizon.py`

---

### E3b — H2: is a correct target memory sufficient? (F11/F12)

**Setup.** 6 memory pairs. Each pair shares the whole **visible** terrain, the same
target at the same place, and the same walk-in, so $O_A = O_B$ *and* $M_A = M_B$ by
construction. They differ only in terrain **hidden inside the shadow of a visible
occluder**, which was never traversed and cannot be seen from the decision point.

**Result (measured, not assumed).**

| claim | result |
| --- | ---: |
| decision-point renders are pixel-identical | **6 / 6** (max pixel diff = 0) |
| target memory is identical | **6 / 6** |
| optimal action differs | **4 / 6** |
| successful action sets are disjoint | **0 / 6** |
| some policy fails in A but succeeds in B | **0 / 6** |

A representative pair (`mem001`): world A wins with `[right, right_hold]`, optimal
`right`; world B wins with `[left, right, left_hold, right_hold]`, optimal `left`.

**Conclusion, stated precisely — this is where overclaiming is easy:**

1. **Holds (information level, strong).** $O_A=O_B$ and $M_A=M_B$, yet
   $a_A^\* \neq a_B^\*$. **A correct target memory does not determine the optimal
   action**; the optimum depends on terrain behind the occluder that is invisible now
   and was never observed.
2. **Does not hold (behavioural level).** The successful sets **overlap**: `right_hold`
   works in both worlds. So H2 does **not** show that no fixed rule can succeed. This is
   the key difference from H1:

   | | $O_A=O_B$ | $M_A=M_B$ | disjoint winning sets | $a_A^\*\neq a_B^\*$ |
   | --- | :---: | :---: | :---: | :---: |
   | **H1** (E1) | yes | **no** | **yes** | yes |
   | **H2** (E3b) | yes | **yes** | no | yes |

3. **H2 does not separate the baselines behaviourally.** On these pairs all six classic
   policies choose `toward` and fail in **both** worlds (they drive at the remembered
   target and hit the wall). H2 is therefore a pure information-theoretic statement,
   and it must not be presented as "existing methods fail here".

**Two construction failures worth recording**, because both produced pairs that looked
fine geometrically and proved nothing:

* the first wall was too wide for one 1.8 m strafe to clear, so *both* worlds were
  unsolvable;
* with a symmetric wall the two detours tie and the physical gait asymmetry decides,
  leaving the optimal action identical. Making the +y detour strictly better requires a
  **visible** lateral asymmetry (the wall offset in y), which in turn forces the hidden
  feature further back, where the shadow magnifies enough to hide it *and* lie across
  the detour's sight line.

> Evidence: `memory_pairs.json`, `memory_sufficiency.json`, `memory_pairs.py`,
> `memory_sufficiency.py`.

### E3c — H3: does geometric visibility predict physical re-observability? (F13/F14)

**This is the experiment that decides whether a World Model is needed at all.**

**Setup.** For every candidate action at every decision state, two quantities:

* $G(s,a)$ — would the target be visible if the robot reached the **ideal** pose this
  action nominally produces (kinematic displacement + line-of-sight oracle)? This is
  what an active-perception / next-best-view method assumes it can evaluate.
* $P(s,a)$ — is the target actually re-observed when the primitive is **executed**?

24 terrains x 10 actions = **240 pairs**.  We also record the pose error between the
predicted and executed end pose, so the gap is attributed rather than asserted.

**Result.**

| quantity | value |
| --- | ---: |
| confusion | `G0P0`=171, `G0P1`=11, **`G1P0`=25**, `G1P1`=33 |
| agreement rate | 0.85 |
| $G=1$ and $P=0$ | **10.4 %** of all pairs |
| **share of geometric positives that are wrong** | **43.1 %** (25 of 58) |
| $G=0$ and $P=1$ | 4.6 % (geometry under-predicts too; not a one-sided bias) |
| **oracle at the *executed* pose agrees with $P$** | **97.9 %** |
| pose error, gap rows vs agreeing rows | **0.89 m vs 0.58 m** |
| pose error breakdown (gap rows) | forward 0.52 m, lateral 0.34 m, **height 0.40 m**, yaw 5.5 deg |

**Conclusion.**

> **43 % of the actions geometry endorses fail physically.** Putting the same oracle at
> the **executed** pose instead makes it agree with the physical outcome **97.9 %** of
> the time. The discrepancy therefore lies **not in the visibility model but in the pose
> prediction**: an action's nominal displacement differs from where the robot actually
> ends up by 0.89 m on average.

This rules out geometric viewpoint quality as an action-value proxy *and* identifies what
is missing: not a better visibility test, but a model of **where an action takes you and
what is visible from there**. GAP 3 in the paper's framing is therefore confirmed
quantitatively.

One detail matters for the method direction: **height error is 0.40 m**, which is gait
bob and step climbing - *systematic*, not noise. A learned model is therefore not the
only candidate; part of this gap is analytically modellable. That is a question for E4,
not an assumption to make now.

> Evidence: `geometry_gap.json` (240 rows with `G`, `P`, `G_actual` and four pose errors).
>
> Bug fixed here: when a kinematic pose falls outside the terrain the slab oracle returns
> a $-10^9$ sentinel, which poisoned the pose-error statistics at the $10^7$ m scale.
> Off-terrain candidates are now $G=0$ and excluded from the error statistics.

### E4 — What information is actually missing? (F15–F19)

H1, H2 and H3 remove the simple explanations one at a time. E4 asks what is left,
in four parts sharing one dataset and one protocol: **24 terrains x 10 states x 10
actions = 2240** `(state, action)` pairs, every one **physically executed**,
positive rate 0.143. Splits are **by terrain**, never random - a terrain family must
not appear on both sides.

```powershell
python tasks/task3_reobservability/consequence_dataset.py --states 10
python tasks/task3_reobservability/consequence_model.py --planner
```

#### E4-1 Information ablation (leave-one-terrain-out)

| inputs | ROC-AUC | PR-AUC | Brier | ECE | pose MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| $O$ | 0.759 | 0.360 | 0.117 | 0.076 | 0.436 |
| $M$ | 0.852 | 0.639 | 0.077 | 0.039 | 0.461 |
| $O+M$ | 0.854 | 0.600 | 0.084 | 0.056 | 0.425 |
| **$O+M+a$** | **0.934** | **0.748** | **0.062** | **0.022** | **0.148** |

> The key contrast is $O+M$ vs $O+M+a$: **+0.080 AUC**. Action conditioning adds
> information that is not already there. And **$O+M \approx M$** (0.854 vs 0.852):
> adding the observation to the memory buys essentially nothing. The information is
> in the *action*, not in seeing more.

Paired-world accuracy is also reported because a good average AUC can hide the very
ambiguity H1/H2 construct: on the 16 paired-world decision states the $O+M+a$ model
scores **0.8125**, and it **switches its chosen action in 8 of 8 pairs** - the thing a
memoryless rule provably cannot do.

#### E4-2 Analytic vs learned pose consequence

H3 located the gap: the oracle at the executed pose agrees with physics 97.9 % of the
time, so what fails is $a \rightarrow S$. E4-2 therefore predicts the executed
**body-frame pose delta** rather than the visibility directly.

| component | analytic (nominal) | learned | ratio |
| --- | ---: | ---: | ---: |
| forward | 0.829 | **0.152** | 5.5x |
| lateral | 0.418 | **0.145** | 2.9x |
| **height** | **0.090** | 0.149 | analytic wins |
| yaw | 0.538 | **0.147** | 3.7x |
| **mean** | **0.469** | **0.148** | **3.2x** |

> The learned model wins decisively on the components that decide success, and
> **loses on height** - the analytic model predicts "no change", and height changes
> are small. This must not be written as "learning is simply better".

#### E4-3 Prediction to planning (same multi-step protocol, 24 terrains)

| method | RSR@1 | RSR@3 | RSR@5 | RSR@10 |
| --- | ---: | ---: | ---: | ---: |
| six classic baselines | 0.167 | 0.250 | 0.250 | **0.250** |
| **LearnedConsequence (E4)** | **0.708** | **0.708** | **0.792** | **0.792** |
| Oracle (privileged) | 0.833 | 0.875 | 0.875 | 0.875 |
| OracleSearch (privileged) | 0.750 | 1.000 | 1.000 | 1.000 |

> **RSR@10 rises from 0.250 to 0.792 (3.2x)** and RSR@1 from 0.167 to 0.708, with no
> privileged information: only $O$, $M$ and the candidate actions.

**The failure mode changed, and that matters more than the success rate.** The learned
planner fails with `out_of_bounds` 2, `fallen` 2, `budget_exhausted` 1. It has learned
*where an action takes it and what is visible from there*; it has **not** learned to
predict **losing stability or leaving the terrain**. That is a different failure class,
and it is a *consequence-prediction* gap, not a candidate-generation gap. On this
evidence the natural next module is a better action-conditioned consequence model -
**not** a VLM.

#### E4-4 History ablation

| $H$ | AUC | Brier | pose MAE |
| --- | ---: | ---: | ---: |
| H0 | 0.930 | 0.063 | 0.177 |
| H1 | 0.929 | 0.061 | 0.165 |
| H2 | 0.926 | 0.063 | 0.157 |
| **H4** | 0.934 | 0.063 | **0.147** |
| H8 | 0.934 | 0.062 | 0.148 |

> Visibility AUC is **flat at ~0.93 across every horizon** - history does not help
> decide *whether* the target can be seen. But pose MAE improves from 0.177 to 0.147
> and **saturates at H4** - history helps predict *where you will end up*. The two
> targets diverge exactly as H3 predicted: the missing piece is physical consequence,
> not perceptual history. And only a short motion window is needed, not a long one.

> Evidence: `consequence_dataset.json`, `consequence_model.json`.
>
> Bug found and fixed while building this: `rollout()` restores the simulation state
> on exit (the API's guarantee), so reading the environment pose **after** a rollout
> returns the *start* pose. The executed end pose must come from `rollout.curve[-1]`.
> The first dataset had `delta_pose` identically zero for this reason.

### E4b - Is the safety consequence useful, and is it usable? (F22/F23)

E4-3's remaining failures are `out_of_bounds` and `fallen`, so the natural next
consequence is **safety**.

**Label design first.** A binary *fall* label is unusable here: **0 falls in 720
single-action rollouts**, so there are no positives to learn from. The heads target
`out_of_bounds` (from the executed end pose) and `upright_min` (minimum uprightness
along the action), both defined for every action.

| quantity | value |
| --- | ---: |
| positive rate | 0.0826 (185 / 2240) |
| ROC-AUC | **0.900** |
| PR-AUC | 0.606 |
| Brier / ECE | 0.051 / 0.042 |
| `upright_min` regression MAE | 0.0289 |

**Oracle ablation** (224 decision states, one-step success):

| planner | one-step success |
| --- | ---: |
| `Learned-V` (visibility only, = E4-3) | **0.3661** (82/224) |
| `Learned-V+S` (hard veto, lambda=1e3) | 0.3125 (70/224) |
| soft penalty lambda = 0.5 / 1 / 2 / 5 | 0.3571 / 0.3438 / 0.3482 / 0.3438 |
| `Oracle-V` (true visibility, no safety) | 0.3884 (87/224) |
| `Oracle-S` (true visibility + true safety) | **0.4107** (92/224) |

**A precise negative result, worth more than "add a safety head":**

> **The safety term is genuinely valuable** (oracle: 0.388 to **0.411**, +0.023),
> **but the learned safety head makes the planner worse at every penalty weight
> tried** (0.366 to 0.31-0.36), **including the soft ones**. The bottleneck is
> therefore **not a missing safety concept** - it is the head's **calibration at the
> decision boundary**. AUC 0.900 shows the *ranking* is good, and good ranking is
> **not enough** to use a quantity as a decision term.

This is the prediction-to-planning transfer failure: the information is present and
the planner still cannot use it. The correct next move is **calibration /
uncertainty-aware use of the head**, not more data and not a new modality.

### E4c - Is the safety information useless, or is the way we use it useless? (F24/F25)

E4b left a precise puzzle: the safety term is worth having (Oracle-S 0.411 > Oracle-V
0.388), the safety head is predictive (AUC 0.900), and yet the planner gets **worse** at
every penalty weight. Two explanations fit - the signal carries nothing usable, or a
probability is not a decision. E4c **fixes the prediction and varies only the rule**.

**Diagnosis: the miscalibration is concentrated exactly where a threshold acts.**

| variant | overall ECE | **ECE at the decision band** | mean p |
| --- | ---: | ---: | ---: |
| raw | 0.042 | **0.197** | 0.056 |
| **temperature** | **0.016** | **0.099** | 0.085 |
| isotonic | 0.028 | **0.220** | 0.081 |

The average looks healthy at 0.042 while the band |p - 0.5| <= 0.15 - the only region
where a threshold changes anything - sits at **0.197**, a factor of **4.7**. The diagnosis
is therefore right about *where* the problem is, and temperature scaling **halves** it
(0.197 to 0.099).

**And fixing it buys nothing.**

| variant | best point on the tau sweep | coverage there | beats no-safety (0.3661)? |
| --- | ---: | ---: | :---: |
| raw | 0.3661 | 0.004 | no |
| temperature | 0.3661 | 0.004 | no |
| isotonic | 0.3661 | 0.000 | no |

**No variant beats not using safety at any tau or any coverage**, and every variant's
optimum falls at coverage ~ 0 - i.e. "best not to use it". The most aggressive setting
(tau = 0.05, coverage 0.098) drops to **0.339-0.357**.

> **The binding constraint is not calibration.** Halving the decision-band ECE produced
> exactly zero planning gain. What limits this is the **joint visibility-safety decision
> boundary** - not whether the probability is accurate, but whether an accurate
> probability should be allowed to override the visibility ranking at all.

This is the branch in which a decision-aware fix still fails, and it is the more
informative outcome: it rules out "just calibrate it better" as the next step.
Calibrators are fitted **out-of-fold by terrain**; fitting a calibrator on the same
predictions it is scored on would manufacture the improvement it is meant to demonstrate.

> Evidence: `safety_calibration.json`, `paper/figures/fig5_safety_calibration.pdf`.

### E5 - Cross-mechanism generalisation

预测器已训（§7），**尚未接进 planner**。待补：

- [ ] `Learned` 注入冻结接口，在 E2 的同一条件下与全部 baseline 对比
- [ ] 报 `RSR@N`、$T_{\text{reobs}}$、`action cost`，并与 oracle 的**推理代价**对比
      （oracle 每决策点 $K$ 次物理 rollout，学习式一次前向）
- [ ] 目标：`Learned ≈ Oracle` **且** `Learned > existing baselines`

## E5 — Does the predictor generalise across terrain mechanisms? ⬜

对应 failure type E。三个 setting：

| setting | train | test | 验证什么 |
| --- | --- | --- | --- |
| A 实例泛化 | 同族 seed 1–8 | 同族 seed 9–12 | 是否只是记忆 |
| B 混合机制 | wall + trench + barrier | pit + waves | 能否跨实例泛化 |
| C 留一机制 | wall + trench + barrier + waves | pit（循环） | **最有说服力** |

现状（§7）：留一组交叉下 IID 1.00、OOD **0.50 < 朴素基线 0.75**。
**在 Setting C 完成之前，不能声称学到的是 re-observability 的原理而不是地形相关记忆。**

---

## 复现命令

```powershell
# E1
python tasks/task3_reobservability/history_pairs.py --pairs 8 --seed 11
python tasks/task3_reobservability/history.py --pair all --render-dir tasks/task3_reobservability/generated/history_camera

# E2
python tasks/task3_reobservability/test_api.py
python tasks/task3_reobservability/benchmark_api.py --protocol B --max-steps 10
python tasks/task3_reobservability/benchmark_api.py --protocol A --max-steps 10

# E3
python tasks/task3_reobservability/history_horizon.py

# E4/E5
python tasks/task3_reobservability/visibility_model.py
```
