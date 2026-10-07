# Findings F1–F8（速查 + 证据指针）

> 这张表是论文的论点总账。**每一行都必须能指到证据文件或复现命令。**
> 状态列：✅ 已可写进论文 / 🟡 可写但要标注限制 / ⬜ 未做，不能写。

| # | 发现（论文里的写法） | 实验 | 关键数字 | 状态 | 证据 |
| --- | --- | --- | --- | --- | --- |
| **F1** | Visibility is **non-monotonic** along a traverse | 沿路径采样 oracle | $N_{\mathrm{flip}} \in \{2,3,4\}$，8 个机制全部非单调；`basin` 的方向与 `stairs_up` **相反** | ✅ | `benchmark.json`, `figure_benchmark.py` |
| **F2** | **Current observation is insufficient** for the optimal re-observation action | 8 对成对世界 | 观测**逐像素相同**（mean diff 0.0, max 0, 100% 像素一致, 目标像素 0/0）；最优动作相反 8/8；**没有任何动作在两个世界都成功** 8/8 | ✅ | `history_necessity.json`, `history_pairs.json`, `history_camera/` |
| **F3** | **Target memory alone is insufficient** — knowing where ≠ knowing how | 16 个成对世界上的决策规则 | history-only（朝记忆目标走）**0%**；多头 baseline 全部 **0/16**，即使给 10 步预算；18 次失败全是 B 类（belief 正确、动作错） | ✅ | `benchmark_multistep_B.json` |
| **F4** | **Heading preservation ≠ target-relative spatial preservation** | IMU 朝向锁定 baseline | 16 个世界里 **0%**（10 步预算下仍 0/16） | ✅ | `benchmark_multistep_B.json` |
| **F5** | **Geometric viewpoint quality ≠ physical re-observability** | NBV（几何重投影，给了精确高度场） | 成对世界 **0/16**；唯一从闭环获益但代价最高（2.29 步 / 6.86 s / 3.33 m ≈ oracle 的 2.2×/2.2×/1.7×）；2 次 D 类失败 | ✅ | `benchmark_multistep_B.json`, Experiment 0 的 `climb` vs `climb_toward` |
| **F6** | History has **causal decision value**, not just correlation | 打乱记忆的对照 | 有记忆 1.00 → 打乱 0.4375 → 无记忆 0.50（成对世界上） | ✅ | `visibility_model.json` |
| **F7** | The learned predictor does **not** generalise across terrain mechanisms | 跨地形族留一组 | 成对世界 1.00，**未见地形族 0.50 < 朴素基线 0.75** | 🟡 | `visibility_model.json` |
| **F8** | Classic search methods must be evaluated in a **closed-loop budget**, and doing so **does not rescue them** | protocol A vs B | 多步把 `RSR@1 0.17` 抬到 `RSR@3 0.25`；但成对世界**仍然 0/16**；闭环相对开环只在 NBV 上有增益（0.25→0.29） | ✅ | `benchmark_multistep_{A,B}.json` |
| **F9** | **All 24 terrains are solvable**；the greedy oracle is an incomplete reference, so the upper bound is **1.00 not 0.88** | Solvability audit（3 口径） | `solvable_greedy` 20、`solvable_by_search` 4、**`unsolvable` 0**；`RSR on solvable ≡ RSR` | ✅ | `solvability.json`, `benchmark_multistep_{A,B}_scored.json` |
| **F10** | **A planner's failure is easily mistaken for a property of the benchmark** | Solvability audit | 我们曾判定 4/24 不可解；beam search 用 2–3 步序列全部解决（`forward,forward` / `back,back` / `right,right` / `toward,left_hold,back`）。是参考规划器不完备，不是地形 | ✅ | `solvability.json` |
| **F11** | **A correct target memory does not determine the optimal action** | H2 memory pairs | $O_A=O_B$（max pixel diff **0**，6/6）**且** $M_A=M_B$（6/6），但 $a_A^\*\neq a_B^\*$（**4/6**）。差的是**藏在可见遮挡体阴影里、当前不可见也从未走过**的地形 | ✅ | `memory_pairs.json`, `memory_sufficiency.json` |
| **F12** | H1 and H2 are **not the same claim**: H2 is information-level only | H1 vs H2 对照 | H1 可行动作集**不相交**（0/16 固定规则能成功）；H2 动作集**有交集**（`right_hold` 两边都行），只是**最优**不同；在 H2 上六个 baseline 全部选 `toward` 并在**两个世界都失败**，所以 H2 不区分策略行为 | ✅ | `memory_sufficiency.json` |
| **F13** | **Geometric visibility does not predict physical re-observability**, and the gap is **pose prediction, not the visibility model** | H3（240 个 $(s,a)$ 对） | **$G=1$ 的动作里 43.1 % 物理失败**（58 个几何认可 → 25 个失败）；$G=0$ 但 $P=1$ 4.6 %；**同一个 oracle 放到实际执行位姿上，与 $P$ 一致 97.9 %** | ✅ | `geometry_gap.json` |
| **F14** | The pose error is **large and systematic**, so this is modellable | H3 误差分解 | 标称位移 vs 实际执行位姿平均差 **0.89 m**（gap 行）vs 0.58 m（一致行）；分解：前向 0.52 m、横向 0.34 m、**高度 0.40 m**、偏航 5.5° | ✅ | `geometry_gap.json` |
| **F15** | **Action conditioning adds real information; more observation does not** | E4-1 信息消融 | $O$=0.759, $M$=0.852, $O{+}M$=0.854, **$O{+}M{+}a$=0.934** AUC（留一地形，2240 样本）。**$O{+}M\approx M$**，$O{+}M{+}a$ 比 $O{+}M$ **+0.080** | ✅ | `consequence_model.json` |
| **F16** | **The learned pose model beats the analytic one 3.2×, but not on height** | E4-2 | 平均 MAE 0.469 → **0.148**；前向 5.5×、横向 2.9×、偏航 3.7×；**高度 analytic 0.090 反而更好（learned 0.149）** | ✅ | `consequence_model.json` |
| **F17** | **An action-conditioned planner nearly triples RSR without privilege** | E4-3 | RSR@10 **0.250 → 0.792**，RSR@1 **0.167 → 0.708**；只用 $O$、$M$、候选动作。**8/8 对成对世界里切换动作** | ✅ | `consequence_model.json` |
| **F18** | The learned planner's **remaining failures are a different class** | E4-3 终止统计 | 失败 = `out_of_bounds` 2、`fallen` 2、`budget_exhausted` 1。模型不会预测**失稳/越界**——这是**物理后果**的另一面，不是候选动作不够 | ✅ | `consequence_model.json` |
| **F19** | **History helps predict where you end up, not whether you see the target** | E4-4 | AUC 全程平坦 ~0.93（H0→H8）；位姿 MAE **0.177 → 0.147**，在 **H4 饱和** | ✅ | `consequence_model.json` |
| **F20** | **H1 is structurally unavailable in the y-symmetric mechanisms - a proof, not a failed search** | Mechanism Support Audit | `stairs_up`/`trench`/`pit`/`waves`/`basin` 的高度场 $\max\lvert h(x,y)-h(x,-y)\rvert$ = **0.0000 m**，单侧致胜集 **0**（27 次遮挡试验全是 `both`）。$y$ 对称无法表达方向性遮挡孔 | ✅ | `mechanism_audit.json` |
| **F21** | Only mechanisms with a **finite-y occluder** can carry H1 | Mechanism Support Audit | `barrier` 单侧率 **0.0952**（21 次遮挡试验：plus_y 1、minus_y 1），可镜像成对 ✓；`perlin` 虽有 0.2–0.3 m 的 $y$ 不对称，仍**全是 both** | ✅ | `mechanism_audit.json` |
| **F22** | **The safety head is learnable, and the safety *term* is valuable - but the learned version destroys value at every weight** | E4-5 oracle 消融 | `Oracle-S` **0.411** > `Oracle-V` 0.388（+0.023，安全项有价值）；但 `Learned-V` 0.366 → `Learned-V+S` **0.312**，软惩罚 $\lambda\in\{0.5,1,2,5\}$ 也全是 0.34–0.36。安全头 AUC 0.900 但**决策边界校准不够** | ✅ | `consequence_model.json` |
| **F23** | **A binary fall label carries no signal; use a continuous margin** | E4-5 标签设计 | 720 次单动作 rollout **0 次摔倒** → 二值标签无正例；改用 `out_of_bounds`（185/2240 = 8.3 % 正例）与 `upright_min`（MAE 0.029），两者处处有定义 | ✅ | `consequence_dataset.json` |
| **F24** | **The safety head is well calibrated on average and badly calibrated exactly where the threshold acts** | E4-5b 诊断 | 整体 ECE **0.042**，但**决策带内 ECE 0.197**（差 4.7×）。温度缩放把决策带 ECE 减半到 **0.099**；isotonic 反而更差（**0.220**）。所以"平均校准好"掩盖了决策边界上的失准 | ✅ | `safety_calibration.json` |
| **F25** | **Fixing the boundary calibration gives zero planning benefit - the limit is the decision rule, not the calibration** | E4-5b τ 扫描 | 三个变体在**任意** $\tau$ / 任意 coverage 下都**不超过**不用安全的 0.3661；每个变体的最优点都在 **coverage ≈ 0**（等于不用）。温度缩放把决策带 ECE 减半，规划收益 **0.000** | ✅ | `safety_calibration.json` |
| **F26** | **Re-observation is intrinsically sequential - a one-step model provably cannot solve it** | E4d | 特权参考需要多步的地形有 6 个（`pit`/`trench`/`waves`/…）。H=1 **解不了** `stairs_up` 与 `pit`，而 H≥2 **能稳定解掉** | ✅ | `horizon_planner.json` |
| **F27** | **But longer lookahead is a net loss: it rescues 2 terrains and breaks 8** | E4d | H2/H3/H5 相对 H1：成功率差 −0.208 / **0.000** / −0.125，**CI 全部含 0**，代价 **×2.09 / ×1.78 / ×1.87**。救回 `stairs_up`+`pit`，弄坏 `basin`/`hist001_right`/`hist013_left`/`hist017_left`/`hist026_right`/`hist036_left`/`hist040_left`/`hist040_right` | ✅ | `horizon_planner.json` |
| **F28** | **The chained score is monotonically optimistic in $H$, so lookahead adds ways to be wrong** | E4d 机制 | 链式分数 $\max_k \hat p_k$ 对 $H$ 单调乐观；一步位姿误差 **0.30–0.76 m** 与一个原语位移 **0.5–1.8 m** **同量级**，链到第 2–3 步预测已接近无信息 | ✅ | `horizon_planner.json` |

---

## 论文主线用哪三个

用户明确建议主线是 **F2 + F3 + F5**，因为它们构成一个完整的逻辑闭环：

$$
\underbrace{O_t \text{ insufficient}}_{\text{F2}}
\;+\;
\underbrace{M_t \text{ alone insufficient}}_{\text{F3}}
\;+\;
\underbrace{\text{geometric } V \neq \text{physical } V}_{\text{F5}}
\;\Longrightarrow\;
\boxed{\text{need } P(V_{t+H}\mid O_t, M_t, a) \text{ measured after execution}}
$$

F1 是问题的**前提**（否则问题不存在），F4 是 F2 的一个具体攻击面，F6 是历史因果性的
**证伪对照**，F7 是**限制**（必须写，不能吹），F8 是**协议公平性**的交代。

---

## 引用纪律

- F7 必须在正文里同时给出 IID（1.00）与 OOD（0.50）**以及朴素基线的 OOD（0.75）**，
  否则会被读成「能泛化」。
- F5 的 2.2× 代价是 **NBV 唯一继续获益**的证据，也是它被归为 D（低效）的依据，两个数字要一起给。
- 所有 oracle 数字标注为 **privileged upper bound / teacher**，不是可部署方法。
- 全部结果来自 **simulation**， locomotion 权重是第三方（MIT），未在真机验证。

---

## 与用户提供的状态表的差异（需要用户确认）

用户给的状态表里有几处与当前仓库不一致，我按**仓库实际数字**写论文：

| 用户表中写 | 仓库实际 | 说明 |
| --- | --- | --- |
| Shuffled memory 下降到 0.375 | **0.4375** | 标记点位置修正后重跑过全部实验 |
| OOD 0.50 | **0.50**，但朴素基线是 **0.75**（不是 0.60） | 基线数字也随重跑变了 |
| History Horizon ⏳ 下一实验 | **单步版已完成**（§8），多步版待做 | 单步版结论：H4 达天花板、AUC 在 H64 饱和 |
| Multi-step benchmark ⏳ | **已完成**（protocol A/B，24 地形 × 8 策略） | 见 F8 |
| Learned predictor ✅ 初版 | predictor **已训**，但**未接进 planner** | E4 只能写 preliminary |
| `perlin` 6 次翻转 | **4 次** | `_walkable` 别名 bug 修好后地形起伏变了 |
