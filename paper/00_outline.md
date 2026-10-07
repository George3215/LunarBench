> **2026-10-04 审计：以下为历史材料，不可作为当前论文证据表。** 主实验使用几何标签与真值目标记忆；Random 为 8 场景，Learned 为 24 场景，原跨集合比较无效。修订报告、补充相机实验及新稿见 [audit_20261004/AUDIT.md](audit_20261004/AUDIT.md)。

# 00 — 论文大纲与状态（working document）

## 稿件完成度（本轮更新）

| 章节 | 状态 | 说明 |
| --- | --- | --- |
| 01 Introduction | ✅ 已成稿 | 约 1400 words；`[CIT:]` 已在 §2 核实 |
| 02 Related Work | ✅ 已成稿 | 4 条线 + 定位；引用**经网络核实并给 URL**，未核实的卷号页码**刻意留空** |
| 03 Problem Formulation | ✅ 已成稿 | 含 RSR@K 严格定义 + 两个伴随指标 |
| 04 Benchmark | ✅ 已成稿 | 8 类遮挡机制 + 五轴 + Perlin 诚实声明 |
| 05 Baselines | ✅ 已成稿 | 冻结接口 + 统计规范 + 配对检验表 |
| 06 Failure Analysis | ✅ 已成稿 | 六类标签（含新增 `F_physical_termination`）+ 交叉表 + 可分性 |
| 07 Method | 🟡 架构骨架 | E4-3 已写入；**刻意不定稿**（安全项未解决） |
| 08 Experiments | ✅ 已成稿 | H1/H2/H3/E4/E4b |
| 09 Discussion | ✅ 已成稿 | 四个讨论结论 + L1–L16 限制 |
| 10 Conclusion | ✅ 已成稿 | Finding 1–6 + 由失败推出的未来工作 |

**仍待实验补的**：`E_ood`（跨机制留一）、完整后果模型的安全项、动作空间 sanity check。

## A 已完成：decision-aware calibration（E4-5b）

问题：**安全信息本身无效，还是"把概率直接当决策项"的方式无效？**
做法：固定预测、只变化用法——raw / temperature / isotonic × $\tau$ 扫描，
报 $RSR@1(\tau)$ 与 $Coverage(\tau)$，校准器**逐地形留一拟合**。

| 变体 | 整体 ECE | **决策带内 ECE** | 最优点 | coverage | 超过不用安全？ |
| --- | ---: | ---: | ---: | ---: | :---: |
| raw | 0.042 | **0.197** | 0.3661 | 0.004 | ✗ |
| **temperature** | **0.016** | **0.099** | 0.3661 | 0.004 | ✗ |
| isotonic | 0.028 | 0.220 | 0.3661 | 0.000 | ✗ |

> **诊断对了一半**：校准问题确实集中在决策边界（0.042 → 0.197，差 4.7×），
> 温度缩放也确实把它**减半**（0.099）。
> **但规划收益是 0.000**——任意 $\tau$、任意 coverage 都不超过不用安全的 0.3661，
> 且每个变体的最优点都在 **coverage ≈ 0**（等于不用）。

**结论：瓶颈不是校准，是可见性与安全性的联合决策边界。**
这正是预设的"仍然无改善"分支——**排除了"再把校准做好一点"这条路**。

**下一步（B）**：Action horizon ablation $H \in \{1,2,3,5\}$，
把 $P(V_{t+1}|a_t)$ 推广到 $P(V_{t+H}|a_{t:t+H})$。
理由：`trench: forward→forward`、`pit: back→back`、16 个历史世界都是多步问题，
所以 B 很可能解释 $RSR@1 \to RSR@10$ 之间的剩余 gap。

## 本轮的关键修正

1. **新增 failure class `F_physical_termination`。** 第一版把摔倒/越界记为
   `B_action_wrong`，这是**错误归因**：动作是基于合理预测选出的，错的是对**稳定性**的
   预测。修正后：经典 baseline 的失败几乎全是 `B`；`LearnedConsequence` 的 5 次失败里
   **4 次是 `F`**。**失败类型随方法改变**——这是"后果预测是缺失环节"的直接证据。
2. **RSR@K 的定义被强制检查。** `rsr_at` 直接从可见性轨迹求值，契约测试 `[10]` 断言
   它与 `success and steps_used<=K` 在 `waves`/`basin`/`stairs_up` × $K=1..6$ 上相等。
3. **地形改为 8 类遮挡机制 + 五轴**，措辞由 `mechanism_table.py` **机器强制检查**
   （禁用 "challenging/hard/difficult/adversarial terrain"）。
4. **统计口径**：配对 bootstrap + Wilson 区间；五条规则式 baseline **逐地形结果完全相同**
   （差 0.0，CI [0,0]，标记 `identical` 而非报 $p$）；`NBV vs HistoryOnly` **不显著**
   （+0.042，CI [0.00,0.125]，$p$=0.356）。

---

> 目标期刊专题（**由用户提供，我未独立核实**）：Frontiers Research Topic
> *Closed-Loop Robot Autonomy for Embodied Intelligence*
> <https://www.frontiersin.org/research-topics/82866/closed-loop-robot-autonomy-for-embodied-intelligence>
> 用户说明截稿 **2027-01-09**、仍在收稿。**投稿前请自行核对专题页面的日期与范围。**

---

## 一句话定位

> 机器人**曾经看到**目标，之后因为**地形造成的遮挡**而看不到；我们证明「靠近 ≠ 更可见」，
> 证明**当前观测、单独的目标记忆、几何视角质量三者都不足以决定下一步动作**，
> 并给出一个**动作条件化的物理可重观测性** benchmark 与预测器。

## 中心问题

> **When a previously observed target becomes occluded by terrain, what information
> and action reasoning are required for a robot to physically restore target
> visibility?**

因果链（就是方法部分的组织方式）：

$$a_{t:t+H} \rightarrow S_{t+H} \rightarrow V_{t+H}, \qquad \text{决策时只有 } (O_t, M_t)$$

## 状态：Phase 1 与 Phase 2 都已完成

| 因果问题 | 实验 | 结果 |
| --- | --- | --- |
| 当前视觉是否足够？ | **H1** | ✗ 像素相同、动作相反、可行集**不相交**，8/8 |
| target memory 是否足够？ | **H2** | ✗ $O_A{=}O_B$ **且** $M_A{=}M_B$，最优动作不同，4/6 |
| geometry 是否足够？ | **H3** | ✗ 几何认可的动作 **43 % 物理失败**；gap 全在位姿预测 |
| action 是否带来新信息？ | **E4-1** | ✓ $O{+}M{+}a$ **0.934** vs $O{+}M$ 0.854；且 $O{+}M\approx M$ |
| 需要学习吗？ | **E4-2** | 部分：学习 3.2× 优于解析，但**高度上解析更好** |
| 能否转化为任务收益？ | **E4-3** | ✓ RSR@10 **0.250 → 0.792**（非特权） |
| 需要时序动力学吗？ | **E4-4** | AUC 全程平坦；位姿 MAE 在 **H4 饱和** |

**E4-3 的失败模式是决定方法方向的关键**：剩余失败是 `out_of_bounds` / `fallen`——
模型学会了「动作把我带到哪里、从那里能否看到目标」，但**不会预测失稳/越界**。
这属于**后果预测**的另一面，**不是候选动作不够**。

$$\Rightarrow \text{目前证据指向 Route 1（action-conditioned consequence model），尚不支持 VLM}$$

所以下一步**不是 VLM**，而是把 E4-3 的失败模式补上：预测**终止/失稳**，而不只是位姿。


**不要写成**「我们提出了一个 World Model」——那太普通，而且 WoMAP / NavWAM 已经做过。

---

## 标题候选

1. **Closed-Loop Target Re-observability Planning under Terrain-Induced Partial Observability**（用户倾向）
2. Action-Conditioned Target Re-observability Planning in Terrain-Rich Environments

作者倾向 1。理由是它把「闭环」和「地形导致的非单调可观测性」都放进标题。

---

## 主线逻辑（论文的骨架）

```text
        Existing methods
   ┌───────────┼───────────┐
   ▼           ▼           ▼
Lost-target  Object      NBV
  search    permanence
   └───────────┼───────────┘
               ▼
   Terrain-induced setting
               ▼
   Non-monotonic observability      ← Finding F1
               ▼
      Three assumptions fail
   ┌───────────┼───────────┐
   ▼           ▼           ▼
current RGB  target      geometric
insufficient direction   viewpoint
             insufficient insufficient
   └───────────┼───────────┘   ← Findings F2 / F3 / F5
               ▼
  Need action-conditioned prediction
      of future physical re-observability
```

**论文主张的是「统一问题的交叉条件」，不是「第一次做某个组件」。**

> 现有工作分别研究 lost-target search、active perception / NBV、object permanence、
> world-model planning；我们把它们放进**同一个**问题
> （曾经看到的目标 + 地形遮挡 + 非单调可见性 + 物理动作），
> 发现**不同机制在不同 failure mode 下失效**。

用词纪律：**写 "a lack of unified analysis"，不要写 "nobody has studied"**。

---

## 章节状态表

| 章节 | 文件 | 状态 | 依赖 |
| --- | --- | --- | --- |
| 00 Outline | `00_outline.md` | ✅ 本文件 | — |
| 01 Introduction | `01_introduction.md` | ✅ 初稿（英文，约 950 词） | — |
| 02 Related Work | `02_related_work.md` | 🟡 骨架 + 定位段 | 需要补引用 |
| 03 Problem Formulation | `03_problem_formulation.md` | ✅ 初稿 | — |
| 04 Benchmark | `04_benchmark.md` | ✅ 初稿（数字已用当前代码复核） | — |
| 05 Unified Baselines | `05_baselines.md` | ✅ 初稿 | — |
| 06 Failure Analysis | `06_failure_analysis.md` | 🟡 骨架，F1–F5 已有证据 | Failure taxonomy 定量分析待做 |
| 07 Method | `07_method.md` | 🟡 骨架，初步结果可填 | Learned scorer 未接线 |
| 08 Experiments | `08_experiments.md` | 🟡 E1/E2/E3 有结果，E4/E5 待补 | 多步版 History Horizon |
| 09 Discussion / Limitations | `09_discussion.md` | 🟡 骨架，限制清单已完整 | — |
| 10 Conclusion | `10_conclusion.md` | ⬜ 空 | 等 E4/E5 |
| Findings 速查 | `findings.md` | ✅ F1–F8 + 证据指针 | — |

图与表：`figures/README.md`（统计图由 `paper_figures.py` 生成，PDF+PNG）、`tables/`。

**统计规范**（全图统一）：每个比例带 **Wilson 区间**；每个比较用**对地形重采样的配对
bootstrap**；矢量输出。数字同时写 CSV / LaTeX，避免表格与图脱节。

**配对检验已经给出两条必须写进论文的结论**（`tables/table_paired_tests.csv`）：

1. 五条规则式 baseline 在全部 24 个地形上**逐个地形结果完全相同**（配对差 0.0，CI [0,0]）。
   它们不是"差不多"，而是**成功/失败的地形集合一致**。
2. **没有任何经典 baseline 显著优于其他经典 baseline**：`NBV vs HistoryOnly` 差 +0.042，
   95 % CI **[0.00, 0.125]**，$p=0.356$。只有两个特权 oracle 显著（$p<5\times10^{-5}$）。

---

## 实验状态（与论文主张一一对应）

| 实验 | 回答什么 | 状态 | 能否进论文 |
| --- | --- | --- | --- |
| **E1** 当前观测是否充分 | $O_A=O_B$ 逐像素相同但 $a_A^*\neq a_B^*$ | ✅ 8 对，8/8 | **核心结果** |
| **E2** 现有机制能否解决 | 24 地形 × 8 策略，多步 protocol B | ✅ 已跑 | **核心结果**（failure 分析） |
| **E3** 需要多少历史 | H0/H1/H2/H4/H8/H16/H32/H64/Hfull | 🟡 单步版已跑，**多步版待重做** | 单步版可先写，注明协议 |
| **E4** 动作条件化预测是否有效 | Learned vs baselines | 🟡 predictor 已训，**未接进 planner** | 只能写 preliminary |
| **E5** 是否跨地形机制泛化 | Setting A/B/C | ⬜ 未做 | 只能写已知的 OOD 现象 |

**E1 与 E2 已足以支撑论文主体；E3 单步版可写但要标注协议；E4/E5 是「待补」位置。**

---

## 现在就能写的四块（用户指定今天完成）

1. **Introduction**（`01_introduction.md`）— 4 段，已写
2. **Problem Formulation**（`03_problem_formulation.md`）— $O_t, M_t, A, V, a^*$、RSR@K、$T_{\text{reobs}}$、$N_{\text{flip}}$ 已定义
3. **Benchmark**（`04_benchmark.md`）— 8 个 visibility mechanism、非单调表、成对世界
4. **History Necessity**（并入 `08_experiments.md` 的 E1，完整初稿在 `04_benchmark.md` §4.4 与 `findings.md` F2）

---

## 写作纪律（自用 checklist）

- [ ] 每个数字都能指到 `generated/*.json` 或某个 `figure_*.py`（见 `findings.md` 证据列）
- [ ] 不写 "first to"、"nobody has"、"novel" 这类无法证伪的说法
- [ ] 凡是用 simulator 的结论，明确写 simulator
- [ ] `Oracle` 一律标注为 privileged upper bound / teacher，不是可部署策略
- [ ] OOD 结果必须同时给出 IID 数字与朴素基线数字（否则会被读成「能泛化」）
- [ ] 多步协议的公平性（等时长原语、统一预算、统一终止）必须写进 Benchmark 章节
- [ ] 单步协议的旧数字不得单独引用（§9.6 已废弃）

---

## 已知需要补的实验（按优先级）

| 优先级 | 事项 | 为什么必须 |
| --- | --- | --- |
| ✅ | ~~Solvability audit~~ | 24/24 可解，分母无需修正；上界改为 1.00（F9/F10） |
| ✅ | ~~H2 记忆充分性~~ | $O_A=O_B$ 且 $M_A=M_B$ 但 $a_A^*\neq a_B^*$（F11/F12） |
| ✅ | ~~H3 几何 vs 物理~~ | 几何认可的 43% 物理失败；gap 全在位姿预测（F13/F14） |
| ✅ | ~~多步 RSR@K 协议~~ | protocol A/B，等时长原语，统一预算与终止（F8） |
| **P0** | **E4-1 信息消融** $O$ / $M$ / $O{+}M$ / $O{+}M{+}a$ | 判定「action-conditioned 是否带来新信息」。核心比较是 $O{+}M$ vs $O{+}M{+}a$ |
| **P0** | **E4-2 analytic vs learned 位姿后果** | H3 已定位缺口在 $a\to S$，所以预测 $\Delta$pose 而不是直接预测可见性 |
| **P0** | **E4-3 预测 → 规划**（RSR@1/3/5/10） | AUC 不是目标，能否转化为实际 re-observation 才是 |
| **P0** | **E4-4 历史消融** $H_0/H_1/H_2/H_4/H_8$ | H3 量到 0.40 m 高度误差与步态有关，单帧可能不知道在爬升还是已稳定 |
| P0 | H1 系统化：4–6 个机制 × 3–5 对 | **先做 Mechanism Support Audit**，见下方警告 |
| P1 | Learned scorer 接进 planner（E4） | 接口已留好；先做 M0/M1/M2/M3 四组消融：$O$ / $M$ / $O{+}M$ / $O{+}M{+}a$ |
| P1 | Cross-mechanism OOD（Setting A/B/C，leave-one-mechanism-out） | E5 + failure type E |
| P2 | 预测完整可见性轨迹 $\hat V^a_{t:t+H}$ 而非二值 | 从 binary 升级成真正的 counterfactual |
| P2 | 真机 / Sim2Real | 专题欢迎，但目前不是硬要求 |

**关于 H1 系统化必须先验证的一件事**：$O_A=O_B$ 要求两个世界在决策点**看起来一样**，
而 $a_A^*\neq a_B^*$ 要求可选动作在两侧**不对称**。y 方向不变的地形
（`trench`/`pit`/`basin`/`waves`）的揭示几何是 y 不变的，**可能根本无法构造 H1 对**。
所以先要测「哪些机制在原理上支持 H1」，而不是假设都能做——
否则会把「构造不出来」误写成「现象不普适」。

---

## 目标时间线（用户建议，我照此执行）

- **10 月**：论文主体（Intro / Problem / Benchmark / History Necessity / Baselines / Failure Analysis）
- **11 月**：补齐 E3 多步版、E4、E5，投稿
- 截稿 2027-01-09，**不必为赶 10 月牺牲实验完整性**
