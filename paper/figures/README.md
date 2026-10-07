# figures/ — 论文图

分两类：

- **统计图**（`fig1`–`fig4`）：由 `paper_figures.py` 从实验 JSON **直接生成**，
  带置信区间与检验，输出 **PDF（投稿用）+ PNG（查看用）**。数字不会与图脱节。
- **诊断图**（`paper/figures/` 之外的 `docs/images/task3_*.png`）：解释实验怎么跑的
  （接触表、多面板示意），用于补充材料或 rebuttal。

```powershell
python tasks/task3_reobservability/paper_figures.py
# 只重画一部分
python tasks/task3_reobservability/paper_figures.py --only benchmark geometry
```

## 统计规范（三条，全图统一）

1. **每个比例都带区间。** 二项比例用 **Wilson score interval**，不用正态近似——
   本文好几个结果贴在 0 % / 100 % 上，正态近似在那里会给出越界区间并低估不确定性。
2. **每个比较都是配对检验。** 所有策略跑在**同一批地形**上，所以差异用
   **对地形做有放回重采样**的配对 bootstrap，看配对差的分布。非配对检验会忽略
   "地形"这个高方差共享因子。
3. **矢量输出。** PDF 给投稿，PNG 给自己看；数字另外写 CSV 与 LaTeX，
   保证表格不会与图漂移。

## 图清单

| 文件 | 内容 | 数据来源 |
| --- | --- | --- |
| `fig1_benchmark.pdf` | (a) RSR@K vs 预算，Wilson 区间；(b) 按地形组拆分的 dot plot + 区间 | `benchmark_multistep_B_scored.json` |
| `fig2_paired_worlds.pdf` | H1 与 H2 的判据达成率（带区间与计数） | `history_necessity.json`, `memory_sufficiency.json` |
| `fig3_geometry_gap.pdf` | (a) $G$ vs $P$ 混淆矩阵；(b) 位姿误差按 $G/P$ 分组 + bootstrap 检验 | `geometry_gap.json` |
| `fig4_history_horizon.pdf` | (a) 决策 RSR vs $H_k$；(b) OOF AUC vs $H_k$ | `history_horizon.json` |

## 关键读法

- **fig1(b) 是全文最重要的一张。** 在 8 个多地形场景上六个经典 baseline 都落在 0.75
  （NBV 0.625），而在 16 个**成对历史世界**上它们全部落在 **0**，只有
  `Oracle`（0.81）与 `OracleSearch`（1.00）离开 0。**区分力全部来自右半边。**
- **fig1(a) 的注释**：五条规则式 baseline（Reactive / HeadingLock / HistoryOnly /
  LostTargetSearch / ObjectPermanence）在每个预算上**完全重合**——这本身就是结论，
  所以图例只给一条并加了箭头说明，而不是叠五个看不见的图例项。
- **fig2 把 H1 与 H2 的区别画出来**：H1 三条判据全 8/8（含"可行动作集不相交"），
  H2 是观测 6/6、记忆 6/6、最优动作 4/6，但**不相交 0/6**。
  这直接说明 H2 是**信息层面**的结论，不是行为层面的。
- **fig3(a)** 橙色框是失败格：**几何认可但物理失败 25 例**。
  **(b)** 给出机制：该格的位姿预测误差显著高于成功格
  （$\Delta=+0.28$ m，95 % CI $[+0.11,+0.45]$，bootstrap $p=0.0005$），
  而**同一个 oracle 放到实际执行位姿上**与 $P$ 一致 97.9 %。

## 表格

| 文件 | 内容 |
| --- | --- |
| `table_benchmark.csv` / `.tex` | 每个策略的 RSR@1/3/5/10 与 RSR@10 的 95 % CI |
| `table_paired_tests.csv` | 每个策略 vs `NBV` / `HistoryOnly` 的配对 bootstrap（差值、区间、$p$） |

## 配对检验暴露的一条必须写进论文的结论

`table_paired_tests.csv` 不是走过场，它给出两个硬结论：

1. **五条规则式 baseline 在全部 24 个地形上结果是"逐个地形完全相同"的。**
   `Reactive` / `HeadingLock` / `LostTargetSearch` / `ObjectPermanence` 相对
   `HistoryOnly` 的配对差是 **0.0，CI [0, 0]，`identical = True`**。
   也就是说它们**不是"差不多"，而是成功/失败的地形集合完全一致**——
   在这种地形分布下，这四种"机制上不同"的方法**做出的是同一个决策**。
   配对 bootstrap 在这里没有方差可重采样，所以**不报 $p$ 值**（报了也没有意义），
   标记为 `identical`。
2. **`NBV` 相对它们的优势不显著。** `NBV vs HistoryOnly`：
   差 **+0.042**，95 % CI **[0.00, 0.125]**，$p=0.356$。
   区间含 0，**不能**声称 NBV 更好。只有两个特权 oracle 显著：
   `Oracle vs NBV` $+0.583$、`OracleSearch vs NBV` $+0.708$，均 $p<5\times10^{-5}$。

所以正确的表述是：**在 24 个地形上，没有任何经典 baseline 显著优于任何其他经典 baseline；
只有特权参考显著优于全部经典方法。**

## 投稿前要做的

- [ ] 按期刊字号要求调 `figsize`（当前 7.2 in 双栏 / 3.6 in 单栏）
- [ ] 若要求 Times，`matplotlib.rcParams["font.family"] = "serif"` + `mathtext.fontset = "stix"`
- [ ] 检查颜色在灰度打印下是否仍可区分（当前 Okabe-Ito 调色板在灰度下部分重合，
      fig1(a) 已用线型区分，fig1(b) 需要改成不同 marker）
- [ ] fig2 的 `H1 no common action` 与 `H2 no common action` 需要正文明确解释差异
- [ ] 正文引用 `table_paired_tests.csv` 的两个结论时，必须同时给出 CI 与"不显著"的表述
