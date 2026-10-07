# 02 — Related Work

> 状态：**已成稿**。引用条目经网络核实，均给出 URL；未核实的细节（页码、卷号）
> **刻意不写**，避免编造。
> 用词纪律：写 **a lack of unified analysis**，不写 *nobody has studied* / *first* / *novel*。

---

## 2.1 Lost-target search

A robot that has lost a target maintains a belief over where it might be and selects
search actions to reduce that uncertainty.

- Radmard & Croft, *Active target search for high dimensional robotic systems*,
  Autonomous Robots — <https://dl.acm.org/doi/10.1007/s10514-015-9539-8>
- Radmard, Meger, Law & Croft, *Resolving Occlusion in Active Visual Target Search
  of High-Dimensional Robotic Systems*, IEEE T-RO — <https://dl.acm.org/doi/10.1109/TRO.2018.2796577>
- *Where did it go? Regaining a lost target for robot visual servoing* —
  <https://www.semanticscholar.org/paper/53ebeac5cfc2e14b5cffb8712189f8195064300b>

**What this line answers.** Given a lost target and a belief over its location, which
search action best reduces uncertainty.

**What it does not answer.** What to do when the terrain makes the correct
re-observation action point **away** from the belief. That is exactly the situation in
our H1 construction: the belief is correct, the bearing to the target is correct, and
the action that restores visibility is on the *opposite* side. A belief-directed search
policy is not mis-implemented in that situation - it is answering a different question.
Our `LostTargetSearch` and `ObjectPermanence` baselines implement this line and score
$RSR@10 = 0.250$, and - the sharper result - they are **outcome-identical on every one
of the 24 terrains** (paired difference $0.0$, CI $[0,0]$).

## 2.2 Active perception and next-best-view

A second line predicts visibility or information gain at candidate viewpoints and picks
the best one.

- *OA-NBV: Occlusion-Aware Next-Best-View Planning for Human-Centered Active
  Perception on Mobile Robots* — <https://arxiv.org/abs/2603.11072>
- *LEAP-NBV: Lightweight Edge Active-Perception for Foundation-Model Next-Best-View
  Planning* — <https://arxiv.org/pdf/2609.23974>

**What this line answers.** Where to stand so that the target is visible.

**What it does not answer.** Whether the robot, after physically executing the action
that should take it there, is actually there. Candidate viewpoints in this line are
typically scored with kinematic poses or reprojection, which assumes $a \rightarrow S$
is exact. It is not: we measure that the executed pose differs from the nominal
displacement by **0.89 m** on average (forward 0.52, lateral 0.34, height 0.40,
yaw 5.5°), and that **43.1 %** of the actions geometry endorses fail physically. The
visibility model is not the culprit - the same oracle placed at the *executed* pose
agrees with the physical outcome **97.9 %** of the time.

Our `NBV` baseline implements this line with the exact height field available and scores
$0/16$ on the paired worlds; its advantage over `HistoryOnly` on the main suite is
$+0.042$ with a 95 % CI of $[0.00, 0.125]$ ($p = 0.356$), i.e. **not significant**.

## 2.3 Object permanence and target memory

A third line maintains a belief through non-observation and acts on it.

- *Object Permanence Filter for Robust Tracking with Interactive Robots*, IEEE —
  <https://ieeexplore.ieee.org/abstract/document/10611528/>

**What this line answers.** Where the target probably is while it cannot be seen.

**What it does not answer.** That knowing where the target is determines what to do.
This is the specific gap our H2 experiment isolates: we construct worlds where the
current observation is **pixel-identical** *and* the reconstructed target memory is
**identical**, yet the optimal re-observation action differs (4 of 6 pairs). A correct
memory constrains the target's location without constraining which side of a hidden
occluder is passable.

Our `ObjectPermanence` baseline keeps the constant-velocity filter structure of this
line; with a static target the extrapolation term is zero, so it **collapses onto the
same rule as `LostTargetSearch`** - measured, not assumed: identical per-terrain
outcomes.

**A scope note we state explicitly.** Our $M_t$ is a *structured* representation
(target position and age, reconstructed without noise), not an open-world semantic
memory. Our claims about memory are therefore claims about this representation.

## 2.4 World models for embodied decision making

A fourth line learns action-conditioned consequence predictors and plans with them.

- *WoMAP: World Models for Embodied Open-Vocabulary Object Localization*, CoRL 2025 —
  <https://mlanthology.org/corl/2025/yin2025corl-womap/>
- *Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in
  Robotics* — <https://arxiv.org/pdf/2501.10100>
- *Dreamwalker: Mental Planning for Continuous Vision-Language Navigation* —
  <https://arxiv.org/abs/2308.07498>
- *Predicting Consequences and Reinforcing Navigation Policies with Latent World
  Models*, ECCV — <https://eccv2024.ecva.net/virtual/2026/poster/4904>

**What this line answers.** How to learn a consequence model usable for planning, and
that such models are useful for embodied exploration.

**What our work takes from it, and what it adds.** We do not claim the combination of a
world model with a reasoning module as a contribution - WoMAP and related work already
demonstrate it. Our focus is narrower and physical: **which quantity** must be predicted
for the specific problem of regaining sight of a previously observed target under
terrain-induced occlusion. Our answer is that it is neither the geometric visibility nor
the current visibility, but the **action-conditioned visibility after execution**, and
we arrive at that answer by elimination rather than by assumption (H1, H2, H3).

We also report a caution that this line can use: **an accurate consequence head is not a
usable decision term.** Adding a learned safety head with ROC-AUC 0.900 made our planner
*worse* at every penalty weight we tried, while the same term with ground-truth labels
made it better. Ranking quality and decision-boundary calibration are different
requirements.

## 2.5 Positioning

Each of these four lines supplies a necessary component. We claim no component as
original. What this work does is put lost-target search, active perception /
next-best-view, object permanence, and world-model planning into **one** problem - a
previously observed target, terrain-induced occlusion, non-monotonic visibility, physical
action - under **one observation and action interface, one budget, and one termination
set**, and then measure **which mechanism fails in which failure mode**:

| line | what it supplies | measured failure in our setting |
| --- | --- | --- |
| lost-target search | belief-directed search | belief correct, action wrong: $RSR@10 = 0.250$, 0/16 paired |
| active perception / NBV | geometric viewpoint scoring | $G = 1, P = 0$ in 43.1 % of endorsed actions; NBV not significantly above `HistoryOnly` |
| object permanence | memory through non-observation | memory identical, optimal action differs (H2) |
| world models | action-conditioned prediction | **+0.080 AUC** over $O{+}M$; $RSR@10$ 0.250 $\rightarrow$ 0.792 |

The literature contains **a lack of unified analysis** across these conditions: each
line is evaluated on its own setting, with its own interface and budget, so the
boundaries between them are not visible. The contribution of this paper is to measure
those boundaries - and, in doing so, to identify that the shared missing quantity is
$P(V_{t+H} \mid O_t, M_t, a_{t:t+H})$.

> **Citation status.** Each reference above was located by web search and is given with
> the URL actually retrieved. Bibliographic details that were not retrieved (volume,
> pages, author lists) are deliberately omitted rather than reconstructed, and must be
> completed from the publisher record before submission.
