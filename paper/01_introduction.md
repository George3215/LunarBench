# 01 — Introduction

> 状态：**已成稿**（约 1400 words）。`[CIT: ...]` 处对应 §2 的已核实引用；
> 投稿前需把 `[CIT: ...]` 替换为正式引用条目。

---

Robots that operate in natural terrain rarely enjoy a clear view of what they care
about. A legged platform descending into a crater, a rover traversing a ridge line, or
a manipulator reaching behind a boulder all face the same structural problem: the
terrain itself determines what can be seen from where, and the robot's own motion is
what changes it. **Partial observability in these settings is not sensor noise.** It is
a deterministic consequence of geometry, and it is therefore predictable in principle —
but only if the robot reasons about how its next action will change what it can see.

This paper studies a specific and, we argue, under-analysed instance of that problem.
The robot has already observed a target. At some later time the target is no longer
visible — not because it moved or because the sensor failed, but because the robot's
own position and the intervening terrain now block the line of sight. The robot must
choose an action that makes the target visible again. We call this **target
re-observability**, and it differs from the classical search problem in one decisive
way: *the robot has information about where the target is, and that information is
still not enough.*

The natural intuition is that moving toward a target makes it more likely to be seen.
Terrain breaks that intuition. Consider a robot walking a straight line toward a target
placed beyond a ridge. Measured along the traverse, the visibility sequence is not a
single step from visible to hidden; it alternates. We measure, on eight procedurally
generated terrains, profiles such as

```
stairs_up     V X V X          waves        X V X V X
trench        V X V X          perlin       X V X V X
pit           V X V X          basin        X V X
barrier       V X V X          perlin_rough X V X V X
```

where `V` marks a position from which the target is visible and `X` one from which it
is blocked. The number of transitions ranges from two to four on a single traverse.
Two consequences follow. First, a policy that assumes monotonicity — "get closer" — is
not merely imprecise; it is aimed at the wrong objective. Second, the moment of loss is
not the moment of failure: a robot may lose the target and then re-acquire it purely as
a side effect of continuing forward. We formalise this as the number of visibility
transitions $N_{\mathrm{flip}}$ along a path and use it as a difficulty measure for the
benchmark.

Existing lines of work each supply a necessary component but address a different
question. **Lost-target search** [CIT: Radmard & Croft, Auton. Robots; Radmard et al.,
T-RO 2018] maintains a belief over where the target went and selects search actions to
reduce it; the mechanism is a belief and a search policy. **Occlusion-aware active
perception and next-best-view** [CIT: OA-NBV; LEAP-NBV] select viewpoints by predicting
visibility or information gain from a candidate pose, typically using geometry rather
than execution. **Object permanence** filters [CIT: Object Permanence Filter, IEEE 2024]
maintain a belief through periods of non-observation and act on the filtered estimate.
**World-model planning** for embodied agents [CIT: WoMAP, CoRL 2025; Dreamwalker; Robotic
World Model] learns to predict the consequences of actions and plans with them. Each is a
reasonable answer to its own question. What the literature lacks is a **unified
analysis**: these lines are evaluated in their own settings, with their own observation
and action interfaces and their own budgets, so the boundaries between them are never
visible. We place them on a single task — terrain fixed, target static, action set
identical, only the decision rule differing — and ask *where and why each one breaks*.

We do that here, and the answer is uncomfortable for all of them. We construct **paired
worlds**: two scenes that share terrain, robot pose, heading and — we verify this by
rendering the injected camera on the actual robot model — a **pixel-identical current
observation**, but in which the target was last seen from opposite sides. Across eight
such pairs, the optimal re-observation action is opposite in the two worlds, and there
is **no action that succeeds in both**. Since any function of the current observation is
identical in the two worlds, no memoryless policy can solve both. Target memory is not a
refinement of the current observation; it is *decision-relevant information that the
current observation provably does not contain*.

Knowing where the target is, however, is also not sufficient. Two of the baselines we
evaluate hold a correct belief about the target's position and still fail: the rule that
aims at the remembered target drives straight into the occluder, and the
heading-preservation rule holds a heading that was correct when the target was lost but
is wrong after any lateral displacement. Across 16 paired worlds and five classic
decision rules, evaluated under a multi-step budget with a shared observation and action
interface, **every classic rule scores 0/16** while a privileged counterfactual oracle
scores 13/16. Every failure is of one kind: the belief was right and the action was
wrong.

A third assumption also fails, and it is the one most likely to be silently wrong in
practice. Next-best-view style methods predict the benefit of an action from a
*geometric* re-projection of the believed target into a hypothetical camera pose. We
implement that baseline with the strongest reasonable map — the exact height field — and
it fails in exactly the way our earlier measurement predicted: on this robot the
geometrically endorsed action and the physically successful action disagree, because
executing a primitive involves a controller, tracking error, lateral drift and a
duration, none of which the re-projection models. We quantify this directly over
$240$ (state, action) pairs: **43.1 % of the actions geometry endorses fail physically**.
Crucially, the fault is not in the visibility model — placing the *same* oracle at the
pose the robot **actually reaches** agrees with the physical outcome **97.9 %** of the
time. What fails is the step before it: predicting where an action takes the robot.
**Geometric viewpoint quality is not physical re-observability**, and since the
discrepancy is a property of the closed loop rather than of the sensor, better perception
cannot remove it.

Taken together, the three failures identify what a method for this problem must actually
predict. Not the current visibility ($O_t$ is insufficient), not the target direction
($M_t$ alone is insufficient), and not a geometric viewpoint score. It must predict the
**action-conditioned, physically-executed future observability**:
$P(V_{t+H} = 1 \mid O_t, M_t, a)$, where $V$ is measured after the action has actually
been carried out.

We then test that conclusion positively rather than leaving it as a deduction. On
$2240$ physically executed (state, action) pairs, adding the candidate action to
observation-plus-memory raises the re-observation AUC from $0.854$ to $0.934$
(**+0.080**), while adding the current observation *to* the memory buys almost nothing
($0.852 \rightarrow 0.854$). **The information is in the action, not in seeing more.**
A planner built on the resulting consequence model — prediction of the executed pose
followed by the already-validated visibility oracle — raises $RSR@10$ from $0.250$ to
**$0.792$** and $RSR@1$ from $0.167$ to **$0.708$**, with no privileged information, and
it is the only method we evaluate that **chooses different actions in the two members of
every paired world**. We also report where it still fails, because that failure is
informative: its remaining errors are no longer action-selection errors but
**physical-termination** errors (leaving the terrain, losing stability), i.e. the same
missing-consequence gap one step further on. And we report a negative result that
constrains how such a model may be used: adding a learned safety head with ROC-AUC
$0.900$ made the planner *worse* at every penalty weight we tried, while the same term
with ground-truth labels made it better. An accurate consequence predictor is not
automatically a usable decision term.

This paper contributes:

1. **A formulation** of terrain-induced target re-observability as an action-selection
   problem, with the non-monotonicity of $V$ along a traverse as its defining property,
   and an evaluation metric ($RSR@K$) whose implementation is audited against its own
   definition rather than assumed to match it (§3).
2. **A procedurally generated benchmark** of eight *terrain-induced occlusion
   mechanisms*, organised by physical mechanism and real-world analogue rather than as a
   difficulty ladder, built as exact box geometry so that the analytic visibility oracle
   and the MuJoCo collision surface are the same object; together with (i) a set of
   **paired worlds** that differ in history and are provably identical in the current
   observation, (ii) a second set of pairs that are additionally identical in target
   memory, and (iii) a **mechanism-support audit** that proves which mechanisms can
   express such an ambiguity and which provably cannot (§4).
3. **A frozen observation/action interface** under which every decision rule is evaluated
   on the same terrain, with the same budget, the same termination set and a
   policy-agnostic terrain contract, with all comparisons reported as paired bootstrap
   intervals rather than point estimates (§5).
4. **A failure analysis** that separates the three assumptions, quantifies each, shows
   they are separable rather than three names for one failure, and identifies a failure
   class (`F_physical_termination`) that the first version of our own taxonomy got wrong
   (§6, §8).
5. **A measured answer** to what the missing quantity is: an ablation showing
   $a$ contributes far more than $O$, a consequence model that beats the analytic
   kinematic model $3.2\times$ on mean pose error, and a closed-loop planner that
   improves $RSR@10$ $3.2\times$ without privileged information (§8).

We are explicit about scope. All results are in simulation with a vendored third-party
locomotion policy, on one controller; the oracle is an upper bound and not a deployable
method; the learned consequence model is evaluated with **terrain held out**, which is
generalisation *within* a mechanism family and not across mechanisms; and our
$M_t$ is a structured target-location memory rather than an open-vocabulary semantic one.
The contribution we claim is the formulation, the benchmark, the failure analysis and the
identified quantity — not a world model, and not the combination of a world model with
a language model.
