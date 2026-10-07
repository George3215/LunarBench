# 03 — Problem Formulation

> 状态：**初稿，可直接进论文**。数字与记号已与代码对齐
> （`api.py` / `episode.py` / `benchmark.py`）。

---

## 3.1 Setting

A robot moves on a known terrain surface. A **target** is a static point
$T \in \mathbb{R}^3$ that the robot has observed at least once in the past. At the
current time the target may or may not be visible.

We write visibility as a deterministic function of the robot's state and the terrain:

$$
V(s) \;=\; \mathbb{1}\!\left[\;\text{the line segment from the camera at } s
\text{ to } T \text{ is unobstructed and inside the frustum}\;\right]
$$

$V$ is *not* a function of the target's position alone, and *not* a function of the
robot–target distance. This is the property the paper is about.

## 3.2 Non-monotonicity of visibility

Let $s_0, s_1, \dots, s_n$ be the states along a path. Define the visible/lost sequence
$V(s_0), \dots, V(s_n)$ and the number of transitions

$$
N_{\mathrm{flip}}(s_{0:n}) \;=\; \sum_{i=1}^{n} \mathbb{1}\!\left[V(s_i) \neq V(s_{i-1})\right]
$$

**H1 (non-monotonicity).** There exist terrains and paths for which
$N_{\mathrm{flip}} \geq 2$, i.e. the target is lost and re-acquired along a single
traverse, and for which $V$ is not monotone in the robot–target distance.

Empirically (§4.3) $N_{\mathrm{flip}} \in \{2,3,4\}$ on eight terrain mechanisms.

## 3.3 Observations, memory and the action set

At time $t$ the robot has an **observation**

$$
O_t \;=\; \bigl(\underbrace{\text{RGB}_t,\ \text{depth}_t}_{\text{exteroceptive}},\
\underbrace{\text{proprioception}_t}_{\text{joint } q,\dot q,\ v_b,\ g},\
\underbrace{\text{action history}_{t}}_{\text{including the approach}}\bigr)
$$

and a **target memory**

$$
M_t \;=\; \bigl(\hat T_t,\ \text{age}_t,\ \text{range}_t\bigr)
\quad\text{or}\quad \bot
$$

$M_t$ is the robot's own belief, reconstructed only from observations in which the
target actually appeared. **It is not a privileged quantity**: in our implementation it
is recovered from a sighting as $\hat T = \text{eye} + r\,u(\psi + \beta, \varepsilon)$
using the measured range $r$, bearing $\beta$ and elevation $\varepsilon$. Where $M_t$
is absent we write $M_t = \bot$ and require the policy to behave sensibly; a policy is
never handed the true $T$.

The robot has a fixed, finite **action set**

$$
\mathcal{A} = \{a_1, \dots, a_K\},\qquad K = 10
$$

of parameterised primitives (drive forward/back, strafe left/right, turn left/right,
each with and without heading hold, a memory-aimed primitive, and a heading-lock
primitive). Every action costs the same simulated time $\Delta$ in the multi-step
protocol (§4.5); the distance and turn it produces are recorded rather than equalised.

A **policy** is a map

$$
\pi:\ (O_t, M_t) \;\longmapsto\; a \in \mathcal{A}
$$

with no access to $T$, $V$, the terrain height field, or the outcome of $a$ — except for
the oracle baseline, which is privileged and labelled as such.

## 3.4 The evaluation metric: RSR@K

The benchmark reports a *difficulty curve*, not a single success flag, because the
quantity of interest is **how many physical actions are needed to regain sight of
the target**, not merely whether the episode ended well. For episode $i$ and budget
$K$:

$$r_i(K) = \mathbf{1}\!\left[\exists\, h \le K : V_{t+h} = 1\right], \qquad
RSR@K = \frac{1}{N}\sum_{i=1}^{N} r_i(K)$$

Four policies may all eventually succeed while being very different problems:

```
A: right → visible
B: right → right → visible
C: right → left → back → visible
D: right → right → left → right → …  (beyond the budget)
```

$RSR@1$ asks "can the target be regained in one action", $RSR@3$ "within three".
Note what the definition deliberately does **not** require: visibility to be
*held*. Visibility under these mechanisms is non-monotonic
($1\to0\to1\to0\to1$ under `waves`), and the task is to **regain** sight within a
budget, not to keep it.

**The implementation is audited against this definition rather than assumed to
match it.** `rsr_at` evaluates the definition directly on the recorded per-step
visibility trace, while an episode also terminates at its first re-observation, so
`success and steps_used <= K` gives a second reading. Contract test `[10]` asserts
the two agree for `waves`, `basin` and `stairs_up` at every $K = 1\ldots6$; if the
termination semantics ever change, the test fails instead of silently redefining
the headline number.

### 3.5 Two companion metrics

| metric | definition | why it cannot be omitted |
| --- | --- | --- |
| $RSR@K$ | fraction of episodes that re-observe within $K$ | the difficulty curve |
| $T_{reobs}$ | seconds from loss to the **first** re-observation | separates "wandering then found" from "found immediately" |
| $C_{action}$ | motion cost in metres | prevents a policy scoring well by pacing back and forth |

$C_{action}$ is averaged over **every** episode, not only the successful ones,
otherwise exhausting the budget looks cheap; the success-only figure is reported
alongside. The three together, $RSR@K + T_{reobs} + C_{action}$, are the complete
report; no single scalar is.
Re-observation as a decision problem

Executing $a$ from $s_t$ yields a physical next state $s_{t+1} = f(s_t, a)$ and a
re-observation indicator

$$
R(a \mid s_t) \;=\; \mathbb{1}\!\left[\exists\, \tau \in (t, t+\Delta]:\ V(s_\tau)=1\right]
$$

The **optimal single-step action** is

$$
a^\* \;=\; \arg\max_{a \in \mathcal{A}} \; P\bigl(V_{t+\Delta}=1 \mid O_t, M_t, a\bigr)
\tag{3.1}
$$

The central question of the paper is whether the conditioning set of (3.1) can be
reduced. §4.4 shows that $O_t$ alone cannot: there exist $O_A = O_B$ (pixel-identical)
with $a_A^\* \neq a_B^\*$ and
$\{a : R(a\mid s_A)=1\} \cap \{a : R(a \mid s_B)=1\} = \varnothing$.

## 3.5 Multi-step episodes

A single decision is not a fair test of a search algorithm, so we evaluate **episodes**
of up to $N$ steps. At each step the policy is asked for an action; the environment
executes it; the episode ends on the first of:

| termination | condition |
| --- | --- |
| `success` | $V = 1$ at the end of a step |
| `budget_exhausted` | $N$ steps executed without success |
| `fallen` | the base tipped below the upright threshold |
| `out_of_bounds` | the base left the terrain rectangle |

The reported protocol is **closed loop** (protocol B): the policy is re-asked after
every step with the new observation, i.e. it implements $\pi(a_t \mid O_t, M_t)$. The
open-loop variant (protocol A) asks once and repeats, and is reported only as a control.

**Fairness invariants** (enforced by tests, not by convention): every policy receives
the same $N$, the same per-step duration $\Delta$, the same observation stream length,
and the same termination set; and an action outside $\mathcal{A}$ is an error rather
than a silent fallback.

## 3.6 Multi-step episodes

The protocol, budget, action set and termination set are fixed in §"Benchmark".
The one property that matters here is that the **budget is identical for every
policy**: contract test `[8]` forces a policy that never succeeds and verifies the
full budget is granted, so a difference in $RSR@K$ cannot come from a difference in
how long each policy was allowed to run.

We deliberately **do not** fold the metrics into a single scalar
$J = \mathbb{1}_{\text{reobs}} - \lambda C$: the weights would be a choice made
before the data exist. We do report $RSR@K$ for several $K$ from a single run, since
re-observing at step $k$ implies re-observing within every $K \ge k$.

## 3.7 What is and is not assumed

| Assumed | Not assumed |
| --- | --- |
| the terrain is known to privileged components and to baselines that are given a map | that a deployable policy has the map |
| the target is static | that the target is always visible |
| the robot knows it once saw the target | that it currently knows where the target is |
| the action set is fixed and finite | that actions are reversible or unit-cost |
| the visibility oracle is exact geometry | that geometric visibility equals physical re-observability |

The last row is the assumption this paper is written to attack.
