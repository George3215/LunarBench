# 04 — Benchmark

> 状态：**初稿**。所有数字已用当前代码重新量过（`benchmark.py --mode profile`、
> `terrain_viewer.py --check`、`history.py`、`benchmark_api.py`）。

---

## 4.1 Design constraints

The benchmark is built so that **the visibility oracle and the collision surface are the
same object**. Each terrain is a set of axis-aligned boxes ("slabs"); the analytic height
function is the upper envelope of the box tops, so the ray-cast oracle and MuJoCo's
contact geometry cannot disagree.

This is not a stylistic choice. We first built the terrains as sampled height fields and
measured that MuJoCo's `hfield` collider stalls **both** vendored locomotion policies:
on flat ground with an identical command, the robot advanced 0.076 m on a height field
versus 0.377 m on box ground, while the two surfaces agreed to 0.0 m. Every benchmark
terrain is therefore exact box geometry. A ray-cast surface check
(`benchmark.py --mode surface`, `terrain_viewer.py --check`) verifies this per terrain:
**all 12 slab/box terrains agree with the oracle to 0.0 m**.

A second constraint is walkability. The locomotion policy is a blind, proprioception-only
network; measured, it climbs steps up to about 0.15 m and slopes up to about 30°. Terrain
relief is therefore quantised into treads within that limit. Where this conflicts with
richness we report the conflict rather than hiding it — see the Perlin discussion below.

## 4.2 The benchmark is organised by occlusion mechanism

The suite is not "eight kinds of terrain", and it is not a set of obstacles built to
break a baseline. Each entry is a **terrain-induced target occlusion scenario**: a
situation in which a robot that has already seen a target loses sight of it and must
regain sight by moving. A reviewer should recognise the *situations*, not the traps.

Two framing rules follow, and the first is enforced mechanically:
`mechanism_table.py` refuses to emit the table if the wording "challenging / hard /
difficult / adversarial terrain" appears anywhere in the taxonomy, because the
difficulty is in the target's **visibility**, not in the driving.

| Mechanism | Axis | Position | Real-world analogue | Visibility effect | Physical effect | Typical recovery |
| --- | --- | --- | --- | --- | --- | --- |
| `barrier` | occlusion direction | lateral | walls, boulders, shelving, parked machinery | target behind a finite-width obstacle; the sight line is restorable sideways | lateral bypass; moving closer does not help | side-step around the obstacle edge |
| `trench` | elevation | elongated depression | drainage ditches, service trenches, ravines | the near lip occludes until the robot descends and clears the far rim | the robot's own height changes along the route | continue through the trench |
| `stairs_up` | elevation | discrete steps | staircases, industrial platforms, stepped terrain | visibility depends on height, not horizontal distance | climbing; horizontal progress alone is insufficient | ascend to a higher viewpoint |
| `pit` | spatial scale | local | potholes, craters, excavation pits | descending lowers the viewpoint and the far wall occludes | the viewpoint drops below the surrounding ground | back out of the depression |
| `basin` | spatial scale | global | valleys, mining depressions, large settlement areas | visibility varies continuously over a wide region, not at one lip | the whole local viewpoint field shifts | climb out of the basin |
| `waves` | visibility dynamics | periodic | dunes, wind ripples, rolling natural ground | visibility flips repeatedly along the path, $1\to0\to1\to0\to1$ | repeated small pose changes | closed-loop re-observation; no permanently correct heading |
| `perlin` | terrain complexity | smooth | natural off-road ground, rock fields, unpaved roads | occlusion distributed over local terrain rather than localised | continuous pose change along the route | terrain-conditioned viewpoint prediction |
| `perlin_rough` | terrain complexity | rough | rubble, rock piles, debris fields, rough regolith | irregular occlusion with no consistent geometry | **locomotion uncertainty**: large pose prediction error | consequence prediction, not viewpoint geometry |

**Five roughly orthogonal axes.** A mechanism is a *point* on an axis, not a member
of a category: `pit` and `basin` are both spatial scale, at opposite ends of it.

| Axis | Mechanisms, by position on the axis |
| --- | --- |
| occlusion direction | `barrier` (lateral) |
| elevation | `stairs_up` (discrete steps), `trench` (elongated depression) |
| spatial scale | `pit` (local), `basin` (global) |
| visibility dynamics | `waves` (periodic) |
| terrain complexity | `perlin` (smooth), `perlin_rough` (rough) |

So the suite is **eight representative terrain-induced mechanisms** spanning
directional occlusion, elevation, scale, visibility dynamics and terrain complexity -
not a list of surfaces, and not a difficulty ladder.

`basin` is mechanistically distinct from `pit` and `trench`: those hide a target
*beyond* a far rim, whereas `basin` hides a target *inside* a broad depression, and
the occlusion is a grazing angle rather than a wall. `pit` and `basin` differ by
**scale**, not by kind.

`window_wall` (a pinhole aperture) is used for the paired-world construction rather
than for the main suite; it is a mechanism for *manufacturing* observation ambiguity,
not a task in its own right.

### A note on Perlin terrain and the locomotion limit

`perlin` is genuine 2D Perlin gradient noise summed over fBm octaves, implemented with
NumPy only and reproducible from a seed. Making it *walkable* required three measured
corrections, which we report because they bound what this benchmark can claim:

1. **Quantisation.** The blind policy cannot walk a continuous ramp, so heights are
   quantised into 0.05 m steps.
2. **Slope clamping.** The natural noise slope far exceeds the robot's step limit, so
   the field is projected onto a Lipschitz bound.
3. **Lateral flattening.** With full 2D relief, the robot ends up straddling a ledge
   under its own footprint and stalls — measured, 0.24 m of travel in 6 s. Terrain is
   therefore blended toward its centreline profile ($\beta = 0.35$, rough: 0.45).
   $\beta = 0$ is a $y$-invariant profile, $\beta = 1$ is full 2D.

The honest statement is: **for this locomotion policy, full-2D relief with meaningful
amplitude and quantised walkable treads is not simultaneously achievable.** The benchmark
therefore varies mechanism and $y$-variation together, and we do not claim that
`perlin` exercises arbitrary 2D terrain.

## 4.3 Non-monotonic visibility (Finding F1)

Sampling the visibility oracle along the approach line at 0.05 m resolution gives:

| terrain | $V(x)$ | $N_{\mathrm{flip}}$ |
| --- | --- | ---: |
| `stairs_up` | `V X V X` | 3 |
| `trench` | `V X V X` | 3 |
| `pit` | `V X V X` | 3 |
| `barrier` | `V X V X` | 3 |
| `waves` | `X V X V X` | 4 |
| `basin` | `X V X` | 2 |
| `perlin` | `X V X V X` | 4 |
| `perlin_rough` | `X V X V X` | 4 |

Every mechanism is non-monotone, and the direction of the effect differs: on
`stairs_up` the target is visible from far and lost on approach, on `basin` it is
**hidden from far and revealed by approaching**. The latter is worth emphasising: the
sign of the "closer ⇒ more observable" relation is itself terrain-dependent, so it has to
be measured rather than assumed.

> Evidence: `generated/benchmark.json` (`profile` per scene),
> `figure_benchmark.py` → `docs/images/task3_benchmark.png`.

## 4.4 Paired worlds: identical observation, opposite optimal action (Finding F2)

The decisive construction. Each **pair** consists of two scenes that share

* the same terrain (a wall pierced by a pinhole window),
* the same robot pose, heading and action history at the decision moment,
* the same target distance and altitude,

and differ **only** in which side of the aperture the target sits behind, and therefore
in which side the robot last saw it from.

### The pairs are searched for, not hand-picked

A parameter search accepts a pair only if seven conditions hold, all **purely geometric
and fixed before any physics is run**:

1. at the lost state both targets are hidden by **terrain**, inside the field of view;
2. the **entire marker** is hidden, not merely its centre point;
3. each world has a history vantage on its own side from which its target is visible;
4. the two lateral reveal sets are disjoint and on **opposite sides** of the lost state;
5. each reveal set lies within the reach of the corresponding strafe primitive;
6. each reveal set is at least 0.6 m wide, so a 10 Hz sampler cannot step over it;
7. walking straight forward or backward never reveals either target.

Result: **49 candidate parameter sets, 8 accepted.** Across three seeds (11/23/37) the
search accepted 8/8/8 pairs from 49/35/43 candidates, so the acceptance is a property of
the criteria rather than of one lucky seed. Criteria 2 and 6 were added after the first
version of the search produced pairs that passed geometry but failed for reasons
unrelated to history.

### The observation really is identical

Rendering the camera actually injected into the Go2+Piper model at the shared lost state
and comparing pixels:

| quantity | value across the 8 pairs |
| --- | ---: |
| mean absolute pixel difference (worst pair) | **0.0** |
| maximum pixel difference | **0** |
| fraction of identical pixels (best pair) | **100.0 %** |
| target pixels in world A / world B | **0 / 0** |

### The optimal action is opposite

| quantity | result |
| --- | ---: |
| pairs with equivalent observations | **8 / 8** |
| pairs with opposite optimal actions | **8 / 8** |
| pairs where **no action succeeds in both worlds** | **8 / 8** |

Optimal-action distribution: `right_hold` 7, `left` 5, `left_hold` 3, `right` 1 — spread
over both sides, so there is no majority answer.

Decision-rule success over the 16 worlds:

| rule | RSR |
| --- | ---: |
| reactive (always forward) | **0 %** |
| history-only (aim at remembered target) | **0 %** |
| initial-heading lock (IMU only) | **0 %** |
| oracle counterfactual | **100 %** |

**Interpretation.** Since $O_A = O_B$ is pixel-identical, every function of $O_t$ takes
the same value in both worlds. No memoryless policy can select $a_A^\*$ in world A and
$a_B^\*$ in world B. History is therefore not a refinement that improves accuracy; it is
information the current observation provably does not contain.

> Evidence: `generated/history_pairs.json`, `generated/history_pairs_seed{23,37}.json`,
> `generated/history_necessity.json`, `generated/history_camera/`,
> `figure_history.py` → `docs/images/task3_history_necessity.png`.

## 4.5 The multi-step episode protocol

A single decision from a frozen state is not a fair test of a search algorithm:
lost-target search is *defined* as "turn to face the belief, then close on it", and
scoring only its first step reports a property of the protocol, not of the algorithm. We
therefore evaluate episodes under a shared budget (§3.5).

### Step duration is measured, not chosen

A step must be long enough that a single action *can* re-observe, or the benchmark
degenerates into "chain four moves" and every policy fails at step one for a reason
unrelated to its decision rule. We measured, on seven scenes, how many actions can
re-observe within one step:

| step duration | scenes with at least one working action |
| --- | ---: |
| 1.5 s | **3 / 7** (four scenes had **zero** working actions) |
| 3.0 s | **7 / 7** |
| 4.5 s | 7 / 7 |
| 6.0 s | 7 / 7 |

The greedy oracle solved 6/7 within a five-step budget at 3.0 s, versus 2/7 at 1.5 s.
We use $\Delta = 3.0$ s and $N = 10$.

### The turn quantum must be smaller than the servo deadband

An earlier version of the protocol used the original turn rate (0.8 rad/s for 2 s,
rescaled to 137° per step). Any threshold rule then **oscillates** — we observed
`turn_left, turn_left, turn_right, turn_left, …` — because one step always jumps past
the deadband to the other side. Turn steps are therefore rescaled so one step is 45°,
and every threshold in a baseline is at least that wide. This is a correctness
condition on the action set, not a tuning choice.

> Evidence: `generated/benchmark_multistep_B.json`, `generated/benchmark_multistep_A.json`,
> `episode.py`, `test_api.py`.

## 4.6 Terrain self-check

Every terrain is verified independently of any policy: the MuJoCo contact surface is
ray-cast and compared against the analytic oracle, the robot holds station for 2 s, and
it walks forward for 5 s. **14 of 15 terrains pass**; the twelve slab/box terrains agree
with the oracle to 0.0 m and the robot walks 3.1–4.3 m on each. The single failure is a
legacy sampled height-field terrain with a 6.9 cm surface mismatch, which is reported
rather than silently dropped.

> Evidence: `generated/terrain_check.json`, `terrain_viewer.py --check --terrain all`.
