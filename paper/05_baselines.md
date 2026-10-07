# 05 — Unified Baseline Interface

> 状态：**初稿**。接口已冻结（`API_VERSION = task3-reobservability/1.1`），
> 契约由 `test_api.py` 强制。

---

## 5.1 Why a frozen interface

Comparing decision rules across published work is normally confounded by the fact that
each is evaluated with its own observation, its own action set and its own budget. We
remove that confound by fixing one interface and expressing every baseline through it:

```
Environment                      (owns physics and the camera; policy-agnostic)
    |  reset() / step()
    v
Observation                      (everything a policy may look at)
    |-- current_rgb        H x W x 3 uint8, or None
    |-- current_depth      H x W float32, metres
    |-- proprioception     joint q/dq, base velocity, projected gravity
    |-- action_history     executed actions, including the approach walk-in
    |-- target_memory      where the target was last seen, age, range
    v
ReObservationPolicy.act(observation) -> action_id
    v
Environment.rollout(action_id)   (privileged counterfactual; scoring only)
```

Three properties make the comparison meaningful:

1. **No privileged information in `Observation`.** Ground-truth visibility and the
   target's world position live in a separate `PrivilegedState` that only the evaluator
   and the oracle baseline may read. This is asserted by a contract test, not by
   convention.
2. **The terrain is policy-agnostic.** Terrain modules must not import the API or any
   policy. This is enforced two ways: an AST scan of the terrain modules' imports, and a
   subprocess that imports each terrain module alone and fails if the policy stack comes
   with it.
3. **Counterfactuals do not perturb the episode.** `Environment.rollout` saves and
   restores the integration state and resets the controller. Without this, the oracle's
   own probing would move the robot it is deciding for — a defect the contract test
   caught in the first implementation.

## 5.2 The baselines

Each class implements the **core decision mechanism** of a line of work, not a full
reproduction.

| method | core mechanism | history | map | physics |
| --- | --- | :---: | :---: | :---: |
| `Reactive` | always drive forward | – | – | – |
| `HeadingLock` | hold the heading recorded at the loss moment (IMU only) | – | – | – |
| `HistoryOnly` | aim the heading at the remembered target and drive | ✓ | – | – |
| `LostTargetSearch` | face the belief, then close on it | ✓ | – | – |
| `ObjectPermanence` | constant-velocity belief filter, act on the filtered belief | ✓ | – | – |
| `NBV` | re-project the believed target from a **kinematic** hypothetical pose | ✓ | ✓ | – |
| `LearnedConsequence` | learned action-conditioned executed-pose + visibility model, score candidates, take argmax (E4-3) | ✓ | – | – |
| `Oracle` | physically roll out every candidate, take the fastest success | – | – | ✓ |

`NBV` is deliberately given the **exact height field**: an active-perception method is
allowed a map, and a failure should not be attributable to a weak one. It is not allowed
to execute the controller — that is the point of the baseline.

`ObjectPermanence` keeps the filter rather than being defined as equivalent to
`HistoryOnly`, because on a static target the extrapolation term vanishes and the
collapse is then a **measured** result rather than a stipulated one.

`Oracle` is a privileged upper bound and is never presented as a deployable method. It
costs one physical rollout per candidate per decision.

## 5.3 Evaluation protocol

Terrain catalogue: **8 multi-terrain scenes + 16 paired-history worlds = 24 terrains.**

The multi-terrain scenes alone are insufficient. On six of the eight, every baseline
succeeds, so they cannot separate the policies at all; an earlier version of the results
table showed six policies tied at 0.75. The paired worlds are the discriminating half,
because the correct action there depends on which side the target was last seen from.

Protocol B (closed loop) is the reported protocol: the policy is re-asked after every
step, implementing $\pi(a_t \mid O_t, M_t)$. Protocol A (open loop) asks once and
repeats, and is reported as a control.

### Fairness invariants (asserted by `test_api.py`)

* every primitive costs the same simulated time, so one step is one unit of budget;
* the budget is **actually granted**, verified by forcing the visibility read to report
  "hidden" and checking that a failing episode consumes all $N$ steps;
* one $N$, one termination set, one observation stream length for every policy;
* an `action_id` outside the frozen set is an error, never a silent fallback;
* $\mathrm{RSR@}N$ is monotone in $N$, so one run at the largest budget yields the curve.

### Phase bookkeeping

`action_history` entries tagged with the reserved id `approach` are the walk-in, not
decisions. The episode records `approach_entries`, `reobservation_steps`, `loss_t` and
`decision_t`, so that a history-horizon statement $H_k$ unambiguously means *the last $k$
observations before the decision* rather than "the last $k$ frames of everything".

## 5.4 Statistics

A comparison across decision rules is only meaningful if the uncertainty is reported, so
three rules are applied throughout (§"Figures" gives the implementation).

**Every proportion carries a Wilson score interval**, not a normal approximation. Several
of these results sit at or near $0\,\%$ or $100\,\%$, exactly where the normal interval
produces out-of-range bounds and understates the uncertainty.

**Every comparison is paired.** All policies run on the same terrains, so a difference in
$RSR$ is tested by resampling **terrains** with replacement and looking at the
distribution of the paired difference. An unpaired test would discard the fact that the
terrain is a shared, high-variance factor.

**Two degenerate cases are labelled rather than dressed up as statistics:**

* **Identical inputs.** Five of the baselines make **outcome-identical** choices on every
  one of the 24 terrains - i.e. they succeed and fail on exactly the same terrain set.
  Relative to `HistoryOnly`, `Reactive`, `HeadingLock`, `LostTargetSearch` and
  `ObjectPermanence` each have a paired difference of $0.0$ with interval $[0, 0]$. There
  is no variance to bootstrap, so **no $p$-value is reported**; reporting $p = 1$ would
  present "identical" as "not significant", which is a different statement.
* **All-zero difference distribution.** A $p$-value is bounded as
  $< 5 \times 10^{-5}$ rather than printed as an exact $0$.

**The resulting claims are deliberately weaker than the point estimates suggest.** From
`table_paired_tests.csv`:

| contrast | paired difference | 95 % CI | $p$ |
| --- | ---: | --- | ---: |
| `Reactive` / `HeadingLock` / `LostTargetSearch` / `ObjectPermanence` vs `HistoryOnly` | **0.000** | $[0, 0]$ | identical |
| `NBV` vs `HistoryOnly` | +0.042 | $[0.00, 0.125]$ | **0.356** |
| `Oracle` vs `NBV` | +0.583 | $[0.375, 0.792]$ | $<5\times10^{-5}$ |
| `OracleSearch` vs `NBV` | +0.708 | $[0.542, 0.875]$ | $<5\times10^{-5}$ |

so the correct statement is:

> **On these 24 terrains no classical baseline significantly outperforms any other
> classical baseline; only the privileged references do.**

Without the paired test, `NBV = 0.292 > 0.250` reads as "NBV is better". With it, the
interval contains zero and the claim does not survive. This is the reason the baselines
chapter reports intervals rather than a ranking.

> Evidence: `tasks/task3_reobservability/api.py`, `policies.py`, `episode.py`,
> `test_api.py`, `benchmark_api.py`, `paper_figures.py`;
> `paper/tables/table_paired_tests.csv`.
