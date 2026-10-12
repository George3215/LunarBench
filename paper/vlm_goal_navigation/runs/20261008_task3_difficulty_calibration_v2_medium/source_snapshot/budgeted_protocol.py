"""Budgeted closed-loop target re-observation: protocol and metrics.

Implements the agreed evaluation design. The ordering matters and is enforced by the
output format rather than left to discipline:

  1. **task success** - RSR@K over the budget curve. Reported first, on its own.
  2. **search cost**  - steps, real path length, net displacement, turn, time,
                        reversals, re-observation quality. Reported separately, never
                        merged into the success number.
  3. **discount scores** - linear and exponential, supplementary only.

The rules this module enforces:

* **A step is a step, not a metre and not a decision.** The budget counts action
  primitives. Path length is measured from the executed trajectory, never from the
  commanded displacement: if a "forward 0.5 m" primitive moves the base 0.32 m, the
  cost is 0.32 m.
* **Rotation is not translation.** Turn is reported in radians and never added to
  metres.
* **Failures are not free.** A failed episode contributes zero to every discount
  score, so a method that fails quickly cannot outscore one that succeeds late.
* **The discount never decides success.** ``success`` is a separate boolean and every
  score is conditioned on it.
* **A single pixel is not a success.** ``re-observed`` requires the target to appear
  for a sustained run of frames above a minimum area; ``usable`` additionally requires
  enough area to support downstream use. Both thresholds are configured, not tuned on
  the test set.
* **Episode ends at first success.** Once the sustained-visibility condition is met,
  the episode ends and its cost is recorded at that step. Methods are not allowed to
  keep walking and then be compared.

    python tasks/task3_reobservability/budgeted_protocol.py --run
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
except ImportError:
    import api
    import episode as ep

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "generated" / "budgeted_protocol.json"

#: Evaluation checkpoints. The budget itself is the last of these.
BUDGETS = (1, 3, 5, 10, 25, 50, 100)
STRESS_BUDGET = 150

#: Success thresholds. Fixed here, before any test run, and not adjusted afterwards.
MIN_TARGET_PIXELS = 3        # a single pixel does not count as re-observation
SUSTAINED_FRAMES = 3         # must hold for this many consecutive 10 Hz frames
USABLE_TARGET_PIXELS = 24    # enough area for downstream use, not just detection

#: Discount parameters, fixed a priori.
RHO_STEPS = 0.5              # linear step discount keeps half its value at K
HALF_LIFE_STEPS = 100.0      # exponential half-life, so both curves agree at K
D_MAX = 25.0                 # metres at which the linear distance discount vanishes
RHO_DIST = 0.5
D_HALF_M = 20.0              # half-life of the exponential distance discount


@dataclass
class EpisodeTrace:
    """Everything one episode records. Raw, unmerged."""
    terrain: str
    method: str
    success: bool
    usable: bool
    steps: int
    t_reobs_s: float | None
    path_m: float
    net_m: float
    turn_rad: float
    reversals: int
    target_pixels_at_success: int
    termination: str
    loop_steps: int               # steps spent in an almost-unchanged pose
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def rsr_at(traces: list[EpisodeTrace], k: int) -> float:
    """Fraction of episodes that first re-observed within ``k`` primitives."""
    if not traces:
        return 0.0
    hit = sum(1 for t in traces if t.success and t.steps <= k)
    return hit / len(traces)


def discount_steps(linear: bool, n: int, k: int = BUDGETS[-1]) -> float:
    if linear:
        return max(0.0, 1.0 - RHO_STEPS * n / k)
    return math.exp(-math.log(2.0) / HALF_LIFE_STEPS * n)


def discount_distance(linear: bool, d: float) -> float:
    if linear:
        return max(0.0, 1.0 - RHO_DIST * d / D_MAX)
    return math.exp(-math.log(2.0) / D_HALF_M * d)


def discounted_scores(traces: list[EpisodeTrace]) -> dict:
    """Supplementary only: zero for every failure, by construction."""
    out = {}
    for name, linear in (("linear", True), ("exponential", False)):
        vals = [
            (discount_steps(linear, t.steps) * discount_distance(linear, t.path_m))
            if t.success else 0.0
            for t in traces
        ]
        out[name] = float(np.mean(vals)) if vals else 0.0
        # Step-only and distance-only variants, so a ranking that depends on the
        # discount shape rather than on behaviour is visible rather than hidden.
        out[f"{name}_steps_only"] = float(np.mean(
            [discount_steps(linear, t.steps) if t.success else 0.0 for t in traces]))
        out[f"{name}_distance_only"] = float(np.mean(
            [discount_distance(linear, t.path_m) if t.success else 0.0 for t in traces]))
    return out


def summarise(traces: list[EpisodeTrace]) -> dict:
    ok = [t for t in traces if t.success]
    failed = [t for t in traces if not t.success]
    finite = [t.path_m for t in traces]
    return {
        # -- layer 1: task success, reported alone ------------------------
        "n_episodes": len(traces),
        "rsr": {f"@{k}": round(rsr_at(traces, k), 4) for k in BUDGETS},
        "n_success": len(ok),
        "n_usable": sum(1 for t in traces if t.usable),
        # -- layer 2: search cost, never merged into the success number ---
        "cost": {
            "steps_mean_all": round(float(np.mean([t.steps for t in traces])), 3),
            "steps_mean_success": round(float(np.mean([t.steps for t in ok])), 3) if ok else None,
            "path_m_mean_all": round(float(np.mean(finite)), 3),
            "path_m_median": round(float(np.median(finite)), 3),
            "net_m_mean": round(float(np.mean([t.net_m for t in traces])), 3),
            "path_over_net": round(
                float(np.mean([t.path_m for t in traces])
                      / max(float(np.mean([t.net_m for t in traces])), 1e-6)), 3),
            "turn_rad_total": round(float(np.sum([t.turn_rad for t in traces])), 3),
            "reversals_total": int(sum(t.reversals for t in traces)),
            "t_reobs_s_mean_success": round(
                float(np.mean([t.t_reobs_s for t in ok if t.t_reobs_s is not None])), 3)
                if ok else None,
            "target_pixels_at_success": int(
                np.median([t.target_pixels_at_success for t in ok])) if ok else None,
            "loop_steps_total": int(sum(t.loop_steps for t in traces)),
        },
        # -- layer 3: discounts, supplementary ---------------------------
        "discount": discounted_scores(traces),
        "failure_terminations": {
            k: sum(1 for t in failed if t.termination == k)
            for k in sorted({t.termination for t in failed})
        },
        "n_failed": len(failed),
    }


def paired_report(a: list[EpisodeTrace], b: list[EpisodeTrace],
                  n_resamples: int = 2000, seed: int = 11) -> dict:
    """Paired bootstrap over terrains, as everywhere else in this project."""
    by_terrain_a = {t.terrain: t for t in a}
    by_terrain_b = {t.terrain: t for t in b}
    shared = sorted(set(by_terrain_a) & set(by_terrain_b))
    if not shared:
        return {}
    va = np.array([1.0 if by_terrain_a[t].success else 0.0 for t in shared])
    vb = np.array([1.0 if by_terrain_b[t].success else 0.0 for t in shared])
    diff = va - vb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_resamples, len(shared)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    if np.allclose(va, vb):
        return {"difference": 0.0, "ci": [0.0, 0.0], "p": None, "label": "identical"}
    p = float(np.mean(boot <= 0.0))
    p = min(p, 1 - p) * 2
    return {"difference": round(float(diff.mean()), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(p, 5) if p > 0 else "<5e-05",
            "n_shared_terrains": len(shared)}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def policy_for(name: str, seed: int = 1):
    """Decision rules. Every one draws from the identical frozen action set."""
    from benchmark import BENCHMARK_SCENES  # noqa: F401  (import check)
    import random_baseline
    if name == "Random":
        return random_baseline.RandomPolicy(seed)
    import benchmark as bm
    if hasattr(bm, "POLICIES") and name in bm.POLICIES:
        return bm.POLICIES[name]()
    raise SystemExit(f"unknown policy {name!r}")


def run_episode(policy, environment, terrain: str, method: str,
                budget: int) -> EpisodeTrace:
    """One closed-loop episode under the budget, ending at first sustained success."""
    observation = environment.reset()
    start = np.array(observation.pose.xyz, dtype=float)
    path = 0.0
    turn = 0.0
    reversals = 0
    loop_steps = 0
    previous_xy = start[:2].copy()
    previous_yaw = float(observation.pose.yaw)
    last_direction = None
    streak = 0
    pixels_at_success = 0
    t_reobs = None
    success = False
    usable = False
    step = 0

    for step in range(1, budget + 1):
        action = policy.act(observation)
        environment.step(action)
        now = environment.observe()
        xy = np.array(now.pose.xyz, dtype=float)[:2]
        delta = xy - previous_xy
        # Real displacement, measured from the executed trajectory, never the
        # commanded value: a "forward 0.5 m" primitive that moves the base 0.32 m
        # costs 0.32 m.
        moved = float(np.linalg.norm(delta))
        path += moved
        # Rotation is not translation: turn is accumulated in radians on its own.
        yaw = float(now.pose.yaw)
        turn += abs(math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw)))

        if moved > 1e-3:
            direction = math.atan2(delta[1], delta[0])
            if last_direction is not None and abs(
                    math.atan2(math.sin(direction - last_direction),
                               math.cos(direction - last_direction))) > 2.0:
                reversals += 1
            last_direction = direction
        else:
            loop_steps += 1

        visible_pixels = count_target_pixels(environment, now)
        if visible_pixels >= MIN_TARGET_PIXELS:
            streak += 1
            if streak >= SUSTAINED_FRAMES:
                success = True
                t_reobs = step * ep.STEP_DURATION_S
                pixels_at_success = max(pixels_at_success, visible_pixels)
                usable = visible_pixels >= USABLE_TARGET_PIXELS
                break
        else:
            streak = 0
        previous_xy = xy
        previous_yaw = yaw
        observation = now

    return EpisodeTrace(
        terrain=terrain, method=method, success=success, usable=usable,
        steps=step, t_reobs_s=t_reobs, path_m=path,
        net_m=float(np.linalg.norm(np.array(observation.pose.xyz, dtype=float)[:2]
                                   - start[:2])),
        turn_rad=turn, reversals=reversals,
        target_pixels_at_success=pixels_at_success,
        termination="success" if success else "budget_exhausted",
        loop_steps=loop_steps)


def count_target_pixels(environment, observation) -> int:
    """Target pixels in the rendered frame. Returns 0 when rendering is off."""
    rgb = observation.current_rgb
    if rgb is None:
        return 0
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    return int(np.count_nonzero((r > 140) & (g < 90) & (b < 90)))


def main() -> None:
    p = argparse.ArgumentParser(description="budgeted closed-loop re-observation")
    p.add_argument("--run", action="store_true")
    p.add_argument("--methods", nargs="*", default=["Random"])
    p.add_argument("--budget", type=int, default=BUDGETS[-1])
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    if not args.run:
        print(__doc__)
        return

    from benchmark import BENCHMARK_SCENES
    print(f"=== Budgeted closed-loop target re-observation ===")
    print(f"budget {args.budget} primitives; checkpoints {BUDGETS}; "
          f"success needs {MIN_TARGET_PIXELS} px for {SUSTAINED_FRAMES} frames")
    print(f"step duration {ep.STEP_DURATION_S}s -> {args.budget * ep.STEP_DURATION_S:.0f}s "
          f"simulated per episode; a step is a primitive, not a metre\n")

    report = {}
    traces_by_method = {}
    for method in args.methods:
        traces = []
        for name in BENCHMARK_SCENES:
            scene = api.scene_catalogue()[name]
            environment = api.MuJoCoEnvironment(
                scene, action_table=ep.protocol_action_table(),
                render=True, render_size=(160, 120))
            try:
                traces.append(run_episode(policy_for(method, 1), environment, name,
                                          method, args.budget))
            finally:
                environment.close()
        traces_by_method[method] = traces
        report[method] = summarise(traces)
        s = report[method]
        print(f"  {method:22s} RSR@10={s['rsr']['@10']:.3f} RSR@100={s['rsr']['@100']:.3f} "
              f"success={s['n_success']}/{s['n_episodes']} "
              f"path={s['cost']['path_m_mean_all']:.2f}m "
              f"loop_steps={s['cost']['loop_steps_total']}")

    names = list(traces_by_method)
    if len(names) >= 2:
        report["paired"] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                report["paired"][f"{names[i]} vs {names[j]}"] = paired_report(
                    traces_by_method[names[i]], traces_by_method[names[j]])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    print("\nLayer 1 (success) and layer 2 (cost) are reported separately by design; "
          "the discount scores in layer 3 are supplementary and are zero for every "
          "failure, so a method that fails quickly cannot outscore one that succeeds "
          "late.")


if __name__ == "__main__":
    main()
