"""Generate and validate paired-history worlds for Experiment 1.

Every pair is a pair of worlds that share one terrain, one lost state and one
heading, and differ only in which side of a pinhole window the target sits
behind.  A pair is only accepted if the geometry provably gives the experiment
its teeth, which is checked here rather than assumed:

A. the shared lost state hides *both* targets, and hides them by terrain
   occlusion rather than by leaving the field of view;
B. each world has a history vantage where a *moving* robot would have seen its
   own target (so the difference really is history, not knowledge);
C. the lateral positions that reveal target A and target B are disjoint, sit on
   opposite sides of the lost state, and each lies inside the reach of the
   corresponding strafe primitive;
D. walking straight forward or straight back from the lost state never reveals
   either target.

Parameters that fail any check are discarded.  Nothing about the acceptance is
physical: it is pure line-of-sight geometry, so the search is cheap and can be
run before any MuJoCo rollout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

try:
    from .experiment import visibility
    from .scenes import ROBOT_BASE_HEIGHT_M, Scene
    from .slabs import SlabTerrain, window_wall
except ImportError:
    from experiment import visibility
    from scenes import ROBOT_BASE_HEIGHT_M, Scene
    from slabs import SlabTerrain, window_wall

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESCRIPTOR = ROOT / "tasks" / "task3_reobservability" / "generated" / "history_pairs.json"

# Lateral reach of the strafe primitives, measured from the Experiment 2
# rollouts (vy = +/-0.6 for 6 s gives about 2.9 m).  Keep a safety margin.
LATERAL_REACH_M = 2.6

# The strafe moves at roughly 0.5 m/s and visibility is sampled at 10 Hz, so a
# reveal window narrower than this can be stepped over between samples.  This is
# a sampling-robustness margin, chosen before looking at any physics result.
MIN_REVEAL_WIDTH_M = 0.6

# Radius of the target marker in the injected scene; the whole sphere has to be
# hidden, not just its centre, or a sliver renders and the two worlds stop being
# observationally identical.
TARGET_RADIUS_M = 0.15


@dataclass(frozen=True)
class PairParams:
    pair_id: str
    wall_x: float = 2.0
    wall_width: float = 0.15
    wall_height: float = 3.0
    window_half_y: float = 0.8
    sill_z: float = 1.0
    size_y: float = 8.0
    target_x: float = 5.0
    target_y: float = 3.0
    target_z: float = 2.0
    lost_x: float = 0.0
    lost_y: float = 0.0
    history_x: float | None = None
    history_y: float | None = None

    def terrain_kwargs(self) -> dict:
        return {"wall_x": self.wall_x, "wall_width": self.wall_width,
                "wall_height": self.wall_height, "window_half_y": self.window_half_y,
                "sill_z": self.sill_z, "size_y": self.size_y}


def build_terrain(params: PairParams) -> SlabTerrain:
    return window_wall(name="window_wall", **params.terrain_kwargs())


def _vis(terrain: SlabTerrain, x: float, y: float, yaw: float, target: np.ndarray) -> dict:
    robot = np.array([x, y, terrain.height(x, y) + ROBOT_BASE_HEIGHT_M], dtype=float)
    return visibility(terrain, robot, yaw, 0.0, target)


def _revealing_ys(terrain, lost_x, yaw, target, y_lim, step=0.05) -> np.ndarray:
    ys = np.arange(-y_lim, y_lim + step * 0.5, step)
    return np.array([float(y) for y in ys
                     if _vis(terrain, lost_x, float(y), yaw, target)["visible_oracle"]])


def _sphere_hidden(terrain, x: float, y: float, yaw: float, target: np.ndarray,
                   radius: float = TARGET_RADIUS_M) -> bool:
    """True only if no point of the target marker is observable.

    The oracle labels a single point; the renderer draws the whole sphere.  A
    pair is only observationally identical if the *sphere* is hidden, so sample
    its centre, the six face centres and the eight corners.
    """
    offsets = [np.zeros(3)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            v = np.zeros(3)
            v[axis] = sign * radius
            offsets.append(v)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                offsets.append(np.array([sx, sy, sz]) * radius / np.sqrt(3.0))
    for off in offsets:
        if _vis(terrain, x, y, yaw, target + off)["visible_oracle"]:
            return False
    return True


def _find_history_vantage(terrain, target, lost_xy, yaw, side: int,
                          y_lim: float) -> tuple[float, float] | None:
    """Grid-search a vantage on ``side`` of y where the target is visible."""
    best = None
    for x in np.arange(-5.0, terrain.spec["wall_x_m"] - 0.6, 0.25):
        for y in np.arange(-y_lim + 0.2, y_lim - 0.2, 0.2):
            if side * y <= 0.5:          # must approach from the target's own side
                continue
            if np.hypot(x - lost_xy[0], y - lost_xy[1]) < 1.0:
                continue
            if _vis(terrain, float(x), float(y), yaw, target)["visible_oracle"]:
                # Prefer the vantage furthest from the lost state.
                d = np.hypot(x - lost_xy[0], y - lost_xy[1])
                if best is None or d > best[0]:
                    best = (d, float(x), float(y))
    return None if best is None else (best[1], best[2])


def validate_pair(params: PairParams, yaw: float = 0.0,
                  lateral_reach: float = LATERAL_REACH_M) -> dict:
    terrain = build_terrain(params)
    y_lim = params.size_y / 2 - 0.05
    target_a = np.array([params.target_x, params.target_y, params.target_z])
    target_b = np.array([params.target_x, -params.target_y, params.target_z])
    lost = (params.lost_x, params.lost_y)
    checks: dict = {"pair_id": params.pair_id, "params": params.__dict__.copy()}

    # A. both targets hidden at the shared lost state, by terrain, not by FOV,
    #    and hidden as a whole sphere rather than as a point.
    for tag, target in (("A", target_a), ("B", target_b)):
        r = _vis(terrain, lost[0], lost[1], yaw, target)
        checks[f"lost_hides_{tag}"] = bool((not r["visible_oracle"]) and r["blocked_by_terrain"]
                                           and r["in_camera_fov"])
        checks[f"lost_blocked_{tag}"] = bool(r["blocked_by_terrain"])
        checks[f"lost_hides_sphere_{tag}"] = _sphere_hidden(terrain, lost[0], lost[1], yaw, target)

    # C. lateral reveal sets.
    set_a = _revealing_ys(terrain, lost[0], yaw, target_a, y_lim)
    set_b = _revealing_ys(terrain, lost[0], yaw, target_b, y_lim)
    checks["reveal_A_range"] = None if set_a.size == 0 else [round(float(set_a.min()), 3),
                                                             round(float(set_a.max()), 3)]
    checks["reveal_B_range"] = None if set_b.size == 0 else [round(float(set_b.min()), 3),
                                                             round(float(set_b.max()), 3)]
    lo_a = params.lost_y - lateral_reach
    hi_b = params.lost_y + lateral_reach
    checks["reveal_nonempty"] = bool(set_a.size and set_b.size)
    checks["reveal_disjoint"] = bool(set_a.size and set_b.size
                                     and not np.intersect1d(set_a, set_b).size)
    checks["reveal_opposite_sides"] = bool(
        set_a.size and set_b.size and np.all(set_a < params.lost_y)
        and np.all(set_b > params.lost_y))
    checks["reveal_A_reachable_by_right_strafe"] = bool(
        set_a.size and set_a.min() >= lo_a and set_a.max() <= params.lost_y)
    checks["reveal_B_reachable_by_left_strafe"] = bool(
        set_b.size and set_b.min() >= params.lost_y and set_b.max() <= hi_b)
    width_a = 0.0 if set_a.size < 2 else float(set_a.max() - set_a.min())
    width_b = 0.0 if set_b.size < 2 else float(set_b.max() - set_b.min())
    checks["reveal_width_A_m"] = round(width_a, 3)
    checks["reveal_width_B_m"] = round(width_b, 3)
    checks["reveal_wide_enough"] = bool(width_a >= MIN_REVEAL_WIDTH_M
                                        and width_b >= MIN_REVEAL_WIDTH_M)

    # D. straight forward / straight back never reveal.
    straight_a = straight_b = False
    for x in np.arange(params.lost_x - 4.5, params.wall_x + 0.01, 0.05):
        if _vis(terrain, float(x), params.lost_y, yaw, target_a)["visible_oracle"]:
            straight_a = True
        if _vis(terrain, float(x), params.lost_y, yaw, target_b)["visible_oracle"]:
            straight_b = True
    checks["straight_path_never_reveals"] = bool(not straight_a and not straight_b)

    # B. history vantages.
    hist_a = _find_history_vantage(terrain, target_a, lost, yaw, -1, y_lim)
    hist_b = _find_history_vantage(terrain, target_b, lost, yaw, +1, y_lim)
    checks["history_A"] = None if hist_a is None else [round(v, 3) for v in hist_a]
    checks["history_B"] = None if hist_b is None else [round(v, 3) for v in hist_b]
    checks["history_exists"] = bool(hist_a is not None and hist_b is not None)

    required = ["lost_hides_A", "lost_hides_B", "lost_hides_sphere_A", "lost_hides_sphere_B",
                "reveal_nonempty", "reveal_disjoint", "reveal_opposite_sides",
                "reveal_A_reachable_by_right_strafe", "reveal_B_reachable_by_left_strafe",
                "reveal_wide_enough", "straight_path_never_reveals", "history_exists"]
    checks["accepted"] = bool(all(checks[k] for k in required))
    checks["failed"] = [k for k in required if not checks[k]]
    return checks


def scenes_for(params: PairParams, yaw: float = 0.0) -> list[Scene]:
    """The two mirror worlds of one accepted pair."""
    terrain = build_terrain(params)
    y_lim = params.size_y / 2 - 0.05
    target_a = np.array([params.target_x, params.target_y, params.target_z])
    target_b = np.array([params.target_x, -params.target_y, params.target_z])
    hist_a = _find_history_vantage(terrain, target_a, (params.lost_x, params.lost_y), yaw, -1, y_lim)
    hist_b = _find_history_vantage(terrain, target_b, (params.lost_x, params.lost_y), yaw, +1, y_lim)
    common = dict(terrain_profile="window_wall", terrain_params=params.terrain_kwargs(),
                  yaw_rad=yaw, lost_xy=(params.lost_x, params.lost_y),
                  target_z_absolute=params.target_z, target_height_m=params.target_z,
                  history_saw_target=True)
    left = Scene(name=f"{params.pair_id}_left", target_xy=(params.target_x, params.target_y),
                 approach_xy=hist_a, history_xy=hist_a,
                 mechanism="target last seen through the window on the +y side", **common)
    right = Scene(name=f"{params.pair_id}_right", target_xy=(params.target_x, -params.target_y),
                  approach_xy=hist_b, history_xy=hist_b,
                  mechanism="target last seen through the window on the -y side", **common)
    return [left, right]


def search(n_pairs: int = 8, seed: int = 7, attempts: int = 600,
           lateral_reach: float = LATERAL_REACH_M) -> tuple[list[PairParams], list[dict]]:
    """Randomised search over parameters; returns (accepted, all_checks)."""
    rng = np.random.default_rng(seed)
    accepted: list[PairParams] = []
    log: list[dict] = []
    for i in range(attempts):
        if len(accepted) >= n_pairs:
            break
        p = PairParams(
            pair_id=f"hist{i:03d}",
            wall_x=float(rng.uniform(1.6, 2.4)),
            window_half_y=float(rng.uniform(0.6, 1.0)),
            sill_z=float(rng.uniform(0.90, 1.15)),
            size_y=float(rng.uniform(7.0, 9.0)),
            target_x=float(rng.uniform(4.5, 5.5)),
            target_y=float(rng.uniform(2.6, 3.4)),
            target_z=float(rng.uniform(1.8, 2.2)),
            lost_x=float(rng.uniform(-0.5, 0.5)),
            lost_y=float(rng.uniform(-0.4, 0.4)),
        )
        checks = validate_pair(p, lateral_reach=lateral_reach)
        if checks["accepted"]:
            accepted.append(p)
        log.append(checks)
    return accepted, log


def write_descriptor(path: Path, accepted: list[PairParams], log: list[dict]) -> dict:
    payload = {
        "task": "TASK3_experiment1_pair_search",
        "lateral_reach_m": LATERAL_REACH_M,
        "n_attempts": len(log),
        "n_accepted": len(accepted),
        "acceptance_criteria": [
            "lost state hides both targets by terrain occlusion inside the FOV",
            "the whole target marker is hidden, not just its centre point",
            "each world has a history vantage on its own side where the target is visible",
            "the lateral reveal sets are disjoint and on opposite sides of the lost state",
            "each reveal set lies inside the reach of the corresponding strafe primitive",
            f"each reveal set is at least {MIN_REVEAL_WIDTH_M} m wide so a 10 Hz sampler "
            "cannot step over it",
            "walking straight forward or back never reveals either target",
        ],
        "pairs": [p.__dict__ for p in accepted],
        "all_checks": log,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_descriptor(path: Path = DEFAULT_DESCRIPTOR) -> list[PairParams]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [PairParams(**p) for p in payload["pairs"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="search for paired-history worlds")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attempts", type=int, default=600)
    ap.add_argument("--output", type=Path, default=DEFAULT_DESCRIPTOR)
    args = ap.parse_args()
    accepted, log = search(args.pairs, args.seed, args.attempts)
    payload = write_descriptor(args.output, accepted, log)
    print(f"attempts={payload['n_attempts']} accepted={payload['n_accepted']}")
    for p in accepted:
        chk = next(c for c in log if c["pair_id"] == p.pair_id)
        print(f"  {p.pair_id}: wall_x={p.wall_x:.2f} win±{p.window_half_y:.2f} "
              f"sill={p.sill_z:.2f} target=({p.target_x:.2f},{p.target_y:.2f},{p.target_z:.2f}) "
              f"lost=({p.lost_x:+.2f},{p.lost_y:+.2f})  "
              f"revealA={chk['reveal_A_range']} revealB={chk['reveal_B_range']}")
    if not accepted:
        fails = {}
        for c in log:
            for k in c["failed"]:
                fails[k] = fails.get(k, 0) + 1
        print("no accepted pairs; failure counts:", fails)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
