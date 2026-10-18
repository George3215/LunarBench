"""TASK3 H2: is a correct target memory sufficient to choose the action?

Experiment 1 (H1) showed that the *current observation* does not determine the
optimal re-observation action, because the two worlds differ in where the target
was last seen from.  The obvious reply is: "then just remember where the target
is".  H2 tests that reply directly, and it is strictly stronger.

Each **memory pair** consists of two worlds that share

* the same terrain in the whole visible forward frustum, so that the rendered
  camera image at the decision point is pixel-identical;
* the same target, at the same place;
* the same approach and therefore the same target memory (position, age, range);

and differ **only in terrain hidden from the decision point**, placed inside the
shadow of a visible occluder.  Because the difference is invisible *now* and was
never traversed, it is present neither in $O_t$ nor in $M_t$ - yet it decides
which action works.

And the difference is not decorative: the hidden feature is placed so that it
blocks one lateral detour route.  World B is free on both sides, world A is
blocked on one, so

    a_B* = left        a_A* = right

with $O_A = O_B$ and $M_A = M_B$.  That is

    target localisation  !=  re-observation planning

The relation to Experiment 1 is the point of the experiment:

===========  ==========  ==========  ==================
experiment   O_A = O_B   M_A = M_B   a_A* != a_B*
===========  ==========  ==========  ==================
H1           yes         **no**      yes
H2 (here)    yes         **yes**     yes
===========  ==========  ==========  ==================
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from .slabs import RESOLUTION_M, Slab, SlabTerrain
except ImportError:
    from slabs import RESOLUTION_M, Slab, SlabTerrain

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESCRIPTOR = Path(__file__).resolve().parent / "generated" / "memory_pairs.json"

ROBOT_EYE_Z = 0.46          # base 0.30 + camera mount 0.16, as everywhere else in TASK3
ROBOT_BASE_Z = 0.30
LATERAL_REACH_M = 2.6       # a 3 s strafe at 0.6 m/s, minus margin
#: Where the robot walks in from, relative to the decision point.  It has to be
#: off to one side, because behind the wall the target is never visible; and it
#: has to be mostly *ahead* of the decision point, because the walk-in heading is
#: the direction of travel, so a purely lateral approach would start with the
#: target outside the field of view and produce no sighting at all.
HISTORY_BACK_M = 3.5
HISTORY_SIDE_M = 2.5


@dataclass(frozen=True)
class PairParams:
    """Geometry of one memory pair.  Everything here is visible in both worlds."""

    pair_id: str
    #: the visible occluder, identical in both worlds
    wall_x: float
    wall_half_y: float
    wall_height: float
    wall_width: float
    #: The wall is offset in y so that one lateral detour is strictly easier than
    #: the other.  Without the offset the two detours are symmetric, the physical
    #: gait asymmetry decides the tie, and the optimal action ends up the same in
    #: both worlds even though the hidden wall blocked one of them.
    wall_center_y: float
    #: the target, identical in both worlds
    target_x: float
    target_y: float
    target_z: float
    #: the decision point
    lost_x: float
    lost_y: float
    #: the hidden feature, present in world A only.  Must sit inside the shadow
    #: of the visible wall as seen from the decision point.
    hidden_x: float
    hidden_y0: float
    hidden_y1: float
    hidden_height: float
    size_x: float = 12.0
    size_y: float = 6.0


def build_terrain(params: PairParams, side: str, name: str = "") -> SlabTerrain:
    """``side`` 'a' includes the hidden feature, 'b' does not.

    The floor, the visible wall and the target are identical, so the two worlds
    differ only in geometry that the decision-point camera cannot see.
    """
    if side not in ("a", "b"):
        raise ValueError(side)
    x_min, x_max = -params.size_x / 2, params.size_x / 2
    y_half = params.size_y / 2
    wall_x1 = params.wall_x + params.wall_width
    slabs = [
        Slab(x_min, x_max, -y_half, y_half, 0.0, 0.0, 1.5, "ground"),
        Slab(params.wall_x, wall_x1,
             params.wall_center_y - params.wall_half_y,
             params.wall_center_y + params.wall_half_y,
             params.wall_height, params.wall_height, 1.5, "wall_visible"),
    ]
    hidden_present = side == "a"
    if hidden_present:
        slabs.append(Slab(params.hidden_x, params.hidden_x + params.wall_width,
                          params.hidden_y0, params.hidden_y1,
                          params.hidden_height, params.hidden_height, 1.5,
                          "wall_hidden"))
    label = name or params.pair_id
    spec = {
        "profile": f"memory_pair_{label}_{side}",
        "pair_id": params.pair_id, "side": side,
        "visible_wall_x_m": params.wall_x, "visible_wall_half_y_m": params.wall_half_y,
        "visible_wall_height_m": params.wall_height,
        "hidden_wall_x_m": params.hidden_x if hidden_present else None,
        "hidden_wall_y_m": [params.hidden_y0, params.hidden_y1] if hidden_present else None,
        "hidden_wall_height_m": params.hidden_height if hidden_present else None,
        "size_m": [params.size_x, params.size_y], "resolution_m": RESOLUTION_M,
    }
    return SlabTerrain(spec["profile"], slabs, spec)


def scene_for_side(params: PairParams, side: str):
    """A :class:`Scene` for one side, with the target at a fixed absolute height.

    The robot walks in from ``history_xy`` off to the -y side, so it *has* seen the
    target before the decision point and carries a non-empty memory.  That vantage
    gives the same sighting in both worlds - the hidden feature sits on the +y
    side and never crosses that line of sight - so $M_A = M_B$ is a construction,
    not an assumption.  Without this the memory would simply be empty in both
    worlds and "identical memory" would be vacuous.
    """
    try:
        from .scenes import Scene
    except ImportError:
        from scenes import Scene
    vantage = (params.lost_x - HISTORY_BACK_M, params.lost_y - HISTORY_SIDE_M)
    return Scene(
        name=f"{params.pair_id}_{'left' if side == 'a' else 'right'}",
        terrain_profile=f"memory_pair_{params.pair_id}_{side}",
        target_xy=(params.target_x, params.target_y),
        target_z_absolute=params.target_z,
        target_height_m=params.target_z,
        approach_xy=vantage,
        lost_xy=(params.lost_x, params.lost_y),
        history_xy=vantage,
        yaw_rad=0.0,
        mechanism="terrain hidden inside the shadow of a visible occluder",
        notes=(f"world {'A (hidden wall present)' if side == 'a' else 'B (hidden wall absent)'}",),
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _eye(x: float, y: float, z: float = ROBOT_EYE_Z) -> np.ndarray:
    return np.array([x, y, z])


def shadow_contains(params: PairParams, point: np.ndarray) -> bool:
    """Is ``point`` hidden from the decision point by the visible wall?"""
    eye = _eye(params.lost_x, params.lost_y)
    if point[0] <= params.wall_x:
        return False
    t = (params.wall_x - eye[0]) / (point[0] - eye[0])
    if not (0.0 < t < 1.0):
        return False
    y_at = eye[1] + t * (point[1] - eye[1])
    z_at = eye[2] + t * (point[2] - eye[2])
    return (abs(y_at - params.wall_center_y) <= params.wall_half_y
            and z_at <= params.wall_height)


def detour_threshold(params: PairParams, terrain) -> dict:
    """Smallest |y| at which each lateral detour first reveals the target.

    Measured geometrically by scanning the strafe line, so the search can require
    that the +y detour is *easier* than the -y one.  Without that asymmetry the
    two detours tie and the physical gait decides, which leaves the optimal action
    identical in both worlds.
    """
    from experiment import visibility
    target = np.array([params.target_x, params.target_y, params.target_z])
    out = {}
    for sign, tag in ((1.0, "plus_y"), (-1.0, "minus_y")):
        threshold = None
        for step in range(1, 41):
            y = params.lost_y + sign * 0.05 * step
            x = params.lost_x
            if not (terrain.y[0] <= y <= terrain.y[-1]):
                break
            z = float(terrain.height(x, y)) + ROBOT_BASE_Z
            if visibility(terrain, np.array([x, y, z]), params_yaw(params), 0.0,
                          target)["visible_oracle"]:
                threshold = abs(y - params.lost_y)
                break
        out[tag] = threshold
    return out


def kinematic_outcomes(params: PairParams, terrain) -> dict:
    """Which actions would put the target in view, under the kinematic model.

    Cheap pre-filter.  Its purpose is to guarantee the pair is *solvable by a
    single lateral detour*, because the benchmark scores one action per step and
    a wall that needs two strafes would make every world unsolvable - which is
    what the first version of this search produced.
    """
    from experiment import visibility
    try:
        from . import episode as ep
    except ImportError:
        import episode as ep
    table = ep.protocol_action_table()
    lost = np.array([params.lost_x, params.lost_y, ROBOT_BASE_Z])
    target = np.array([params.target_x, params.target_y, params.target_z])
    out = {}
    for action_id, primitive in table.items():
        dx = primitive.vx * primitive.duration_s
        dy = primitive.vy * primitive.duration_s
        x = params.lost_x + dx
        y = params.lost_y + dy
        if not (terrain.x[0] <= x <= terrain.x[-1] and terrain.y[0] <= y <= terrain.y[-1]):
            out[action_id] = False
            continue
        z = float(terrain.height(x, y)) + ROBOT_BASE_Z
        reading = visibility(terrain, np.array([x, y, z]), params_yaw(params), 0.0, target)
        out[action_id] = bool(reading["visible_oracle"])
    return out


def params_yaw(params: PairParams) -> float:
    return 0.0


def validate_pair(params: PairParams, terrain_a, terrain_b) -> dict:
    """Geometric acceptance criteria, fixed before any physics is run."""
    from experiment import visibility

    checks: dict = {}
    target = np.array([params.target_x, params.target_y, params.target_z])
    lost = np.array([params.lost_x, params.lost_y, ROBOT_BASE_Z])

    # 1. the target is hidden by terrain at the decision point, in both worlds
    for tag, terrain in (("A", terrain_a), ("B", terrain_b)):
        reading = visibility(terrain, lost, 0.0, 0.0, target)
        checks[f"lost_hides_target_{tag}"] = bool(
            (not reading["visible_oracle"]) and reading["blocked_by_terrain"]
            and reading["in_camera_fov"])

    # 2. the hidden wall exists only in A, and is entirely inside the wall's shadow
    hidden = np.array([params.hidden_x, params.hidden_y0, 0.0])
    corner_top = params.hidden_height
    sample_points = [np.array([params.hidden_x + dx, y, z])
                     for dx in (0.0, params.wall_width)
                     for y in (params.hidden_y0, params.hidden_y1)
                     for z in (0.02, corner_top - 0.02)]
    checks["hidden_fully_in_shadow"] = all(shadow_contains(params, p) for p in sample_points)
    checks["hidden_only_in_A"] = (len(terrain_a.slabs) == len(terrain_b.slabs) + 1)

    # 3. the two worlds are geometrically identical in front of the hidden wall:
    #    every slab of B must also be in A, with the same top
    tops_b = sorted((round(s.x0, 6), round(s.x1, 6), round(s.y0, 6), round(s.y1, 6),
                     round(max(s.z0, s.z1), 6)) for s in terrain_b.slabs)
    tops_a = sorted((round(s.x0, 6), round(s.x1, 6), round(s.y0, 6), round(s.y1, 6),
                     round(max(s.z0, s.z1), 6)) for s in terrain_a.slabs)
    checks["B_is_a_subset_of_A"] = all(item in tops_a for item in tops_b)

    # 4. the visible wall must block the direct line, so a lateral detour is needed
    checks["visible_wall_blocks"] = abs(params.target_y - params.lost_y) < params.wall_half_y + 0.6

    # 5. the hidden wall must sit on exactly one side, so one detour is blocked
    checks["hidden_on_one_side"] = (params.hidden_y0 > 0.1 or params.hidden_y1 < -0.1)
    checks["hidden_within_lateral_reach"] = (
        min(abs(params.hidden_y0), abs(params.hidden_y1)) <= LATERAL_REACH_M)
    checks["hidden_covers_a_detour_side"] = (params.hidden_y1 - params.hidden_y0) >= 0.25

    # 6. the pair must be solvable by ONE lateral detour, and the hidden wall must
    #    take exactly one of the two detours away.  Without this the search happily
    #    returns pairs where nothing succeeds in either world.
    kin_a = kinematic_outcomes(params, terrain_a)
    kin_b = kinematic_outcomes(params, terrain_b)
    win_a = {k for k, v in kin_a.items() if v}
    win_b = {k for k, v in kin_b.items() if v}
    checks["B_solvable_kinematically"] = bool(win_b)
    checks["A_solvable_kinematically"] = bool(win_a)
    checks["left_detour_works_in_B"] = bool(kin_b.get("left") or kin_b.get("left_hold"))
    checks["left_detour_blocked_in_A"] = not bool(kin_a.get("left") or kin_a.get("left_hold"))
    checks["right_detour_works_in_A"] = bool(kin_a.get("right") or kin_a.get("right_hold"))

    # 7. the walk-in must actually produce a sighting, or the memory is empty in
    #    both worlds and "identical memory" would be vacuous
    vantage = np.array([params.lost_x - HISTORY_BACK_M, params.lost_y - HISTORY_SIDE_M,
                        ROBOT_BASE_Z])
    travel = math.atan2(HISTORY_SIDE_M, HISTORY_BACK_M)
    checks["vantage_sees_target"] = bool(
        visibility(terrain_a, vantage, travel, 0.0, target)["visible_oracle"])

    # 8. the +y detour must be strictly easier in world B, otherwise the optimal
    #    action is the same in both worlds and the pair proves nothing
    threshold_b = detour_threshold(params, terrain_b)
    threshold_a = detour_threshold(params, terrain_a)
    checks["plus_y_detour_easier_in_B"] = bool(
        threshold_b["plus_y"] is not None and threshold_b["minus_y"] is not None
        and threshold_b["plus_y"] < threshold_b["minus_y"] - 0.15)
    checks["plus_y_detour_blocked_in_A"] = bool(
        threshold_a["plus_y"] is None
        or (threshold_a["minus_y"] is not None
            and threshold_a["plus_y"] > threshold_a["minus_y"]))
    checks["detour_thresholds_B"] = threshold_b
    checks["detour_thresholds_A"] = threshold_a

    required = ["lost_hides_target_A", "lost_hides_target_B", "hidden_fully_in_shadow",
                "hidden_only_in_A", "B_is_a_subset_of_A", "visible_wall_blocks",
                "hidden_on_one_side", "hidden_within_lateral_reach",
                "hidden_covers_a_detour_side", "B_solvable_kinematically",
                "A_solvable_kinematically", "left_detour_works_in_B",
                "left_detour_blocked_in_A", "right_detour_works_in_A",
                "vantage_sees_target", "plus_y_detour_easier_in_B",
                "plus_y_detour_blocked_in_A"]
    checks["passed"] = all(bool(checks[k]) for k in required)
    checks["required"] = required
    return checks


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def sample_params(rng: np.random.Generator, index: int) -> PairParams:
    wall_x = float(rng.uniform(1.8, 2.8))
    wall_height = float(rng.uniform(1.2, 1.8))
    # Narrow enough that ONE strafe step (1.8 m) clears the wall edge; a wider
    # wall needs two strafes and the pair becomes unsolvable in one step.
    wall_half_y = float(rng.uniform(0.30, 0.50))
    # Place the wall so its +y edge sits just past the robot's line, which makes
    # the +y detour strictly easier than the -y one.  The edge position is what
    # matters, so it is drawn directly rather than the centre.
    plus_y_edge = float(rng.uniform(0.16, 0.40))
    wall_center_y = plus_y_edge - wall_half_y
    wall_width = float(rng.uniform(0.15, 0.3))
    lost_x = float(rng.uniform(-0.6, 0.2))
    target_x = float(rng.uniform(4.5, 6.0))
    target_z = float(rng.uniform(0.35, 0.7))
    # The hidden wall sits behind the visible one, on the +y side, inside its
    # shadow.  The shadow interval at x_h follows from the shadow condition:
    #   |y_wall(c) - y_ray| <= h   at x = wall_x,  with t = (wall_x-ex)/(x_h-ex)
    # so the blocked y-range at x_h is [(c-h)/t, (c+h)/t] for an eye at y = 0.
    # Getting this wrong is what made the first asymmetric search accept nothing.
    # The hidden wall goes far enough back that the shadow magnifies: at x_h the
    # blocked interval is [(c-h)/t, (c+h)/t] with t = (wall_x-ex)/(x_h-ex), and a
    # small t is what lets a blocker sit inside the shadow *and* lie across the
    # +y detour's sight line.  Placed just behind the wall it cannot do both.
    hidden_x = float(rng.uniform(wall_x + 0.8, target_x - 0.5))
    t = (wall_x - lost_x) / (hidden_x - lost_x)
    lo = (wall_center_y - wall_half_y) / t
    hi = (wall_center_y + wall_half_y) / t
    lo = max(lo, 0.05)
    hi = min(hi, LATERAL_REACH_M)
    # aim at where the +y detour's sight line actually crosses this x
    crossing = 1.8 * (1.0 - (hidden_x - lost_x) / (target_x - lost_x))
    span = max(0.0, hi - lo)
    y0 = float(np.clip(crossing - 0.20, lo, max(lo, hi - 0.4)))
    y1 = float(min(hi, y0 + 0.45)) if span >= 0.4 else float(hi)
    return PairParams(
        pair_id=f"mem{index:03d}", wall_x=wall_x, wall_half_y=wall_half_y,
        wall_height=wall_height, wall_width=wall_width, wall_center_y=wall_center_y,
        target_x=target_x, target_y=0.0, target_z=target_z,
        lost_x=lost_x, lost_y=0.0, hidden_x=hidden_x,
        hidden_y0=y0, hidden_y1=y1, hidden_height=wall_height)


def search(n_pairs: int, seed: int, attempts: int) -> tuple[list[PairParams], list[dict]]:
    rng = np.random.default_rng(seed)
    accepted: list[PairParams] = []
    log: list[dict] = []
    for attempt in range(attempts):
        if len(accepted) >= n_pairs:
            break
        params = sample_params(rng, attempt)
        terrain_a = build_terrain(params, "a")
        terrain_b = build_terrain(params, "b")
        checks = validate_pair(params, terrain_a, terrain_b)
        entry = {"attempt": attempt, "pair_id": params.pair_id, "passed": checks["passed"],
                 "failed": [k for k in checks["required"] if not checks[k]],
                 "params": asdict(params)}
        log.append(entry)
        if checks["passed"]:
            accepted.append(params)
    return accepted, log


def scenes_for(params: PairParams) -> list:
    return [scene_for_side(params, "a"), scene_for_side(params, "b")]


def write_descriptor(params_list: list[PairParams], log: list[dict], path: Path) -> None:
    payload = {
        "task": "TASK3_experiment5_memory_pairs", "n_attempts": len(log),
        "n_accepted": len(params_list),
        "acceptance_criteria": [
            "the decision point hides the target by terrain, inside the FOV, in both worlds",
            "the hidden feature lies entirely inside the shadow of the visible wall",
            "the hidden feature exists in world A only",
            "every slab of world B is also a slab of world A (same top): the visible scene is shared",
            "the visible wall blocks the direct line, so a lateral detour is required",
            "the hidden feature covers exactly one detour side and lies within lateral reach",
        ],
        "pairs": [asdict(p) for p in params_list], "attempts": log,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_descriptor(path: Path | None = None) -> list[PairParams]:
    payload = json.loads((path or DEFAULT_DESCRIPTOR).read_text(encoding="utf-8"))
    return [PairParams(**entry) for entry in payload["pairs"]]


def main() -> None:
    p = argparse.ArgumentParser(description="Search for TASK3 H2 memory pairs")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--attempts", type=int, default=300)
    p.add_argument("--output", type=Path, default=DEFAULT_DESCRIPTOR)
    args = p.parse_args()

    accepted, log = search(args.pairs, args.seed, args.attempts)
    print(f"attempts={len(log)} accepted={len(accepted)}")
    reason_counts: dict[str, int] = {}
    for entry in log:
        for reason in entry["failed"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {count:4d}x  {reason}")
    for params in accepted[:12]:
        print(f"  {params.pair_id}: wall_x={params.wall_x:.2f} h={params.wall_height:.2f} "
              f"half_y={params.wall_half_y:.2f} target=({params.target_x:.2f},"
              f"{params.target_z:.2f}) lost_x={params.lost_x:+.2f} "
              f"hidden x={params.hidden_x:.2f} y=[{params.hidden_y0:.2f},"
              f"{params.hidden_y1:.2f}] h={params.hidden_height:.2f}")
    write_descriptor(accepted, log, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
