"""TASK3 H3: does geometric visibility predict physical re-observability?

This is the experiment that decides whether a World Model is needed at all.  For
every candidate action at every decision state we compute two things:

``G(s, a)``
    Would the target be visible if the robot reached the **ideal** pose this
    action nominally produces?  Computed with a kinematic displacement and the
    line-of-sight oracle.  This is what an active-perception / next-best-view
    method assumes it can evaluate.
``P(s, a)``
    Is the target actually re-observed when the primitive is **executed** by the
    controller?

The interesting cell is ``G=1, P=0``: geometry says the viewpoint is good and the
robot does not see the target.  The module also records the **pose error** between
the predicted and the executed end pose, so the gap is attributed rather than
asserted - if ``G != P`` the reason should be visible in the pose error, and if it
is not, the oracle itself is the problem.

Reported:

* confusion matrix over all (state, action) pairs
* ``G=1, P=0`` rate, and the same rate conditioned on the terrain
* pose prediction error (forward, lateral, yaw, height) split by agreement
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
    from .experiment import visibility
except ImportError:
    import api
    import episode as ep
    from experiment import visibility

ROOT = Path(__file__).resolve().parents[2]
BASE_Z = 0.30


def kinematic_pose(pose, primitive, terrain):
    """The ideal end pose this action nominally produces, or ``None`` off-terrain.

    Nominal displacement and yaw rate, then dropped onto the terrain.  This is
    exactly the model ``NBV`` uses, and exactly what an active-perception
    candidate evaluation assumes.

    Returning ``None`` off the terrain matters: the slab height oracle returns a
    large negative sentinel for a point no slab covers, so without the bounds
    check a candidate that would drive off the edge produces a pose "error" of
    a billion metres and poisons the attribution statistics.
    """
    dx = primitive.vx * primitive.duration_s
    dy = primitive.vy * primitive.duration_s
    x = pose.x + dx * math.cos(pose.yaw) - dy * math.sin(pose.yaw)
    y = pose.y + dx * math.sin(pose.yaw) + dy * math.cos(pose.yaw)
    yaw = pose.yaw + primitive.wz * primitive.duration_s
    if not (terrain.x[0] <= x <= terrain.x[-1] and terrain.y[0] <= y <= terrain.y[-1]):
        return None, float(yaw)
    z = float(terrain.height(x, y)) + BASE_Z
    if not np.isfinite(z) or z < -10.0:
        return None, float(yaw)
    return np.array([x, y, z]), float(yaw)


def visibility_at(terrain, target, xyz, yaw) -> bool:
    return bool(visibility(terrain, np.asarray(xyz, dtype=float), float(yaw), 0.0,
                           target)["visible_oracle"])


def measure(scene, controller: str, ground: str) -> list[dict]:
    environment = api.MuJoCoEnvironment(scene, controller_name=controller,
                                        ground=ground,
                                        action_table=ep.protocol_action_table())
    try:
        observation = environment.reset()
        target = environment.target
        terrain = environment.terrain
        rows = []
        for action_id in api.ACTION_IDS:
            primitive = environment.action_table[action_id]
            predicted_xyz, predicted_yaw = kinematic_pose(observation.pose, primitive,
                                                          terrain)
            off_terrain = predicted_xyz is None
            g = (False if off_terrain
                 else visibility_at(terrain, target, predicted_xyz, predicted_yaw))

            rollout = environment.rollout(action_id)
            if rollout.curve:
                last = rollout.curve[-1]
                actual_xyz = np.asarray(last["xyz"], dtype=float)
                actual_yaw = float(last["yaw_rad"])
            else:
                actual_xyz, actual_yaw = predicted_xyz, predicted_yaw
            p = bool(rollout.reobserved)

            # pose prediction error, in the *body* frame of the start pose
            if off_terrain or actual_xyz is None or not np.all(np.isfinite(actual_xyz)):
                forward_err = lateral_err = yaw_err = height_err = float("nan")
                pose_err = float("nan")
            else:
                d = actual_xyz - predicted_xyz
                c, s = math.cos(-observation.pose.yaw), math.sin(-observation.pose.yaw)
                forward_err = d[0] * c - d[1] * s
                lateral_err = d[0] * s + d[1] * c
                yaw_err = math.atan2(math.sin(actual_yaw - predicted_yaw),
                                     math.cos(actual_yaw - predicted_yaw))
                height_err = actual_xyz[2] - predicted_xyz[2]
                pose_err = float(np.linalg.norm(d))
            rows.append({
                "terrain": scene.name, "action": action_id,
                "G": int(g), "P": int(p),
                "off_terrain": bool(off_terrain),
                "G_actual": int(visibility_at(terrain, target, actual_xyz, actual_yaw)),
                "forward_err_m": round(float(forward_err), 4),
                "lateral_err_m": round(float(lateral_err), 4),
                "yaw_err_rad": round(float(yaw_err), 4),
                "height_err_m": round(float(height_err), 4),
                "pose_err_m": round(float(pose_err), 4),
                "t_reobs_s": rollout.t_reobs_s,
                "fallen": bool(rollout.fallen),
            })
        return rows
    finally:
        environment.close()


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    cell = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for r in rows:
        cell[(r["G"], r["P"])] += 1
    agree = cell[(0, 0)] + cell[(1, 1)]
    gap = cell[(1, 0)]
    missed = cell[(0, 1)]

    def err(key, subset):
        values = [abs(r[key]) for r in subset
                  if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        if not values:
            return None
        return round(float(np.mean(values)), 4)

    gap_rows = [r for r in rows if r["G"] == 1 and r["P"] == 0]
    agree_rows = [r for r in rows if r["G"] == r["P"] and not r.get("off_terrain")]
    by_terrain: dict[str, dict] = {}
    for r in rows:
        t = by_terrain.setdefault(r["terrain"], {"n": 0, "G1P0": 0, "G1": 0})
        t["n"] += 1
        t["G1P0"] += int(r["G"] == 1 and r["P"] == 0)
        t["G1"] += int(r["G"] == 1)

    return {
        "n": n,
        "confusion": {"G0_P0": cell[(0, 0)], "G0_P1": cell[(0, 1)],
                      "G1_P0": cell[(1, 0)], "G1_P1": cell[(1, 1)]},
        "agreement_rate": round(agree / n, 4),
        "geometric_overconfidence_rate": round(gap / n, 4),
        "geometric_miss_rate": round(missed / n, 4),
        "gap_share_of_geometric_positives": round(gap / max(1, gap + cell[(1, 1)]), 4),
        "pose_error_gap_rows_m": err("pose_err_m", gap_rows),
        "pose_error_agree_rows_m": err("pose_err_m", agree_rows),
        "pose_error_breakdown_gap": {
            "forward_m": err("forward_err_m", gap_rows),
            "lateral_m": err("lateral_err_m", gap_rows),
            "yaw_rad": err("yaw_err_rad", gap_rows),
            "height_m": err("height_err_m", gap_rows),
        },
        "G_actual_equals_P_rate": round(
            float(np.mean([r["G_actual"] == r["P"] for r in rows])), 4),
        "by_terrain": by_terrain,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 H3: geometric vs physical visibility")
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--controller", default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/geometry_gap.json"))
    args = p.parse_args()

    catalogue = api.scene_catalogue()
    if args.scenes:
        catalogue = {s: catalogue[s] for s in args.scenes}

    rows: list[dict] = []
    for name, scene in catalogue.items():
        scene_rows = measure(scene, args.controller, args.ground)
        rows.extend(scene_rows)
        gap = sum(1 for r in scene_rows if r["G"] == 1 and r["P"] == 0)
        geo = sum(1 for r in scene_rows if r["G"] == 1)
        print(f"  {name:18s} G=1: {geo:2d}  of which P=0: {gap:2d}", flush=True)

    summary = summarise(rows)
    print("\n=== H3 geometry vs physical re-observability ===")
    print(f"  (state, action) pairs        : {summary['n']}")
    print(f"  confusion                    : {summary['confusion']}")
    print(f"  agreement rate               : {summary['agreement_rate']}")
    print(f"  G=1 but P=0 (gap)            : {summary['geometric_overconfidence_rate']}")
    print(f"  gap share of all G=1         : {summary['gap_share_of_geometric_positives']}")
    print(f"  G=0 but P=1                  : {summary['geometric_miss_rate']}")
    print(f"  mean pose error, gap rows    : {summary['pose_error_gap_rows_m']} m")
    print(f"  mean pose error, agree rows  : {summary['pose_error_agree_rows_m']} m")
    print(f"  pose error breakdown (gap)   : {summary['pose_error_breakdown_gap']}")
    print(f"  oracle-at-executed-pose == P : {summary['G_actual_equals_P_rate']}")

    report = {"task": "TASK3_experiment6_geometry_gap",
              "controller": args.controller, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
