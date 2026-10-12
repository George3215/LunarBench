"""Is H1 a property of the terrain family, or only of the pairs we hand-built?

H1's evidence is strong but it was constructed by hand for one mechanism.  The
obvious review question is whether observation-equivalent / opposite-action pairs
exist *structurally*, or whether the construction simply picked favourable cases.

This module answers that by testing, per mechanism, the property that H1 requires.
H1 needs two worlds that look identical at the decision point yet whose winning
actions point in **opposite lateral directions**.  That is only possible if the
mechanism can produce a world whose winning set is **one-sided** - every action
that reveals the target moves the robot the same way in y.  So:

* **Level 0** ``y_symmetry``   does ``h(x, y) == h(x, -y)``?  A y-symmetric height
  field cannot express a directional aperture at all, so it cannot support H1.
* **Level 1** ``one_sided``    across many target placements, how often is the
  geometric winning set non-empty and entirely on one side?
* **Level 2** ``mirrorable``   do one-sided worlds come in mirror pairs, which is
  what lets both members of a pair be constructed from the same visible occluder?

A mechanism that never produces a one-sided winning set cannot support H1, and
this audit says so with a count rather than an argument.  Note the direction of
the claim: a high one-sided rate is **necessary but not sufficient** - it shows the
mechanism *can* express the ambiguity, not that an observation-equivalent pair was
found for it.  Only ``window_wall`` has actually been searched end to end.

    python tasks/task3_reobservability/mechanism_audit.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
    from .experiment import visibility
    from .geometry_gap import kinematic_pose, visibility_at
except ImportError:
    import api
    import episode as ep
    from experiment import visibility
    from geometry_gap import kinematic_pose, visibility_at

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "mechanism_audit.json"

#: Target y offsets tried per mechanism.  A y-symmetric mechanism will produce the
#: same winner set for +y and -y, which is exactly what the audit is looking for.
TARGET_Y_OFFSETS = (-2.5, -1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5, 2.5)
TARGET_X_OFFSETS = (-0.6, 0.0, 0.6)


def base_scenes() -> dict:
    return {k: s for k, s in api.scene_catalogue().items()
            if not k.startswith(("hist", "mem"))}


def y_symmetry(terrain, samples: int = 120) -> float:
    """Max ``|h(x, y) - h(x, -y)|`` over the reachable region.

    A value near zero means the height field cannot express a directional
    aperture, so no amount of pair search will find a one-sided world in it.
    """
    xs = np.linspace(terrain.x[0] + 1.0, terrain.x[-1] - 1.0, samples // 2)
    ys = np.linspace(0.0, min(4.0, terrain.y[-1] - 0.5), samples // 2)
    worst = 0.0
    for x in xs:
        for y in ys:
            a = float(terrain.height(x, y))
            b = float(terrain.height(x, -y))
            if a < -10.0 or b < -10.0:
                continue
            worst = max(worst, abs(a - b))
    return worst


def signature(scene, terrain) -> dict:
    """Geometric winner set at the decision point, with lateral directions.

    The target is taken from the built environment, not rebuilt from the scene:
    ``target_z_absolute`` is ``None`` for most scenes, because the anchor height is
    derived from the terrain.  Rebuilding it from the scene silently produced a
    target at ``z = NaN``, which made every winner set empty and made the whole
    audit report "no mechanism supports H1".
    """
    environment = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table(),
                                        terrain=terrain)
    try:
        observation = environment.reset()
        target = np.asarray(environment.target, dtype=float)
        winners, directions = [], {}
        for action_id in api.ACTION_IDS:
            primitive = environment.action_table[action_id]
            predicted, yaw = kinematic_pose(observation.pose, primitive, terrain)
            if predicted is None:
                continue
            if visibility_at(terrain, target, predicted, yaw):
                winners.append(action_id)
                directions[action_id] = float(primitive.vy * primitive.duration_s)
        return {"winners": winners, "lateral": directions,
                "target": np.round(target, 3).tolist(),
                "hidden_at_decision": not visibility(
                    terrain, observation.pose.xyz, observation.pose.yaw, 0.0,
                    target)["visible_oracle"]}
    finally:
        environment.close()


def sidedness(entry: dict) -> str:
    """Is the winning set entirely on one lateral side, both, or empty?"""
    winners = entry["winners"]
    if not winners:
        return "none"
    lat = [entry["lateral"][a] for a in winners]
    if all(v > 0.05 for v in lat):
        return "plus_y"
    if all(v < -0.05 for v in lat):
        return "minus_y"
    return "both"


def audit_scene(name: str, scene, trials: int) -> dict:
    from benchmark import build_scene_terrain
    terrain = build_scene_terrain(scene)
    sym = y_symmetry(terrain)

    counts = {"none": 0, "plus_y": 0, "minus_y": 0, "both": 0}
    hidden_counts = {"none": 0, "plus_y": 0, "minus_y": 0, "both": 0}
    hidden_count = 0
    trials_run = 0
    for dx in TARGET_X_OFFSETS:
        for dy in TARGET_Y_OFFSETS:
            tx, ty = scene.target_xy
            probe = replace(scene, target_xy=(tx + dx, ty + dy),
                            name=f"{name}_probe")
            try:
                terrain_probe = build_scene_terrain(probe)
                entry = signature(probe, terrain_probe)
            except Exception:
                continue
            trials_run += 1
            side = sidedness(entry)
            counts[side] += 1
            # H1's precondition is that the target is *hidden* at the decision
            # point; only those trials can carry the ambiguity at all.
            if entry["hidden_at_decision"]:
                hidden_count += 1
                hidden_counts[side] += 1
            if trials_run >= trials:
                break
        if trials_run >= trials:
            break

    one_sided = hidden_counts["plus_y"] + hidden_counts["minus_y"]
    return {
        "mechanism": name,
        "y_symmetry_m": round(float(sym), 5),
        "y_symmetric": bool(sym < 1e-3),
        "trials": trials_run,
        "hidden_at_decision": hidden_count,
        "winner_sets": counts,
        "winner_sets_when_hidden": hidden_counts,
        "one_sided_rate_when_hidden": (round(one_sided / hidden_count, 4)
                                       if hidden_count else None),
        "supports_h1_level2": bool(one_sided > 0),
        "mirror_pairs_available": bool(hidden_counts["plus_y"] > 0
                                       and hidden_counts["minus_y"] > 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="H1 mechanism support audit")
    p.add_argument("--trials", type=int, default=27)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    rows = []
    for name, scene in base_scenes().items():
        row = audit_scene(name, scene, args.trials)
        rows.append(row)
        marker = "SUPPORTS" if row["supports_h1_level2"] else "no       "
        print(f"  {name:14s} {marker}  y-sym={row['y_symmetry_m']:.4f} m  "
              f"hidden={row['hidden_at_decision']:2d} one-sided={row['one_sided_rate_when_hidden']}  "
              f"sets(hidden)={row['winner_sets_when_hidden']}  mirror={row['mirror_pairs_available']}",
              flush=True)

    supported = [r["mechanism"] for r in rows if r["supports_h1_level2"]]
    symmetric = [r["mechanism"] for r in rows if r["y_symmetric"]]
    print(f"\nmechanisms that can express a one-sided winner set: {supported}")
    print(f"y-symmetric height fields (cannot express a directional aperture): "
          f"{symmetric}")

    report = {"task": "TASK3_h1_mechanism_audit",
              "note": "high one-sided rate is necessary but not sufficient for H1; "
                      "only window_wall has been searched end to end",
              "mechanisms_supporting_level2": supported,
              "y_symmetric_mechanisms": symmetric,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
