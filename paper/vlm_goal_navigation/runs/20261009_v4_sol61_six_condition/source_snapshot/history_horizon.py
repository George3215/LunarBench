"""TASK3 Experiment 4: how much history is actually necessary?

Experiment 1 showed that the current observation alone does not determine the
optimal re-observation action.  That leaves the question this module answers:

    How much history, exactly, does the decision need?

The horizon is defined once and never redefined per condition:

    H_k = {o_{t-k+1}, ..., o_t}          an observation is recorded at 10 Hz

An observation is ``o = (pose, target_seen, range, bearing, elevation)``.  The
robot is *not* handed the target's world position: it has to recover it from an
observation in which the target was actually seen.  That recovery is what makes
the horizon matter - a window that does not reach back past the last sighting
contains no information about where the target is.

Fair-comparison property: the feature vector has **the same dimension for every
horizon**.  The window is summarised into a fixed number of slots (span,
displacement, turn, whether the target was seen, age of the sighting, the
remembered target in the current body frame).  Only how far back the window
reaches changes, never the feature definition, so a reviewer cannot object that
H4 was given a richer representation than H0.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .benchmark import BENCHMARK_ACTIONS, build_scene_terrain
    from .experiment import target_anchor, visibility
    from .scenes import ROBOT_BASE_HEIGHT_M, scene_for
except ImportError:
    from benchmark import BENCHMARK_ACTIONS, build_scene_terrain
    from experiment import target_anchor, visibility
    from scenes import ROBOT_BASE_HEIGHT_M, scene_for

ROOT = Path(__file__).resolve().parents[2]
ACTION_NAMES = sorted(BENCHMARK_ACTIONS)
SAMPLE_HZ = 10.0
SUBSAMPLE_S = 0.5
# Measured forward speed of the vendored policy on flat ground, used to lay the
# approach trace out in time.
APPROACH_SPEED_MPS = 0.8
HORIZONS = [0, 1, 2, 4, 8, 16, 32, 64, 10 ** 6]
HORIZON_LABELS = ["H0", "H1", "H2", "H4", "H8", "H16", "H32", "H64", "Hfull"]


@dataclass
class Observation:
    t: float
    xyz: np.ndarray
    yaw: float
    seen: bool
    # Everything below is what a real robot gets from a detection: a range and a
    # bearing, not a world position.
    rng: float
    bearing: float
    elevation: float


def _observe(terrain, xyz, yaw, target) -> Observation:
    """One observation at a robot base pose, from the line-of-sight oracle."""
    reading = visibility(terrain, np.asarray(xyz, dtype=float), yaw, 0.0, target)
    eye = np.asarray(xyz, dtype=float) + np.array([0.0, 0.0, 0.16])
    d = target - eye
    return Observation(
        t=float("nan"), xyz=np.asarray(xyz, dtype=float), yaw=float(yaw),
        seen=bool(reading["visible_oracle"]), rng=float(reading.get("range_m", np.linalg.norm(d))),
        bearing=math.atan2(d[1], d[0]) - yaw, elevation=math.atan2(d[2], float(np.hypot(d[0], d[1]))),
    )


def _target_from_observation(o: Observation) -> np.ndarray:
    """Recover the target's world position from an observation that saw it.

    This is the only route by which any horizon can know where the target is.
    """
    h = o.rng * math.cos(o.elevation)
    angle = o.yaw + o.bearing
    eye = o.xyz + np.array([0.0, 0.0, 0.16])
    return np.array([eye[0] + h * math.cos(angle), eye[1] + h * math.sin(angle),
                     eye[2] + o.rng * math.sin(o.elevation)])


def approach_trace(terrain, scene, target) -> list[Observation]:
    """Robot walking from where it last saw the target up to the lost anchor.

    A straight line in the plane from the history vantage to the lost state,
    walked at the measured policy speed and sampled at ``SAMPLE_HZ``.  The
    heading starts along the direction of travel and turns smoothly to the
    scene's lost-state heading, so the trace joins the recorded rollout without
    a yaw discontinuity.

    Note the vantage is *not* on the lost state's ``y``: for the paired-history
    worlds the robot last saw the target from off to one side, so taking only
    the vantage's ``x`` and forcing ``y = lost_y`` would start the trace
    somewhere the target was never visible.
    """
    if scene.history_xy is not None:
        start = np.array([float(scene.history_xy[0]), float(scene.history_xy[1])])
    else:
        start = np.array([float(terrain.x[0]) + 0.3, float(scene.lost_xy[1])])
    end = np.array([float(scene.lost_xy[0]), float(scene.lost_xy[1])])
    distance = float(np.linalg.norm(end - start))
    steps = max(2, int(round(distance / APPROACH_SPEED_MPS * SAMPLE_HZ)) + 1)
    travel_yaw = math.atan2(end[1] - start[1], end[0] - start[0])
    turn = math.atan2(math.sin(scene.yaw_rad - travel_yaw),
                      math.cos(scene.yaw_rad - travel_yaw))
    trace = []
    for i, alpha in enumerate(np.linspace(0.0, 1.0, steps)):
        xy = start + alpha * (end - start)
        yaw = travel_yaw + alpha * turn
        obs = _observe(terrain, [xy[0], xy[1], terrain.height(xy[0], xy[1]) + ROBOT_BASE_HEIGHT_M],
                       yaw, target)
        obs.t = (i - (steps - 1)) / SAMPLE_HZ  # negative: time before the lost anchor
        trace.append(obs)
    return trace


def rollout_trace(action: dict, terrain, target) -> list[Observation]:
    """Observations along one recorded rollout.

    The recorder stores visibility, bearing, pose and yaw; range and elevation
    follow from the pose and the target, so the trace carries exactly the same
    fields as the oracle-sampled approach.
    """
    out = []
    for c in action["curve"]:
        xyz = np.asarray(c["xyz"], dtype=float)
        eye = xyz + np.array([0.0, 0.0, 0.16])
        d = target - eye
        out.append(Observation(
            t=float(c["t_s"]), xyz=xyz, yaw=float(c["yaw_rad"]),
            seen=bool(c["visible"]), rng=float(np.linalg.norm(d)),
            bearing=float(c["bearing_rad"]),
            elevation=math.atan2(d[2], float(np.hypot(d[0], d[1])))))
    return out


def _action_vector(name: str) -> np.ndarray:
    p = BENCHMARK_ACTIONS[name]
    onehot = np.zeros(len(ACTION_NAMES), dtype=float)
    onehot[ACTION_NAMES.index(name)] = 1.0
    return np.concatenate([[p.vx, p.vy, p.wz, p.duration_s], onehot])


ACTION_VECTORS = {n: _action_vector(n) for n in ACTION_NAMES}


def window_features(history: list[Observation], k: int) -> np.ndarray:
    """Fixed-dimension summary of the last ``k`` observations.

    The dimension is 5 and does not depend on ``k``; that is what makes the
    horizon comparison fair.
    """
    if k <= 0 or not history:
        return np.zeros(5, dtype=float)
    win = history[-k:]
    now = history[-1]
    span = max(0.0, now.t - win[0].t)
    # displacement of the window start relative to now, in the current body frame
    d = win[0].xyz - now.xyz
    c, s = math.cos(-now.yaw), math.sin(-now.yaw)
    dx, dy = d[0] * c - d[1] * s, d[0] * s + d[1] * c
    turn = math.atan2(math.sin(win[0].yaw - now.yaw), math.cos(win[0].yaw - now.yaw))
    return np.array([math.log1p(len(win)), span, dx, dy, turn], dtype=float)


def memory_features(history: list[Observation], k: int) -> np.ndarray:
    """What the robot knows about the target, using only the last ``k`` observations.

    Zeroed with the first slot set to 0 when the window contains no sighting, so
    "no memory" is representable rather than silently looking like a target at
    the origin.
    """
    out = np.zeros(12, dtype=float)
    if k <= 0 or not history:
        return out
    win = history[-k:]
    now = history[-1]
    idx = None
    for i in range(len(win) - 1, -1, -1):
        if win[i].seen:
            idx = i
            break
    if idx is None:
        return out
    sighting = win[idx]
    tgt = _target_from_observation(sighting)
    eye = now.xyz + np.array([0.0, 0.0, 0.16])
    d = tgt - eye
    c, s = math.cos(-now.yaw), math.sin(-now.yaw)
    body = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2]])
    bearing = math.atan2(d[1], d[0]) - now.yaw
    out[0] = 1.0                                   # seen inside the window
    out[1] = max(0.0, now.t - sighting.t)          # age of the sighting
    out[2] = sighting.xyz[0] - now.xyz[0]          # where the robot was when it saw it
    out[3] = sighting.xyz[1] - now.xyz[1]
    out[4] = math.atan2(math.sin(sighting.yaw - now.yaw), math.cos(sighting.yaw - now.yaw))
    out[5:8] = body
    out[8] = float(np.linalg.norm(d))
    out[9] = math.sin(bearing)
    out[10] = math.cos(bearing)
    out[11] = 1.0 if sighting.seen else 0.0
    return out


def pose_features(o: Observation) -> np.ndarray:
    return np.array([o.xyz[0], o.xyz[1], o.xyz[2], math.sin(o.yaw), math.cos(o.yaw)])


def _scene_index(source: dict) -> dict:
    """Map each recorded world key to its Scene and its group.

    The paired-history worlds are not named profiles: their terrain comes from
    parameter sets found by the search in ``history_pairs``.  Those have to be
    rebuilt from the descriptor rather than looked up by name.  The benchmark
    report is keyed by scene name directly, and the history report by
    ``<pair_id>_<left|right>``.
    """
    out: dict[str, tuple] = {}
    if source.get("pairs"):
        try:
            from .history_pairs import load_descriptor, scenes_for
        except ImportError:
            from history_pairs import load_descriptor, scenes_for
        wanted = {p["pair_id"] for p in source["pairs"]}
        for params in load_descriptor():
            if params.pair_id in wanted:
                for scene in scenes_for(params):
                    out[scene.name] = (scene, params.pair_id)
    for key, entry in (source.get("worlds") or source.get("scenes") or {}).items():
        if key in out:
            continue
        name = entry.get("scene") or entry.get("terrain")
        if not name:
            name = key.rsplit("_", 1)[0] if key.endswith(("_left", "_right")) else key
        out[key] = (scene_for(name), entry.get("pair_id") or name)
    return out


def build_dataset(report_paths: list[Path]) -> list[dict]:
    """One row per (sample, horizon); features are built once per sample."""
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    rows: list[dict] = []
    for source in reports:
        index = _scene_index(source)
        entries = source.get("worlds") or source.get("scenes") or {}
        for key, entry in entries.items():
            scene, group = index[key]
            terrain = build_scene_terrain(scene)
            target = target_anchor(terrain, scene)
            prefix = approach_trace(terrain, scene, target)
            for action in entry["actions"]:
                curve = action["curve"]
                if not curve:
                    continue
                tail = rollout_trace(action, terrain, target)
                visible = [bool(c["visible"]) for c in curve]
                dt = curve[1]["t_s"] - curve[0]["t_s"] if len(curve) > 1 else SUBSAMPLE_S
                stride = max(1, int(round(SUBSAMPLE_S / dt))) if dt > 0 else 1
                for i, c in enumerate(curve):
                    if i % stride:
                        continue
                    # Only states where the target is currently hidden.  A sample
                    # whose label is "the target becomes visible later" while the
                    # target is visible *right now* is not a decision problem -
                    # the current observation alone answers it - and including
                    # those rows lets the model learn "visible now => success"
                    # instead of a counterfactual.  The decision states (i == 0)
                    # are hidden by construction and always kept.
                    if visible[i]:
                        continue
                    history = prefix + tail[:i + 1]
                    rows.append({
                        "group": group, "world": key, "action": action["action"],
                        "source": ("history" if "history" in source["task"] else "benchmark"),
                        "is_decision_state": bool(i == 0),
                        "pose": pose_features(tail[i]),
                        "target_body": target_body_features(tail[i].xyz, tail[i].yaw, target),
                        "action_vec": ACTION_VECTORS[action["action"]],
                        "history": history,
                        "label": int(any(v for v in visible[i:])),
                    })
    return rows


def target_body_features(pose: np.ndarray, yaw: float, target: np.ndarray) -> np.ndarray:
    """The true target in the robot's body frame; a privileged ceiling, not a policy input."""
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([dx * c - dy * s, dx * s + dy * c, target[2] - pose[2],
                     math.hypot(dx, dy)], dtype=float)


def _matrix(rows: list[dict], k: int, block: str = "all") -> np.ndarray:
    """Feature matrix.  ``block`` selects which information sources are live.

    Zeroing a block keeps the dimension identical, so the four conditions differ
    only in what they are allowed to know:

    ``all``           everything the horizon provides
    ``memory``        target memory only; recent-motion summary zeroed
    ``motion``        recent-motion summary only; target memory zeroed
    ``oracle_target`` the true target position with NO history at all - the
                      ceiling, which says whether the task is solvable once the
                      target position is known, independently of any horizon
    """
    pose = np.stack([r["pose"] for r in rows])
    act = np.stack([r["action_vec"] for r in rows])
    if block == "oracle_target":
        return np.hstack([pose, act, np.stack([r["target_body"] for r in rows])])
    win = np.stack([window_features(r["history"], k) for r in rows])
    mem = np.stack([memory_features(r["history"], k) for r in rows])
    if block == "memory":
        win = np.zeros_like(win)
    elif block == "motion":
        mem = np.zeros_like(mem)
    return np.hstack([pose, act, win, mem])


def out_of_fold(X, y, groups, seed=7):
    from sklearn.ensemble import HistGradientBoostingClassifier

    pred = np.zeros(len(y), dtype=float)
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if not test.any() or len(np.unique(y[train])) < 2:
            pred[test] = float(y[train].mean()) if train.any() else 0.0
            continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                             learning_rate=0.08, random_state=seed)
        clf.fit(X[train], y[train])
        pred[test] = clf.predict_proba(X[test])[:, 1]
    return pred


def evaluate(rows: list[dict], k: int, block: str = "all", seed: int = 7) -> dict:
    from sklearn.metrics import roc_auc_score

    X = _matrix(rows, k, block)
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    probs = out_of_fold(X, y, groups, seed)

    by_world: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r["is_decision_state"]:
            by_world.setdefault(r["world"], []).append(i)
    chosen, hits, match = [], 0, 0
    per_source: dict[str, list[int]] = {}
    # Split the decision states by whether the window actually reaches back to a
    # sighting.  This is the mechanism claim: history helps exactly when it
    # contains the sighting, and not otherwise.
    with_memory, without_memory = [], []
    for world, idx in by_world.items():
        best = max(idx, key=lambda i: probs[i])
        pick = rows[best]["action"]
        chosen.append(pick)
        ok = int(bool(rows[best]["label"]))
        hits += ok
        per_source.setdefault(rows[best]["source"], []).append(ok)
        winners = {rows[i]["action"] for i in idx if rows[i]["label"]}
        if pick in winners:
            match += 1
        (with_memory if memory_features(rows[best]["history"], k)[0] > 0
         else without_memory).append(ok)
    n = len(by_world)
    return {
        "horizon": k,
        "n_features": int(X.shape[1]),
        "oof_auc": round(float(roc_auc_score(y, probs)), 4),
        "rsr": round(hits / n, 4),
        "optimal_action_match": round(match / n, 4),
        "rsr_by_source": {s: round(float(np.mean(v)), 4) for s, v in per_source.items()},
        "states_with_memory": len(with_memory),
        "rsr_with_memory": (round(float(np.mean(with_memory)), 4) if with_memory else None),
        "states_without_memory": len(without_memory),
        "rsr_without_memory": (round(float(np.mean(without_memory)), 4)
                               if without_memory else None),
        "n_worlds": n,
        "chosen_actions": chosen,
    }


def sighting_gap(rows: list[dict]) -> dict:
    """How many steps back the last sighting is, at each decision state."""
    gaps = {}
    for r in rows:
        if not r["is_decision_state"]:
            continue
        steps = None
        for back, o in enumerate(reversed(r["history"]), start=1):
            if o.seen:
                steps = back
                break
        gaps[r["world"]] = steps
    return gaps


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 Experiment 4: history horizon")
    p.add_argument("--reports", nargs="*", type=Path, default=[
        Path("tasks/task3_reobservability/generated/history_necessity.json"),
        Path("tasks/task3_reobservability/generated/benchmark.json")])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/history_horizon.json"))
    args = p.parse_args()

    rows = build_dataset(args.reports)
    print(f"samples={len(rows)} groups={len({r['group'] for r in rows})} "
          f"positive_rate={np.mean([r['label'] for r in rows]):.3f}")
    gaps = sighting_gap(rows)
    finite = [v for v in gaps.values() if v is not None]
    print(f"decision states={len(gaps)}  last-sighting gap in steps: "
          f"min={min(finite) if finite else None} "
          f"median={int(np.median(finite)) if finite else None} "
          f"max={max(finite) if finite else None}  never-seen={sum(1 for v in gaps.values() if v is None)}")

    results: dict[str, list] = {}
    for block in ("all", "memory", "motion", "oracle_target"):
        series, prev = [], None
        horizons = [1] if block == "oracle_target" else HORIZONS
        labels = ["no-history"] if block == "oracle_target" else HORIZON_LABELS
        for label, k in zip(labels, horizons):
            r = evaluate(rows, k, block, args.seed)
            r["label"] = label
            r["block"] = block
            r["delta_rsr"] = None if prev is None else round(r["rsr"] - prev, 4)
            prev = r["rsr"]
            series.append(r)
            print(f"  {block:14s} {label:9s} dims={r['n_features']:3d}  "
                  f"AUC={r['oof_auc']:.4f}  RSR={r['rsr']:.4f}  "
                  f"match={r['optimal_action_match']:.4f}  dRSR={r['delta_rsr']}  "
                  f"| mem-avail {r['states_with_memory']:2d} -> "
                  f"{r['rsr_with_memory']}  no-mem {r['states_without_memory']:2d} -> "
                  f"{r['rsr_without_memory']}")
        results[block] = series
        print()

    out = {"task": "TASK3_experiment4_history_horizon",
           "n_samples": len(rows), "n_groups": len({r["group"] for r in rows}),
           "sample_hz": SAMPLE_HZ, "approach_speed_mps": APPROACH_SPEED_MPS,
           "sighting_gap_steps": gaps, "blocks": results,
           "sources": [str(p) for p in args.reports]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
