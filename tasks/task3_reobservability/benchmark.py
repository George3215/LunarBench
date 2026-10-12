"""TASK3 Experiment 2: action-dependent target re-observability.

Experiment 1 asked whether terrain can hide a previously observed target, i.e.
whether V is non-monotonic along the robot's path.  Experiment 2 asks the next
question, and it is the one the research claim actually rests on:

    do different candidate actions really lead to different future observability?

For every scene the benchmark

1. records the oracle visibility profile V(x) along the nominal approach, so the
   non-monotonicity is a measured number rather than an assertion;
2. physically executes every candidate action from the frozen lost state and
   records the full curve V(a, t), not just a final boolean;
3. reduces each curve to RSR, T_reobs and motion cost;
4. applies three decision rules to the same table:
   A. reactive      - always go forward
   B. history-only  - go straight at the remembered target bearing, no lookahead
   C. oracle counterfactual - execute every candidate in imagination, take the best

The physical rollouts are driven by a vendored policy and are the ground truth;
the geometric oracle only labels them.  Nothing here is learned.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .controllers import Controller, Primitive, load_controller
    from .scenes import BENCHMARK_SCENES, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from .slabs import RESOLUTION_M, SlabTerrain, build_terrain
    from .terrain import Terrain, generate
    from .experiment import target_anchor, visibility
    from .go2_piper_camera import build_model
except ImportError:
    from controllers import Controller, Primitive, load_controller
    from scenes import BENCHMARK_SCENES, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from slabs import RESOLUTION_M, SlabTerrain, build_terrain
    from terrain import Terrain, generate
    from experiment import target_anchor, visibility
    from go2_piper_camera import build_model

ROOT = Path(__file__).resolve().parents[2]
FALLEN_UP_Z = 0.25
VISIBILITY_SAMPLE_S = 0.1

# Uniform directional action set: every translation gets the same time budget so
# motion cost and T_reobs are comparable across actions.
BENCHMARK_ACTIONS = {
    "forward": Primitive("forward", vx=1.0, duration_s=6.0,
                         note="walk forward along the current heading"),
    "back": Primitive("back", vx=-0.8, duration_s=6.0,
                      note="retreat along the current heading"),
    "left": Primitive("left", vy=0.6, duration_s=6.0, note="strafe left (+y)"),
    "right": Primitive("right", vy=-0.6, duration_s=6.0, note="strafe right (-y)"),
    "turn_left": Primitive("turn_left", wz=0.8, duration_s=2.0),
    "turn_right": Primitive("turn_right", wz=-0.8, duration_s=2.0),
    "toward": Primitive("toward", vx=0.9, duration_s=6.0, heading_hold=True,
                        note="walk toward the remembered target position"),
    # Baseline D: what an IMU plus "keep going the way I was facing" buys you.
    # No vision memory at all - it holds the world heading recorded at the loss
    # moment via closed-loop yaw correction.
    "heading_lock": Primitive("heading_lock", vx=0.9, duration_s=6.0, heading_hold=True,
                              note="hold the initial loss heading; no target memory"),
    # The same sidesteps with the heading held: no target memory involved, just a
    # controller that does not turn while it translates.
    "left_hold": Primitive("left_hold", vy=0.6, duration_s=6.0, yaw_hold=True,
                           note="strafe left while holding the heading"),
    "right_hold": Primitive("right_hold", vy=-0.6, duration_s=6.0, yaw_hold=True,
                            note="strafe right while holding the heading"),
}
ACTION_ORDER = ["forward", "back", "left", "right", "left_hold", "right_hold",
                "turn_left", "turn_right", "toward", "heading_lock"]


def build_scene_terrain(scene: Scene, seed: int = 7):
    """Slab terrain when the profile is in the kit, else the legacy generator."""
    if scene.terrain_params and scene.terrain_params.get("design_version") == 3:
        from terrain_design_v3 import build_terrain_v3
        return build_terrain_v3(scene.terrain_params)
    if scene.terrain_params is not None:
        return build_terrain(scene.terrain_profile, **scene.terrain_params)
    try:
        return build_terrain(scene.terrain_profile)
    except ValueError:
        return generate(scene.terrain_profile, seed)


def profile_for(scene: Scene, terrain, y: float = 0.0, step: float = 0.05,
                x_range: tuple[float, float] | None = None) -> dict:
    """Oracle visibility along a straight approach line at constant ``y``."""
    target = target_anchor(terrain, scene)
    lo, hi = x_range if x_range else (float(terrain.x[0]) + 0.2, float(terrain.x[-1]) - 0.2)
    xs = np.arange(lo, hi, step)
    rows = []
    for x in xs:
        robot = np.array([x, y, terrain.height(float(x), y) + ROBOT_BASE_HEIGHT_M], dtype=float)
        reading = visibility(terrain, robot, scene.yaw_rad, 0.0, target)
        rows.append({"x": round(float(x), 4), "visible": bool(reading["visible_oracle"]),
                     "blocked": bool(reading["blocked_by_terrain"]),
                     "in_fov": bool(reading["in_camera_fov"])})
    runs, start = [], None
    for i, row in enumerate(rows):
        if i == 0:
            start = i
            continue
        if row["visible"] != rows[i - 1]["visible"]:
            runs.append({"visible": rows[start]["visible"],
                         "x_from": rows[start]["x"], "x_to": rows[i - 1]["x"],
                         "length_m": round(rows[i - 1]["x"] - rows[start]["x"], 3)})
            start = i
    runs.append({"visible": rows[start]["visible"], "x_from": rows[start]["x"],
                 "x_to": rows[-1]["x"], "length_m": round(rows[-1]["x"] - rows[start]["x"], 3)})
    switches = sum(1 for a, b in zip(runs, runs[1:]) if a["visible"] == b["visible"])
    return {"target_xyz": target.tolist(), "y": y, "runs": runs,
            "visible_to_lost_switches": len(runs) - 1,
            "non_monotonic": len(runs) >= 3,
            "rows": rows}


def rollout_with_visibility(model, data, terrain, controller: Controller, primitive: Primitive,
                            target: np.ndarray, seconds: float | None = None,
                            heading_point: np.ndarray | None = None,
                            fallen_up_z: float = FALLEN_UP_Z,
                            sample_observer=None) -> dict:
    """Physically execute a primitive and sample the oracle visibility curve.

    ``heading_point`` overrides what a ``heading_hold`` primitive steers toward.
    The default is the remembered target position; baseline D points it along the
    heading recorded at the loss moment instead, so it uses no target memory.
    """
    import mujoco

    dt, decim = controller.timestep, controller.control_decimation
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    mujoco.mj_forward(model, data)
    start = np.array(data.qpos[:3], dtype=float)
    duration = primitive.duration_s if seconds is None else seconds
    controller.reset()
    target = np.asarray(target, dtype=float)
    target_angles = controller.default_angles.copy()
    steps = int(round(duration / dt))
    sample_every = max(int(VISIBILITY_SAMPLE_S / dt), 1)
    curve, path_cost = [], 0.0
    prev = start.copy()
    upright_min = 1.0
    fallen = False
    start_yaw = _yaw(data)
    for step in range(steps):
        if primitive.heading_hold:
            steer_to = target if heading_point is None else np.asarray(heading_point, dtype=float)
            bearing = math.atan2(steer_to[1] - data.qpos[1], steer_to[0] - data.qpos[0]) - _yaw(data)
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            wz = float(np.clip(primitive.heading_gain * bearing,
                               -primitive.heading_wz_max, primitive.heading_wz_max))
            command = [primitive.vx, primitive.vy, wz]
        elif primitive.yaw_hold:
            err = math.atan2(math.sin(_yaw(data) - start_yaw), math.cos(_yaw(data) - start_yaw))
            wz = float(np.clip(-primitive.yaw_hold_gain * err,
                               -primitive.yaw_hold_wz_max, primitive.yaw_hold_wz_max))
            command = [primitive.vx, primitive.vy, wz]
        else:
            command = [primitive.vx, primitive.vy, primitive.wz]
        torque = (target_angles - data.qpos[7:25]) * controller.kps - data.qvel[6:24] * controller.kds
        data.ctrl[:] = np.clip(torque, lo, hi)
        mujoco.mj_step(model, data)
        xyz = np.array(data.qpos[:3], dtype=float)
        path_cost += float(np.linalg.norm(xyz[:2] - prev[:2]))
        prev = xyz
        upright_min = min(upright_min, float(data.xmat[1].reshape(3, 3)[2, 2]))
        if step % decim == 0:
            target_angles = controller.act(data, command)
        if step % sample_every == 0:
            yaw_now = _yaw(data)
            reading = visibility(terrain, xyz, yaw_now, 0.0, target)
            curve.append({"t_s": round(float(data.time), 3),
                          "visible": bool(reading["visible_oracle"]),
                          "blocked": bool(reading["blocked_by_terrain"]),
                          "in_fov": bool(reading["in_camera_fov"]),
                          "bearing_rad": round(float(reading["bearing_rad"]), 4),
                          "yaw_rad": round(yaw_now, 4),
                          # Uprightness is recorded per sample because it is the
                          # only continuous stability signal available: a fall is a
                          # rare terminal event (0 in 720 single-action rollouts),
                          # so a binary "fallen" head has almost no positives to
                          # learn from, while the margin is present everywhere.
                          "upright": round(float(data.xmat[1].reshape(3, 3)[2, 2]), 4),
                          "xyz": np.round(xyz, 4).tolist()})
            if sample_observer is not None:
                # Observer must use a private MjData copy; sensor reads cannot
                # recompute dynamics in the live state (including solver caches).
                curve[-1]["sensor"] = sample_observer(model, data)
        if not np.isfinite(data.qpos).all() or upright_min < fallen_up_z:
            fallen = True
            break
    end = np.array(data.qpos[:3], dtype=float)
    visible_flags = [c["visible"] for c in curve]
    first = next((c for c in curve if c["visible"]), None)
    return {
        "action": primitive.name,
        "start_xyz": np.round(start, 4).tolist(),
        "end_xyz": np.round(end, 4).tolist(),
        "forward_m": float(end[0] - start[0]),
        "lateral_m": float(end[1] - start[1]),
        "height_gain_m": float(end[2] - start[2]),
        "motion_cost_m": round(path_cost, 4),
        "rsr": bool(any(visible_flags)),
        "t_reobs_s": None if first is None else first["t_s"],
        "fallen": bool(fallen),
        "upright_min": round(float(upright_min), 4),
        "simulated_s": round(float(data.time), 3),
        "visible_fraction": round(float(np.mean(visible_flags)) if visible_flags else 0.0, 4),
        "ever_visible_times_s": [c["t_s"] for c in curve if c["visible"]],
        "curve": curve,
    }


def _yaw(data) -> float:
    q = data.qpos[3:7]
    return float(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _target_bearing(lost_xyz, yaw: float, target: np.ndarray) -> float:
    bearing = math.atan2(target[1] - lost_xyz[1], target[0] - lost_xyz[0]) - yaw
    return math.atan2(math.sin(bearing), math.cos(bearing))


def history_only_action(bearing_rad: float) -> str:
    """Baseline B: act straight at the remembered target, with no lookahead."""
    deg = math.degrees(bearing_rad)
    if -45.0 <= deg <= 45.0:
        return "forward"
    if 45.0 < deg <= 135.0:
        return "left"
    if -135.0 <= deg < -45.0:
        return "right"
    return "back"


def choose_oracle_counterfactual(rows: list[dict]) -> str | None:
    """Baseline C: best RSR, then fastest re-observation, then cheapest motion."""
    ok = [r for r in rows if r["rsr"] and not r["fallen"]]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["t_reobs_s"] if r["t_reobs_s"] is not None else 1e9,
                                  r["motion_cost_m"]))["action"]


def surface_check(terrain, scene: Scene, samples: int = 9) -> dict:
    """Ray-cast the built MuJoCo ground and compare it with the oracle height.

    Rays that land on the target marker are skipped: the check is about the
    ground surface, not about the object standing on it.
    """
    import mujoco

    target = target_anchor(terrain, scene)
    model = build_model(terrain, target)
    data = mujoco.MjData(model)
    base = model.body("base_link").id
    qadr = int(model.jnt_qposadr[model.body_jntadr[base]])
    data.qpos[qadr:qadr + 2] = [-40.0, -40.0]
    mujoco.mj_forward(model, data)
    down = np.array([0.0, 0.0, -1.0])
    xs = np.linspace(float(terrain.x[0]) + 0.5, float(terrain.x[-1]) - 0.5, samples)
    ys = np.linspace(float(terrain.y[0]) + 0.5, float(terrain.y[-1]) - 0.5, samples)
    worst = 0.0
    skipped = 0
    for x in xs:
        for y in ys:
            gid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(model, data, np.array([float(x), float(y), 20.0]), down,
                                 None, 1, -1, gid)
            if int(gid[0]) < 0:
                continue
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) == "task3_target_geom":
                skipped += 1
                continue
            worst = max(worst, abs((20.0 - float(dist)) - float(terrain.height(float(x), float(y)))))
    return {"samples_per_axis": samples, "max_abs_error_m": round(worst, 4),
            "consistent": bool(worst <= 0.02), "target_hits_skipped": skipped,
            "slab_floor_z": None if not isinstance(terrain, SlabTerrain) else round(terrain.floor_z, 3),
            "slab_lowest_bottom_z": None if not isinstance(terrain, SlabTerrain) else round(terrain.bottom_z(), 3)}


def run_scene(scene: Scene, terrain, controller_name: str = "go2rl",
              ground: str = "auto", policy: Path | None = None,
              actions: list[str] | None = None) -> dict:
    import mujoco

    from locomotion import base_yaw, spawn

    target = target_anchor(terrain, scene)
    model = build_model(terrain, target, ground)
    controller = load_controller(controller_name, policy)
    model.opt.timestep = controller.timestep
    data = mujoco.MjData(model)

    profile = profile_for(scene, terrain)

    rows = []
    heading_point = None
    for name in (actions or ACTION_ORDER):
        primitive = BENCHMARK_ACTIONS[name]
        spawn(model, data, terrain, scene.lost_xy, controller, scene.yaw_rad)
        lost_xyz = np.array(data.qpos[:3], dtype=float)
        if name == "heading_lock":
            heading_point = lost_xyz + 10.0 * np.array(
                [math.cos(scene.yaw_rad), math.sin(scene.yaw_rad), 0.0])
        before = visibility(terrain, lost_xyz, base_yaw(data), 0.0, target)
        result = rollout_with_visibility(model, data, terrain, controller, primitive, target,
                                         heading_point=heading_point)
        result["visible_at_lost_state"] = bool(before["visible_oracle"])
        rows.append(result)

    lost_xyz = np.array(rows[0]["start_xyz"], dtype=float)
    bearing = _target_bearing(lost_xyz, scene.yaw_rad, target)
    reactive_action = "forward"
    history_action = history_only_action(bearing)
    counterfactual_action = choose_oracle_counterfactual(rows)
    by_name = {r["action"]: r for r in rows}

    def summarise(name):
        r = by_name.get(name)
        if r is None:
            return None
        return {"action": name, "rsr": r["rsr"], "t_reobs_s": r["t_reobs_s"],
                "motion_cost_m": r["motion_cost_m"], "fallen": r["fallen"]}

    best = min((r for r in rows if r["rsr"] and not r["fallen"]),
               key=lambda r: (r["t_reobs_s"] or 1e9, r["motion_cost_m"]), default=None)
    return {
        "scene": scene.name, "mechanism": scene.mechanism, "terrain": terrain.spec | {"name": terrain.name},
        "ground_kind": "slabs" if isinstance(terrain, SlabTerrain) else ground,
        "controller": controller_name, "policy": controller.asset,
        "target_xyz": target.tolist(),
        "lost_state_xyz": rows[0]["start_xyz"] if rows else None,
        "target_bearing_rad": round(bearing, 4),
        "profile": {k: v for k, v in profile.items() if k != "rows"},
        "surface_check": surface_check(terrain, scene),
        "actions": rows,
        "best_action": None if best is None else best["action"],
        "action_rsr_spread": {"n_success": sum(1 for r in rows if r["rsr"] and not r["fallen"]),
                              "n_total": len(rows)},
        "baselines": {
            "A_reactive_forward": summarise(reactive_action),
            "B_history_only": {"rule": "aim at remembered bearing",
                               "bearing_deg": round(math.degrees(bearing), 2),
                               "chosen": history_action, "outcome": summarise(history_action)},
            # D: an IMU gives the world heading at the loss moment, not the
            # target's position.  If these two rules ever disagree, the
            # difference is exactly what target memory buys.
            "D_initial_heading_lock": {"rule": "hold the loss heading; no target memory",
                                       "chosen": "heading_lock",
                                       "outcome": summarise("heading_lock")},
            "C_oracle_counterfactual": {"chosen": counterfactual_action,
                                        "outcome": None if counterfactual_action is None
                                        else summarise(counterfactual_action)},
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 Experiment 2 benchmark")
    p.add_argument("--mode", choices=["profile", "actions", "surface", "all"], default="all")
    p.add_argument("--scenes", nargs="*", default=BENCHMARK_SCENES)
    p.add_argument("--controller", choices=["go2rl", "leggedmanip"], default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--actions", nargs="*", default=None)
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/benchmark.json"))
    args = p.parse_args()

    report = {"task": "TASK3_experiment2_action_dependent_reobservability",
              "controller": args.controller, "scenes": {}}
    if args.mode == "surface" and args.output.exists():
        # Re-validate the terrain without re-running the physics.
        report = json.loads(args.output.read_text(encoding="utf-8"))
    for name in args.scenes:
        scene = scene_for(name)
        if scene.name != name:
            raise SystemExit(f"no benchmark scene named {name!r}")
        terrain = build_scene_terrain(scene)
        print(f"--- {name}: {scene.mechanism}")
        if args.mode == "surface":
            check = surface_check(terrain, scene)
            report["scenes"].setdefault(name, {})["surface_check"] = check
            print(f"    surface_check {check}")
            continue
        if args.mode in ("profile", "all"):
            prof = profile_for(scene, terrain)
            print("    V(x) runs:", [(r["visible"], r["x_from"], r["x_to"]) for r in prof["runs"]],
                  "switches:", prof["visible_to_lost_switches"])
            report["scenes"].setdefault(name, {})["profile"] = {
                k: v for k, v in prof.items() if k != "rows"}
        if args.mode in ("actions", "all"):
            out = run_scene(scene, terrain, args.controller, args.ground, args.policy, args.actions)
            report["scenes"].setdefault(name, {}).update(
                {k: v for k, v in out.items() if k != "profile"})
            for r in out["actions"]:
                print(f"    {r['action']:11s} rsr={str(r['rsr']):5s} "
                      f"t_reobs={str(r['t_reobs_s']):>6s} cost={r['motion_cost_m']:.2f} "
                      f"fwd={r['forward_m']:+.2f} lat={r['lateral_m']:+.2f} "
                      f"dz={r['height_gain_m']:+.2f} fallen={r['fallen']}")
            def _rsr(entry):
                return None if not entry else entry["rsr"]

            print(f"    best={out['best_action']} "
                  f"baselines A/B/C="
                  f"{_rsr(out['baselines']['A_reactive_forward'])}/"
                  f"{_rsr(out['baselines']['B_history_only']['outcome'])}/"
                  f"{_rsr(out['baselines']['C_oracle_counterfactual']['outcome'])}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Cross-scene matrix: does the best action actually change with the terrain,
    # and does the counterfactual baseline beat the two non-lookahead rules?
    print("\n=== scene x action RSR ===")
    header = "scene".ljust(11) + "".join(a.rjust(13) for a in ACTION_ORDER) + \
             "   best      A   B   D   C"
    print(header)
    for name in args.scenes:
        entry = report["scenes"].get(name, {})
        rows = entry.get("actions")
        if not rows:
            continue
        by = {r["action"]: r for r in rows}
        cells = []
        for a in ACTION_ORDER:
            r = by.get(a)
            cells.append(("-" if r is None else ("1" if r["rsr"] else "0")).rjust(13))
        base = entry["baselines"]

        def flag(x):
            return "?" if x is None else ("1" if x else "0")

        print(f"{name:<11}" + "".join(cells) + f"   {str(entry['best_action']):<9}"
              f"{flag(base['A_reactive_forward']['rsr'])}   "
              f"{flag(base['B_history_only']['outcome']['rsr'])}   "
              f"{flag(base['D_initial_heading_lock']['outcome']['rsr'])}   "
              f"{flag(base['C_oracle_counterfactual']['outcome']['rsr'])}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
