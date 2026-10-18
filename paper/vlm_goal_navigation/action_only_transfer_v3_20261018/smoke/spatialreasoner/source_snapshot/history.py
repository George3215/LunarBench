"""TASK3 Experiment 1: History Necessity.

Experiment 2 showed that different actions lead to different future
observability.  This experiment asks the question that decides whether the
problem is a *decision* problem at all:

    can the current observation alone determine the optimal re-observation action?

The construction is a pair of worlds that are locally identical and differ only
in what the robot saw earlier:

* world A - the target was last seen behind the left lip of a window in a wall
* world B - the same wall, same robot pose, target behind the right lip

Both worlds put the robot at the same lost state, with the same heading, the same
visible terrain and no target in view.  If the current observation were
sufficient, the optimal action would be the same in both.  It is not:

* world A is revealed only by strafing -y
* world B is revealed only by strafing +y
* forward, back and both turns fail in both worlds

The observation-equality claim is *measured*, not asserted: the injected camera
is rendered at the shared lost state in both worlds and the images are compared.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .benchmark import (ACTION_ORDER, BENCHMARK_ACTIONS, build_scene_terrain, choose_oracle_counterfactual,
                            history_only_action, profile_for, rollout_with_visibility)
    from .controllers import load_controller
    from .experiment import target_anchor, visibility
    from .go2_piper_camera import close_scene, create_scene, render_observation, set_robot_pose
    from .scenes import HISTORY_PAIRS, ROBOT_BASE_HEIGHT_M, Scene
except ImportError:
    from benchmark import (ACTION_ORDER, BENCHMARK_ACTIONS, build_scene_terrain, choose_oracle_counterfactual,
                           history_only_action, profile_for, rollout_with_visibility)
    from controllers import load_controller
    from experiment import target_anchor, visibility
    from go2_piper_camera import close_scene, create_scene, render_observation, set_robot_pose
    from scenes import HISTORY_PAIRS, ROBOT_BASE_HEIGHT_M, Scene

ROOT = Path(__file__).resolve().parents[2]


def observe_at_lost_state(scene: Scene, terrain, save_dir: Path | None = None) -> dict:
    """Render the injected camera at the scene's lost state (both worlds)."""
    target = target_anchor(terrain, scene)
    cam = create_scene(terrain, target)
    try:
        x, y = float(scene.lost_xy[0]), float(scene.lost_xy[1])
        robot = np.array([x, y, terrain.height(x, y) + ROBOT_BASE_HEIGHT_M], dtype=float)
        set_robot_pose(cam, robot, scene.yaw_rad)
        cam.renderer.update_scene(cam.data, camera=cam.camera_name)
        rgb = np.asarray(cam.renderer.render()).copy()
        cam.renderer.enable_depth_rendering()
        depth = np.asarray(cam.renderer.render()).copy()
        cam.renderer.disable_depth_rendering()
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / f"{scene.name}_rgb.npy", rgb)
        return {"robot_xyz": np.round(robot, 4).tolist(), "rgb": rgb, "depth": depth}
    finally:
        close_scene(cam)


def compare_observations(a: dict, b: dict) -> dict:
    """Quantify O_A ~= O_B at the shared lost state."""
    ra, rb = a["rgb"].astype(np.int16), b["rgb"].astype(np.int16)
    diff = np.abs(ra - rb)
    a_red = int(np.count_nonzero((ra[..., 0] > 120) & (ra[..., 0] > ra[..., 1] * 1.5)
                                 & (ra[..., 0] > ra[..., 2] * 1.5)))
    b_red = int(np.count_nonzero((rb[..., 0] > 120) & (rb[..., 0] > rb[..., 1] * 1.5)
                                 & (rb[..., 0] > rb[..., 2] * 1.5)))
    return {"mean_abs_pixel_diff": round(float(diff.mean()), 4),
            "max_abs_pixel_diff": int(diff.max()),
            "pixels_identical_fraction": round(float(np.mean(diff.max(axis=2) == 0)), 4),
            "target_pixels_world_A": a_red, "target_pixels_world_B": b_red,
            "both_worlds_show_no_target": bool(a_red == 0 and b_red == 0)}


def run_world(scene: Scene, controller_name: str = "go2rl", ground: str = "auto",
              policy: Path | None = None, actions: list[str] | None = None) -> dict:
    import mujoco

    from locomotion import base_yaw, spawn
    from scenes import CAMERA_OFFSET_XYZ

    terrain = build_scene_terrain(scene)
    target = target_anchor(terrain, scene)
    model = build_scene_model(terrain, target, ground)
    controller = load_controller(controller_name, policy)
    model.opt.timestep = controller.timestep
    data = mujoco.MjData(model)

    rows = []
    action_names = actions or (ACTION_ORDER + ["left_hold", "right_hold"])
    for name in action_names:
        primitive = BENCHMARK_ACTIONS[name]
        spawn(model, data, terrain, scene.lost_xy, controller, scene.yaw_rad)
        lost_xyz = np.array(data.qpos[:3], dtype=float)
        heading_point = None
        if name == "heading_lock":
            heading_point = lost_xyz + 10.0 * np.array(
                [math.cos(scene.yaw_rad), math.sin(scene.yaw_rad), 0.0])
        result = rollout_with_visibility(model, data, terrain, controller, primitive, target,
                                         heading_point=heading_point)
        result["visible_at_lost_state"] = bool(
            visibility(terrain, lost_xyz, base_yaw(data), 0.0, target)["visible_oracle"])
        rows.append(result)

    lost_xyz = np.array(rows[0]["start_xyz"], dtype=float)
    bearing = math.atan2(target[1] - lost_xyz[1], target[0] - lost_xyz[0]) - scene.yaw_rad
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    by_name = {r["action"]: r for r in rows}
    best = min((r for r in rows if r["rsr"] and not r["fallen"]),
               key=lambda r: (r["t_reobs_s"] or 1e9, r["motion_cost_m"]), default=None)

    def row(name):
        r = by_name.get(name)
        return None if r is None else {
            "action": name, "rsr": r["rsr"], "t_reobs_s": r["t_reobs_s"],
            "motion_cost_m": r["motion_cost_m"], "lateral_m": r["lateral_m"],
            "forward_m": r["forward_m"], "fallen": r["fallen"]}

    return {
        "world": scene.name, "target_xyz": target.tolist(), "policy": controller.asset,
        "history_vantage_xy": list(scene.history_xy) if scene.history_xy else None,
        "history_saw_target": scene.history_saw_target,
        "lost_state_xyz": rows[0]["start_xyz"], "yaw_rad": scene.yaw_rad,
        "target_bearing_rad": round(bearing, 4),
        "actions": rows,
        "optimal_action": None if best is None else best["action"],
        "optimal_outcome": None if best is None else row(best["action"]),
        "baselines": {
            "A_reactive_forward": row("forward"),
            "B_history_only_toward": row("toward"),
            "D_initial_heading_lock": row("heading_lock"),
            "C_oracle_counterfactual": row(choose_oracle_counterfactual(rows)),
        },
        "profile": {k: v for k, v in profile_for(scene, terrain).items() if k != "rows"},
    }


def lateral_profile(scene: Scene, terrain, x: float = 0.0, step: float = 0.05) -> dict:
    """V(y) at the lost state's x: which lateral positions reveal the target."""
    target = target_anchor(terrain, scene)
    y_lim = float(terrain.y[-1]) - 0.05
    ys = np.arange(-y_lim, y_lim + step * 0.5, step)
    reveal = []
    for y in ys:
        robot = np.array([x, float(y), terrain.height(x, float(y)) + ROBOT_BASE_HEIGHT_M],
                         dtype=float)
        if visibility(terrain, robot, scene.yaw_rad, 0.0, target)["visible_oracle"]:
            reveal.append(round(float(y), 3))
    return {"x": x, "revealing_y": reveal,
            "revealing_y_range": None if not reveal else [reveal[0], reveal[-1]],
            "n_revealing": len(reveal)}


def build_scene_model(terrain, target, ground):
    from go2_piper_camera import build_model
    return build_model(terrain, target, ground)


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 Experiment 1: history necessity")
    p.add_argument("--pair", default="window", choices=["window", "all"],
                   help="'window' = the canonical documented pair, 'all' = the generated set")
    p.add_argument("--pairs-file", type=Path, default=None,
                   help="descriptor written by history_pairs.py (defaults to the generated one)")
    p.add_argument("--controller", choices=["go2rl", "leggedmanip"], default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--actions", nargs="*", default=None,
                   help="defaults to ACTION_ORDER plus the heading-holding sidesteps")
    p.add_argument("--render-dir", type=Path, default=None)
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/history_necessity.json"))
    args = p.parse_args()

    if args.pair == "window":
        pairs = [HISTORY_PAIRS["window"]]
        search_summary = None
    else:
        from history_pairs import DEFAULT_DESCRIPTOR, load_descriptor, scenes_for
        descriptor = args.pairs_file or DEFAULT_DESCRIPTOR
        payload = json.loads(Path(descriptor).read_text(encoding="utf-8"))
        pairs = [scenes_for(params) for params in load_descriptor(descriptor)]
        search_summary = {"descriptor": str(descriptor),
                          "n_attempts": payload["n_attempts"],
                          "n_accepted": payload["n_accepted"],
                          "lateral_reach_m": payload["lateral_reach_m"],
                          "acceptance_criteria": payload["acceptance_criteria"]}

    report = {"task": "TASK3_experiment1_history_necessity", "pair": args.pair,
              "controller": args.controller, "n_pairs": len(pairs), "worlds": {},
              "pairs": [], "search": search_summary}

    for index, scenes in enumerate(pairs):
        if len(scenes) != 2:
            raise SystemExit("a history pair must contain exactly two worlds")
        pair_id = scenes[0].name.rsplit("_", 1)[0] if args.pair == "all" else "window"
        entry = {"pair_id": pair_id, "worlds": [s.name for s in scenes]}
        observations = {}
        for scene in scenes:
            terrain = build_scene_terrain(scene)
            observations[scene.name] = observe_at_lost_state(scene, terrain, args.render_dir)
        entry["observation_equality"] = compare_observations(observations[scenes[0].name],
                                                             observations[scenes[1].name])

        for scene in scenes:
            out = run_world(scene, args.controller, args.ground, args.policy, args.actions)
            terrain = build_scene_terrain(scene)
            out["lateral_profile"] = lateral_profile(scene, terrain)
            out["pair_id"] = pair_id
            report["worlds"][scene.name] = out

        a_name, b_name = scenes[0].name, scenes[1].name
        wa, wb = report["worlds"][a_name], report["worlds"][b_name]
        overlap = [r["action"] for r in wa["actions"]
                   if r["rsr"] and any(s["action"] == r["action"] and s["rsr"]
                                       for s in wb["actions"])]
        entry["verdict"] = {
            "observations_equivalent": bool(
                entry["observation_equality"]["both_worlds_show_no_target"]
                and entry["observation_equality"]["mean_abs_pixel_diff"] < 1e-9),
            "optimal_action_world_A": wa["optimal_action"],
            "optimal_action_world_B": wb["optimal_action"],
            "optimal_actions_differ": wa["optimal_action"] != wb["optimal_action"]
            and wa["optimal_action"] is not None and wb["optimal_action"] is not None,
            "actions_working_in_both_worlds": overlap,
            "history_independent_action_exists": bool(overlap),
        }
        report["pairs"].append(entry)
        print(f"=== pair {pair_id} [{index + 1}/{len(pairs)}] ===")
        print(f"    observation equality: {entry['observation_equality']}")
        for scene in scenes:
            w = report["worlds"][scene.name]
            hits = [f"{r['action']}(t={r['t_reobs_s']})" for r in w["actions"] if r["rsr"]]
            print(f"    {scene.name:22s} optimal={str(w['optimal_action']):8s} "
                  f"successful={hits or 'none'}")
        print(f"    verdict: {entry['verdict']}")

    report["aggregate"] = aggregate(report)
    print("\n=== aggregate ===")
    for k, v in report["aggregate"].items():
        print(f"    {k}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


def aggregate(report: dict) -> dict:
    """Rates over every world and every pair: does H2 hold, do the baselines hold up?"""
    pairs = report["pairs"]
    worlds = report["worlds"]
    names = list(worlds)
    if not pairs:
        return {}
    baseline_keys = {
        "A_reactive_forward": lambda w: w["baselines"]["A_reactive_forward"],
        "B_history_only_toward": lambda w: w["baselines"]["B_history_only_toward"],
        "D_initial_heading_lock": lambda w: w["baselines"]["D_initial_heading_lock"],
        "C_oracle_counterfactual": lambda w: w["baselines"]["C_oracle_counterfactual"],
    }
    rates = {}
    for label, getter in baseline_keys.items():
        hits = 0
        for name in names:
            row = getter(worlds[name])
            if row and row["rsr"] and not row["fallen"]:
                hits += 1
        rates[label] = round(hits / len(names), 4)

    optimal_counts: dict[str, int] = {}
    for name in names:
        act = worlds[name]["optimal_action"]
        optimal_counts[str(act)] = optimal_counts.get(str(act), 0) + 1

    diffs = [p["observation_equality"]["mean_abs_pixel_diff"] for p in pairs]
    identical = [p["observation_equality"]["pixels_identical_fraction"] for p in pairs]
    return {
        "n_pairs": len(pairs),
        "n_worlds": len(names),
        "pairs_with_equivalent_observations": sum(
            1 for p in pairs if p["verdict"]["observations_equivalent"]),
        "pairs_with_opposite_optimal_actions": sum(
            1 for p in pairs if p["verdict"]["optimal_actions_differ"]),
        "pairs_with_no_history_independent_action": sum(
            1 for p in pairs if not p["verdict"]["history_independent_action_exists"]),
        "H2_hold_rate": round(sum(1 for p in pairs if p["verdict"]["observations_equivalent"]
                                  and p["verdict"]["optimal_actions_differ"]
                                  and not p["verdict"]["history_independent_action_exists"])
                              / len(pairs), 4),
        "camera_mean_abs_pixel_diff_max": round(max(diffs), 6),
        "camera_pixels_identical_fraction_min": round(min(identical), 6),
        "baseline_rsr_over_worlds": rates,
        "optimal_action_counts": optimal_counts,
    }


if __name__ == "__main__":
    main()
