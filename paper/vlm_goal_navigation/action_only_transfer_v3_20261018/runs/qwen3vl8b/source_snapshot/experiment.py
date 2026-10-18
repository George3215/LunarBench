"""Headless TASK3 Experiment 0: privileged oracle vs camera-visible input.

This is intentionally a geometry experiment, not a learned policy.  It asks a
simple counterfactual question: after the target is lost, which candidate
action actually restores line-of-sight?  The oracle state is written alongside
the camera observation so the effect of removing visual evidence is auditable.

The action model here is a *geometric* one: an action is a displacement of the
eye point, not a controller command.  Whether the real Go2+Piper can execute a
displacement is a separate question, measured by ``locomotion.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
import numpy as np

try:  # supports both ``python experiment.py`` and package imports
    from .scenes import CAMERA_OFFSET_XYZ, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from .terrain import Terrain, generate, load_npy
    from .go2_piper_camera import asset_status as go2_piper_asset_status
    from .go2_piper_camera import build_model, close_scene, create_scene, ground_spec
    from .go2_piper_camera import render_observation, set_robot_pose
except ImportError:
    from scenes import CAMERA_OFFSET_XYZ, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from terrain import Terrain, generate, load_npy
    from go2_piper_camera import asset_status as go2_piper_asset_status
    from go2_piper_camera import build_model, close_scene, create_scene, ground_spec
    from go2_piper_camera import render_observation, set_robot_pose


def validate_contact_surface(terrain: Terrain, target: np.ndarray, samples: int = 11,
                             ground: str = "auto") -> dict:
    """Compare the MuJoCo ground surface with the geometric oracle.

    This keeps the privileged label honest: the height used by ``visibility``
    and the height the robot's feet actually touch have to be the same surface.
    The check ray-casts the built model, so it also fails loudly if the ground
    representation cannot be consumed by MuJoCo at all.
    """
    try:
        import mujoco
    except ImportError as exc:
        return {"available": False, "reason": "mujoco_not_installed", "error": str(exc)}
    try:
        model = build_model(terrain, target, ground)
        spec = ground_spec(terrain, ground)
        data = mujoco.MjData(model)
        # Park the robot off the sampled line so only the ground is hit.
        base = model.body("base_link").id
        qadr = int(model.jnt_qposadr[model.body_jntadr[base]])
        data.qpos[qadr:qadr + 2] = [-40.0, -40.0]
        mujoco.mj_forward(model, data)
        down = np.array([0.0, 0.0, -1.0])
        rows = []
        for x in np.linspace(float(terrain.x[0]) + 0.5, float(terrain.x[-1]) - 0.5, samples):
            gid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(model, data, np.array([float(x), 1.5, 20.0]), down, None, 1, -1, gid)
            if int(gid[0]) < 0:
                continue
            hit = 20.0 - float(dist)
            expected = float(terrain.height(float(x), 1.5))
            rows.append({"x_m": round(float(x), 3), "mujoco_surface_z": round(hit, 4),
                         "oracle_height_z": round(expected, 4), "error_m": round(hit - expected, 4)})
        worst = max((abs(r["error_m"]) for r in rows), default=None)
        return {"available": True, "engine": "mujoco", "version": getattr(mujoco, "__version__", "unknown"),
                "ground_kind": spec.kind, "samples": rows, "max_abs_error_m": worst,
                "consistent": bool(worst is not None and worst <= 0.02)}
    except Exception as exc:
        return {"available": False, "reason": "scene_build_failed", "error": repr(exc)}


def _yaw_forward(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw), 0.0])


def robot_anchor(terrain: Terrain, xy: tuple[float, float]) -> np.ndarray:
    """Base-origin position for a robot standing at ``xy`` on ``terrain``."""
    x, y = float(xy[0]), float(xy[1])
    return np.array([x, y, terrain.height(x, y) + ROBOT_BASE_HEIGHT_M], dtype=float)


def target_anchor(terrain: Terrain, scene: Scene) -> np.ndarray:
    x, y = float(scene.target_xy[0]), float(scene.target_xy[1])
    if scene.target_z_absolute is not None:
        return np.array([x, y, float(scene.target_z_absolute)], dtype=float)
    return np.array([x, y, terrain.height(x, y) + scene.target_height_m], dtype=float)


def visibility(terrain: Terrain, robot: np.ndarray, yaw: float, pitch: float,
               target: np.ndarray, hfov: float = math.radians(87), vfov: float = math.radians(58),
               camera_height: float = CAMERA_OFFSET_XYZ[2]) -> dict:
    """Geometric line-of-sight oracle for one eye pose.

    ``robot`` is the base origin; the eye is offset by the same mount used by
    the injected MuJoCo camera, so the oracle and the rendered RGB-D agree.
    """
    eye = robot + np.array([0.0, 0.0, camera_height])
    d = target - eye
    horizontal = max(float(np.linalg.norm(d[:2])), 1e-9)
    bearing = math.atan2(d[1], d[0]) - yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    target_pitch = math.atan2(float(d[2]), horizontal)
    in_fov = abs(bearing) <= hfov / 2 and abs(target_pitch - pitch) <= vfov / 2
    blocked = False
    if in_fov:
        for alpha in np.linspace(0.04, 0.96, 80):
            p = eye + alpha * d
            # Leave a small clearance for the target ray and robot body.
            if terrain.height(float(p[0]), float(p[1])) + 0.035 > p[2]:
                blocked = True
                break
    return {"visible_oracle": bool(in_fov and not blocked), "in_camera_fov": bool(in_fov),
            "blocked_by_terrain": bool(blocked), "range_m": float(np.linalg.norm(d)),
            "bearing_rad": float(bearing), "target_pitch_rad": float(target_pitch),
            "eye_xyz": eye.tolist()}


def action_state(state: dict, action: str, terrain: Terrain, scene: Scene) -> dict:
    """Apply one geometric candidate action to a copy of the state."""
    out = dict(state)
    if action == "turn_left": out["yaw"] += scene.turn_rad
    elif action == "turn_right": out["yaw"] -= scene.turn_rad
    elif action == "look_up": out["pitch"] += math.radians(25)
    elif action == "forward": out["robot"] = out["robot"] + _yaw_forward(out["yaw"]) * scene.forward_m
    elif action == "climb": out["robot"] = out["robot"] + _yaw_forward(out["yaw"]) * scene.climb_m
    elif action == "back": out["robot"] = out["robot"] - _yaw_forward(out["yaw"]) * scene.back_m
    else: raise ValueError(action)
    out["robot"] = out["robot"].copy()
    out["robot"][2] = terrain.height(float(out["robot"][0]), float(out["robot"][1])) + ROBOT_BASE_HEIGHT_M
    return out


def run(terrain: Terrain, vision_mode: str = "blackout", seed: int = 7,
        robot_backend: str = "go2_piper", save_camera: bool = False,
        camera_output_dir: Path | None = None, scene: Scene | None = None,
        ground: str = "auto") -> dict:
    scene = scene or scene_for(terrain.name)
    # Both anchors are fixed by the scene so profile comparisons are paired and
    # repeatable: the same robot, target and lost state for every candidate.
    approach_robot = robot_anchor(terrain, scene.approach_xy)
    lost_robot = robot_anchor(terrain, scene.lost_xy)
    target = target_anchor(terrain, scene)
    state = {"robot": lost_robot, "yaw": scene.yaw_rad, "pitch": 0.0}
    approach_state = {"robot": approach_robot, "yaw": scene.yaw_rad, "pitch": 0.0}
    approach_observation = visibility(terrain, approach_robot, scene.yaw_rad, 0.0, target)
    initial = visibility(terrain, lost_robot, scene.yaw_rad, 0.0, target)
    actions = ["turn_left", "turn_right", "forward", "climb", "back"]
    if robot_backend == "proxy":
        # Go2+Piper has no pan/tilt joint; only the primitive proxy can pitch.
        actions.insert(2, "look_up")
    camera_scene = None
    camera_setup = {"backend": robot_backend, "available": False}
    if robot_backend == "go2_piper":
        try:
            camera_scene = create_scene(terrain, target, ground=ground)
            camera_setup.update({"available": True, "camera_name": camera_scene.camera_name,
                                 "model": "third_party/LeggedManip_Lab/mujoco/robots/go2_piper/go2piper.xml"})
        except Exception as exc:
            camera_setup.update({"reason": "go2_piper_camera_build_failed", "error": repr(exc)})

    def camera_observation(state: dict, stem: str) -> dict:
        if camera_scene is None:
            return {"source": "unavailable", "rgb_available": False, "depth_available": False,
                    "target_visible_observed": False}
        set_robot_pose(camera_scene, state["robot"], state["yaw"])
        rendered = render_observation(camera_scene,
                                      camera_output_dir if save_camera else None,
                                      stem=stem)
        if vision_mode == "full":
            return rendered
        return {
            "source": rendered["source"],
            "camera_name": rendered["camera_name"],
            "rgb_available": False,
            "depth_available": False,
            "target_visible_observed": False,
            "evidence_masked": True,
        }

    rows = []
    try:
        approach_camera = camera_observation(approach_state, "approach")
        initial_camera = camera_observation(state, "lost")
        for action in actions:
            nxt = action_state(state, action, terrain, scene)
            oracle = visibility(terrain, nxt["robot"], nxt["yaw"], nxt["pitch"], target)
            # In full mode this is rendered Go2+Piper camera evidence. In
            # blackout mode the render exists only to exercise the sensor path;
            # its target evidence is deliberately hidden from the decision.
            camera = camera_observation(nxt, action)
            rows.append({"action": action, "oracle": oracle, "camera": camera,
                         "robot_xyz": nxt["robot"].tolist(), "yaw_rad": nxt["yaw"], "pitch_rad": nxt["pitch"]})
    finally:
        if camera_scene is not None:
            close_scene(camera_scene)
    oracle_choice = next((r["action"] for r in rows if r["oracle"]["visible_oracle"]), None)
    camera_choice = next((r["action"] for r in rows if r["camera"]["target_visible_observed"]), None)
    return {"task": "TASK3_target_reobservability", "scene": scene.name,
            "terrain": terrain.spec | {"name": terrain.name},
            "seed": seed, "vision_mode": vision_mode, "robot_backend": robot_backend,
            "ground_request": ground, "ground_kind": ground_spec(terrain, ground).kind,
            "target_xyz": target.tolist(),
            "robot_base_height_m": ROBOT_BASE_HEIGHT_M,
            "camera_mount_xyz": list(CAMERA_OFFSET_XYZ),
            "approach_state": {"robot_xyz": approach_robot.tolist(), "yaw_rad": scene.yaw_rad,
                               "pitch_rad": 0.0, "xy": list(scene.approach_xy)},
            "lost_state": {"robot_xyz": lost_robot.tolist(), "yaw_rad": scene.yaw_rad,
                           "pitch_rad": 0.0, "xy": list(scene.lost_xy)},
            "approach_observation": approach_observation, "initial_observation": initial,
            "approach_camera": approach_camera, "initial_camera": initial_camera,
            "candidates": rows,
            "privileged_oracle_choice": oracle_choice, "camera_only_choice": camera_choice,
            "camera_dropout_changes_decision": vision_mode != "full" and oracle_choice != camera_choice,
            "scene_notes": list(scene.notes),
            "contact_surface_validation": validate_contact_surface(terrain, target, ground=ground),
            "go2_piper_asset_status": go2_piper_asset_status(),
            "camera_setup": camera_setup,
            "notes": ["oracle is geometry ground truth, not a learned world model",
                      "camera_mode=blackout intentionally removes RGB/depth target evidence",
                      "Go2+Piper upstream XML has no native camera; TASK3 injects a fixed forward MuJoCo camera",
                      "candidate actions are geometric eye displacements, not controller commands"]}


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 target re-observability smoke test")
    p.add_argument("--terrain", default="staircase10",
                   choices=["flat", "crater", "stairs", "undulating", "staircase10"])
    p.add_argument("--omnilrs-dem", type=Path, help="optional OmniLRS-exported .npy DEM")
    p.add_argument("--vision-mode", choices=["full", "blackout"], default="blackout")
    p.add_argument("--robot-backend", choices=["go2_piper", "proxy"], default="go2_piper")
    p.add_argument("--ground", choices=["auto", "boxes", "hfield"], default="auto",
                   help="MuJoCo ground representation; boxes is exact for the staircase")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("tasks/task3_reobservability/generated/smoke.json"))
    p.add_argument("--save-terrain", action="store_true")
    p.add_argument("--save-camera", action="store_true",
                   help="save rendered Go2+Piper RGB/depth arrays as .npy files")
    args = p.parse_args()
    terrain = load_npy(args.omnilrs_dem) if args.omnilrs_dem else generate(args.terrain, args.seed)
    if args.save_terrain:
        terrain.save(args.output.parent / "terrain")
    camera_output_dir = args.output.parent / "camera" if args.save_camera else None
    report = run(terrain, args.vision_mode, args.seed, args.robot_backend,
                 args.save_camera, camera_output_dir, ground=args.ground)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "scene": report["scene"],
                      "ground_kind": report["ground_kind"],
                      "approach_visible": report["approach_observation"]["visible_oracle"],
                      "lost_visible": report["initial_observation"]["visible_oracle"],
                      "oracle_choice": report["privileged_oracle_choice"],
                      "camera_only_choice": report["camera_only_choice"],
                      "camera_dropout_changes_decision": report["camera_dropout_changes_decision"]}, indent=2))


if __name__ == "__main__":
    main()
