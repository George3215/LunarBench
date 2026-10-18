"""TASK3 locomotion test: execute the candidate actions with the real robot model.

Experiment 0 evaluates candidate actions as geometric displacements of the eye
point.  That is only meaningful if the repository's Go2+Piper can actually
execute those displacements in the same terrain.  This module closes that gap:
it drives the vendored MuJoCo Go2+Piper with a vendored policy from
:mod:`controllers`, headless, and measures what each candidate action really does.

Two modes:

``actions``
    Characterise every candidate primitive on the scene terrain: how far it
    moves the robot, how much it turns, whether the robot stays upright, and
    whether it climbs the stairs.

``episode``
    Replay the Experiment 0 story with physics instead of teleports: spawn where
    the target was visible, walk forward until the geometric oracle declares the
    target lost, then run each candidate action from that lost state and report
    which one physically re-observes the target.

Nothing here trains a model, reads a real camera, or authorises real robot
motion.  Both policies are upstream artefacts already vendored in this
repository; TASK3 only supplies the terrain and the candidate commands.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

try:
    from .controllers import CANDIDATE_ORDER, Controller, load_controller
    from .scenes import CAMERA_OFFSET_XYZ, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from .terrain import STAIRCASE10_RISE_M, STAIRCASE10_STEPS, STAIRCASE10_TREAD_M
    from .terrain import Terrain, generate, load_npy
    from .experiment import visibility
    from .go2_piper_camera import build_model
except ImportError:
    from controllers import CANDIDATE_ORDER, Controller, load_controller
    from scenes import CAMERA_OFFSET_XYZ, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from terrain import STAIRCASE10_RISE_M, STAIRCASE10_STEPS, STAIRCASE10_TREAD_M
    from terrain import Terrain, generate, load_npy
    from experiment import visibility
    from go2_piper_camera import build_model

ROOT = Path(__file__).resolve().parents[2]
FALLEN_UP_Z = 0.25

# The visible/lost transition is a grazing condition, so the *first* frame that
# reports "lost" sits exactly on the boundary: a 3 cm gait bob flips it back.
# The physical lost state is therefore defined as "walk on until the robot has
# committed this far past the transition, then come to a stop and settle".
LOSS_COMMIT_M = 0.75
LOSS_SETTLE_S = 1.2

CANDIDATES = CANDIDATE_ORDER


def candidates_for(controller: Controller) -> list[str]:
    return [name for name in CANDIDATE_ORDER if name in controller.primitives]


def policy_available(name: str) -> dict:
    """Report whether the policy file for a controller is present."""
    from controllers import ControllerRegistry
    return ControllerRegistry().status(name)


def spawn(model, data, terrain: Terrain, xy, controller: Controller, yaw: float = 0.0) -> None:
    """Place the robot standing on the terrain at ``xy`` facing ``yaw``."""
    import mujoco

    mujoco.mj_resetData(model, data)
    x, y = float(xy[0]), float(xy[1])
    data.qpos[:3] = [x, y, terrain.height(x, y) + ROBOT_BASE_HEIGHT_M]
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    data.qpos[7:] = controller.default_angles
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def base_upright(data) -> float:
    return float(data.xmat[1].reshape(3, 3)[2, 2])


def base_yaw(data) -> float:
    q = data.qpos[3:7]
    return float(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


@dataclass
class RolloutResult:
    action: str
    start_xyz: list
    end_xyz: list
    displacement_m: float
    forward_m: float
    height_gain_m: float
    yaw_change_rad: float
    upright_min: float
    fallen: bool
    finite: bool
    simulated_s: float
    max_contacts: int
    trace: list = field(default_factory=list)


def run_primitive(model, data, terrain: Terrain, controller: Controller, primitive,
                  target: np.ndarray | None = None, seconds: float | None = None,
                  trace_every_s: float = 0.5,
                  fallen_up_z: float = FALLEN_UP_Z) -> RolloutResult:
    """Execute one named action primitive on an existing ``(model, data)`` pair.

    ``target`` is the remembered world position of the target.  It is only
    consulted by primitives with ``heading_hold``, which steer toward it; the
    robot has not observed the target since it was lost.
    """
    import mujoco

    dt, decim = controller.timestep, controller.control_decimation
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    mujoco.mj_forward(model, data)
    start = np.array(data.qpos[:3], dtype=float)
    start_yaw = base_yaw(data)
    duration = primitive.duration_s if seconds is None else seconds
    controller.reset()
    target = None if target is None else np.asarray(target, dtype=float)
    target_angles = controller.default_angles.copy()
    steps = int(round(duration / dt))
    trace = []
    upright_min = 1.0
    max_contacts = 0
    fallen = False
    start_yaw = base_yaw(data)
    for step in range(steps):
        if primitive.heading_hold and target is not None:
            # Steer toward where the target was last seen (memory, not vision).
            bearing = math.atan2(target[1] - data.qpos[1], target[0] - data.qpos[0]) - base_yaw(data)
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            wz = float(np.clip(primitive.heading_gain * bearing,
                               -primitive.heading_wz_max, primitive.heading_wz_max))
            command = [primitive.vx, primitive.vy, wz]
        elif primitive.yaw_hold:
            err = math.atan2(math.sin(base_yaw(data) - start_yaw),
                             math.cos(base_yaw(data) - start_yaw))
            wz = float(np.clip(-primitive.yaw_hold_gain * err,
                               -primitive.yaw_hold_wz_max, primitive.yaw_hold_wz_max))
            command = [primitive.vx, primitive.vy, wz]
        else:
            command = [primitive.vx, primitive.vy, primitive.wz]
        torque = (target_angles - data.qpos[7:]) * controller.kps - data.qvel[6:] * controller.kds
        data.ctrl[:] = np.clip(torque, lo, hi)
        mujoco.mj_step(model, data)
        max_contacts = max(max_contacts, int(data.ncon))
        upright_min = min(upright_min, base_upright(data))
        if step % decim == 0:
            target_angles = controller.act(data, command)
        if not np.isfinite(data.qpos).all() or base_upright(data) < fallen_up_z:
            fallen = True
            break
        if step % max(int(trace_every_s / dt), 1) == 0:
            trace.append({"t_s": round(float(data.time), 3),
                          "xyz": np.round(np.array(data.qpos[:3], dtype=float), 3).tolist(),
                          "yaw_rad": round(base_yaw(data), 3),
                          "upright_z": round(base_upright(data), 3)})
    end = np.array(data.qpos[:3], dtype=float)
    delta = end - start
    return RolloutResult(
        action=primitive.name, start_xyz=np.round(start, 4).tolist(), end_xyz=np.round(end, 4).tolist(),
        displacement_m=float(np.linalg.norm(delta)),
        forward_m=float(delta[0]), height_gain_m=float(delta[2]),
        yaw_change_rad=float(math.atan2(math.sin(base_yaw(data) - start_yaw),
                                        math.cos(base_yaw(data) - start_yaw))),
        upright_min=float(upright_min), fallen=bool(fallen),
        finite=bool(np.isfinite(data.qpos).all()), simulated_s=float(data.time),
        max_contacts=int(max_contacts), trace=trace)


def stair_level(x: float, terrain: Terrain | None = None) -> int:
    """Which step of the staircase the given x sits on (0 = apron, N = landing)."""
    tread = STAIRCASE10_TREAD_M
    steps = STAIRCASE10_STEPS
    if terrain is not None:
        tread = float(terrain.spec.get("step_tread_m", tread))
        steps = int(terrain.spec.get("step_count", steps))
    return int(np.clip(np.floor(x / tread) + 1, 0, steps))


def observe(model, data, terrain: Terrain, target: np.ndarray) -> dict:
    """Geometric oracle reading at the robot's current physical state."""
    xyz = np.array(data.qpos[:3], dtype=float)
    obs = visibility(terrain, xyz, base_yaw(data), 0.0, target)
    obs["base_xyz"] = np.round(xyz, 4).tolist()
    # Only the staircase profile defines discrete step levels.
    obs["stair_level"] = stair_level(float(xyz[0]), terrain) if terrain.name == "staircase10" else None
    obs["upright_z"] = round(base_upright(data), 3)
    return obs


def prepare(terrain: Terrain, scene: Scene, ground: str = "auto",
            controller_name: str = "go2rl", policy: Path | None = None):
    """Build the MuJoCo scene and load the requested controller."""
    import mujoco

    from experiment import target_anchor

    target = target_anchor(terrain, scene)
    model = build_model(terrain, target, ground)
    controller = load_controller(controller_name, policy)
    model.opt.timestep = controller.timestep
    data = mujoco.MjData(model)
    return model, data, target, controller


def characterise_actions(terrain: Terrain, scene: Scene, actions: list[str],
                         ground: str = "auto", controller_name: str = "go2rl",
                         policy: Path | None = None) -> dict:
    """Mode ``actions``: run every primitive once from the scene's lost state."""
    model, data, target, controller = prepare(terrain, scene, ground, controller_name, policy)
    results = []
    for name in actions:
        primitive = controller.primitives[name]
        spawn(model, data, terrain, scene.lost_xy, controller, scene.yaw_rad)
        before = observe(model, data, terrain, target)
        result = run_primitive(model, data, terrain, controller, primitive, target=target)
        after = observe(model, data, terrain, target)
        results.append({"action": name,
                        "command": {"vx": primitive.vx, "vy": primitive.vy, "wz": primitive.wz,
                                    "duration_s": primitive.duration_s, "note": primitive.note},
                        "before": before, "after": after, "rollout": result.__dict__})
    return {"mode": "actions", "terrain": terrain.spec | {"name": terrain.name}, "scene": scene.name,
            "ground_request": ground, "controller": controller_name, "policy": controller.asset,
            "control_hz": round(1.0 / controller.control_dt, 1),
            "target_xyz": target.tolist(), "actions": results}


def episode(terrain: Terrain, scene: Scene, max_approach_s: float = 20.0,
            ground: str = "auto", controller_name: str = "go2rl",
            policy: Path | None = None) -> dict:
    """Mode ``episode``: physical walk-in, target loss, then candidate actions."""
    import mujoco

    model, data, target, controller = prepare(terrain, scene, ground, controller_name, policy)
    dt, decim = controller.timestep, controller.control_decimation
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    spawn(model, data, terrain, scene.approach_xy, controller, scene.yaw_rad)
    start = observe(model, data, terrain, target)

    # 1. Walk in with the real policy: approach until the target is lost, then
    #    keep walking a fixed commit distance and settle, so the frozen lost
    #    state is not sitting on the visible/lost boundary.
    controller.reset()
    target_angles = controller.default_angles.copy()
    forward = controller.primitives["forward"]
    command = [forward.vx, 0.0, 0.0]
    steps = int(round(max_approach_s / dt))
    phase = "approach"
    loss_x = None
    commit_started_s = None
    walk_trace = []
    for step in range(steps):
        torque = (target_angles - data.qpos[7:]) * controller.kps - data.qvel[6:] * controller.kds
        data.ctrl[:] = np.clip(torque, lo, hi)
        mujoco.mj_step(model, data)
        if step % decim == 0:
            target_angles = controller.act(data, command)
            reading = observe(model, data, terrain, target)
            x = float(data.qpos[0])
            if step % int(0.5 / dt / decim) == 0:
                walk_trace.append({"t_s": round(float(data.time), 3), "phase": phase,
                                   "base_xyz": reading["base_xyz"],
                                   "visible_oracle": reading["visible_oracle"],
                                   "blocked_by_terrain": reading["blocked_by_terrain"]})
            if phase == "approach" and not reading["visible_oracle"]:
                phase = "commit"
                loss_x = x
            elif phase == "commit":
                if abs(x - loss_x) >= LOSS_COMMIT_M:
                    phase = "settle"
                    command = [0.0, 0.0, 0.0]
                    commit_started_s = float(data.time)
            elif phase == "settle" and float(data.time) - commit_started_s >= LOSS_SETTLE_S:
                break
        if base_upright(data) < FALLEN_UP_Z or not np.isfinite(data.qpos).all():
            break
    loss = observe(model, data, terrain, target)
    loss_state = {"detected": loss_x is not None, "phase_reached": phase,
                  "first_lost_x_m": None if loss_x is None else round(float(loss_x), 4),
                  "commit_distance_m": LOSS_COMMIT_M, "settle_s": LOSS_SETTLE_S,
                  "t_s": round(float(data.time), 3),
                  "base_xyz": loss["base_xyz"], "stair_level": loss["stair_level"]}

    # 2. From the physical lost state, run each candidate action for real.
    loss_yaw_rad = base_yaw(data)
    candidates = []
    for name in candidates_for(controller):
        spawn(model, data, terrain, tuple(loss["base_xyz"][:2]), controller, loss_yaw_rad)
        before = observe(model, data, terrain, target)
        result = run_primitive(model, data, terrain, controller, controller.primitives[name],
                               target=target)
        after = observe(model, data, terrain, target)
        candidates.append({"action": name, "before": before, "after": after,
                           "reobserved": bool(after["visible_oracle"] and not result.fallen),
                           "rollout": result.__dict__})

    reached = next((c["action"] for c in candidates if c["reobserved"]), None)
    return {"mode": "episode", "terrain": terrain.spec | {"name": terrain.name}, "scene": scene.name,
            "ground_request": ground, "controller": controller_name, "policy": controller.asset,
            "control_hz": round(1.0 / controller.control_dt, 1),
            "target_xyz": target.tolist(), "start": start, "walk_trace": walk_trace,
            "lost_state": loss_state, "candidates": candidates,
            "physically_reobserved_by": reached,
            "reactive_baseline_would_do": "forward",
            "notes": ["the walk-in is driven by the vendored policy, not by TASK3 code",
                      "candidate actions are executed physically; the oracle only labels the result",
                      "a candidate marked fallen=True did not complete and cannot count as re-observation"]}


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 Go2+Piper locomotion test")
    p.add_argument("--mode", choices=["actions", "episode"], default="actions")
    p.add_argument("--controller", choices=["go2rl", "leggedmanip"], default="go2rl",
                   help="which vendored locomotion policy drives the robot")
    p.add_argument("--policy", type=Path, default=None,
                   help="override the policy file for the chosen controller")
    p.add_argument("--terrain", default="staircase10",
                   choices=["flat", "crater", "stairs", "undulating", "staircase10"])
    p.add_argument("--omnilrs-dem", type=Path, help="optional OmniLRS-exported .npy DEM")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--actions", nargs="*",
                   default=["stand", "forward", "turn_left", "turn_right", "climb", "climb_toward", "back"])
    p.add_argument("--ground", choices=["auto", "boxes", "hfield"], default="auto",
                   help="MuJoCo ground representation; hfield is the generic sampled path")
    p.add_argument("--step-rise", type=float, default=None,
                   help="override the staircase10 step rise (m) to probe climbability")
    p.add_argument("--step-tread", type=float, default=None,
                   help="override the staircase10 step tread (m)")
    p.add_argument("--step-count", type=int, default=None,
                   help="override the staircase10 step count")
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/locomotion.json"))
    args = p.parse_args()
    terrain = load_npy(args.omnilrs_dem) if args.omnilrs_dem else generate(
        args.terrain, args.seed, step_rise=args.step_rise, step_tread=args.step_tread,
        step_count=args.step_count)
    scene = scene_for(terrain.name)
    status = policy_available(args.controller)
    if not status["present"]:
        raise SystemExit(f"policy missing for controller {args.controller}: {status}")
    if args.mode == "actions":
        report = characterise_actions(terrain, scene, args.actions, args.ground,
                                      args.controller, args.policy)
    else:
        report = episode(terrain, scene, ground=args.ground,
                         controller_name=args.controller, policy=args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.mode == "actions":
        summary = {r["action"]: {"forward_m": round(r["rollout"]["forward_m"], 3),
                                 "height_gain_m": round(r["rollout"]["height_gain_m"], 3),
                                 "yaw_change_rad": round(r["rollout"]["yaw_change_rad"], 3),
                                 "stair_level": r["after"]["stair_level"],
                                 "visible_after": r["after"]["visible_oracle"],
                                 "fallen": r["rollout"]["fallen"]} for r in report["actions"]}
    else:
        summary = {"lost_state": report["lost_state"],
                   "physically_reobserved_by": report["physically_reobserved_by"],
                   "candidates": {c["action"]: {"reobserved": c["reobserved"],
                                                "forward_m": round(c["rollout"]["forward_m"], 3),
                                                "height_gain_m": round(c["rollout"]["height_gain_m"], 3),
                                                "stair_level": c["after"]["stair_level"],
                                                "fallen": c["rollout"]["fallen"]}
                                  for c in report["candidates"]}}
    print(json.dumps({"output": str(args.output), "mode": args.mode,
                      "controller": args.controller, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
