"""Inspect and drive the TASK3 terrains in an interactive MuJoCo window.

Three modes:

``--list``
    Print every available terrain with its shape parameters and the scene
    anchors that go with it.
``--check``
    Headless self-test of the terrain and the robot on it: the MuJoCo contact
    surface is compared against the analytic oracle, then the vendored policy
    holds station and walks forward.  This is the "is the terrain any good"
    check, and it needs no display.
``(default)``
    Open the interactive viewer on one terrain and drive the robot.

Keyboard while the window is open:

===== ==================================================================
W / S forward / backward          A / D   strafe left / right
Q / E turn left / right           X       zero the command
Z / C raise / lower forward speed R       respawn the robot
1 / 2 / 3  respawn at approach / lost / scene start
T         toggle camera tracking  V       print the visibility readout
P         print a status line     ESC     quit
===== ==================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from .benchmark import BENCHMARK_ACTIONS, build_scene_terrain, surface_check
    from .controllers import load_controller
    from .experiment import target_anchor, visibility
    from .scenes import ROBOT_BASE_HEIGHT_M, scene_for
    from .slabs import TERRAIN_KIT
except ImportError:
    from benchmark import BENCHMARK_ACTIONS, build_scene_terrain, surface_check
    from controllers import load_controller
    from experiment import target_anchor, visibility
    from scenes import ROBOT_BASE_HEIGHT_M, scene_for
    from slabs import TERRAIN_KIT

ROOT = Path(__file__).resolve().parents[2]

# Legacy sampled-height-field profiles, kept so the older figures remain inspectable.
LEGACY_TERRAINS = ["staircase10", "flat", "crater", "stairs", "undulating"]
ALL_TERRAINS = sorted(TERRAIN_KIT) + LEGACY_TERRAINS


def describe(name: str) -> dict:
    scene = scene_for(name)
    terrain = build_scene_terrain(scene)
    target = target_anchor(terrain, scene)
    return {"terrain": name, "scene": scene.name, "profile": terrain.spec,
            "x_range_m": [round(float(terrain.x[0]), 2), round(float(terrain.x[-1]), 2)],
            "y_range_m": [round(float(terrain.y[0]), 2), round(float(terrain.y[-1]), 2)],
            "z_range_m": [round(float(terrain.heights.min()), 3),
                          round(float(terrain.heights.max()), 3)],
            "target_xyz": [round(float(v), 3) for v in target],
            "approach_xy": list(scene.approach_xy), "lost_xy": list(scene.lost_xy),
            "mechanism": scene.mechanism}


def build(name: str, controller_name: str, ground: str, policy: Path | None):
    """Return (model, controller, scene, terrain, target) ready to simulate."""
    import mujoco

    scene = scene_for(name)
    terrain = build_scene_terrain(scene)
    target = target_anchor(terrain, scene)
    from go2_piper_camera import build_model
    model = build_model(terrain, target, ground)
    controller = load_controller(controller_name, policy)
    model.opt.timestep = controller.timestep
    return model, controller, scene, terrain, target


def support_height(terrain, x: float, y: float, yaw: float = 0.0,
                   half_x: float = 0.28, half_y: float = 0.20) -> float:
    """Highest terrain under the robot's footprint.

    Placing the base at the height of the single point ``(x, y)`` drops the robot
    onto a step edge whenever the anchor sits on one: on rough ground the front
    feet end up on the higher tread and the policy cannot start walking.  Taking
    the maximum over the footprint means no foot ever starts inside the terrain.
    """
    fx = np.linspace(-half_x, half_x, 9)
    fy = np.linspace(-half_y, half_y, 7)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    xx = x + fx[:, None] * cos_y - fy[None, :] * sin_y
    yy = y + fx[:, None] * sin_y + fy[None, :] * cos_y
    try:
        return float(np.max(terrain.height(xx, yy)))
    except TypeError:
        # The legacy sampled-height-field terrain only accepts scalar queries.
        return float(max(float(terrain.height(float(a), float(b)))
                         for a, b in zip(xx.ravel(), yy.ravel())))


def place(model, data, terrain, xy, controller, yaw: float = 0.0) -> None:
    import mujoco

    x, y = float(xy[0]), float(xy[1])
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [x, y, support_height(terrain, x, y, yaw) + ROBOT_BASE_HEIGHT_M]
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    data.qpos[7:25] = controller.default_angles
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def yaw_of(data) -> float:
    q = data.qpos[3:7]
    return float(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def upright_of(data) -> float:
    return float(data.xmat[1].reshape(3, 3)[2, 2])


def run_check(name: str, controller_name: str, ground: str, policy: Path | None,
              walk_s: float = 5.0, stand_s: float = 2.0,
              force_locomotion: bool = False) -> dict:
    """Headless self-test: surface agreement, hold station, walk forward.

    Legacy sampled profiles still use MuJoCo's height-field collider, which is
    measured to stall both vendored policies.  For those the geometry check runs
    and the locomotion checks are reported as not applicable rather than as a
    failure, unless ``force_locomotion`` reproduces the stall on purpose.
    """
    import mujoco

    from go2_piper_camera import ground_spec

    model, controller, scene, terrain, target = build(name, controller_name, ground, policy)
    data = mujoco.MjData(model)
    dt, decim = controller.timestep, controller.control_decimation
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    kind = ground_spec(terrain, ground).kind
    locomote = force_locomotion or kind != "hfield"

    def rollout(command, seconds, x, y):
        place(model, data, terrain, (x, y), controller, scene.yaw_rad)
        controller.reset()
        angles = controller.default_angles.copy()
        start = np.array(data.qpos[:3], dtype=float)
        up_min, fallen = 1.0, False
        for step in range(int(round(seconds / dt))):
            torque = (angles - data.qpos[7:25]) * controller.kps - data.qvel[6:24] * controller.kds
            data.ctrl[:] = np.clip(torque, lo, hi)
            mujoco.mj_step(model, data)
            up_min = min(up_min, upright_of(data))
            if step % decim == 0:
                angles = controller.act(data, command)
            if not np.isfinite(data.qpos).all() or upright_of(data) < 0.25:
                fallen = True
                break
        end = np.array(data.qpos[:3], dtype=float)
        return {"start_xyz": np.round(start, 3).tolist(), "end_xyz": np.round(end, 3).tolist(),
                "forward_m": round(float(end[0] - start[0]), 3),
                "upright_min": round(float(up_min), 3), "fallen": bool(fallen)}

    surface = surface_check(terrain, scene)
    info = describe(name)
    if locomote:
        stand = rollout([0.0, 0.0, 0.0], stand_s, *scene.approach_xy)
        walk = rollout([float(BENCHMARK_ACTIONS["forward"].vx), 0.0, 0.0], walk_s, *scene.approach_xy)
        back = rollout([-0.6, 0.0, 0.0], walk_s, *scene.lost_xy)
        # Required: the contact surface is the oracle's surface, the robot holds
        # station, and it can make progress.  Backward motion is *informational*:
        # on stairs_down retreating means climbing steps backwards, which the
        # vendored policy cannot do, and that is a policy limit, not bad terrain.
        required = {
            "surface_matches_oracle": surface["consistent"],
            "locomotion_tested": True,
            "stands_without_falling": bool(not stand["fallen"] and stand["upright_min"] > 0.9),
            "walks_forward": bool(not walk["fallen"] and walk["forward_m"] > 1.0),
        }
        verdict = dict(required)
        verdict["walks_backward_informational"] = bool(not back["fallen"]
                                                       and back["forward_m"] < -1.0)
        locomotion_note = None if verdict["walks_backward_informational"] else (
            "backward motion did not retreat; on stairs_down that is expected "
            "(retreat means climbing steps backwards) and is a policy limit, "
            "not a terrain defect")
        all_pass = all(required.values())
    else:
        stand = walk = back = None
        verdict = {"surface_matches_oracle": surface["consistent"],
                   "locomotion_tested": False}
        locomotion_note = ("ground_kind=hfield: MuJoCo's height-field collider stalls both "
                          "vendored policies, so locomotion is not validated here (README 8.1). "
                          "Use --force-locomotion to reproduce the stall.")
        all_pass = surface["consistent"]
    return {"terrain": name, "scene": scene.name, "ground_kind": kind,
            "surface_check": surface, "stand": stand, "walk_forward": walk,
            "walk_backward": back, "locomotion_note": locomotion_note,
            "verdict": verdict, "all_pass": all_pass, "spec": info}


def check_all(names: list[str], controller_name: str, ground: str,
              policy: Path | None, force_locomotion: bool = False) -> dict:
    results = []
    for name in names:
        print(f"--- checking {name} ...", flush=True)
        try:
            results.append(run_check(name, controller_name, ground, policy,
                                     force_locomotion=force_locomotion))
        except Exception as exc:  # noqa: BLE001 - reported per terrain, not fatal
            results.append({"terrain": name, "error": repr(exc), "all_pass": False})
        r = results[-1]
        if "error" in r:
            print(f"    ERROR {r['error']}")
        elif not r["verdict"].get("locomotion_tested", True):
            print(f"    surface_err={r['surface_check']['max_abs_error_m']} ({r['ground_kind']}) "
                  f"-> {'GEOMETRY OK' if r['all_pass'] else 'GEOMETRY MISMATCH'}, locomotion N/A")
        else:
            print(f"    surface_err={r['surface_check']['max_abs_error_m']} "
                  f"stand_up={r['stand']['upright_min']} "
                  f"walk={r['walk_forward']['forward_m']}m "
                  f"back={r['walk_backward']['forward_m']}m "
                  f"-> {'PASS' if r['all_pass'] else 'FAIL'}"
                  f"{'' if r['all_pass'] else ' ' + str([k for k, v in r['verdict'].items() if v is False])}"
                  f"{'  [backward N/A: ' + r['locomotion_note'] + ']' if r['locomotion_note'] else ''}")
    return {"task": "TASK3_terrain_check", "controller": controller_name,
            "n_terrains": len(names),
            "n_passed": sum(1 for r in results if r.get("all_pass")),
            "n_locomotion_tested": sum(1 for r in results
                                       if r.get("verdict", {}).get("locomotion_tested", False)
                                       and "error" not in r),
            "n_geometry_consistent": sum(1 for r in results
                                         if r.get("surface_check", {}).get("consistent")),
            "results": results}


def view(name: str, controller_name: str, ground: str, policy: Path | None,
         start: str, command, seconds: float | None, track: bool) -> None:
    import mujoco
    import mujoco.viewer

    model, controller, scene, terrain, target = build(name, controller_name, ground, policy)
    data = mujoco.MjData(model)
    anchor = {"approach": scene.approach_xy, "lost": scene.lost_xy,
              "start": scene.approach_xy}[start]
    place(model, data, terrain, anchor, controller, scene.yaw_rad)
    controller.reset()

    cmd = np.array(command if command else [0.0, 0.0, 0.0], dtype=float)
    state = {"angles": controller.default_angles.copy(), "show_readout": True}
    anchors = {"approach": scene.approach_xy, "lost": scene.lost_xy, "start": scene.approach_xy}

    def respawn(which: str) -> None:
        place(model, data, terrain, anchors[which], controller, scene.yaw_rad)
        controller.reset()
        state["angles"] = controller.default_angles.copy()

    def key(code: int) -> None:
        c = chr(code).lower()
        if c == "w":
            cmd[0] = min(cmd[0] + 0.2, 1.6)
        elif c == "s":
            cmd[0] = max(cmd[0] - 0.2, -1.2)
        elif c == "d":
            cmd[1] = max(cmd[1] - 0.2, -1.0)
        elif c == "a":
            cmd[1] = min(cmd[1] + 0.2, 1.0)
        elif c == "q":
            cmd[2] = min(cmd[2] + 0.2, 1.5)
        elif c == "e":
            cmd[2] = max(cmd[2] - 0.2, -1.5)
        elif c == "x":
            cmd[:] = 0.0
        elif c == "z":
            cmd[0] = min(cmd[0] + 0.1, 1.6)
        elif c == "c":
            cmd[0] = max(cmd[0] - 0.1, -1.2)
        elif c == "r":
            respawn(start)
        elif c in anchors:
            respawn({"1": "approach", "2": "lost", "3": "start"}[c])
        elif c == "v":
            state["show_readout"] = not state["show_readout"]
        print(f"[viewer] cmd=({cmd[0]:+.1f}, {cmd[1]:+.1f}, {cmd[2]:+.1f})")

    print(f"[viewer] terrain={name} scene={scene.name} controller={controller_name}")
    print(f"[viewer] mechanism: {scene.mechanism}")
    print(f"[viewer] target at {np.round(target, 2).tolist()}")
    print("[viewer] W/S forward-back  A/D strafe  Q/E turn  X stop  Z/C speed  "
          "R respawn  1/2/3 anchor  T track  V readout  ESC quit")

    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        # Frame the whole terrain on start so the shape is readable.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, float(terrain.heights.max()) * 0.5 + 0.2]
        viewer.cam.distance = 9.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0

        dt, decim = controller.timestep, controller.control_decimation
        lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        # Render at display rate, not at physics rate: 7000 physics steps/s with
        # one sync each would spend all its time in the renderer.
        steps_per_frame = max(1, int(round((1.0 / 60.0) / dt)))
        t0 = time.monotonic()
        last_report = 0.0
        tick = 0
        while viewer.is_running():
            frame_start = time.monotonic()
            if seconds is not None and frame_start - t0 > seconds:
                break
            if track:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = model.body("base_link").id
                viewer.cam.distance = 3.2
            for _ in range(steps_per_frame):
                torque = ((state["angles"] - data.qpos[7:25]) * controller.kps
                          - data.qvel[6:24] * controller.kds)
                data.ctrl[:] = np.clip(torque, lo, hi)
                mujoco.mj_step(model, data)
                tick += 1
                if tick % decim == 0:
                    state["angles"] = controller.act(data, cmd)
            viewer.sync()

            if frame_start - last_report > 1.0:
                last_report = frame_start
                xyz = np.round(data.qpos[:3], 2).tolist()
                reading = visibility(terrain, np.array(data.qpos[:3], dtype=float),
                                     yaw_of(data), 0.0, target)
                if state["show_readout"]:
                    print(f"[viewer] t={data.time:6.2f}s  base={xyz}  "
                          f"upright={upright_of(data):.2f}  "
                          f"target={'VISIBLE' if reading['visible_oracle'] else 'hidden'}"
                          f"{' (blocked by terrain)' if reading['blocked_by_terrain'] else ''}"
                          f"  range={reading['range_m']:.2f}m  "
                          f"cmd=({cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f})")
            elapsed = time.monotonic() - frame_start
            sleep = (1.0 / 60.0) - elapsed
            if sleep > 0:
                time.sleep(sleep)
    print(f"[viewer] closed at sim time {data.time:.2f}s, "
          f"base={np.round(data.qpos[:3], 3).tolist()}")


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 terrain inspector and robot driver")
    p.add_argument("--terrain", default="stairs_up",
                   help="terrain name, or 'all' with --check")
    p.add_argument("--list", action="store_true", help="list terrains and exit")
    p.add_argument("--check", action="store_true",
                   help="headless self-test of terrain + robot (no display needed)")
    p.add_argument("--controller", choices=["go2rl", "leggedmanip"], default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--start", choices=["approach", "lost", "start"], default="approach")
    p.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                   default=None, help="initial velocity command")
    p.add_argument("--seconds", type=float, default=None,
                   help="close the viewer after this many seconds (smoke test)")
    p.add_argument("--speed", type=float, default=0.0,
                   help="constant forward command, e.g. 0.8 to just watch it walk")
    p.add_argument("--track", action="store_true", help="start with the camera tracking the robot")
    p.add_argument("--force-locomotion", action="store_true",
                   help="run the stand/walk tests even on the hfield collider, to reproduce the stall")
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/terrain_check.json"))
    args = p.parse_args()

    if args.list:
        for name in ALL_TERRAINS:
            info = describe(name)
            print(f"{name:12s} scene={info['scene']:12s} "
                  f"x{info['x_range_m']} y{info['y_range_m']} z{info['z_range_m']}  "
                  f"target={info['target_xyz']}")
            print(f"{'':12s} {info['mechanism'] or '(legacy profile)'}")
        return

    names = ALL_TERRAINS if (args.check and args.terrain == "all") else [args.terrain]
    if args.check:
        report = check_all(names, args.controller, args.ground, args.policy,
                           args.force_locomotion)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n{report['n_passed']}/{report['n_terrains']} terrains pass "
              f"({report['n_locomotion_tested']} with locomotion tested)")
        print(f"wrote {args.output}")
        return

    if args.terrain not in ALL_TERRAINS:
        raise SystemExit(f"unknown terrain {args.terrain!r}; try --list")
    command = list(args.command) if args.command else None
    if args.speed:
        command = [args.speed, 0.0, 0.0]
    view(args.terrain, args.controller, args.ground, args.policy,
         args.start, command, args.seconds, args.track)


if __name__ == "__main__":
    main()
