"""E4 dataset: (O, M, a) -> executed pose consequence and re-observation.

H3 established where the missing information is.  The visibility oracle is not the
problem: placed at the pose the robot **actually reaches**, it agrees with the
physical outcome 97.9 % of the time.  What is wrong is the step before it - the
assumption that an action moves the robot to its nominal displacement.  So this
dataset records, for every (state, action):

* the observation-side features a policy actually has (pose, proprioception,
  short pose history),
* the target memory at that state,
* the action,
* and two targets: the **executed body-frame pose delta** and whether the target
  was physically re-observed.

Recording the pose delta rather than the visibility lets E4 ask the question H3
points at - can we predict ``a -> S_future`` - and then reuse the already-validated
oracle for ``S -> V`` instead of learning the whole chain end to end.

States are collected by wandering each terrain so the model sees more than one
pose per scene; every (state, action) pair is then executed for real.

    python tasks/task3_reobservability/consequence_dataset.py
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
except ImportError:
    import api
    import episode as ep

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
#: How many distinct start states per terrain.  More states means more data but a
#: longer run; each state costs |actions| physical rollouts.
STATES_PER_TERRAIN = 10
HISTORY_LEN = 8


def wander_actions() -> list[str]:
    """A fixed, diverse action cycle used to reach interesting states."""
    return ["forward", "left", "forward", "right", "back", "forward",
            "turn_left", "forward", "turn_right", "forward"]


def body_frame_delta(start_pose, end_pose) -> np.ndarray:
    """Pose change expressed in the *start* body frame: (forward, lateral, dz, dyaw)."""
    dx = end_pose.x - start_pose.x
    dy = end_pose.y - start_pose.y
    c, s = math.cos(-start_pose.yaw), math.sin(-start_pose.yaw)
    dyaw = math.atan2(math.sin(end_pose.yaw - start_pose.yaw),
                      math.cos(end_pose.yaw - start_pose.yaw))
    return np.array([dx * c - dy * s, dx * s + dy * c,
                     end_pose.z - start_pose.z, dyaw], dtype=float)


def memory_features(observation) -> list[float] | None:
    """Target memory in the body frame, or ``None`` when there is no belief.

    ``None`` rather than zeros: a policy with no memory and a policy whose memory
    is at the origin must not look the same to the model.
    """
    memory = observation.target_memory
    if memory is None:
        return None
    d = memory.xyz - observation.pose.xyz
    c, s = math.cos(-observation.pose.yaw), math.sin(-observation.pose.yaw)
    body = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2]])
    bearing = math.atan2(d[1], d[0]) - observation.pose.yaw
    return [float(body[0]), float(body[1]), float(body[2]),
            float(np.linalg.norm(d)), math.sin(bearing), math.cos(bearing),
            float(memory.age_s)]


def observation_features(observation, history: list[list[float]]) -> list[float]:
    """The observation side: pose, proprioception, and a short pose history."""
    pose = observation.pose
    gravity = observation.proprioception[-3:]
    flat_history = []
    for entry in history[-HISTORY_LEN:]:
        flat_history.extend(entry)
    while len(flat_history) < 4 * HISTORY_LEN:
        flat_history = [0.0, 0.0, 0.0, 0.0] + flat_history
    return [pose.x, pose.y, pose.z, math.sin(pose.yaw), math.cos(pose.yaw),
            *[float(g) for g in gravity], *flat_history]


def collect(controller: str, ground: str, states_per_terrain: int) -> list[dict]:
    rows: list[dict] = []
    catalogue = api.scene_catalogue()
    for name, scene in catalogue.items():
        environment = api.MuJoCoEnvironment(scene, controller_name=controller,
                                            ground=ground,
                                            action_table=ep.protocol_action_table())
        try:
            observation = environment.reset()
            snapshots = [ep.snapshot(environment)]
            observed = [observation]
            history: list[list[float]] = []
            for action_id in wander_actions()[:states_per_terrain - 1]:
                previous = environment.observe().pose
                environment.step(action_id)
                now = environment.observe()
                if environment.privileged().target_visible is None:
                    break
                if float(environment.data.xmat[1].reshape(3, 3)[2, 2]) < 0.25:
                    break
                history.append([float(v) for v in
                                body_frame_delta(previous, now.pose)])
                observed.append(now)
                snapshots.append(ep.snapshot(environment))

            n_states = len(snapshots)
            for state_id in range(n_states):
                ep.restore(environment, snapshots[state_id])
                environment.controller.reset()
                start = environment.observe()
                mem = memory_features(start)
                hist = history[:state_id]
                for action_id in api.ACTION_IDS:
                    ep.restore(environment, snapshots[state_id])
                    environment.controller.reset()
                    rollout = environment.rollout(action_id)
                    # ``rollout`` restores the simulation state on exit - that is
                    # the frozen API's guarantee, so the episode is not advanced by
                    # scoring.  The executed end pose therefore has to come from
                    # the recorded curve, not from the environment afterwards.
                    if not rollout.curve:
                        continue
                    last = rollout.curve[-1]
                    end_xyz = np.asarray(last["xyz"], dtype=float)
                    end_yaw = float(last["yaw_rad"])
                    from api import Pose
                    end_pose = Pose(float(end_xyz[0]), float(end_xyz[1]),
                                    float(end_xyz[2]), end_yaw)
                    # Safety labels are part of the consequence, not an add-on: an
                    # action that reveals the target and then leaves the terrain is
                    # not a usable re-observation action.  Both are derived from the
                    # executed end pose, since ``rollout`` restores the state.
                    margin = 0.3
                    terrain_x = environment.terrain.x
                    terrain_y = environment.terrain.y
                    oob = not (terrain_x[0] + margin <= end_xyz[0] <= terrain_x[-1] - margin
                               and terrain_y[0] + margin <= end_xyz[1] <= terrain_y[-1] - margin)
                    rows.append({
                        "terrain": name,
                        "mechanism": scene.mechanism.split(";")[0].strip() or name,
                        "state_id": state_id,
                        "start_pose": [start.pose.x, start.pose.y, start.pose.z,
                                       start.pose.yaw],
                        "observation": observation_features(start, hist),
                        "memory": mem,
                        "action": action_id,
                        "action_vec": [float(environment.action_table[action_id].vx),
                                       float(environment.action_table[action_id].vy),
                                       float(environment.action_table[action_id].wz),
                                       float(environment.action_table[action_id].duration_s)],
                        "delta_pose": [float(v) for v in
                                       body_frame_delta(start.pose, end_pose)],
                        "reobserved": bool(rollout.reobserved),
                        "t_reobs_s": rollout.t_reobs_s,
                        "fallen": bool(rollout.fallen),
                        "out_of_bounds": bool(oob),
                        "unsafe": bool(rollout.fallen or oob),
                        # Continuous stability margin at the end of the action.
                        # A binary fall label is useless here: 0 falls in 720
                        # single-action rollouts, so it has no positives to learn
                        # from, while the margin is defined everywhere.
                        "upright_end": float(last.get("upright", 1.0)),
                        "upright_min": float(min(s.get("upright", 1.0)
                                                 for s in rollout.curve)),
                    })
        finally:
            environment.close()
        print(f"  {name:18s} states={n_states} rows={len(rows)}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build the E4 consequence dataset")
    p.add_argument("--controller", default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--states", type=int, default=STATES_PER_TERRAIN)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    rows = collect(args.controller, args.ground, args.states)
    positive = sum(1 for r in rows if r["reobserved"])
    with_memory = sum(1 for r in rows if r["memory"] is not None)
    print(f"\nrows={len(rows)}  reobserved={positive} ({positive / len(rows):.3f})  "
          f"with memory={with_memory} ({with_memory / len(rows):.3f})")
    payload = {"task": "TASK3_e4_consequence_dataset", "controller": args.controller,
               "states_per_terrain": args.states, "history_len": HISTORY_LEN,
               "action_ids": list(api.ACTION_IDS), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
