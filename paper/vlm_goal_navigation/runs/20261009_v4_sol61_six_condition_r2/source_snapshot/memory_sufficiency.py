"""TASK3 H2 runner: is a correct target memory sufficient to choose the action?

The geometric search in ``memory_pairs`` only guarantees that the two worlds share
their visible geometry.  Three claims still have to be *measured*:

1. ``O_A = O_B``   - render the injected camera at the decision point in both
                     worlds and compare pixels.
2. ``M_A = M_B``   - compare the reconstructed target memory (position, age,
                     range) at the decision point.
3. ``a_A* != a_B*`` - physically execute every candidate action in both worlds and
                     compare the successful sets and the fastest success.

A pair is only accepted if all three hold and both worlds are solvable.

    python tasks/task3_reobservability/memory_sufficiency.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
    from .memory_pairs import PairParams, build_terrain, scene_for_side
    from .policies import build_policies
except ImportError:
    import api
    import episode as ep
    from memory_pairs import PairParams, build_terrain, scene_for_side
    from policies import build_policies

ROOT = Path(__file__).resolve().parents[2]


def observe(scene, params: PairParams, side: str, render: bool = True) -> dict:
    """Camera image, memory and pose at the decision point of one world."""
    import mujoco
    from go2_piper_camera import build_model

    terrain = build_terrain(params, side)
    scene = scene_for_side(params, side)
    from experiment import target_anchor
    target = target_anchor(terrain, scene)
    model = build_model(terrain, target, "auto")
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    x, y = scene.lost_xy
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [x, y, terrain.height(x, y) + 0.30]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = 0.0
    mujoco.mj_forward(model, data)

    out = {"target": target.tolist()}
    if render:
        from go2_piper_camera import CAMERA_NAME
        renderer = mujoco.Renderer(model, height=240, width=320)
        try:
            renderer.update_scene(data, camera=CAMERA_NAME)
            out["rgb"] = np.asarray(renderer.render()).copy()
        finally:
            renderer.close()
    return out


def compare(a: dict, b: dict) -> dict:
    ra, rb = a.get("rgb"), b.get("rgb")
    if ra is None or rb is None:
        return {"compared": False}
    diff = np.abs(ra.astype(np.int16) - rb.astype(np.int16))
    return {
        "compared": True,
        "mean_abs_pixel_diff": float(diff.mean()),
        "max_abs_pixel_diff": int(diff.max()),
        "pixels_identical_fraction": float((diff.max(axis=2) == 0).mean()),
    }


def memory_at_decision(scene, params: PairParams, side: str) -> dict:
    """The target memory the frozen API would hand a policy here."""
    environment = api.MuJoCoEnvironment(scene_for_side(params, side),
                                        action_table=ep.protocol_action_table(),
                                        terrain=build_terrain(params, side))
    try:
        observation = environment.reset()
        memory = observation.target_memory
        return {
            "present": memory is not None,
            "x": None if memory is None else round(memory.x, 6),
            "y": None if memory is None else round(memory.y, 6),
            "z": None if memory is None else round(memory.z, 6),
            "range_m": None if memory is None else round(memory.range_m, 6),
            "approach_entries": sum(1 for r in observation.action_history
                                    if r.action_id == api.APPROACH_ACTION_ID),
        }
    finally:
        environment.close()


def outcomes(params: PairParams, side: str, controller: str, ground: str) -> dict:
    """Physically execute every candidate from the decision point in one world."""
    environment = api.MuJoCoEnvironment(scene_for_side(params, side),
                                        controller_name=controller, ground=ground,
                                        action_table=ep.protocol_action_table(),
                                        terrain=build_terrain(params, side))
    try:
        environment.reset()
        result = {}
        for action_id in api.ACTION_IDS:
            rollout = environment.rollout(action_id)
            result[action_id] = {"reobserved": bool(rollout.reobserved),
                                 "t_reobs_s": rollout.t_reobs_s,
                                 "path_cost_m": round(rollout.path_cost_m, 4)}
        return result
    finally:
        environment.close()


def policy_choices(params: PairParams, side: str, controller: str, ground: str,
                   outcomes: dict) -> dict:
    """What each baseline would do here, and whether that action succeeds.

    The policies are re-run live because their choice depends on the observation
    (memory, pose), not on the outcome table.
    """
    from policies import build_policies
    scene = scene_for_side(params, side)
    terrain = build_terrain(params, side)
    environment = api.MuJoCoEnvironment(scene, controller_name=controller,
                                        ground=ground,
                                        action_table=ep.protocol_action_table(),
                                        terrain=terrain)
    try:
        observation = environment.reset()
        policies = build_policies()
        policies["NBV"].attach(terrain, scene)
        out = {}
        for name, policy in policies.items():
            policy.reset(observation)
            action = policy.act(observation)
            out[name] = {"action": action,
                         "success": bool(outcomes.get(action, {}).get("reobserved"))}
        return out
    finally:
        environment.close()


def optimal(successes: dict) -> str | None:
    winners = [a for a, r in successes.items() if r["reobserved"]]
    if not winners:
        return None
    return min(winners, key=lambda a: (successes[a]["t_reobs_s"] or 1e9,
                                       successes[a]["path_cost_m"]))


def run_pair(params: PairParams, controller: str, ground: str, render: bool) -> dict:
    a = observe(None, params, "a", render)
    b = observe(None, params, "b", render)
    equality = compare(a, b)
    mem_a = memory_at_decision(None, params, "a")
    mem_b = memory_at_decision(None, params, "b")
    memory_identical = (mem_a["present"] and mem_b["present"]
                        and mem_a["x"] == mem_b["x"] and mem_a["y"] == mem_b["y"]
                        and mem_a["z"] == mem_b["z"])
    out_a = outcomes(params, "a", controller, ground)
    out_b = outcomes(params, "b", controller, ground)
    win_a = [k for k, v in out_a.items() if v["reobserved"]]
    win_b = [k for k, v in out_b.items() if v["reobserved"]]
    opt_a, opt_b = optimal(out_a), optimal(out_b)
    pol_a = policy_choices(params, "a", controller, ground, out_a)
    pol_b = policy_choices(params, "b", controller, ground, out_b)

    verdict = {
        "observations_identical": bool(equality.get("compared")
                                       and equality["max_abs_pixel_diff"] == 0),
        "memory_identical": bool(memory_identical),
        "optimal_actions_differ": bool(opt_a is not None and opt_b is not None
                                       and opt_a != opt_b),
        "both_solvable": bool(win_a and win_b),
        "no_common_successful_action": bool(win_a and win_b
                                            and not set(win_a) & set(win_b)),
        "some_policy_fails_in_A_but_not_B": bool(
            any((not pol_a[n]["success"]) and pol_b[n]["success"] for n in pol_a)),
    }
    return {
        "pair_id": params.pair_id,
        "observation_equality": equality,
        "memory_world_A": mem_a, "memory_world_B": mem_b,
        "successful_actions_A": win_a, "successful_actions_B": win_b,
        "optimal_action_A": opt_a, "optimal_action_B": opt_b,
        "outcomes_A": out_a, "outcomes_B": out_b,
        "policy_choices_A": pol_a, "policy_choices_B": pol_b,
        "verdict": verdict,
        "accepted": (verdict["observations_identical"] and verdict["memory_identical"]
                     and verdict["optimal_actions_differ"] and verdict["both_solvable"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 H2 memory-sufficiency runner")
    p.add_argument("--pairs-file", type=Path,
                   default=Path("tasks/task3_reobservability/generated/memory_pairs.json"))
    p.add_argument("--controller", default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/memory_sufficiency.json"))
    args = p.parse_args()

    payload = json.loads(args.pairs_file.read_text(encoding="utf-8"))
    params_list = [PairParams(**entry) for entry in payload["pairs"]]
    if args.limit:
        params_list = params_list[:args.limit]

    results = []
    for params in params_list:
        print(f"=== pair {params.pair_id}", flush=True)
        row = run_pair(params, args.controller, args.ground, args.render)
        eq = row["observation_equality"]
        print(f"    observation: max pixel diff "
              f"{eq.get('max_abs_pixel_diff')}  identical {eq.get('pixels_identical_fraction')}")
        print(f"    memory A present={row['memory_world_A']['present']} "
              f"B present={row['memory_world_B']['present']} identical="
              f"{row['verdict']['memory_identical']}")
        print(f"    A optimal={row['optimal_action_A']} wins={row['successful_actions_A']}")
        print(f"    B optimal={row['optimal_action_B']} wins={row['successful_actions_B']}")
        print(f"    verdict: {row['verdict']}")
        results.append(row)

    n = len(results)
    aggregate = {
        "n_pairs": n,
        "pairs_with_identical_observations":
            sum(1 for r in results if r["verdict"]["observations_identical"]),
        "pairs_with_identical_memory":
            sum(1 for r in results if r["verdict"]["memory_identical"]),
        "pairs_with_different_optimal_actions":
            sum(1 for r in results if r["verdict"]["optimal_actions_differ"]),
        "pairs_with_no_common_successful_action":
            sum(1 for r in results if r["verdict"]["no_common_successful_action"]),
        "pairs_accepted": sum(1 for r in results if r["accepted"]),
    }
    report = {"task": "TASK3_experiment5_memory_sufficiency",
              "controller": args.controller, "aggregate": aggregate, "pairs": results}
    print("\n=== aggregate ===")
    for key, value in aggregate.items():
        print(f"    {key}: {value}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
