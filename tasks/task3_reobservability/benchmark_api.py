"""Run every frozen-API policy on every TASK3 terrain.

    python tasks/task3_reobservability/benchmark_api.py
    python tasks/task3_reobservability/benchmark_api.py --scenes stairs_up pit basin

Writes ``generated/benchmark_api.json`` and prints a policy x terrain RSR table.

This is the P0 "Existing Methods Failure" harness: all baselines share the same
observation, the same action set and the same scoring, and the terrain is never
told which policy is running.  The per-scene breakdown is what makes the failure
regimes visible - a single averaged number hides exactly the thing we want.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
    from .policies import Oracle, build_policies
    from .scenes import BENCHMARK_SCENES, scene_for
except ImportError:
    import api
    import episode as ep
    from policies import Oracle, build_policies
    from scenes import BENCHMARK_SCENES, scene_for


def run(scene_names: list[str], controller: str, ground: str,
        with_nbv_geometry: bool = True, max_steps: int = ep.DEFAULT_MAX_STEPS,
        protocol: str = "B") -> dict:
    catalogue = api.scene_catalogue()
    if scene_names:
        missing = [s for s in scene_names if s not in catalogue]
        if missing:
            raise SystemExit(f"unknown scenes {missing}; known: {sorted(catalogue)}")
        catalogue = {s: catalogue[s] for s in scene_names}
    report = {
        "api_version": api.API_VERSION,
        "controller": controller,
        "protocol": protocol,
        "max_steps": max_steps,
        "step_duration_s": ep.STEP_DURATION_S,
        "action_ids": list(api.ACTION_IDS),
        "terrains": {},
        "summary": {},
    }
    policy_names = list(build_policies()) + ["Oracle"]
    episodes: dict[str, list] = {p: [] for p in policy_names}

    for name, scene in catalogue.items():
        scene_name = scene.name
        print(f"--- {scene_name}", flush=True)
        environment = api.MuJoCoEnvironment(scene, controller_name=controller,
                                            ground=ground,
                                            action_table=ep.protocol_action_table())
        try:
            observation = environment.reset()
            # Reference: what a greedy oracle achieves on this terrain.  Measured
            # once and shared, because it does not depend on the policy.
            ref_steps, ref_t, ref_winners = ep.reference_episode(environment, max_steps)
            initial = {a: environment.rollout(a) for a in api.ACTION_IDS}
            physically_endorsed = [a for a, r in initial.items() if r.reobserved]
            ep.restore(environment, ep.snapshot(environment))

            entry = {
                "mechanism": scene.mechanism,
                "oracle_reference": {"steps": ref_steps, "t_reobs_s": ref_t},
                "successful_first_actions": physically_endorsed,
                "episodes": {},
            }
            policies = build_policies()
            if with_nbv_geometry:
                policies["NBV"].attach(environment.terrain, scene)
            policies["Oracle"] = Oracle(environment)

            for pname, policy in policies.items():
                episode = ep.run_episode(policy, environment, max_steps=max_steps,
                                         protocol=protocol)
                episode.physically_endorsed = physically_endorsed
                episode.oracle_steps = ref_steps
                episode.oracle_t_reobs_s = ref_t
                # Only NBV decides from a geometric viewpoint proxy, so only NBV
                # can fail *because of* that proxy.
                episode.failure_type = ep.classify_failure(
                    episode, ref_steps, ref_t, physically_endorsed,
                    geometry_driven=getattr(policy, "uses_geometry_proxy", False))
                episodes[pname].append(episode)
                entry["episodes"][pname] = {
                    "termination": episode.termination,
                    "success": episode.success,
                    "steps_used": episode.steps_used,
                    "actions": episode.actions,
                    "t_reobs_s": episode.t_reobs_s,
                    "motion_cost_m": episode.total_motion_cost_m,
                    "yaw_rad": episode.total_yaw_rad,
                    "failure_type": episode.failure_type,
                    "memory_at_start": episode.memory_at_start,
                    "geometric_endorsed": episode.geometric_endorsed,
                }
                print(f"    {pname:18s} {episode.termination:18s} "
                      f"steps={episode.steps_used} "
                      f"{'ok' if episode.success else 'fail'}"
                      f"{'  ' + episode.failure_type if episode.failure_type else ''}")
            report["terrains"][scene_name] = entry
        finally:
            environment.close()

    for pname, rows in episodes.items():
        report["summary"][pname] = ep.summarise(rows)
    return report


def print_matrix(report: dict) -> None:
    scenes = list(report["terrains"])
    names = list(report["summary"])
    width = max(len(n) for n in names) + 2
    print(f"\n=== multi-step episode results (protocol {report['protocol']}, "
          f"budget {report['max_steps']} x {report['step_duration_s']} s) ===")
    header = ("policy".ljust(width) + "RSR@1".rjust(8) + "RSR@3".rjust(8)
              + "RSR@5".rjust(8) + "RSR@10".rjust(8) + "steps".rjust(8)
              + "T_reobs".rjust(9) + "cost_m".rjust(9) + "  failures")
    print(header)
    for name in names:
        s = report["summary"][name]
        taxonomy = ",".join(f"{k.split('_')[0]}:{v}" for k, v in
                            sorted(s["failure_taxonomy"].items())) or "-"
        print(f"{name.ljust(width)}"
              f"{s['rsr@1']:>8.2f}{s['rsr@3']:>8.2f}{s['rsr@5']:>8.2f}{s['rsr@10']:>8.2f}"
              f"{(s['mean_steps_when_solved'] if s['mean_steps_when_solved'] is not None else 0):>8}"
              f"{(s['mean_t_reobs_s'] if s['mean_t_reobs_s'] is not None else 0):>9}"
              f"{s['mean_motion_cost_m']:>9.2f}  {taxonomy}")
    print("\nterminations per policy:")
    for name in names:
        print(f"  {name:18s} {report['summary'][name]['terminations']}")


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 frozen-API benchmark")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="defaults to every terrain: the multi-terrain scenes plus "
                        "the 16 paired-history worlds")
    p.add_argument("--controller", default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--no-nbv-geometry", dest="nbv_geometry_off", action="store_true",
                   help="deny NBV the height field, to show how much the map is worth")
    p.add_argument("--max-steps", type=int, default=ep.DEFAULT_MAX_STEPS,
                   help="action budget shared by every policy (default 10)")
    p.add_argument("--protocol", choices=["A", "B"], default="B",
                   help="A = decide once and repeat open loop; B = re-decide every step")
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/benchmark_api.json"))
    args = p.parse_args()

    scenes = args.scenes
    started = time.monotonic()
    report = run(scenes, args.controller, args.ground, not args.nbv_geometry_off,
                 max_steps=args.max_steps, protocol=args.protocol)
    report["wall_time_s"] = round(time.monotonic() - started, 1)
    print_matrix(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
