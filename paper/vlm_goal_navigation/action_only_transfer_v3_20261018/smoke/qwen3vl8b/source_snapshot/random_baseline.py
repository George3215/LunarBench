"""Uniform-random reference on the complete frozen scene catalogue.

Every policy must use the same scene set, action table, budget and termination
rules for a comparison. Random is a reference, not a mathematical lower bound.

Audit 2026-10-04: the archived result evaluated only eight base worlds although
the learned result covered 24. The original JSON is retained as historical data;
this corrected runner enumerates api.scene_catalogue() and saves per-world rows.
All actions are syntactically valid commands; this does not certify that they are
collision-free or remain inside the terrain. Physical terminations count as failures.

    python tasks/task3_reobservability/random_baseline.py --output NEW_RESULT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
except ImportError:
    import api
    import episode as ep

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "random_baseline.json"
SEEDS = (1, 2, 3)


class RandomPolicy:
    """Uniform choice over the frozen action set."""

    def __init__(self, seed: int, name: str = "Random"):
        self.name = name
        self.requires = frozenset({"proprioception"})
        self.rng = np.random.default_rng(seed)

    def reset(self, observation) -> None:
        pass

    def act(self, observation) -> str:
        # Sample from the frozen set. It is the same set every other method draws
        # from, so the comparison isolates action selection rather than action
        # availability.
        return str(self.rng.choice(api.ACTION_IDS))


def run(seed: int, max_steps: int) -> tuple[list, dict]:
    BENCHMARK_SCENES = list(api.scene_catalogue())
    episodes, rows = [], []
    for name in BENCHMARK_SCENES:
        scene = api.scene_catalogue()[name]
        policy = RandomPolicy(seed)
        environment = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table())
        try:
            episode = ep.run_episode(policy, environment, max_steps=max_steps,
                                     protocol="B")
            episode.failure_type = ep.classify_failure(episode, None, None, [])
            episodes.append(episode)
            rows.append({"terrain": name, "success": bool(episode.success),
                         "steps": episode.steps_used,
                         "termination": episode.termination,
                         "cost_m": episode.total_motion_cost_m,
                         "failure": episode.failure_type})
        finally:
            environment.close()
    summary = ep.summarise(episodes)
    summary["per_terrain"] = rows
    return episodes, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Uniform-random baseline")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    BENCHMARK_SCENES = list(api.scene_catalogue())
    print(f"=== Random baseline, {len(BENCHMARK_SCENES)} terrains, "
          f"budget {args.max_steps} x 3.0 s ===\n")
    print(f"  {'seed':>5s} {'RSR@1':>7s} {'RSR@3':>7s} {'RSR@5':>7s} {'RSR@10':>7s} "
          f"{'cost_m':>7s}  taxonomy")

    results = {}
    for seed in SEEDS:
        _, summary = run(seed, args.max_steps)
        results[f"seed{seed}"] = summary
        print(f"  {seed:>5d} {summary['rsr@1']:>7.3f} {summary['rsr@3']:>7.3f} "
              f"{summary['rsr@5']:>7.3f} {summary['rsr@10']:>7.3f} "
              f"{summary['mean_motion_cost_m']:>7.3f}  {summary['failure_taxonomy']}")

    mean = {k: round(float(np.mean([r[k] for r in results.values()])), 4)
            for k in ("rsr@1", "rsr@3", "rsr@5", "rsr@10")}
    cost = round(float(np.mean([r["mean_motion_cost_m"] for r in results.values()])), 3)
    print(f"\n  mean over seeds: RSR@1 {mean['rsr@1']} RSR@3 {mean['rsr@3']} "
          f"RSR@5 {mean['rsr@5']} RSR@10 {mean['rsr@10']}  cost {cost} m")
    print("Compare only with reruns using this same scene catalogue and code version.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"task": "TASK3_random_baseline", "seeds": list(SEEDS),
         "terrains": BENCHMARK_SCENES, "by_seed": results,
         "mean_over_seeds": mean, "mean_cost_m": cost,
         "note": "uniform-random reference; syntactic action mask not needed "
                 "because the frozen action set is finite and every member is "
                 "executable at every state"},
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
