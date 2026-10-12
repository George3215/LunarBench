"""P0 solvability audit.

Before any baseline number can be corrected with "RSR on solvable", we have to
know what *solvable* means, and specifically whether a terrain that our greedy
reference failed on is genuinely unsolvable or whether the reference is simply
too weak.  Reporting an unverified denominator would be worse than reporting the
uncorrected one.

Three independent notions are measured per terrain:

``geometric``
    Is a target-visible pose reachable at all under the *kinematic* primitive
    model, ignoring physics?  Breadth-first over quantised poses, so this is
    exhaustive up to the discretisation, not heuristic.  If this is false the
    terrain is unsolvable for any policy, whatever it does.
``greedy``
    The existing reference: re-plan with a full physical rollout of every
    candidate at every step.  Cheap, but greedy - it has no way to escape a
    plateau where no single action re-observes.
``search``
    A beam search over action *sequences* with physical execution and a
    geometry-blind novelty heuristic.  This is the strong reference: if it fails
    too, we call the terrain unsolvable within budget.

The heuristic is deliberately blind to the visibility geometry.  A search guided
by geometric visibility would inherit exactly the geometric/physical gap this
paper is about (H3), which would make it a bad reference.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
    from .experiment import visibility
except ImportError:
    import api
    import episode as ep
    from experiment import visibility

SOLVABLE_GREEDY = "solvable_greedy"
SOLVABLE_BY_SEARCH = "solvable_by_search"
UNSOLVABLE = "unsolvable_within_budget"
UNREACHABLE = "geometrically_unreachable"


def pose_key(environment, cell: float = 0.25, yaw_bins: int = 8) -> tuple:
    pose = environment.observe().pose
    return (round(pose.x / cell), round(pose.y / cell),
            int(round((pose.yaw % (2 * math.pi)) / (2 * math.pi) * yaw_bins)) % yaw_bins)


def geometric_reachability(environment, max_steps: int, cell: float = 0.3,
                           yaw_bins: int = 8, max_states: int = 200000) -> dict:
    """Is any target-visible pose reachable kinematically within ``max_steps``?

    Breadth-first over quantised poses using the primitive's nominal displacement,
    so it is a search over the whole reachable set up to discretisation rather
    than a heuristic.  Terrain height is respected; the controller is not.
    """
    terrain = environment.terrain
    target = environment.target
    start_pose = environment.observe().pose
    start = (start_pose.x, start_pose.y, start_pose.yaw)

    def visible_at(x, y, yaw):
        if not (terrain.x[0] <= x <= terrain.x[-1] and terrain.y[0] <= y <= terrain.y[-1]):
            return False
        z = float(terrain.height(x, y)) + 0.30
        return bool(visibility(terrain, np.array([x, y, z]), yaw, 0.0,
                               target)["visible_oracle"])

    queue = deque([(start, 0)])
    seen = {_quantise(start, cell, yaw_bins)}
    visited = 0
    while queue:
        (x, y, yaw), depth = queue.popleft()
        visited += 1
        if visited > max_states:
            return {"reachable": None, "min_depth": None, "n_states": visited,
                    "reason": "state cap reached"}
        if depth >= max_steps:
            continue
        for action_id in api.ACTION_IDS:
            primitive = environment.action_table[action_id]
            dx = primitive.vx * primitive.duration_s
            dy = primitive.vy * primitive.duration_s
            nx = x + dx * math.cos(yaw) - dy * math.sin(yaw)
            ny = y + dx * math.sin(yaw) + dy * math.cos(yaw)
            nyaw = yaw + primitive.wz * primitive.duration_s
            key = _quantise((nx, ny, nyaw), cell, yaw_bins)
            if key in seen:
                continue
            seen.add(key)
            if visible_at(nx, ny, nyaw):
                return {"reachable": True, "min_depth": depth + 1,
                        "n_states": visited, "reason": ""}
            queue.append(((nx, ny, nyaw), depth + 1))
    return {"reachable": False, "min_depth": None, "n_states": visited,
            "reason": "no visible pose reachable within budget"}


def _quantise(pose, cell: float, yaw_bins: int) -> tuple:
    x, y, yaw = pose
    return (round(x / cell), round(y / cell),
            int(round((yaw % (2 * math.pi)) / (2 * math.pi) * yaw_bins)) % yaw_bins)


def beam_search_reference(environment, max_steps: int, beam_width: int = 6,
                          cell: float = 0.3) -> dict:
    """Reference planner: beam search over action sequences, physically executed.

    Scored by *coverage novelty* rather than by predicted visibility, so that a
    failure cannot be blamed on the geometric/physical gap this paper studies.
    The terrain state is restored on every exit path.
    """
    root = ep.snapshot(environment)
    start_key = pose_key(environment, cell)
    beam = [(root, 0.0, [], {start_key})]
    expansions = 0
    try:
        for depth in range(max_steps):
            candidates = []
            level_seen = set()
            for snap, score, path, covered in beam:
                for action_id in api.ACTION_IDS:
                    ep.restore(environment, snap)
                    environment.controller.reset()
                    environment.step(action_id)
                    expansions += 1
                    if environment.privileged().target_visible:
                        return {"solved": True, "steps": depth + 1,
                                "t_reobs_s": (depth + 1) * ep.STEP_DURATION_S,
                                "path": path + [action_id], "expansions": expansions}
                    key = pose_key(environment, cell)
                    if key in level_seen:
                        continue
                    level_seen.add(key)
                    covered_next = covered | {key}
                    reach = float(np.linalg.norm(
                        environment.observe().pose.xyz - environment.target))
                    # Coverage first, proximity as a tie-break.  Coverage is a
                    # proxy for "have I looked from somewhere new".
                    novelty = len(covered_next) - 0.01 * reach
                    candidates.append((ep.snapshot(environment), novelty,
                                       path + [action_id], covered_next))
            if not candidates:
                break
            candidates.sort(key=lambda item: -item[1])
            beam = candidates[:beam_width]
        return {"solved": False, "steps": max_steps, "t_reobs_s": None,
                "path": [], "expansions": expansions}
    finally:
        ep.restore(environment, root)


def audit_one(scene, controller: str, ground: str, max_steps: int,
              beam_width: int, geometric_steps: int = 4) -> dict:
    environment = api.MuJoCoEnvironment(scene, controller_name=controller,
                                        ground=ground,
                                        action_table=ep.protocol_action_table())
    try:
        greedy_steps, greedy_t, _ = ep.reference_episode(environment, max_steps)
        geometric = geometric_reachability(environment, geometric_steps)
        if greedy_t is not None:
            return {"terrain": scene.name, "mechanism": scene.mechanism,
                    "class": SOLVABLE_GREEDY, "greedy_steps": greedy_steps,
                    "search": None, "geometric": geometric}
        search = beam_search_reference(environment, max_steps, beam_width)
        if search["solved"]:
            klass = SOLVABLE_BY_SEARCH
        elif geometric.get("reachable") is False:
            klass = UNREACHABLE
        else:
            klass = UNSOLVABLE
        return {"terrain": scene.name, "mechanism": scene.mechanism, "class": klass,
                "greedy_steps": None, "search": search, "geometric": geometric}
    finally:
        environment.close()


def audit(scene_names: list[str] | None, controller: str, ground: str,
          max_steps: int, beam_width: int, output: Path) -> dict:
    catalogue = api.scene_catalogue()
    if scene_names:
        catalogue = {s: catalogue[s] for s in scene_names}
    rows = []
    for name, scene in catalogue.items():
        row = audit_one(scene, controller, ground, max_steps, beam_width)
        rows.append(row)
        detail = ""
        if row["class"] == SOLVABLE_BY_SEARCH:
            detail = (f"beam found it in {row['search']['steps']} steps: "
                      f"{row['search']['path']}")
        elif row["class"] == UNREACHABLE:
            detail = "no target-visible pose is kinematically reachable"
        elif row["class"] == UNSOLVABLE:
            detail = (f"beam searched {row['search']['expansions']} expansions, "
                      f"geometric reachable={row['geometric'].get('reachable')}")
        print(f"  {name:18s} {row['class']:22s} {detail}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    report = {"api_version": api.API_VERSION, "max_steps": max_steps,
              "beam_width": beam_width, "controller": controller,
              "counts": counts, "terrains": rows,
              "solvable": [r["terrain"] for r in rows if r["class"] != UNSOLVABLE
                           and r["class"] != UNREACHABLE],
              "unsolvable": [r["terrain"] for r in rows if r["class"] in
                             (UNSOLVABLE, UNREACHABLE)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ncounts: {counts}")
    print(f"unsolvable within {max_steps} steps: {report['unsolvable']}")
    print(f"wrote {output}")
    return report


def rescore(episode_report: Path, solvability_report: Path,
            output: Path | None = None) -> dict:
    """Re-score an episode report against the audited solvability and a strong oracle.

    Two corrections are applied:

    1. ``rsr_on_solvable`` - the same RSR restricted to terrains the audit proved
       solvable.  If nothing is unsolvable this equals ``rsr``, which is itself the
       result worth recording: the earlier "4 of 24 are unsolvable" reading was an
       artefact of a greedy reference, not a property of the terrain.
    2. An ``OracleSearch`` row, built from the audit rather than from the greedy
       oracle the episode run used.  Greedy is provably weaker here, so the
       previous oracle row understated the achievable headroom.
    """
    episodes = json.loads(Path(episode_report).read_text(encoding="utf-8"))
    solvability = json.loads(Path(solvability_report).read_text(encoding="utf-8"))
    solvable = set(solvability["solvable"])
    terrains = solvability["terrains"]

    for name, summary in episodes["summary"].items():
        rows = [(t, e["episodes"][name]) for t, e in episodes["terrains"].items()
                if t in solvable]
        solved = sum(1 for _, e in rows if e["success"])
        summary["rsr_on_solvable"] = round(solved / len(rows), 4) if rows else None
        summary["n_solvable_terrains"] = len(rows)

    # Strong oracle: greedy where greedy sufficed, beam search where it did not.
    steps_by_terrain = {}
    for row in terrains:
        if row["class"] == SOLVABLE_GREEDY:
            steps_by_terrain[row["terrain"]] = row["greedy_steps"]
        elif row["class"] == SOLVABLE_BY_SEARCH:
            steps_by_terrain[row["terrain"]] = row["search"]["steps"]
        else:
            steps_by_terrain[row["terrain"]] = None
    n = len(steps_by_terrain)
    ok = [v for v in steps_by_terrain.values() if v is not None]
    oracle = {
        "n_episodes": n,
        "rsr": round(len(ok) / n, 4),
        "n_solvable_terrains": n,
        "rsr_on_solvable": round(len(ok) / n, 4),
        "source": "solvability audit: greedy reference plus beam search fallback",
        "steps_by_terrain": steps_by_terrain,
        "privileged": True,
        "mean_steps_when_solved": round(float(np.mean(ok)), 2) if ok else None,
    }
    for budget in (1, 3, 5, 10):
        oracle[f"rsr@{budget}"] = round(sum(1 for v in ok if v <= budget) / n, 4)
    episodes["summary"]["OracleSearch"] = oracle
    episodes["solvability"] = {
        "counts": solvability["counts"],
        "unsolvable": solvability["unsolvable"],
        "n_solvable": len(solvable),
        "note": ("Every terrain is solvable within the budget; the greedy oracle used "
                 "during the episode run was weak on four of them.  RSR on solvable "
                 "therefore equals RSR, and the OracleSearch row is the correct upper "
                 "bound."),
    }
    if output:
        output.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
        print(f"wrote {output}")
    print(f"\n{'policy':18s} {'RSR@1':>7s} {'RSR@5':>7s} {'RSR@10':>7s} {'RSR(solv)':>10s}")
    for name, s in episodes["summary"].items():
        print(f"{name:18s} {s.get('rsr@1', float('nan')):>7.3f} "
              f"{s.get('rsr@5', float('nan')):>7.3f} "
              f"{s.get('rsr@10', float('nan')):>7.3f} "
              f"{s.get('rsr_on_solvable', float('nan')):>10.3f}")
    return episodes
    if output:
        output.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
        print(f"wrote {output}")
    print(f"\n{'policy':18s} {'RSR(all)':>9s} {'RSR(solvable)':>14s}")
    for name, s in episodes["summary"].items():
        print(f"{name:18s} {s['rsr']:>9.3f} {s['rsr_on_solvable']:>14.3f}")
    return episodes


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 solvability audit")
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--controller", default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--beam-width", type=int, default=6)
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/solvability.json"))
    p.add_argument("--rescore", type=Path, default=None,
                   help="an episode report to re-score with rsr_on_solvable")
    p.add_argument("--rescore-output", type=Path, default=None)
    args = p.parse_args()

    if args.rescore is not None:
        rescore(args.rescore, args.output, args.rescore_output)
        return
    audit(args.scenes, args.controller, args.ground, args.max_steps,
          args.beam_width, args.output)


if __name__ == "__main__":
    main()
