"""Contract tests for the frozen TASK3 benchmark API.

Run directly - no pytest needed:

    python tasks/task3_reobservability/test_api.py

The tests exist to keep the two properties the benchmark depends on:

1.  A terrain never knows which policy is running.  This is checked by reading
    the terrain modules' source and by importing them in a subprocess to confirm
    they do not pull the policy or API modules in.
2.  An observation never carries privileged information, and a policy can only
    choose from the frozen action set.

A failure exits non-zero, so this can gate a commit.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULE_DIR))

#: Modules that describe the world.  None of them may depend on a decision rule.
TERRAIN_MODULES = ["slabs.py", "scenes.py", "terrain.py", "go2_piper_camera.py",
                   "controllers.py"]
FORBIDDEN_IN_TERRAIN = ("policies", "api", "history", "visibility_model",
                        "history_horizon", "benchmark_api")

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


# ---------------------------------------------------------------------------
# 1. terrain does not know about policies
# ---------------------------------------------------------------------------

def test_terrain_is_policy_agnostic() -> None:
    print("\n[1] terrain modules must not reference any policy")
    for name in TERRAIN_MODULES:
        source = (MODULE_DIR / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        hits = [m for m in imported
                if any(bad in m.split(".")[-1] for bad in FORBIDDEN_IN_TERRAIN)]
        check(not hits, f"{name}: imports {hits or 'no policy/api module'}")

    for name in TERRAIN_MODULES:
        source = (MODULE_DIR / name).read_text(encoding="utf-8")
        offenders = [bad for bad in ("ReObservationPolicy", "POLICY_REGISTRY")
                     if re.search(rf"\b{bad}\b", source)]
        check(not offenders, f"{name}: no policy symbols in source ({offenders or 'clean'})")


def test_terrain_imports_standalone() -> None:
    """Importing the terrain must not drag the policy stack in."""
    print("\n[2] importing terrain must not import the policy stack")
    for name in ("slabs", "scenes", "terrain"):
        code = (
            "import sys;"
            f"sys.path.insert(0, r'{MODULE_DIR}');"
            f"sys.path.insert(0, r'{ROOT}');"
            f"import {name};"
            "bad=[m for m in sys.modules if m.split('.')[-1] in "
            "('policies','api','history_horizon','visibility_model')];"
            "print('LEAK', bad) if bad else print('CLEAN')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, cwd=str(ROOT))
        out = (proc.stdout or "").strip().splitlines()
        verdict = out[-1] if out else f"no output ({proc.stderr.strip()[:120]})"
        check(verdict == "CLEAN", f"import {name} -> {verdict}")


# ---------------------------------------------------------------------------
# 2. the frozen interfaces
# ---------------------------------------------------------------------------

def test_action_set_is_frozen() -> None:
    print("\n[3] the candidate action set is frozen and shared")
    import api
    import benchmark
    check(set(api.ACTION_IDS) == set(benchmark.BENCHMARK_ACTIONS),
          "api.ACTION_IDS == benchmark.BENCHMARK_ACTIONS")
    check(len(api.ACTION_IDS) == len(set(api.ACTION_IDS)),
          "action ids are unique")
    check("primitive" not in api.ACTION_IDS, "no stray entries")


def test_observation_has_no_privileged_fields() -> None:
    print("\n[4] Observation must not carry privileged information")
    import dataclasses

    import api
    fields = {f.name for f in dataclasses.fields(api.Observation)}
    forbidden = {"target_xyz", "visible", "target_visible", "oracle", "success",
                 "label", "outcome", "future_visible"}
    overlap = fields & forbidden
    check(not overlap, f"Observation fields = {sorted(fields)}; forbidden present: "
                       f"{sorted(overlap) or 'none'}")
    expected = {"current_rgb", "current_depth", "proprioception",
                "action_history", "target_memory", "pose", "t_s"}
    check(expected <= fields, f"Observation exposes the contract fields "
                              f"(missing {sorted(expected - fields) or 'none'})")
    priv = {f.name for f in dataclasses.fields(api.PrivilegedState)}
    check("target_visible" in priv, "ground truth lives in PrivilegedState")


def test_policies_conform() -> None:
    print("\n[5] every policy conforms to ReObservationPolicy")
    import api
    import policies
    check(isinstance(policies.Reactive(), api.ReObservationPolicy),
          "Reactive is a ReObservationPolicy")
    for name, factory in policies.POLICY_REGISTRY.items():
        policy = factory()
        check(isinstance(policy, api.ReObservationPolicy), f"{name} conforms")
        check(callable(getattr(policy, "reset", None))
              and callable(getattr(policy, "act", None)),
              f"{name} implements reset/act")
        check(policy.requires <= {"current_rgb", "current_depth", "proprioception",
                                  "action_history", "target_memory", "pose", "t_s"},
              f"{name}.requires only names observation fields")
    check("Oracle" not in policies.POLICY_REGISTRY,
          "Oracle is excluded from the deployable registry")


# ---------------------------------------------------------------------------
# 3. behaviour
# ---------------------------------------------------------------------------

def test_policies_run_on_scenes() -> None:
    print("\n[6] every policy returns a legal action on real scenes")
    import api
    import policies

    scenes = ["stairs_up", "basin", "perlin"]
    for scene_name in scenes:
        environment = api.make_environment(scene_name)
        try:
            observation = environment.reset()
            check(observation.pose is not None, f"{scene_name}: observation has a pose")
            check(observation.target_memory is None or
                  observation.target_memory.range_m > 0,
                  f"{scene_name}: target memory is either absent or has a range")
            check(len(observation.action_history) > 0,
                  f"{scene_name}: the approach was replayed into action history")
            for name, factory in policies.POLICY_REGISTRY.items():
                policy = factory()
                if hasattr(policy, "attach"):
                    policy.attach(environment.terrain, environment.scene_spec)
                policy.reset(observation)
                action = policy.act(observation)
                check(action in api.ACTION_IDS,
                      f"{scene_name}/{name} -> {action!r}")
        finally:
            environment.close()


def test_oracle_is_the_upper_bound() -> None:
    print("\n[7] the oracle scores at least as well as anything else")
    import api
    from policies import Oracle

    environment = api.make_environment("basin")
    try:
        observation = environment.reset()
        rows = {a: environment.rollout(a) for a in api.ACTION_IDS}
        winners = [a for a, r in rows.items() if r.reobserved]
        oracle = Oracle(environment)
        oracle.reset(observation)
        choice = oracle.act(observation)
        if winners:
            check(choice in winners, f"oracle chose {choice!r} among {winners}")
        else:
            check(True, "no action succeeds here, oracle falls back")
        # The privileged path must not have moved the robot.
        check(environment.privileged().t_s < 1e-6,
              "counterfactual rollouts do not advance the episode")
    finally:
        environment.close()


def test_multistep_budget_is_identical() -> None:
    """The check the research plan asks for before any baseline conclusion.

    Every policy must get the same action budget, the same number of observation
    updates, the same per-step duration and the same termination rule.  This is
    verified by driving the environment with a fixed action sequence and with a
    policy, and comparing the episode accounting.
    """
    print("\n[8] every policy gets an identical budget and termination rule")
    import api
    import episode as ep
    import policies

    budget = 5
    environment = api.MuJoCoEnvironment(
        api.scene_catalogue()["perlin"], action_table=ep.protocol_action_table())
    try:
        # Same injected action sequence -> identical accounting regardless of who
        # is nominally in charge.
        sequence = ["forward", "left", "turn_left", "back", "right"]
        reference = ep.run_episode(policies.Reactive(), environment, max_steps=budget,
                                   protocol="B", decide_actions=sequence)
        check(reference.steps_used <= budget,
              f"the budget is an upper bound (used {reference.steps_used} <= {budget})")
        check((reference.termination == ep.BUDGET) == (reference.steps_used == budget),
              f"termination and step count agree ({reference.termination}, "
              f"{reference.steps_used} steps)")

        # The budget must actually be granted, not merely capped.  Forcing the
        # visibility read to always say "hidden" removes the scene from the
        # question and tests the loop itself.
        import dataclasses
        original = environment.privileged

        def never_visible():
            state = original()
            return dataclasses.replace(state, target_visible=False)

        environment.privileged = never_visible
        try:
            forced = ep.run_episode(policies.Reactive(), environment, max_steps=budget,
                                    protocol="B", decide_actions=["turn_left"] * budget)
            check(forced.steps_used == budget and forced.termination == ep.BUDGET,
                  f"a never-succeeding episode consumes the whole budget "
                  f"({forced.steps_used}/{budget}, {forced.termination})")
        finally:
            environment.privileged = original

        durations = {a: environment.action_table[a].duration_s for a in api.ACTION_IDS}
        check(len(set(durations.values())) == 1,
              f"all primitives cost the same time: {sorted(set(durations.values()))}")

        for name, factory in list(policies.POLICY_REGISTRY.items()) + [("Oracle", None)]:
            if name == "Oracle":
                policy = policies.Oracle(environment)
            else:
                policy = factory()
                if hasattr(policy, "attach"):
                    policy.attach(environment.terrain, environment.scene_spec)
            episode = ep.run_episode(policy, environment, max_steps=budget, protocol="B")
            check(episode.steps_used <= budget,
                  f"{name} never exceeds the budget ({episode.steps_used} <= {budget})")
            check(episode.termination in (ep.SUCCESS, ep.BUDGET, ep.FALLEN,
                                          ep.OUT_OF_BOUNDS),
                  f"{name} terminates on a declared condition ({episode.termination})")
            check(episode.success == (episode.termination == ep.SUCCESS),
                  f"{name}: success and termination agree")
            check(len(episode.actions) == episode.steps_used,
                  f"{name}: one recorded action per executed step")

        # Protocol A commits once; protocol B re-asks.  Both respect the budget.
        open_loop = ep.run_episode(policies.LostTargetSearch(), environment,
                                   max_steps=budget, protocol="A")
        check(len(set(open_loop.actions)) == 1,
              f"protocol A repeats one action ({open_loop.actions})")
        closed_loop = ep.run_episode(policies.LostTargetSearch(), environment,
                                     max_steps=budget, protocol="B")
        check(closed_loop.steps_used <= budget,
              f"protocol B respects the budget ({closed_loop.steps_used})")
    finally:
        environment.close()


def test_rsr_curve_is_monotone() -> None:
    print("\n[9] the RSR@N curve is monotone in N by construction")
    import episode as ep

    class Always:
        name = "Always"
        requires = frozenset({"proprioception"})

        def reset(self, observation):
            pass

        def act(self, observation):
            return "forward"

    import api
    environment = api.MuJoCoEnvironment(
        api.scene_catalogue()["basin"], action_table=ep.protocol_action_table())
    try:
        results = [ep.run_episode(Always(), environment, max_steps=5, protocol="B")
                   for _ in range(2)]
        curve = [ep.rsr_at(results, n) for n in (1, 2, 3, 4, 5)]
        check(all(b >= a for a, b in zip(curve, curve[1:])),
              f"RSR@N is non-decreasing: {curve}")
        check(ep.rsr_at(results, 5) == round(sum(e.success for e in results)
                                             / len(results), 4),
              "RSR at the full budget equals the success rate")
    finally:
        environment.close()


def test_rsr_matches_its_definition() -> None:
    """RSR@K must equal the stated definition, evaluated on the visibility trace.

    ``RSR@K = (1/N) sum_i 1[ exists h <= K : V_{t+h} = 1 ]``

    The implementation reads the trace directly, but an episode also breaks at its
    first re-observation, so a second reading is available via
    ``success and steps_used <= K``.  Checking that the two agree turns the
    equivalence from an assumption into a test: if episode termination ever stops
    happening at first re-observation, this fails immediately rather than quietly
    changing what the headline number means.
    """
    print("\n[10] RSR@K equals its definition on the visibility trace")
    import api
    import episode as ep

    class Sweep:
        """Cycles actions so that visibility is non-monotonic where the terrain allows."""

        name = "Sweep"
        requires = frozenset({"proprioception"})

        def __init__(self):
            self.k = 0

        def reset(self, observation):
            self.k = 0

        def act(self, observation):
            action = api.ACTION_IDS[self.k % len(api.ACTION_IDS)]
            self.k += 1
            return action

    for scene_name in ("waves", "basin", "stairs_up"):
        environment = api.MuJoCoEnvironment(
            api.scene_catalogue()[scene_name], action_table=ep.protocol_action_table())
        try:
            results = [ep.run_episode(Sweep(), environment, max_steps=6, protocol="B")
                       for _ in range(2)]
        finally:
            environment.close()
        for n in range(1, 7):
            from_trace = round(sum(1 for e in results
                                   if any(s.target_visible for s in e.steps[:n]))
                               / len(results), 4)
            from_termination = round(sum(
                1 for e in results if e.success and e.steps_used <= n) / len(results), 4)
            check(ep.rsr_at(results, n) == from_trace,
                  f"{scene_name}: RSR@{n} is computed from the trace")
            check(from_trace == from_termination,
                  f"{scene_name}: RSR@{n} trace and termination agree "
                  f"({from_trace} vs {from_termination})")
        # the first re-observation step must be recoverable and consistent
        for episode in results:
            first = ep.first_reobservation_step(episode)
            check((first is not None) == episode.success,
                  f"{scene_name}: first re-observation exists iff the episode succeeded")
            check(len(ep.visibility_trace(episode)) == len(episode.steps),
                  f"{scene_name}: the visibility trace covers every step")


def test_terrain_taxonomy_is_complete() -> None:
    """Every benchmark scene must carry a documented occlusion mechanism."""
    print("\n[11] every benchmark scene has an occlusion mechanism entry")
    from scenes import BENCHMARK_SCENES, OCCLUSION_MECHANISMS, mechanism_table

    for name in BENCHMARK_SCENES:
        check(name in OCCLUSION_MECHANISMS,
              f"{name} has a mechanism entry")
        entry = OCCLUSION_MECHANISMS[name]
        for field in ("label", "axis", "position", "real_world", "visibility_effect",
                      "physical_effect", "typical_recovery", "task"):
            check(bool(entry.get(field)), f"{name}.{field} is documented")
    rows = mechanism_table()
    check(len(rows) == len(BENCHMARK_SCENES),
          f"the mechanism table covers all {len(BENCHMARK_SCENES)} scenes")
    factors = {row["axis"] for row in rows}
    check(len(factors) >= 4,
          f"the mechanisms span at least four orthogonal axes: {sorted(factors)}")


def main() -> int:
    print(f"TASK3 frozen API contract tests")
    test_terrain_is_policy_agnostic()
    test_terrain_imports_standalone()
    test_action_set_is_frozen()
    test_observation_has_no_privileged_fields()
    test_policies_conform()
    test_policies_run_on_scenes()
    test_oracle_is_the_upper_bound()
    test_multistep_budget_is_identical()
    test_rsr_curve_is_monotone()
    test_rsr_matches_its_definition()
    test_terrain_taxonomy_is_complete()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
