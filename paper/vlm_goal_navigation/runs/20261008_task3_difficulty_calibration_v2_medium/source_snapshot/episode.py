"""TASK3 multi-step episode protocol.

A single decision from a fixed state is not a fair test of a search algorithm.
``LostTargetSearch`` is literally defined as "turn to face the belief, *then*
close on it"; scoring only its first step reports a property of the protocol,
not of the algorithm.  This module runs an episode instead:

    reset -> decision -> step -> target visible?
                               |-- yes -> success
                               `-- no  -> decision -> step -> ... -> budget

Two protocols share one budget, one observation model and one termination rule:

**Protocol A (open loop)**
    The policy is asked once and its action is repeated for the whole budget
    without re-observing.  This is the "commit to a plan" behaviour.
**Protocol B (closed loop)**
    The policy is re-asked after every step on the new observation.  This is
    :math:`\\pi(a_t \\mid H_t)` and is the protocol the benchmark now reports.

Fairness rules, enforced by ``test_api.py``:

* every policy gets the same ``max_steps`` and the same per-step duration;
* every policy sees the same observation stream length;
* every policy stops on the same four conditions;
* a policy that returns an illegal action is an error, not a silent fallback.

Metrics kept separate rather than folded into one score: success, ``T_reobs``,
action cost at the end of the episode, yaw change, and the number of steps.  A
combined :math:`J = 1_{reobs} - \\lambda C` is deliberately not reported, because
the weights would be a choice made before the data exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from . import api
    from .experiment import visibility
except ImportError:
    import api
    from experiment import visibility

#: One step of every primitive costs this much simulated time.
#:
#: The value is not arbitrary.  A step has to be long enough that a single action
#: *can* re-observe, otherwise the benchmark degenerates into "chain four moves"
#: and every policy fails at step one for a reason that has nothing to do with its
#: decision rule.  Measured on seven scenes, counting how many actions re-observe
#: within one step:
#:
#:     step    scenes with at least one working action
#:     1.5 s   3 of 7   (and zero working actions on four scenes)
#:     3.0 s   7 of 7
#:     4.5 s   7 of 7
#:     6.0 s   7 of 7
#:
#: The greedy oracle solved 6 of 7 within a five-step budget at 3.0 s, versus 2 of
#: 7 at 1.5 s.  3.0 s keeps the per-step reach close to the original 6 s benchmark
#: while still leaving room for multi-step search to matter.
STEP_DURATION_S = 3.0

#: How far one turn step should rotate.  This has to be small enough for a
#: closed-loop policy to servo on: a deadband can only converge if one step does
#: not jump clean over it.
TURN_PER_STEP_RAD = math.radians(45.0)

#: The yaw *command* that actually produces :data:`TURN_PER_STEP_RAD` in one step,
#: one constant per direction because the vendored policy turns asymmetrically.
#:
#: This has to be measured rather than derived from a nominal rate.  The policy's
#: command-to-yaw map is strongly non-linear, and the two directions differ.
#: Measured over one 3 s step on ``waves``::
#:
#:     command |wz|   turn_left   turn_right
#:          0.20          2.6 deg
#:          0.26          3.7 deg      <- what a nominal-rate rescaling produced
#:          0.40          7.0 deg
#:          0.45         10.3 deg      18.5 deg
#:          0.50         14.3 deg      24.5 deg
#:          0.55         22.0 deg      31.7 deg
#:          0.60         29.4 deg      28.5 deg
#:          0.65         33.3 deg      44.5 deg   <- chosen for turn_right
#:          0.75         45.5 deg      61.6 deg   <- chosen for turn_left
#:          1.20        138.0 deg
#:
#: Deriving the command from ``TURN_PER_STEP_RAD / duration`` assumed a 1:1
#: command-to-rad/s mapping and delivered 3.7 deg instead of 45 - twelve times too
#: little, which left threshold-based policies unable to ever close on a bearing.
TURN_COMMAND_PER_STEP = {"turn_left": 0.75, "turn_right": -0.65}


def protocol_action_table(duration_s: float = STEP_DURATION_S) -> dict:
    """The primitive table the multi-step protocol executes.

    Same action ids as the frozen set, but every action lasts ``duration_s`` and
    the turn rate is rescaled so one step turns :data:`TURN_PER_STEP_RAD`.  This
    is a protocol choice, not an API change: the environment simply executes the
    table it is handed.
    """
    from dataclasses import replace

    from benchmark import BENCHMARK_ACTIONS
    table = {}
    for name, primitive in BENCHMARK_ACTIONS.items():
        if name in ("turn_left", "turn_right") and primitive.wz != 0.0:
            wz = TURN_COMMAND_PER_STEP[name]
            table[name] = replace(primitive, duration_s=duration_s, wz=wz,
                                  note=f"protocol turn: one step = "
                                       f"{math.degrees(TURN_PER_STEP_RAD):.0f} deg "
                                       f"at command {wz:+.2f}")
        else:
            table[name] = replace(primitive, duration_s=duration_s)
    return table
#: Default budget.  ``RSR@1/3/5/10`` are all derived from a single run at the
#: largest budget, because a success at step k is also a success at any N >= k.
DEFAULT_MAX_STEPS = 10

SUCCESS = "success"
BUDGET = "budget_exhausted"
FALLEN = "fallen"
OUT_OF_BOUNDS = "out_of_bounds"

# Failure taxonomy, mirroring the research plan.
BELIEF_ABSENT = "A_belief_absent"
BELIEF_WRONG = "A_belief_wrong"
ACTION_WRONG = "B_action_wrong"
LOCAL_VIEWPOINT = "C_local_viewpoint"
INEFFICIENT = "D_inefficient"
OOD = "E_ood"          # requires a train/test split, not produced here
#: The action was selected for a defensible reason and the physical outcome was
#: not anticipated: the episode ended by falling or by leaving the terrain.
#: Distinct from ``B``, where a perfectly good action existed and was not chosen.
PHYSICAL_TERMINATION = "F_physical_termination"


@dataclass
class StepRecord:
    index: int
    action_id: str
    phase: str              # "reobservation"
    t_episode_s: float
    target_visible: bool
    fallen: bool
    displacement_m: float
    turn_rad: float
    motion_cost_m: float
    upright_min: float


@dataclass
class Episode:
    scene: str
    policy: str
    protocol: str
    max_steps: int
    termination: str
    success: bool
    steps_used: int
    t_reobs_s: float | None
    total_motion_cost_m: float
    total_yaw_rad: float
    total_displacement_m: float
    final_upright: float
    steps: list[StepRecord] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    phase_boundary: dict = field(default_factory=dict)
    geometric_endorsed: list[str] = field(default_factory=list)
    physically_endorsed: list[str] = field(default_factory=list)
    oracle_steps: int | None = None
    oracle_t_reobs_s: float | None = None
    failure_type: str | None = None
    #: Whether the robot had any target belief when the episode started.  Kept on
    #: the episode rather than probed later so the taxonomy is reproducible.
    memory_at_start: bool = True
    note: str = ""


def upright(data) -> float:
    return float(data.xmat[1].reshape(3, 3)[2, 2])


def out_of_bounds(environment, margin: float = 0.3) -> bool:
    terrain = environment.terrain
    x, y = float(environment.data.qpos[0]), float(environment.data.qpos[1])
    return not (terrain.x[0] + margin <= x <= terrain.x[-1] - margin
                and terrain.y[0] + margin <= y <= terrain.y[-1] - margin)


def geometric_endorsed_actions(environment, observation) -> list[str]:
    """Actions whose *kinematic* pose would put the target in view.

    This is the classic active-perception prediction, and it is what
    ``NextBestView`` uses.  Comparing it with the physically successful set is
    how the ``local viewpoint`` failure gets detected rather than asserted.
    """
    terrain = environment.terrain
    target = environment.target
    endorsed = []
    for action_id in api.ACTION_IDS:
        primitive = environment.action_table[action_id]
        yaw = observation.pose.yaw
        dx = primitive.vx * primitive.duration_s
        dy = primitive.vy * primitive.duration_s
        x = observation.pose.x + dx * math.cos(yaw) - dy * math.sin(yaw)
        y = observation.pose.y + dx * math.sin(yaw) + dy * math.cos(yaw)
        yaw_after = yaw + primitive.wz * primitive.duration_s
        if not (terrain.x[0] <= x <= terrain.x[-1] and terrain.y[0] <= y <= terrain.y[-1]):
            continue
        z = float(terrain.height(x, y)) + 0.30
        if visibility(terrain, np.array([x, y, z]), yaw_after, 0.0, target)["visible_oracle"]:
            endorsed.append(action_id)
    return endorsed


def snapshot(environment):
    """Save the integration state, so a reference search can be undone."""
    import mujoco
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    buffer = np.empty(mujoco.mj_stateSize(environment.model, spec), dtype=np.float64)
    mujoco.mj_getState(environment.model, environment.data, buffer, spec)
    return spec, buffer


def restore(environment, saved) -> None:
    import mujoco
    spec, buffer = saved
    mujoco.mj_setState(environment.model, environment.data, buffer, spec)
    mujoco.mj_forward(environment.model, environment.data)
    environment.controller.reset()


def reference_episode(environment, max_steps: int) -> tuple[int, float | None, list[str]]:
    """What the oracle best-first policy would need: steps and time to re-observe.

    The oracle is re-planned greedily at every step, so this is an achievable
    reference, not an offline optimum.  The episode state is saved and restored,
    so this can be measured once per terrain and shared by every policy.
    """
    saved = snapshot(environment)
    try:
        environment.controller.reset()
        for k in range(1, max_steps + 1):
            outcomes = {a: environment.rollout(a) for a in api.ACTION_IDS}
            winners = [a for a, r in outcomes.items() if r.reobserved]
            if not winners:
                continue
            best = min(winners, key=lambda a: (outcomes[a].t_reobs_s or 1e9,
                                               outcomes[a].path_cost_m))
            environment.step(best)
            if environment.privileged().target_visible:
                return k, float(k * STEP_DURATION_S), winners
        return max_steps, None, []
    finally:
        restore(environment, saved)


def run_episode(policy, environment, max_steps: int = DEFAULT_MAX_STEPS,
                protocol: str = "B", decide_actions: list[str] | None = None) -> Episode:
    """One multi-step episode under a fixed budget.

    ``protocol`` ``"A"`` asks the policy once and repeats the answer without
    re-observing; ``"B"`` re-asks after every step.  ``decide_actions`` injects a
    pre-decided action sequence (used by the contract test to verify that the
    budget is identical across policies).
    """
    if protocol not in ("A", "B"):
        raise ValueError(f"protocol must be 'A' or 'B', got {protocol!r}")
    observation = environment.reset()
    policy.reset(observation)
    start_pose = observation.pose
    memory_at_start = observation.target_memory is not None
    approved = geometric_endorsed_actions(environment, observation)

    steps: list[StepRecord] = []
    actions: list[str] = []
    termination = BUDGET
    t_reobs = None
    committed: str | None = None
    total_cost = total_disp = total_yaw = 0.0
    previous_pose = start_pose

    for k in range(max_steps):
        if decide_actions is not None:
            action_id = decide_actions[k]
            if action_id not in api.ACTION_IDS:
                raise ValueError(f"injected action {action_id!r} is not in the frozen set")
        elif protocol == "A":
            if committed is None:
                committed = policy.act(observation)
            action_id = committed
        else:
            action_id = policy.act(observation)
        if action_id not in api.ACTION_IDS:
            raise ValueError(f"policy {policy.name!r} returned {action_id!r}, which "
                             f"is not in the frozen action set")
        actions.append(action_id)
        observation = environment.step(action_id)
        reading = environment.privileged()
        pose = observation.pose
        displacement = float(np.hypot(pose.x - previous_pose.x, pose.y - previous_pose.y))
        turn = float(math.atan2(math.sin(pose.yaw - previous_pose.yaw),
                                math.cos(pose.yaw - previous_pose.yaw)))
        record = StepRecord(
            index=k + 1, action_id=action_id, phase="reobservation",
            t_episode_s=observation.t_s, target_visible=reading.target_visible,
            fallen=upright(environment.data) < 0.25,
            displacement_m=displacement, turn_rad=turn,
            motion_cost_m=displacement,
            upright_min=upright(environment.data))
        steps.append(record)
        total_cost += displacement
        total_disp += displacement
        total_yaw += abs(turn)
        previous_pose = pose

        if reading.target_visible:
            termination, success = SUCCESS, True
            t_reobs = observation.t_s
            break
        if record.fallen:
            termination, success = FALLEN, False
            break
        if out_of_bounds(environment):
            termination, success = OUT_OF_BOUNDS, False
            break
    else:
        success = False

    episode = Episode(
        scene=environment.name, policy=policy.name, protocol=protocol,
        max_steps=max_steps, termination=termination, success=success,
        steps_used=len(steps), t_reobs_s=t_reobs,
        total_motion_cost_m=round(total_cost, 4),
        total_yaw_rad=round(total_yaw, 4),
        total_displacement_m=round(total_disp, 4),
        final_upright=round(upright(environment.data), 4),
        steps=steps, actions=actions,
        phase_boundary={
            # How much of the observation stream is the walk-in versus the search.
            # History Horizon needs this split: "H4" must mean four observations
            # before the decision, not "the last four frames of everything".
            "approach_entries": sum(1 for r in observation.action_history
                                    if r.action_id == api.APPROACH_ACTION_ID),
            "reobservation_steps": len(steps),
            "loss_t_s": 0.0,        # the episode starts at the lost state
            "decision_t_s": 0.0,
        },
        geometric_endorsed=approved, physically_endorsed=[],
        memory_at_start=memory_at_start)
    return episode


def classify_failure(episode: Episode, reference_steps: int | None,
                     reference_t: float | None, physically_endorsed: list[str],
                     geometry_driven: bool = False) -> str | None:
    """Assign one of the taxonomy labels.

    ``A_belief_absent``  the robot had no target belief at the decision
    ``C_local_viewpoint`` the policy *is* driven by a geometric viewpoint proxy,
                          its first action is one that proxy endorses, and physics
                          rejects it - so the proxy is the cause, not the search
    ``E_physical_termination`` the episode ended by falling or leaving the terrain
    ``B_action_wrong``    belief present, geometry not the explanation, still wrong
    ``D_inefficient``     it succeeds, but takes far longer than the reference

    ``geometry_driven`` gates ``C`` deliberately.  Without it, any policy whose
    blind first action happens to coincide with a geometric prediction would be
    labelled as a viewpoint failure, which attributes a coincidence to a
    mechanism the policy does not have.

    ``E_physical_termination`` exists because lumping falls and boundary exits into
    ``B`` misattributes them.  A consequence-driven planner can pick an action that
    its model genuinely expects to reveal the target and be right about the
    visibility while being wrong about **stability** - the action was not chosen
    badly, the physical outcome was not predicted.  That is a different defect from
    "the belief was right and the search picked wrong", and the learned planner is
    exactly where it shows up.
    """
    if episode.success:
        if reference_t is not None and episode.t_reobs_s is not None:
            if episode.t_reobs_s > 2.0 * reference_t and episode.steps_used > 1:
                return INEFFICIENT
        return None
    if not episode.memory_at_start:
        return BELIEF_ABSENT
    if episode.termination in (FALLEN, OUT_OF_BOUNDS):
        return PHYSICAL_TERMINATION
    if (geometry_driven and episode.actions
            and episode.actions[0] in episode.geometric_endorsed
            and physically_endorsed
            and episode.actions[0] not in physically_endorsed):
        return LOCAL_VIEWPOINT
    return ACTION_WRONG


def rsr_at(results: list[Episode], n: int) -> float:
    """Re-observation Success Rate within ``n`` actions.

    The definition is the one the paper states, evaluated literally:

        r_i(K) = 1[ exists h <= K : V_{t+h} = 1 ]
        RSR@K  = (1/N) * sum_i r_i(K)

    It is computed from the recorded per-step visibility trace rather than from
    ``success and steps_used <= K``.  The two agree today because an episode breaks
    at its first re-observation, but deriving the metric from the trace means the
    metric stays correct if that ever changes - and it makes the quantity
    auditable step by step instead of resting on a termination convention.

    Note what the definition deliberately does *not* require: visibility to be
    held.  Visibility in this benchmark is non-monotonic (``1 -> 0 -> 1 -> 0 -> 1``
    under ``waves``), and the task is to *regain* sight of the target within the
    budget, not to keep it.
    """
    if not results:
        return 0.0
    return round(sum(1 for e in results
                     if any(s.target_visible for s in e.steps[:n])) / len(results), 4)


def first_reobservation_step(episode: Episode) -> int | None:
    """Index of the first step at which the target was re-observed, else ``None``."""
    for record in episode.steps:
        if record.target_visible:
            return record.index
    return None


def visibility_trace(episode: Episode) -> list[int]:
    """The per-step visibility indicator, for auditing non-monotonic behaviour."""
    return [1 if s.target_visible else 0 for s in episode.steps]


def summarise(results: list[Episode], budget_points=(1, 3, 5, 10)) -> dict:
    n = len(results)
    if not n:
        return {"n_episodes": 0}
    solved = [e for e in results if e.success]
    failed = [e for e in results if not e.success]
    taxonomy: dict[str, int] = {}
    for e in results:
        if e.failure_type:
            taxonomy[e.failure_type] = taxonomy.get(e.failure_type, 0) + 1
    terminations: dict[str, int] = {}
    for e in results:
        terminations[e.termination] = terminations.get(e.termination, 0) + 1
    return {
        "n_episodes": n,
        "rsr": round(len(solved) / n, 4),
        **{f"rsr@{b}": rsr_at(results, b) for b in budget_points},
        "mean_steps_when_solved": (round(float(np.mean([e.steps_used for e in solved])), 2)
                                   if solved else None),
        "mean_t_reobs_s": (round(float(np.mean([e.t_reobs_s for e in solved
                                                if e.t_reobs_s is not None])), 3)
                           if solved else None),
        # Action cost is reported over *every* episode, not only the solved ones,
        # so a policy cannot look cheap by running out of budget; the solved-only
        # figure sits alongside.  RSR alone cannot separate a policy that
        # re-observes in one step from one that wanders for ten.
        "mean_motion_cost_m": round(float(np.mean([e.total_motion_cost_m
                                                   for e in results])), 3),
        "mean_motion_cost_on_success_m": (round(float(np.mean(
            [e.total_motion_cost_m for e in solved])), 3) if solved else None),
        "terminations": terminations,
        "failure_taxonomy": taxonomy,
        "n_failed": len(failed),
    }
