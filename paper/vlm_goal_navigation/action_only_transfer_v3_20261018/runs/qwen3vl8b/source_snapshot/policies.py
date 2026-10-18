"""TASK3 baselines, all behind the frozen :class:`ReObservationPolicy` interface.

Each class implements the *core decision mechanism* of a published line of work,
not a full reproduction.  The purpose is to put them on one task with one action
set so their failure regimes can be compared:

===============  =========================================================
``Reactive``     drive forward, ignore everything
``HeadingLock``  hold the heading recorded at the loss moment (IMU only)
``HistoryOnly``  aim the heading at the remembered target and drive
``LostTargetSearch``
                 classic lost-target search: face the belief, then approach it
``NBV``          next-best-view: score each candidate by re-projecting the
                 remembered target from a *kinematic* hypothetical pose, with
                 no physics at all
``ObjectPermanence``
                 maintain a belief with a constant-velocity filter and act on
                 the filtered belief
``Learned``      a trained action-conditioned re-observability scorer
``Oracle``       physically roll out every candidate and take the best
===============  =========================================================

Only ``Oracle`` may call ``environment.rollout`` during a decision.

A note on what this comparison is designed to expose: ``NBV`` is the classic
"is this viewpoint promising" rule, and on this benchmark the promising
viewpoint is computed geometrically.  Experiment 0 already measured that the
geometric answer and the physically executed answer disagree, so ``NBV`` is
expected to fail in a way that is *not* about perception at all.  That is the
``local viewpoint optimisation`` failure mode this file exists to quantify.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from .api import ACTION_IDS, Observation, ReObservationPolicy, TargetMemory
    from .benchmark import BENCHMARK_ACTIONS
    from .experiment import action_state, visibility
except ImportError:
    from api import ACTION_IDS, Observation, ReObservationPolicy, TargetMemory
    from benchmark import BENCHMARK_ACTIONS
    from experiment import action_state, visibility


def _bearing_to(observation: Observation, point: np.ndarray) -> float:
    """Signed bearing from the robot's heading to a world point, radians."""
    d = point - observation.pose.xyz
    angle = math.atan2(float(d[1]), float(d[0])) - observation.pose.yaw
    return math.atan2(math.sin(angle), math.cos(angle))


def _turn_action(bearing: float,
                 threshold_rad: float = math.radians(45)) -> str | None:
    """Pick a turn when the bearing is outside the deadband.

    The default deadband equals the multi-step protocol's turn quantum (45 deg).
    That is a correctness requirement, not a tuning choice: a deadband smaller
    than one step's rotation makes the servo oscillate forever, because it always
    overshoots to the other side.  With ``threshold >= quantum`` a single turn
    always lands inside the band.
    """
    if bearing > threshold_rad:
        return "turn_left"
    if bearing < -threshold_rad:
        return "turn_right"
    return None


# ---------------------------------------------------------------------------
# Policies that ignore history
# ---------------------------------------------------------------------------

@dataclass
class Reactive:
    """Always drive forward.  The null hypothesis of the whole task."""

    name: str = "Reactive"
    requires: frozenset[str] = frozenset({"proprioception"})

    def reset(self, observation: Observation) -> None:
        pass

    def act(self, observation: Observation) -> str:
        return "forward"


@dataclass
class HeadingLock:
    """Baseline D: keep the heading that was held when the target was lost.

    The primitive set already contains ``heading_lock``, which closes the loop on
    the initial yaw, so this baseline is one line.  It is the strongest thing an
    IMU alone can do, and it is the attack Experiment 1 had to defeat.
    """

    name: str = "HeadingLock"
    requires: frozenset[str] = frozenset({"proprioception"})

    def reset(self, observation: Observation) -> None:
        pass

    def act(self, observation: Observation) -> str:
        return "heading_lock"


# ---------------------------------------------------------------------------
# Policies that use target memory
# ---------------------------------------------------------------------------

@dataclass
class HistoryOnly:
    """Aim the heading at the remembered target and drive there.

    This is the "belief is enough" hypothesis: if you know where the target is,
    go to it.  It has no notion of what the terrain will do to the sight line.
    """

    name: str = "HistoryOnly"
    requires: frozenset[str] = frozenset({"proprioception", "target_memory"})

    def reset(self, observation: Observation) -> None:
        pass

    def act(self, observation: Observation) -> str:
        if observation.target_memory is None:
            return "forward"
        return "toward"


@dataclass
class LostTargetSearch:
    """Radmard & Croft style search: face the belief, then close on it.

    The mechanism is visual servoing on the belief - turn until the belief is
    inside a small angular window, then drive at it.  Unlike :class:`HistoryOnly`
    it explicitly decides *whether* to turn first, which is the part that
    matters when the target was lost off to one side.
    """

    name: str = "LostTargetSearch"
    requires: frozenset[str] = frozenset({"proprioception", "target_memory"})
    face_threshold_rad: float = math.radians(45)
    close_range_m: float = 4.0

    def reset(self, observation: Observation) -> None:
        pass

    def act(self, observation: Observation) -> str:
        if observation.target_memory is None:
            return "forward"
        bearing = _bearing_to(observation, observation.target_memory.xyz)
        turn = _turn_action(bearing, self.face_threshold_rad)
        if turn is not None:
            return turn
        return "toward" if observation.target_memory.range_m > 1.5 else "forward"


class ObjectPermanence:
    """Belief filter with a constant-velocity model, then act on the belief.

    The filter is the distinguishing part: a static-target tracker would just
    hold the last sighting, whereas an object-permanence filter extrapolates.
    With a static target the extrapolation term is zero, which is exactly the
    point - **on this benchmark object permanence collapses onto the
    history-only rule**.  Keeping the filter makes that a measured result rather
    than an assumption.
    """

    name = "ObjectPermanence"
    requires = frozenset({"proprioception", "target_memory", "action_history"})

    def __init__(self, velocity_decay: float = 0.9, **_ignored):
        self.velocity_decay = velocity_decay
        self._velocity = np.zeros(3)
        self._previous: np.ndarray | None = None
        self._dt = 1.0

    def reset(self, observation: Observation) -> None:
        self._velocity = np.zeros(3)
        self._previous = (observation.target_memory.xyz.copy()
                          if observation.target_memory is not None else None)
        history = observation.action_history
        self._dt = (sum(r.duration_s for r in history) / len(history)) if history else 1.0

    def belief(self, observation: Observation) -> np.ndarray | None:
        if observation.target_memory is None:
            return None
        measured = observation.target_memory.xyz
        if self._previous is not None:
            observed_velocity = (measured - self._previous) / max(self._dt, 1e-6)
            self._velocity = (self.velocity_decay * self._velocity
                              + (1 - self.velocity_decay) * observed_velocity)
        self._previous = measured.copy()
        return measured + self._velocity * observation.target_memory.age_s

    def act(self, observation: Observation) -> str:
        belief = self.belief(observation)
        if belief is None:
            return "forward"
        bearing = _bearing_to(observation, belief)
        turn = _turn_action(bearing)
        return turn if turn is not None else "toward"


class NextBestView:
    """Score candidates by a *geometric* re-projection, with no physics.

    This is the classic active-perception rule: for each candidate, imagine the
    camera pose it would produce and ask whether the believed target would be
    inside the frustum and unoccluded.  The imagined pose is a kinematic
    displacement, which is the standard assumption and also the assumption
    Experiment 0 measured to be wrong on this robot.
    """

    name = "NBV"
    requires = frozenset({"proprioception", "target_memory"})
    #: This baseline decides from a geometric viewpoint prediction, so it is the
    #: only one that can fail *because of* that prediction (failure type C).
    uses_geometry_proxy = True

    def __init__(self, nominal_speed_mps: float = 0.8, **_ignored):
        #: Metres of forward travel per second of an action, used to place the
        #: hypothetical pose.  Deliberately a nominal constant: NBV does not know
        #: how the controller will actually move.
        self.nominal_speed_mps = nominal_speed_mps
        self._terrain = None
        self._scene = None

    def reset(self, observation: Observation) -> None:
        pass

    def attach(self, terrain, scene) -> None:
        """Give NBV the *terrain geometry* only - never physics or outcomes.

        An active-perception method is allowed a map; it is not allowed to run
        the controller.  Handing it the height field is the strongest reasonable
        version of the baseline, so a failure cannot be blamed on a weak map.
        """
        self._terrain = terrain
        self._scene = scene

    def imagine(self, observation: Observation, action_id: str):
        """Hypothetical eye pose after a kinematic displacement by ``action_id``."""
        if self._terrain is None:
            return None
        primitive = BENCHMARK_ACTIONS[action_id]
        state = {"robot": observation.pose.xyz.copy(), "yaw": observation.pose.yaw,
                 "pitch": 0.0}
        dx = primitive.vx * primitive.duration_s
        dy = primitive.vy * primitive.duration_s
        state["robot"] = state["robot"] + np.array([
            dx * math.cos(state["yaw"]) - dy * math.sin(state["yaw"]),
            dx * math.sin(state["yaw"]) + dy * math.cos(state["yaw"]), 0.0])
        state["yaw"] += primitive.wz * primitive.duration_s
        state["robot"][2] = float(self._terrain.height(float(state["robot"][0]),
                                                       float(state["robot"][1]))) + 0.30
        return state

    def act(self, observation: Observation) -> str:
        belief = (observation.target_memory.xyz if observation.target_memory is not None
                  else None)
        if belief is None or self._terrain is None:
            return "forward"
        best, best_score = "forward", -1e9
        for action_id in ACTION_IDS:
            state = self.imagine(observation, action_id)
            reading = visibility(self._terrain, state["robot"], state["yaw"],
                                 state["pitch"], belief)
            # Information gain proxy: unoccluded and centred and close.
            if not reading["visible_oracle"]:
                continue
            centred = 1.0 - min(1.0, abs(reading["bearing_rad"]) / math.radians(45))
            closeness = 1.0 / max(reading["range_m"], 0.5)
            score = centred + 0.5 * closeness
            if score > best_score:
                best, best_score = action_id, score
        return best


# ---------------------------------------------------------------------------
# Learned and oracle
# ---------------------------------------------------------------------------

class Learned:
    """A trained action-conditioned re-observability scorer.

    The scorer is injected rather than imported so this module stays free of
    training code and the interface stays frozen: a model is a callable
    ``(observation, action_id, context) -> float``.  See
    :func:`tasks.task3_reobservability.learned_policy.make_scorer` for the
    concrete model trained on the recorded rollouts.

    ``context`` carries the terrain the policy was attached to, so a scorer that
    wants the height field can use it - a learned visibility model legitimately
    may - while policies that do not need it never see it.
    """

    def __init__(self, scorer, name: str = "Learned", requires=None, **kwargs):
        self.name = name
        self.scorer = scorer
        self.requires = frozenset(requires or {"proprioception", "target_memory"})
        self._terrain = None
        self._scene = None

    def attach(self, terrain, scene) -> None:
        self._terrain = terrain
        self._scene = scene

    def reset(self, observation: Observation) -> None:
        pass

    def act(self, observation: Observation) -> str:
        context = {"terrain": self._terrain, "scene": self._scene}
        scored = [(self.scorer(observation, a, context), a) for a in ACTION_IDS]
        return max(scored)[1]


class Oracle:
    """Physically roll out every candidate and take the fastest success.

    Privileged and expensive: one physical rollout per candidate per decision.
    It is the upper bound, not a policy.
    """

    name = "Oracle"
    requires: frozenset[str] = frozenset({"proprioception", "target_memory"})

    def __init__(self, environment, **kwargs):
        self.environment = environment

    def reset(self, observation: Observation) -> None:
        pass

    def act(self, observation: Observation) -> str:
        outcomes = {a: self.environment.rollout(a) for a in ACTION_IDS}
        winners = [a for a, r in outcomes.items() if r.reobserved]
        if not winners:
            return "forward"
        return min(winners, key=lambda a: (outcomes[a].t_reobs_s or 1e9,
                                           outcomes[a].path_cost_m))


#: Registry used by the runner and the contract test.  ``Oracle`` needs the
#: environment, so it is constructed by the runner rather than listed here.
POLICY_REGISTRY = {
    "Reactive": Reactive,
    "HeadingLock": HeadingLock,
    "HistoryOnly": HistoryOnly,
    "LostTargetSearch": LostTargetSearch,
    "ObjectPermanence": ObjectPermanence,
    "NBV": NextBestView,
}


def build_policies(include_learned_scorer=None) -> dict:
    """Instantiate every non-privileged baseline, in reporting order."""
    out = {name: factory() for name, factory in POLICY_REGISTRY.items()}
    if include_learned_scorer is not None:
        out["Learned"] = Learned(include_learned_scorer)
    return out
