"""TASK3 frozen benchmark API.

Everything downstream of this module talks to a terrain only through
:class:`Environment`, and to a decision rule only through
:class:`ReObservationPolicy`.  The point is that **a terrain never knows which
policy is running**, so adding a baseline can never require touching terrain
code, and adding a terrain can never require touching policy code.

    Environment                          (owns physics, owns the camera)
        |  reset()/step() give an Observation
        v
    Observation                          (everything a policy may look at)
        |-- current_rgb        W x H x 3 uint8, or None if not requested
        |-- current_depth      W x H float32 metres, or None
        |-- proprioception     joint q/dq, base velocity, projected gravity
        |-- action_history     the actions already executed this episode
        |-- target_memory      where the target was last seen, if ever
        v
    ReObservationPolicy.act(observation) -> action_id
        |
        v
    Environment.rollout(action_id)       (privileged counterfactual, scoring only)

Contract, version ``API_VERSION``:

1.  An :class:`Observation` never contains a ground-truth visibility flag, a
    target world position, or anything else the robot could not measure.  The
    oracle facts live in :class:`PrivilegedState`, which only the evaluator and
    the ``Oracle`` baseline may touch.
2.  ``Environment.rollout`` is the only way to learn what an action would do.
    It costs a physical rollout, which is why only the oracle baseline is
    allowed to call it during a decision.
3.  Policies declare which observation fields they need through
    :attr:`ReObservationPolicy.requires`, so the evaluator can avoid rendering
    RGB and depth for a policy that only uses proprioception.
4.  ``action_id`` is always a member of ``environment.actions``; the evaluator
    rejects anything else instead of silently coercing it.

The terrain side of the contract is one-way: ``slabs``, ``scenes`` and
``terrain`` must not import this module or any policy.  ``test_api.py`` asserts
that, because it is the property that keeps the benchmark honest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

try:
    from .benchmark import BENCHMARK_ACTIONS, build_scene_terrain, rollout_with_visibility
    from .controllers import Primitive, load_controller
    from .experiment import target_anchor
    from .scenes import ROBOT_BASE_HEIGHT_M, Scene, scene_for
except ImportError:
    from benchmark import BENCHMARK_ACTIONS, build_scene_terrain, rollout_with_visibility
    from controllers import Primitive, load_controller
    from experiment import target_anchor
    from scenes import ROBOT_BASE_HEIGHT_M, Scene, scene_for

API_VERSION = "task3-reobservability/1.1"

#: The frozen candidate action set.  Changing this changes the benchmark, so it
#: lives here and nowhere else; `benchmark.py` imports its primitives from the
#: same table for the historical experiments.
ACTION_IDS: tuple[str, ...] = tuple(BENCHMARK_ACTIONS)

#: ``action_history`` entries with this id are the walk-in before the episode
#: began, not decisions.  An evaluator needs to tell the approach phase from the
#: re-observation phase, and History Horizon needs to say whether $H_k$ reaches
#: across the loss.  The id is a reserved marker, not a candidate action.
APPROACH_ACTION_ID = "approach"


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """Base pose in the world frame, metres and radians."""

    x: float
    y: float
    z: float
    yaw: float

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class TargetMemory:
    """The robot's own belief about the target, built from what it has seen.

    ``age_s`` is how long ago the sighting happened; a policy may legitimately
    discount an old sighting.  Nothing here is privileged: it is exactly the
    information a detector plus odometry would produce.
    """

    x: float
    y: float
    z: float
    age_s: float
    range_m: float

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class ActionRecord:
    """One executed action and the robot's displacement while it ran."""

    action_id: str
    duration_s: float
    displacement_m: float
    turn_rad: float


@dataclass(frozen=True)
class Observation:
    """Everything a policy is allowed to look at.  No privileged fields."""

    t_s: float
    pose: Pose
    proprioception: np.ndarray
    action_history: tuple[ActionRecord, ...] = ()
    target_memory: TargetMemory | None = None
    current_rgb: np.ndarray | None = None
    current_depth: np.ndarray | None = None

    def has(self, field_name: str) -> bool:
        return getattr(self, field_name, None) is not None


@dataclass(frozen=True)
class PrivilegedState:
    """Ground truth, for scoring and for the oracle baseline only."""

    t_s: float
    pose: Pose
    target_xyz: np.ndarray
    target_visible: bool
    target_blocked_by_terrain: bool
    target_in_fov: bool
    range_m: float
    bearing_rad: float


@dataclass
class Rollout:
    """The measured result of executing one action from a fixed state."""

    action_id: str
    reobserved: bool
    t_reobs_s: float | None
    path_cost_m: float
    forward_m: float
    lateral_m: float
    fallen: bool
    curve: list = field(default_factory=list)


@dataclass
class EpisodeResult:
    """One policy decision, scored against the measured counterfactuals."""

    scene: str
    policy: str
    action_id: str
    success: bool
    is_optimal: bool
    n_successful_actions: int
    best_action_id: str | None
    t_reobs_s: float | None
    path_cost_m: float
    fallback: bool = False


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class ReObservationPolicy(Protocol):
    """A decision rule.  Implementations must not touch terrain or physics."""

    name: str
    #: Observation fields the policy reads; the evaluator uses this to decide
    #: whether it has to pay for RGB and depth rendering.
    requires: frozenset[str]

    def reset(self, observation: Observation) -> None:
        """Start a new episode.  Called exactly once before the first ``act``."""

    def act(self, observation: Observation) -> str:
        """Return an ``action_id`` from ``environment.actions``."""


class Environment(Protocol):
    """A terrain episode.  Deliberately policy-agnostic."""

    name: str
    actions: tuple[str, ...]

    def reset(self) -> Observation: ...

    def step(self, action_id: str) -> Observation: ...

    def observe(self) -> Observation: ...

    def privileged(self) -> PrivilegedState: ...

    def rollout(self, action_id: str) -> Rollout: ...


# ---------------------------------------------------------------------------
# Environment implementation
# ---------------------------------------------------------------------------

class MuJoCoEnvironment:
    """Concrete :class:`Environment` over one TASK3 scene.

    Holds the MuJoCo model, the controller and the camera renderer.  It exposes
    no notion of a policy: the only state it keeps is the robot, the terrain and
    what the robot has seen.
    """

    def __init__(self, scene: Scene, controller_name: str = "go2rl",
                 ground: str = "auto", policy_path: Path | None = None,
                 render: bool = False, render_size: tuple[int, int] = (160, 120),
                 action_duration_s: float | None = None,
                 action_table: dict | None = None,
                 terrain=None):
        import mujoco

        self.scene_spec = scene
        self.name = scene.name
        # A caller may hand in an already built terrain.  The scene-name registry
        # only covers the named profiles; experiments that generate terrains from
        # parameters (paired worlds, hidden-feature pairs) build theirs directly
        # and pass it here rather than registering a profile per instance.
        self.terrain = terrain if terrain is not None else build_scene_terrain(scene)
        self.target = target_anchor(self.terrain, scene)
        from go2_piper_camera import build_model
        self.model = build_model(self.terrain, self.target, ground)
        self.controller = load_controller(controller_name, policy_path)
        self.model.opt.timestep = self.controller.timestep
        self.data = mujoco.MjData(self.model)
        self.actions = ACTION_IDS
        # The primitive table this environment executes.  A multi-step protocol
        # overwrites every duration with one common value, so "one step" costs the
        # same simulated time whatever the policy picks - otherwise a 2 s turn
        # would be five times cheaper than a 10 s drive and the comparison would
        # reward the cheap primitive rather than the good decision.
        self.action_table = dict(BENCHMARK_ACTIONS)
        if action_duration_s is not None:
            from dataclasses import replace as _replace
            self.action_table = {k: _replace(v, duration_s=float(action_duration_s))
                                 for k, v in self.action_table.items()}
        if action_table is not None:
            # A protocol may hand in a fully built table (e.g. with a rescaled
            # turn rate); the ids must still be the frozen set, because policies
            # choose by id.
            missing = set(ACTION_IDS) - set(action_table)
            if missing:
                raise ValueError(f"action_table is missing {sorted(missing)}")
            self.action_table = dict(action_table)
        self.action_duration_s = action_duration_s
        self.render = render
        self._renderer = None
        self._render_size = render_size
        self._history: list[ActionRecord] = []
        self._memory: TargetMemory | None = None
        self._t0 = 0.0
        self._mujoco = mujoco
        # Buffer for save/restore around counterfactual rollouts.  Without it a
        # scored rollout would silently advance the robot, so the oracle's own
        # probing would move the very state it is deciding from.
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_buf = np.empty(
            mujoco.mj_stateSize(self.model, self._state_spec), dtype=np.float64)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, xy: tuple[float, float] | None = None,
              replay_approach: bool = True) -> Observation:
        mujoco = self._mujoco
        anchor = xy if xy is not None else self.scene_spec.lost_xy
        x, y = float(anchor[0]), float(anchor[1])
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [x, y, self.terrain.height(x, y) + ROBOT_BASE_HEIGHT_M]
        yaw = self.scene_spec.yaw_rad
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qpos[7:25] = self.controller.default_angles
        self.data.qvel[:] = 0.0
        self.controller.reset()
        mujoco.mj_forward(self.model, self.data)
        self._history = []
        self._memory = None
        self._t0 = float(self.data.time)
        if replay_approach:
            self._replay_approach()
        return self.observe()

    def _replay_approach(self) -> None:
        """Record how the robot arrived, so the episode starts with real history.

        A policy that only gets the current frame cannot know the target was ever
        visible, which is the whole point of the task.  The approach is replayed
        along the straight line from the history vantage at the measured policy
        speed, sampling the same line-of-sight oracle the rest of TASK3 uses, and
        is appended to ``action_history`` as a single entry per sample.  The
        sighting that ends it is what fills ``target_memory``.
        """
        from experiment import visibility
        scene = self.scene_spec
        start = (np.array(scene.history_xy, dtype=float) if scene.history_xy is not None
                 else np.array([float(self.terrain.x[0]) + 0.3, scene.lost_xy[1]]))
        end = np.array(scene.lost_xy, dtype=float)
        span = float(np.linalg.norm(end - start))
        steps = max(2, int(round(span / 0.8 * 10.0)) + 1)
        travel = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
        turn = float(np.arctan2(np.sin(scene.yaw_rad - travel),
                                np.cos(scene.yaw_rad - travel)))
        previous, sighting_pose = None, None
        for i, alpha in enumerate(np.linspace(0.0, 1.0, steps)):
            xy = start + alpha * (end - start)
            yaw = travel + alpha * turn
            z = self.terrain.height(float(xy[0]), float(xy[1])) + ROBOT_BASE_HEIGHT_M
            reading = visibility(self.terrain, np.array([xy[0], xy[1], z]), yaw, 0.0,
                                 self.target)
            if reading["visible_oracle"]:
                sighting_pose = (float(xy[0]), float(xy[1]), float(z), yaw)
            if previous is not None:
                self._history.append(ActionRecord(
                    action_id="approach", duration_s=1.0 / 10.0,
                    displacement_m=float(np.linalg.norm(xy - previous)),
                    turn_rad=float((turn / (steps - 1)) if steps > 1 else 0.0)))
            previous = xy
        if sighting_pose is not None:
            eye = np.array(sighting_pose[:3]) + np.array([0.0, 0.0, 0.16])
            self._memory = TargetMemory(
                x=float(self.target[0]), y=float(self.target[1]), z=float(self.target[2]),
                age_s=0.0, range_m=float(np.linalg.norm(self.target - eye)))

    # -- observation -------------------------------------------------------

    def _pose(self) -> Pose:
        q = self.data.qpos
        yaw = float(np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]),
                               1 - 2 * (q[5] ** 2 + q[6] ** 2)))
        return Pose(float(q[0]), float(q[1]), float(q[2]), yaw)

    def _proprioception(self) -> np.ndarray:
        gravity = self.data.xmat[1].reshape(3, 3)[:, 2].copy()
        return np.concatenate([self.data.qpos[7:25], self.data.qvel[6:24],
                               self.data.qvel[:3], gravity])

    def _render(self):
        if not self.render:
            return None, None
        if self._renderer is None:
            w, h = self._render_size
            self._renderer = self._mujoco.Renderer(self.model, height=h, width=w)
        from go2_piper_camera import CAMERA_NAME
        self._renderer.update_scene(self.data, camera=CAMERA_NAME)
        rgb = np.asarray(self._renderer.render()).copy()
        self._renderer.enable_depth_rendering()
        try:
            depth = np.asarray(self._renderer.render()).copy()
        finally:
            self._renderer.disable_depth_rendering()
        return rgb, depth

    def observe(self) -> Observation:
        rgb, depth = self._render()
        return Observation(
            t_s=float(self.data.time) - self._t0,
            pose=self._pose(),
            proprioception=self._proprioception(),
            action_history=tuple(self._history),
            target_memory=self._memory,
            current_rgb=rgb,
            current_depth=depth,
        )

    def privileged(self) -> PrivilegedState:
        from experiment import visibility
        pose = self._pose()
        reading = visibility(self.terrain, pose.xyz, pose.yaw, 0.0, self.target)
        return PrivilegedState(
            t_s=float(self.data.time) - self._t0, pose=pose, target_xyz=self.target.copy(),
            target_visible=bool(reading["visible_oracle"]),
            target_blocked_by_terrain=bool(reading["blocked_by_terrain"]),
            target_in_fov=bool(reading["in_camera_fov"]),
            range_m=float(reading.get("range_m", 0.0)),
            bearing_rad=float(reading.get("bearing_rad", 0.0)),
        )

    # -- acting ------------------------------------------------------------

    def step(self, action_id: str) -> Observation:
        """Execute an action on the real robot state and remember what happened."""
        self._check(action_id)
        before = self._pose()
        starting_visible = self.privileged().target_visible
        result = rollout_with_visibility(self.model, self.data, self.terrain,
                                         self.controller, self.action_table[action_id],
                                         self.target)
        after = self._pose()
        disp = float(np.hypot(after.x - before.x, after.y - before.y))
        self._history.append(ActionRecord(
            action_id=action_id,
            duration_s=float(self.action_table[action_id].duration_s),
            displacement_m=disp,
            turn_rad=float(np.arctan2(np.sin(after.yaw - before.yaw),
                                      np.cos(after.yaw - before.yaw)))))
        self._update_memory(result, starting_visible)
        return self.observe()

    def rollout(self, action_id: str) -> Rollout:
        """Privileged counterfactual: physically run ``action_id`` from here.

        This is the expensive call the oracle baseline and the evaluator use; it
        is not part of what a deployable policy may do per decision.

        The simulation state is saved and restored, so scoring an action never
        advances the episode.  The controller is reset afterwards for the same
        reason: the vendored policies keep the last commanded action, and a
        counterfactual would otherwise leak into the next real step.
        """
        self._check(action_id)
        mujoco = self._mujoco
        mujoco.mj_getState(self.model, self.data, self._state_buf, self._state_spec)
        self.controller.reset()
        try:
            result = rollout_with_visibility(self.model, self.data, self.terrain,
                                             self.controller, self.action_table[action_id],
                                             self.target)
        finally:
            mujoco.mj_setState(self.model, self.data, self._state_buf, self._state_spec)
            mujoco.mj_forward(self.model, self.data)
            self.controller.reset()
        return Rollout(
            action_id=action_id,
            reobserved=bool(result["rsr"]),
            t_reobs_s=result.get("t_reobs_s"),
            path_cost_m=float(result["motion_cost_m"]),
            forward_m=float(result["forward_m"]),
            lateral_m=float(result["lateral_m"]),
            fallen=bool(result["fallen"]),
            curve=list(result["curve"]),
        )

    def _check(self, action_id: str) -> None:
        if action_id not in self.actions:
            raise KeyError(f"{action_id!r} is not in the frozen action set "
                           f"{self.actions}")

    def _update_memory(self, result: dict, starting_visible: bool) -> None:
        """Update target memory from the sighting recorded during the rollout."""
        sighting = None
        for sample in result["curve"]:
            if sample["visible"]:
                sighting = sample
                break
        if sighting is None:
            if starting_visible:
                pose = self._pose()
                self._memory = TargetMemory(float(self.target[0]), float(self.target[1]),
                                            float(self.target[2]), age_s=0.0,
                                            range_m=float(np.linalg.norm(
                                                self.target - pose.xyz)))
            return
        xyz = np.asarray(sighting["xyz"], dtype=float)
        eye = xyz + np.array([0.0, 0.0, 0.16])
        d = self.target - eye
        self._memory = TargetMemory(
            x=float(self.target[0]), y=float(self.target[1]), z=float(self.target[2]),
            age_s=float(self.data.time) - self._t0 - float(sighting["t_s"]),
            range_m=float(np.linalg.norm(d)))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ---------------------------------------------------------------------------
# Registry and evaluation
# ---------------------------------------------------------------------------

def make_environment(scene_name: str | Scene, **kwargs) -> MuJoCoEnvironment:
    """Build an environment from a scene name or a Scene object."""
    scene = scene_name if isinstance(scene_name, Scene) else scene_for(scene_name)
    return MuJoCoEnvironment(scene, **kwargs)


def scene_catalogue(include_history: bool = True) -> dict[str, Scene]:
    """Every terrain the frozen benchmark runs on.

    The multi-terrain scenes alone are not enough: on six of the eight, every
    baseline happens to succeed, so they cannot tell the policies apart at all.
    The paired-history worlds are the discriminating half, because the correct
    action there depends on which side the target was last seen from.
    """
    try:
        from .scenes import BENCHMARK_SCENES
    except ImportError:
        from scenes import BENCHMARK_SCENES
    out: dict[str, Scene] = {name: scene_for(name) for name in BENCHMARK_SCENES}
    if include_history:
        try:
            from .history_pairs import load_descriptor, scenes_for
        except ImportError:
            from history_pairs import load_descriptor, scenes_for
        for params in load_descriptor():
            for scene in scenes_for(params):
                out[scene.name] = scene
    return out


def evaluate(policy: ReObservationPolicy, environment: MuJoCoEnvironment,
             check_actions: bool = True) -> EpisodeResult:
    """One decision, scored against every measured counterfactual.

    The policy sees only the :class:`Observation`.  The counterfactuals are
    computed after the choice is made, so a policy cannot peek.
    """
    observation = environment.reset()
    policy.reset(observation)
    action_id = policy.act(observation)
    if check_actions and action_id not in environment.actions:
        raise ValueError(f"policy {policy.name!r} returned {action_id!r}, which is "
                         f"not in the frozen action set")

    outcomes = {a: environment.rollout(a) for a in environment.actions}
    winners = [a for a, r in outcomes.items() if r.reobserved]
    best = min(winners, key=lambda a: (outcomes[a].t_reobs_s or 1e9,
                                       outcomes[a].path_cost_m)) if winners else None
    chosen = outcomes[action_id]
    return EpisodeResult(
        scene=environment.name, policy=policy.name, action_id=action_id,
        success=chosen.reobserved, is_optimal=action_id == best,
        n_successful_actions=len(winners), best_action_id=best,
        t_reobs_s=chosen.t_reobs_s, path_cost_m=chosen.path_cost_m)


def summarise(results: list[EpisodeResult]) -> dict:
    """Aggregate episode results into the reporting metrics."""
    n = len(results)
    if not n:
        return {"n_episodes": 0}
    solvable = [r for r in results if r.n_successful_actions > 0]
    return {
        "n_episodes": n,
        "n_solvable": len(solvable),
        "rsr": round(sum(r.success for r in results) / n, 4),
        "rsr_on_solvable": (round(sum(r.success for r in solvable) / len(solvable), 4)
                            if solvable else None),
        "optimal_action_rate": round(sum(r.is_optimal for r in results) / n, 4),
        "mean_t_reobs_s": (round(float(np.mean([r.t_reobs_s for r in results
                                                if r.t_reobs_s is not None])), 3)
                           if any(r.t_reobs_s is not None for r in results) else None),
        "mean_path_cost_m": round(float(np.mean([r.path_cost_m for r in results])), 3),
    }
