"""Locomotion controller backends for TASK3.

Two vendored policies are available. Both were trained by other projects and are
used here unchanged; TASK3 only supplies the terrain and the velocity command.

``leggedmanip``
    ``third_party/LeggedManip_Lab`` Go2+Piper TorchScript policy, 18 actions
    (12 legs + 6 arm), 50 Hz, measured to stop at any step taller than ~0.04 m.

``go2rl``
    ``wty-yy/go2_rl_gym`` MoE-CTS Go2 policy, 12 actions, 50 Hz at a 500 Hz
    physics timestep, trained on randomised terrain that includes 0.20-0.23 m
    stairs.  When run with our Go2+Piper model it drives the robot at 0.88 m/s
    on flat and climbs the full ten-step 0.15 m staircase.

Each backend owns the constants that must match how its weights were trained:
physics timestep, control decimation, PD gains, default pose and action scale.
Nothing here trains or fine-tunes a model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# IsaacLab joint order (FR, FL, RR, RL) -> MuJoCo XML order (FL, FR, RL, RR).
ISAAC_TO_MUJOCO = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Both upstream projects use the same standing pose for the twelve leg joints.
LEG_DEFAULT = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                        0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
ARM_DEFAULT = np.zeros(6, dtype=np.float32)
DEFAULT18 = np.concatenate([LEG_DEFAULT, ARM_DEFAULT])


@dataclass(frozen=True)
class Primitive:
    """A named candidate action expressed as a velocity command plus duration.

    ``heading_hold`` turns the primitive into a target-memory action: instead of
    a blind constant yaw rate it steers toward the world position where the
    target was last seen.  That is memory, not perception - the target has not
    been observed since it was lost.

    ``yaw_hold`` is different and needs no memory at all: it keeps the heading
    the robot had when the action started.  The vendored policy turns as it
    strafes (measured up to 0.27 rad over a 6 s sidestep), which is enough to
    push a revealed target back out of the 87 deg field of view; a sidestep that
    holds heading is both the physically sensible primitive and the fair test.
    """

    name: str
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    duration_s: float = 1.0
    heading_hold: bool = False
    heading_gain: float = 1.5
    heading_wz_max: float = 1.0
    yaw_hold: bool = False
    yaw_hold_gain: float = 1.5
    yaw_hold_wz_max: float = 0.8
    note: str = ""


COMMON_PRIMITIVES = {
    "stand": Primitive("stand", duration_s=1.5, note="hold position, no velocity command"),
    "turn_left": Primitive("turn_left", wz=0.8, duration_s=0.8),
    "turn_right": Primitive("turn_right", wz=-0.8, duration_s=0.8),
    "forward": Primitive("forward", vx=0.55, duration_s=1.0),
    "back": Primitive("back", vx=-0.40, duration_s=7.0,
                      note="retreat along the approach, away from the occluder"),
}

# ``go2rl`` actually climbs, so its step-climbing command is longer and faster
# than the stall-against-the-step command the flat-ground policy needed.
GO2RL_PRIMITIVES = COMMON_PRIMITIVES | {
    "climb": Primitive("climb", vx=1.0, duration_s=8.0,
                       note="walk forward over the step block on a fixed heading"),
    "climb_toward": Primitive("climb_toward", vx=0.9, duration_s=8.0, heading_hold=True,
                              note="walk forward over the step block while steering toward "
                                   "the remembered target bearing"),
}
LEGGEDMANIP_PRIMITIVES = COMMON_PRIMITIVES | {
    "climb": Primitive("climb", vx=0.60, duration_s=6.0,
                       note="walk forward into the step block"),
}

# Candidate actions of Experiment 0, in the order they are reported.  A
# controller that does not define one simply does not offer it.
CANDIDATE_ORDER = ["turn_left", "turn_right", "forward", "climb", "climb_toward", "back"]


class Controller:
    """Common interface: one 50 Hz update returns 18 target joint angles."""

    name: str = "controller"
    timestep: float = 0.005
    control_decimation: int = 4
    kps: np.ndarray
    kds: np.ndarray
    default_angles: np.ndarray
    primitives: dict
    asset: dict

    @property
    def control_dt(self) -> float:
        return self.timestep * self.control_decimation

    def reset(self) -> None:
        raise NotImplementedError

    def act(self, data, command) -> np.ndarray:
        """Return the 18 target joint angles for the next control period."""
        raise NotImplementedError


class LeggedManipController(Controller):
    """Upstream Go2+Piper TorchScript policy from ``third_party/LeggedManip_Lab``.

    Observation layout (210 dims, 3-step history) reproduced from that project's
    MuJoCo deploy script: base angular velocity * 0.2, projected gravity,
    joint positions relative to default, joint velocities * 0.05, last action,
    velocity command, position command.
    """

    name = "leggedmanip"
    timestep = 0.005
    control_decimation = 4
    kps = np.array([35.0] * 12 + [40.0, 40.0, 40.0, 20.0, 20.0, 20.0], dtype=np.float32)
    kds = np.array([0.8] * 12 + [1.0] * 6, dtype=np.float32)
    default_angles = DEFAULT18
    primitives = LEGGEDMANIP_PRIMITIVES
    num_hist = 3
    base_ang_vel_scale = 0.2
    joint_vel_scale = 0.05
    action_scale = 0.25

    def __init__(self, path: Path):
        import torch

        self.torch = torch
        torch.set_grad_enabled(False)
        self.policy = torch.jit.load(str(path))
        self.asset = {"name": self.name, "policy": str(path),
                      "source": "third_party/LeggedManip_Lab/mujoco/deploy/policy/go2_piper/policy.pt"}
        self.reset()

    def reset(self) -> None:
        torch = self.torch
        self.action = torch.zeros(1, 18)
        self.history = {k: torch.zeros(1, d * self.num_hist) for k, d in
                        (("ang", 3), ("grav", 3), ("joint_pos", 18), ("joint_vel", 18),
                         ("action", 18), ("vel_cmd", 3), ("pos_cmd", 7))}

    def _roll(self, key, new, width):
        torch = self.torch
        self.history[key] = torch.cat([self.history[key], new], dim=-1)[:, width:]

    def act(self, data, command) -> np.ndarray:
        torch = self.torch
        qj = torch.tensor(data.qpos[7:], dtype=torch.float32).unsqueeze(0)
        dqj = torch.tensor(data.qvel[6:], dtype=torch.float32).unsqueeze(0) * self.joint_vel_scale
        quat = torch.tensor(data.qpos[3:7], dtype=torch.float32)
        qw, qx, qy, qz = quat
        gravity = torch.tensor([2.0 * (-qz * qx + qw * qy),
                                -2.0 * (qz * qy + qw * qx),
                                1.0 - 2.0 * (qw * qw + qz * qz)]).unsqueeze(0)
        omega = torch.tensor(data.qvel[3:6], dtype=torch.float32).unsqueeze(0) * self.base_ang_vel_scale
        qj_rel = qj - torch.tensor(self.default_angles)
        leg_pos = qj_rel[:, :12][:, ISAAC_TO_MUJOCO]
        leg_vel = dqj[:, :12][:, ISAAC_TO_MUJOCO]
        self._roll("ang", omega, 3)
        self._roll("grav", gravity, 3)
        self._roll("joint_pos", torch.cat([leg_pos, qj_rel[:, 12:]], dim=-1), 18)
        self._roll("joint_vel", torch.cat([leg_vel, dqj[:, 12:]], dim=-1), 18)
        self._roll("action", self.action, 18)
        self._roll("vel_cmd", torch.tensor([command], dtype=torch.float32), 3)
        self._roll("pos_cmd", torch.zeros(1, 7), 7)
        obs = torch.cat([self.history[k] for k in
                         ("ang", "grav", "joint_pos", "joint_vel", "action", "vel_cmd", "pos_cmd")],
                        dim=-1).float().clamp(-100.0, 100.0)
        self.action = self.policy(obs).clamp(-20.0, 20.0)
        leg_action = self.action[:, :12][:, ISAAC_TO_MUJOCO]
        action = torch.cat([leg_action, self.action[:, 12:]], dim=-1)
        return (action * self.action_scale + torch.tensor(self.default_angles)).numpy().ravel()


class Go2RlGymController(Controller):
    """``wty-yy/go2_rl_gym`` MoE-CTS Go2 policy, used unmodified.

    Observation layout (45 dims) and constants copied from that project's
    ``deploy/deploy_mujoco/configs/go2.yaml``.  Its ``model_joint_names`` equal
    its ``mujoco_joint_names``, so indices map straight onto our MuJoCo XML
    order (FL, FR, RL, RR) with no remapping -- confirmed by the flat rollout.
    The six Piper arm joints are not part of the policy and are held by PD.
    """

    name = "go2rl"
    timestep = 0.002
    control_decimation = 10
    kps = np.full(18, 20.0, dtype=np.float32)
    kds = np.full(18, 0.5, dtype=np.float32)
    default_angles = DEFAULT18
    primitives = GO2RL_PRIMITIVES
    action_scale = 0.25
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
    num_actions = 12

    def __init__(self, path: Path):
        import torch

        self.torch = torch
        torch.set_grad_enabled(False)
        self.policy = torch.jit.load(str(path))
        self.asset = {"name": self.name, "policy": str(path),
                      "source": "wty-yy/go2_rl_gym deploy/pre_train/go2/"
                                "go2_moe_cts_high_slope_thre_164k_0.6715.pt"}
        self.reset()

    def reset(self) -> None:
        # The exported MoE-CTS module owns an observation history tensor in
        # addition to this wrapper's last action. Both must be reset so the
        # outcome of a branch cannot depend on the candidate executed before it.
        self.policy.reset()
        self.policy.eval()
        self.action = np.zeros(self.num_actions, dtype=np.float32)

    def act(self, data, command) -> np.ndarray:
        torch = self.torch
        q = data.qpos[3:7]
        qw, qx, qy, qz = q
        gravity = np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx),
                            1 - 2 * (qw * qw + qz * qz)], dtype=np.float32)
        obs = np.concatenate([
            data.qvel[3:6] * self.ang_vel_scale,
            gravity,
            np.asarray(command, dtype=np.float32) * self.cmd_scale,
            (data.qpos[7:19] - LEG_DEFAULT) * self.dof_pos_scale,
            data.qvel[6:18] * self.dof_vel_scale,
            self.action,
        ]).astype(np.float32)
        out = self.policy(torch.from_numpy(obs).unsqueeze(0))
        action = out[0] if isinstance(out, tuple) else out
        self.action = action.detach().numpy().squeeze().astype(np.float32)
        target = self.action * self.action_scale + LEG_DEFAULT
        return np.concatenate([target, ARM_DEFAULT]).astype(np.float32)


@dataclass
class ControllerRegistry:
    leggedmanip: Path = ROOT / "third_party" / "LeggedManip_Lab" / "mujoco" / "deploy" / "policy" / "go2_piper" / "policy.pt"
    go2rl: Path = ROOT / "assets" / "robots" / "go2_rl_gym" / "policy" / "go2_moe_cts_high_slope_thre_164k_0.6715.pt"
    extra: dict = field(default_factory=dict)

    def resolve(self, name: str) -> Path:
        if name in self.extra:
            return Path(self.extra[name])
        if name not in ("leggedmanip", "go2rl"):
            raise KeyError(f"unknown controller {name!r}; known: leggedmanip, go2rl")
        return getattr(self, name)

    def status(self, name: str) -> dict:
        path = self.resolve(name)
        return {"controller": name, "present": path.exists(), "policy": str(path)}


def load_controller(name: str, path: Path | None = None) -> Controller:
    registry = ControllerRegistry()
    if path is not None:
        registry.extra[name] = Path(path)
    resolved = registry.resolve(name)
    if not resolved.exists():
        raise FileNotFoundError(f"policy for controller {name!r} not found: {resolved}")
    if name == "leggedmanip":
        return LeggedManipController(resolved)
    if name == "go2rl":
        return Go2RlGymController(resolved)
    raise KeyError(f"unknown controller {name!r}")
