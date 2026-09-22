"""TASK2 的 Policy 接口：观测、动作、两层决策协议。

设计约束
--------
1. **两层分离**。高层只回答"下一块放哪"（`HighLevelPolicy.decide`），低层只回答
   "末端下一步去哪"（`LowLevelPolicy.step`）。两层各自可换实现，互不知道对方是什么。
2. **策略不碰 MuJoCo**。Policy 只读 `Observation`、只写 `Decision` / `Command`；
   IK、物理步进、碰撞开关、成功判定都在执行器里。这样 VLM 策略可以放在另一个进程，
   也保证了"策略不会偷偷读引擎对象"。
3. **特权状态显式标注**。`Observation.stones` 是仿真真值（脚本策略要用），
   VLM 这类"只该看图像"的策略应当在 prompt 里忽略它——接口不替策略做这个决定，
   但 `Observation.image_only()` 提供了一个剥掉特权信息的视图。
4. **失败要能兜底**。任何策略都可能超时、返回垃圾或抛异常；执行器只认 `Command`
   的结构合法性，异常一律由 `FallbackPolicy` 包装处理，不静默吞掉（计数并记录）。

动作空间
--------
低层输出的是**末端目标位姿**（绝对，世界系），不是关节力矩，也不是原始像素动作：
`Command.tcp_pos` / `tcp_rot` 是 TCP 位姿目标，`Command.grip_width` 是夹爪开口宽度（米，
None 表示保持）。执行器负责阻尼最小二乘 IK + 位置伺服 + 物理步进。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

Phase = Literal[
    "idle", "approach", "descend", "grasp", "lift", "transport", "place", "release", "retreat", "done"
]

#: 低层策略在 `step()` 里可以改变的目标类型。
GoalKind = Literal["home", "pick", "place", "retreat", "hold"]


# --------------------------------------------------------------------------------------
# 观测
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraFrame:
    """一路相机的 RGB-D 帧。`depth` 单位是米，0 表示没有命中（超出 max_range）。"""

    name: str
    rgb: np.ndarray | None  # (H, W, 3) uint8
    depth: np.ndarray | None  # (H, W) float32
    fovy_deg: float
    width: int
    height: int
    #: 相机在世界系下的位姿（cam_xpos / cam_xmat）
    pos: np.ndarray | None = None
    rot: np.ndarray | None = None

    def jpeg_bytes(self, quality: int = 88, max_side: int | None = 768) -> bytes:
        """编码成 JPEG，供 VLM/日志使用。`max_side` 会等比缩放，不改变宽高比。"""
        if self.rgb is None:
            raise ValueError(f"相机 {self.name} 没有 RGB 数据")
        from io import BytesIO

        from PIL import Image

        image = Image.fromarray(np.ascontiguousarray(self.rgb))
        if max_side is not None and max(image.size) > max_side:
            scale = max_side / max(image.size)
            size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(size, Image.BILINEAR)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    def depth_stats(self) -> dict[str, float]:
        if self.depth is None:
            return {"valid_ratio": 0.0, "min_m": 0.0, "max_m": 0.0, "median_m": 0.0}
        valid = self.depth[self.depth > 0.0]
        if valid.size == 0:
            return {"valid_ratio": 0.0, "min_m": 0.0, "max_m": 0.0, "median_m": 0.0}
        return {
            "valid_ratio": float(valid.size / self.depth.size),
            "min_m": float(valid.min()),
            "max_m": float(valid.max()),
            "median_m": float(np.median(valid)),
        }


@dataclass(frozen=True)
class StoneState:
    """石块的真值状态。`graspable` 由夹爪开口与质量上限算出来，是执行器的判断，不是策略的。"""

    name: str
    pos: np.ndarray  # (3,) 世界系
    quat: np.ndarray  # (4,) w x y z
    size: np.ndarray  # (3,) 长 宽 厚
    mass: float
    graspable: bool = True
    #: 抓取质量 0..1（由抓取规划给出：带宽是否留出闭合行程、侧面是否够平行）。
    #: 只用布尔"能不能夹"不够——这批石头里"够窄"和"够平"经常只有一个成立。
    grasp_quality: float = 0.0

    @property
    def thinnest(self) -> float:
        return float(np.min(self.size))


@dataclass(frozen=True)
class PlanSlot:
    """堆叠计划里的一个位置（第几层、该层第几块）。"""

    course: int
    slot_index: int
    target_pos: np.ndarray
    target_quat: np.ndarray

    @property
    def key(self) -> tuple[int, int]:
        return (int(self.course), int(self.slot_index))


@dataclass(frozen=True)
class Candidate:
    """高层候选：一块石头 × 一个空位。VLM 策略就是在这种候选列表里选。"""

    stone: StoneState
    slot: PlanSlot
    score: float = 0.0
    note: str = ""

    def describe(self) -> str:
        size = "x".join(f"{value * 1000:.0f}" for value in self.stone.size)
        target = " ".join(f"{value:+.3f}" for value in self.slot.target_pos)
        return (
            f"stone={self.stone.name} size_mm={size} mass_kg={self.stone.mass:.2f} "
            f"course={self.slot.course} slot={self.slot.slot_index} target_xyz=[{target}]"
        )


@dataclass(frozen=True)
class Observation:
    """一步观测。所有数组都是拷贝出来的，策略改不到引擎内部状态。"""

    time: float
    phase: Phase
    joints: dict[str, float]
    tcp_pos: np.ndarray
    tcp_rot: np.ndarray
    gripper_width: float
    cameras: dict[str, CameraFrame]
    stones: tuple[StoneState, ...] = ()
    held_stone: str | None = None
    instruction: str = ""
    step_index: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def camera_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.cameras))

    def image_only(self, instruction: str | None = None) -> "Observation":
        """剥掉特权信息（石块真值、关节角）——给"只许看图像"的策略用。"""
        return Observation(
            time=self.time,
            phase=self.phase,
            joints={},
            tcp_pos=self.tcp_pos,
            tcp_rot=self.tcp_rot,
            gripper_width=self.gripper_width,
            cameras=self.cameras,
            stones=(),
            held_stone=self.held_stone,
            instruction=self.instruction if instruction is None else instruction,
            step_index=self.step_index,
            info={},
        )


# --------------------------------------------------------------------------------------
# 动作
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """高层输出：把哪块石头放到哪个空位，或者停机。"""

    action: Literal["place", "stop"] = "place"
    stone: str | None = None
    slot: tuple[int, int] | None = None
    target_pos: np.ndarray | None = None
    target_quat: np.ndarray | None = None
    source: str = ""
    reason: str = ""

    @property
    def is_stop(self) -> bool:
        return self.action == "stop"


@dataclass(frozen=True)
class Command:
    """低层输出：TCP 的绝对目标位姿 + 可选夹爪开口宽度。"""

    tcp_pos: np.ndarray
    tcp_rot: np.ndarray
    grip_width: float | None = None
    done: bool = False
    #: True = 这一步只做**纯平移**（雅可比位置解），不要重新解 IK。
    #: 抬起时必须这样：重新解 IK 会让末端在抬升过程中横向漂几毫米，
    #: 夹住的石头会被拖出去（实测：合拢时夹住 160 N，一抬就滑掉，指垫直接合到 38 mm）。
    pure_translation: bool = False
    note: str = ""
    source: str = ""


@dataclass(frozen=True)
class Goal:
    """执行器当前要达成的事，交给低层策略。

    `pick_pos` / `place_pos` 是 **TCP 位姿目标**（世界系），不是"指垫中心应该在哪"：
    指垫与 TCP 之间的固定偏置由执行器换算（它才知道夹爪几何），策略只在 TCP 空间里
    规划。搞混这两个空间会让末端稳定地停在偏 3 cm 的位置上。
    """

    kind: GoalKind
    stone: StoneState | None = None
    pick_pos: np.ndarray | None = None
    pick_rot: np.ndarray | None = None
    place_pos: np.ndarray | None = None
    place_rot: np.ndarray | None = None
    #: 接近方向（世界系单位向量）：夹爪从哪一侧推进到目标位姿。
    #: 默认竖直向下（从上方下爪）；台面上抓平放的石头要改成水平侧向——
    #: 指尖比指垫中心低 36 mm，从上方下爪时这个偏移会变成"往台面里插"。
    approach_direction: np.ndarray | None = None
    #: 侧向站位的退让距离（米）
    standoff_m: float = 0.12
    #: 抓取规划给出的截面带宽（米）。预合拢必须比它宽，否则指垫会在推进阶段
    #: 直接撞上石头并把它推走——实测就是这个原因让"合爪成功但抬不起来"。
    grasp_band_width: float | None = None
    grip_width: float | None = None
    approach_height: float = 0.18
    travel_height: float = 0.34
    timeout_s: float = 6.0

    def describe(self) -> str:
        return f"{self.kind}(stone={self.stone.name if self.stone else None})"


# --------------------------------------------------------------------------------------
# 协议
# --------------------------------------------------------------------------------------


@runtime_checkable
class HighLevelPolicy(Protocol):
    """决定"下一块石头放哪"。每块石头问一次，可以返回 stop 提前结束。"""

    name: str

    def reset(self, context: dict[str, Any]) -> None: ...

    def decide(self, observation: Observation, candidates: Sequence[Candidate]) -> Decision: ...


@runtime_checkable
class LowLevelPolicy(Protocol):
    """决定"末端下一步去哪"。执行器每控制周期问一次，直到 `Command.done`。"""

    name: str

    def reset(self, context: dict[str, Any]) -> None: ...

    def step(self, observation: Observation, goal: Goal) -> Command: ...


@dataclass
class PolicyPair:
    """一个策略包：高层 + 低层 + 说明。`run.py --policy` 选的就是它。"""

    name: str
    high: HighLevelPolicy
    low: LowLevelPolicy
    description: str = ""
    needs_cameras: tuple[str, ...] = ()
    requires_network: bool = False

    def reset(self, context: dict[str, Any]) -> None:
        self.high.reset(context)
        self.low.reset(context)


class FallbackLowLevel:
    """把任意低层策略包成"异常/超时不许中断整个任务"。

    不是静默兜底：每次回退都计数并记在 `events` 里，报告会写出来。
    """

    def __init__(self, primary: LowLevelPolicy, backup: LowLevelPolicy, max_failures: int = 20):
        self.name = f"{primary.name}+fallback({backup.name})"
        self.primary = primary
        self.backup = backup
        self.max_failures = max_failures
        self.events: list[dict[str, Any]] = []
        self.failures = 0

    def reset(self, context: dict[str, Any]) -> None:
        self.primary.reset(context)
        self.backup.reset(context)
        self.failures = 0
        self.events = []

    def step(self, observation: Observation, goal: Goal) -> Command:
        if self.failures >= self.max_failures:
            return self.backup.step(observation, goal)
        try:
            command = self.primary.step(observation, goal)
        except Exception as error:  # noqa: BLE001 - 策略是外部代码，什么都可能抛
            self.failures += 1
            self.events.append(
                {"phase": observation.phase, "goal": goal.describe(), "error": f"{type(error).__name__}: {error}"}
            )
            return self.backup.step(observation, goal)
        if not _command_is_valid(command):
            self.failures += 1
            self.events.append({"phase": observation.phase, "goal": goal.describe(), "error": "invalid command"})
            return self.backup.step(observation, goal)
        return command


def _command_is_valid(command: Any) -> bool:
    if not isinstance(command, Command):
        return False
    pos = np.asarray(command.tcp_pos, dtype=float)
    rot = np.asarray(command.tcp_rot, dtype=float)
    if pos.shape != (3,) or rot.shape != (3, 3):
        return False
    if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
        return False
    if command.grip_width is not None and not np.isfinite(command.grip_width):
        return False
    return True
