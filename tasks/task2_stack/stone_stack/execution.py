"""执行器：把 Policy 的输出变成物理动作，并负责判定成功。

分工（这是本文件唯一重要的事）
------------------------------
- **Policy 决定"做什么"**：高层给（石头，槽位，目标位姿），低层给"末端下一步去哪"。
- **执行器决定"怎么发生"**：IK、位置伺服、物理步进、接触开关、重置石块、计时、
  成功判定、报告。策略拿不到 `MjModel`，也改不了物理——VLM 策略因此可以放在另一个
  进程里，而"策略偷偷读引擎状态"这种事故在结构上就不可能发生。

成功判定沿用旧 TASK2 的阈值（`lift_gain > 0.045`、`xy < 0.13`、`z < 0.10`、
`stacked = course 0 or final_z > 0.075`），这样新旧结果可以直接对比。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cameras import CameraRig
from .policy.base import (
    Candidate,
    Decision,
    Goal,
    Observation,
    PlanSlot,
    PolicyPair,
    StoneState,
)
from .robots.control import RobotController
from .robots.profile import ArmProfile
from .robots.scene import BuiltScene


@dataclass
class ExecutionConfig:
    """执行参数。默认值与旧 TASK2 执行器对齐，便于对比。"""

    #: 一个策略步伺服多久（秒）
    servo_seconds: float = 0.30
    #: 单个 Goal 的策略步上限（防止策略不收敛时死循环）
    policy_timeout_s: float = 25.0
    #: 放石头后静置时间
    settle_seconds: float = 0.70
    #: 把石头重置到料区后静置时间
    reset_seconds: float = 0.25
    #: 每个 placement 的抓取重试次数（失败后收紧夹爪重试）
    grasp_retries: int = 1
    #: 放置时是否在有接触后提前停止下降（叠上层石头时有用）
    contact_aware_place: bool = True
    #: 判定阈值（与旧执行器一致）
    lift_threshold_m: float = 0.045
    place_tolerance_xy_m: float = 0.13
    place_tolerance_z_m: float = 0.10
    stacked_height_m: float = 0.075
    #: 相机
    capture_cameras: bool = True
    camera_names: tuple[str, ...] = ("wrist", "top", "front")
    #: 侧向抓取的站位退让距离（米）
    side_standoff_m: float = 0.12
    #: 抓取前把石头对齐到抓取轴（把料区当定位窝），避免合拢时把石头推走
    nest_align: bool = True
    #: 合拢量 = 截面带宽 × 这个比例（越小夹得越紧）；重试时自动再收紧 0.15。
    #: 实测 0.85 太松（接触力只有 ~10 N），0.45 能到 ~100 N 并稳定吊住石头。
    grasp_close_fraction: float = 0.0
    #: 抓取接近方式：
    #:   "vertical" = 夹爪竖直向下（从正上方下爪），TCP 的 approach 轴对世界 -Z；
    #:   "side"     = 侧向抓取，夹爪水平推进（`grasp_plan` 给的横向向量）。
    #: 只有 "vertical" 会用到 profile 的 `approach_tilt_deg`（UR5e 默认 0° = 笔直向下）。
    grasp_approach: str = "vertical"
    #: 每步都存帧（调试用，很慢）
    save_frames: Path | None = None
    #: 日志回调
    log: Callable[[str], None] | None = None


@dataclass
class PlacementRecord:
    """一次放置的完整记录，字段名与旧 TASK2 报告尽量对齐。"""

    name: str
    course: int
    slot_index: int
    target_pos: list[float]
    target_quat: list[float]
    pick_pos: list[float]
    lifted_pos: list[float]
    final_pos: list[float]
    final_quat: list[float]
    lift_gain_m: float
    target_xy_error_m: float
    target_z_error_m: float
    placed: bool
    stacked: bool
    attempts: int
    policy_source: str
    policy_reason: str
    duration_s: float
    policy_steps: int
    grasp_plan: str = ""
    servo_metrics: list[dict] = field(default_factory=list)


class Executor:
    """一次 TASK2 运行。"""

    def __init__(
        self,
        built: BuiltScene,
        profile: ArmProfile,
        pair: PolicyPair,
        config: ExecutionConfig | None = None,
        cameras: CameraRig | None = None,
        viewer=None,
    ):
        import mujoco  # noqa: F401  (保证模型已被编译过)

        self.built = built
        self.model = built.model
        self.data = built.data
        self.profile = profile
        self.pair = pair
        self.config = config or ExecutionConfig()
        self.workcell = built.workcell
        self.controller = RobotController(self.model, self.data, profile)
        #: 实时 viewer（`run.py --view`）。没有就完全不影响原有行为。
        self.viewer = viewer
        self._last_sync = 0.0
        #: `TASK2_TRACE=1` 时逐步打印低层动作与抓取诊断。
        self.trace = os.environ.get("TASK2_TRACE", "").strip().lower() in ("1", "true", "yes", "on")
        # 装了自造平行夹爪（slide 驱动、面永远平行）时，夹爪指令改发给那两根滑轨
        self.synthetic_gripper = getattr(built, "synthetic_gripper", None)
        if self.synthetic_gripper is not None:
            self.controller.use_gripper_actuators(("pg_jaw_act_0", "pg_jaw_act_1"))
        self.cameras = cameras
        self._log = self.config.log or (lambda message: None)
        self._stone_geom: dict[str, int] = {
            stone: int(self.model.geom(f"{stone}_geom").id) for stone in self.stone_names()
        }
        self._stone_qpos: dict[str, int] = {
            stone: int(self.model.jnt_qposadr[self.model.joint(f"{stone}_free").id])
            for stone in self.stone_names()
        }
        self._sizes: dict[str, np.ndarray] = {
            stone: np.asarray(self.model.geom_size[self._stone_geom[stone]], dtype=float).copy()
            for stone in self.stone_names()
        }
        self._masses: dict[str, float] = {
            stone: float(self.model.body(self.model.geom_bodyid[self._stone_geom[stone]]).mass[0])
            for stone in self.stone_names()
        }
        self.policy_events: list[dict] = []
        # 场景里可能把原本"外八"的指垫换成了平行夹爪（见 scene.measure_parallel_jaws），
        # 接触判定、开口换算、接触开关都要按场景里真正参与接触的那两个 geom 来
        self.pad_geoms = tuple(getattr(built, "pad_geoms", ())) or tuple(profile.gripper.pad_geoms)
        self.pad_thickness_m = float(getattr(built, "pad_thickness_m", 0.0)) or float(
            profile.gripper.pad_thickness_m
        )
        self._pad_geom_ids = [
            int(self.model.geom(name).id) for name in self.pad_geoms if self.model.geom(name).id >= 0
        ]
        self._finger_body_ids = [int(self.model.body(name).id) for name in (profile.gripper.finger_bodies or ())]
        # 夹爪几何是静态量，但必须在已知姿态下量一次并缓存（见 _measure_gripper_clearance）
        self.gripper_geometry = self._measure_gripper_clearance()

    # ------------------------------------------------------------------ 石头状态

    def stone_names(self) -> list[str]:
        return list(self.built.workcell.staging_poses)

    def stone_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        address = self._stone_qpos[name]
        return (
            np.asarray(self.data.qpos[address : address + 3], dtype=float).copy(),
            np.asarray(self.data.qpos[address + 3 : address + 7], dtype=float).copy(),
        )

    def set_stone_pose(self, name: str, pos, quat) -> None:
        address = self._stone_qpos[name]
        self.data.qpos[address : address + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[address + 3 : address + 7] = np.asarray(quat, dtype=float)
        joint = self.model.joint(f"{name}_free").id
        dof = int(self.model.jnt_dofadr[joint])
        self.data.qvel[dof : dof + 6] = 0.0
        self.controller.forward()

    def stone_speed(self, name: str) -> float:
        joint = self.model.joint(f"{name}_free").id
        dof = int(self.model.jnt_dofadr[joint])
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    def _mesh_vertices_body(self, name: str) -> np.ndarray | None:
        """石块网格顶点在 **body 系** 下的坐标。

        实测结论（drop test，见 docs 说明）：MuJoCo 的碰撞/几何用的是
        `body ∘ geom_pos ∘ geom_quat`，**不乘** `mesh_quat`。`mesh_quat` 是编译器
        给 mesh 写的主轴对齐量（某块石头是 (0.0005, 0.659, 0.106, 0.745)），把它也
        乘进去会算出"陷进台面 5.8 cm"的假包围盒——石块实际是平躺的（高 90 mm 而不是
        184 mm）。抓取目标全部来自这里，算错就永远夹空。
        """
        geom_id = self._stone_geom[name]
        mesh_id = int(self.model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        address = int(self.model.mesh_vertadr[mesh_id])
        count = int(self.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(self.model.mesh_vert[address : address + count], dtype=float)
        rotation = _quat_to_mat(self.model.geom_quat[geom_id])
        return vertices @ rotation.T + np.asarray(self.model.geom_pos[geom_id], dtype=float)

    def stone_aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """石块在**世界系**下的轴对齐包围盒。

        为什么不用生成时的名义尺寸（长/宽/厚）：名义尺寸描述的是网格自己的坐标系，
        而网格被 OBB 对齐后的轴向与生成时的假设不一定一致（实测 moonsim 石头的
        网格 Z 向竟然是长轴）。抓取方向和层高都必须按石头**当前实际躺着的样子**算，
        所以统一从"网格顶点 + body 位姿"量世界包围盒。
        """
        local = self._mesh_vertices_body(name)
        body_id = int(self.model.geom_bodyid[self._stone_geom[name]])
        if local is None:
            half = 0.5 * self._sizes[name]
            pos = self.data.xpos[body_id]
            return pos - half, pos + half
        rotation = self.data.xmat[body_id].reshape(3, 3)
        world = local @ rotation.T + self.data.xpos[body_id]
        return world.min(axis=0), world.max(axis=0)

    def _ctrl_for_width(self, inner_width: float) -> tuple[float, ...]:
        """按**净开口**给夹爪指令，用场景实际指垫厚度换算（平行夹爪与原指垫不同）。"""
        if self.synthetic_gripper is not None:
            # 自造平行夹爪：两片平板各自沿滑轨内移 ctrl，净开口 = 2·(half_travel − ctrl) − 板厚
            half = float(self.synthetic_gripper["half_travel"])
            value = float(
                np.clip((2.0 * half - (inner_width + self.pad_thickness_m)) / 2.0, 0.0, half)
            )
            return (value, value)
        gripper = self.profile.gripper
        pad_center = inner_width + self.pad_thickness_m
        raw = (pad_center - gripper.width_intercept) / gripper.width_slope if gripper.width_slope else 0.0
        lo = min(gripper.open_ctrl[0], gripper.close_ctrl[0])
        hi = max(gripper.open_ctrl[0], gripper.close_ctrl[0])
        value = float(np.clip(raw, lo, hi))
        return (value, -value) if len(gripper.actuator_names) > 1 else (value,)

    def _stone_size(self, name: str) -> np.ndarray:
        """世界系下的三轴尺寸 (x, y, z)。"""
        low, high = self.stone_aabb(name)
        return np.asarray(high - low, dtype=float)

    def stone_center(self, name: str) -> np.ndarray:
        """石块在世界系下的**几何中心**（方向包围盒中心）。

        注意不能用自由关节的 body 原点：网格局部原点并不在几何中心上（实测某块石头
        差 5.4 cm），照着 body 原点下爪会让指垫压在台面上、从石头下面合拢——抓取
        全部失败且看不出来。所有抓放目标都用这里返回的中心。
        """
        centre, _, _ = self.stone_obb(name)
        return np.asarray(centre, dtype=float)

    def stone_obb(self, name: str):
        """石块的**有向**包围盒（世界系）：返回 (中心, 三个轴向, 三个尺寸)。

        为什么不能只用轴对齐包围盒：真实月岩是不规则凸包，AABB 的"窄边"常常来自某个
        角，按它下爪时指垫会从石头窄处穿过去、夹到空气（实测夹爪闭合到 51 mm，而 AABB
        说石头宽 110 mm，最后只有台面接触）。方向包围盒给出的才是真正贴着石头的三个
        轴向和尺寸。
        """
        import trimesh

        geom_id = self._stone_geom[name]
        mesh_id = int(self.model.geom_dataid[geom_id])
        body_id = int(self.model.geom_bodyid[geom_id])
        local = self._mesh_vertices_body(name)
        if local is None or mesh_id < 0:
            centre = self.stone_center(name)
            return centre, np.eye(3), self._stone_size(name)
        faces_address = int(self.model.mesh_faceadr[mesh_id])
        faces_count = int(self.model.mesh_facenum[mesh_id])
        faces = np.asarray(self.model.mesh_face[faces_address : faces_address + faces_count], dtype=int)
        rotation = self.data.xmat[body_id].reshape(3, 3)
        world = local @ rotation.T + self.data.xpos[body_id]
        box = trimesh.Trimesh(vertices=world, faces=faces, process=False).bounding_box_oriented
        axes = np.asarray(box.primitive.transform[:3, :3], dtype=float)
        extents = np.asarray(box.primitive.extents, dtype=float)
        centre = np.asarray(box.primitive.transform[:3, 3], dtype=float)
        return centre, axes, extents

    def mesh_world_vertices(self, name: str) -> np.ndarray:
        """石块网格顶点（世界系），含全部几何变换。"""
        local = self._mesh_vertices_body(name)
        if local is None:
            raise KeyError(f"{name}: 碰撞几何不是网格")
        body_id = int(self.model.geom_bodyid[self._stone_geom[name]])
        return local @ self.data.xmat[body_id].reshape(3, 3).T + self.data.xpos[body_id]

    def _measure_gripper_clearance(self) -> tuple[float, float]:
        """在 home 位姿下量一次并缓存：指垫中心到夹爪最低点的高度差、指垫半高。

        必须在**已知姿态**下量。之前直接用当前位姿量，臂可能正好处于倒置/侧置状态
        （上一次伺服留下的），量出来的"指尖下沉"是 71 mm 而不是 35 mm，于是规划器
        判"石头太矮、夹不了"，退回中心抓取——指尖直接插进台面把石头推走 11 cm，
        表现为"夹爪合到 102 mm 却零接触"。
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        self.controller.set_arm_qpos(np.asarray(self.profile.home_qpos, dtype=float))
        self.controller.forward()
        value = self._gripper_clearance_now()
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.controller.forward()
        return value

    def _gripper_clearance_now(self) -> tuple[float, float]:
        """当前位姿下量：指垫中心到夹爪最低点的高度差、指垫自身的半高。

        这是抓取高度的硬约束。实测：石块只有 ~90 mm 高、平放在台面上，按包围盒中心
        下爪时指垫中心在 36–50 mm，而**指尖**比指垫中心低 30+ mm——指尖顶在台面上，
        手指根本合不拢（表现为"指垫停在比指令更宽的位置"），于是石头夹不住；
        继续加大合拢量就是拿位置伺服（kp=150）去挤台面，接触求解直接炸开。
        """
        pads = [int(self.model.geom(n).id) for n in self.pad_geoms[:2]]
        if len(pads) < 2:
            return 0.03, 0.015
        pad_center_z = 0.5 * (float(self.data.geom_xpos[pads[0]][2]) + float(self.data.geom_xpos[pads[1]][2]))
        lowest = pad_center_z
        half_height = 0.0
        for geom_id in range(self.model.ngeom):
            if self.model.geom_contype[geom_id] == 0 and self.model.geom_conaffinity[geom_id] == 0:
                continue
            body = int(self.model.geom_bodyid[geom_id])
            body_name = self.model.body(body).name
            if body_name not in ("right_gripper", "right_hand") and "finger" not in body_name:
                continue
            half = np.abs(self.data.geom_xmat[geom_id].reshape(3, 3)) @ np.asarray(self.model.geom_size[geom_id])
            centre_z = float(self.data.geom_xpos[geom_id][2])
            lowest = min(lowest, centre_z - float(half[2]))
            half_height = max(half_height, float(half[2]))
        return pad_center_z - lowest, half_height

    def grasp_plan(self, name: str, slices: int = 16, safety: float = 0.98, side_approach: bool = True):
        """按**截面**而不是包围盒挑抓取高度、闭合方向与合拢宽度。

        实测教训：AABB 说石头宽 110 mm，指垫实际合到 88–118 mm 之间就停了，
        抬起时石头不动——因为包围盒的宽度往往来自某个角/棱，指垫所在高度的真实截面
        要窄得多。截面窄 → 指垫夹到空气；命令继续合拢 → 撞上更宽的截面并被位置伺服
        （kp=150）硬挤，接触求解直接炸开（实测抬升量 -1e4～-1e5 mm）。

        判据是"指垫实际占据的那条带里，石头的最大宽度能不能放进夹爪开口"：
        放不进 → 指垫根本合不到石头两侧（判不可行，让上层换一块石头）；
        放得进 → 记下这条带的最大宽度，合拢量按它的 92% 给，指垫压在最宽处，
        接触是部分贴合也没关系——不规则石头本来就不可能整条带均匀接触。
        """
        grip = self.profile.gripper
        limit = grip.max_opening_m * safety
        # "舒服"的带宽：留出至少 25% 的闭合行程，否则指垫只是贴到石头表面、
        # 一点夹紧力都没有（实测：石头带宽 92–93 mm、净开口 95 mm 时，抬起时石头
        # 一动不动，因为位置伺服一到位就没有力了）。
        comfort = grip.max_opening_m * 0.92
        drop, pad_half = self.gripper_geometry
        # 侧向抓取时夹爪转了 90°：指垫那 62 mm 的长边变成水平（沿接近方向），
        # 竖直方向的半高只剩 14.7 mm（实测 UR5e 指垫 29 mm 高），指尖的 36 mm 下沉
        # 也变成水平伸出。于是抓取高度从"必须 ≥42 mm"放宽到"必须 ≥22 mm"，
        # 台面上平放的矮石头终于能抓中段。
        side = bool(side_approach)
        if side:
            drop = 0.0
            pad_half = 0.0147
        centre, axes, extents = self.stone_obb(name)
        vertices = self.mesh_world_vertices(name)
        z_low = float(vertices[:, 2].min())
        z_high = float(vertices[:, 2].max())
        band = max((z_high - z_low) / slices, 0.02)

        # 候选闭合方向：在水平面里直接扫一圈（每 15°），不去信 OBB 的轴。
        # 实测教训：按 OBB"最窄水平轴"选出来的方向，用顶点直接量出来反而是石头
        # 最长的那一侧（170+ mm），指垫下爪时整个压在石头上、把石头推翻。
        # 直接扫角度 + 按指垫带宽统计宽度，"哪边窄"就是量出来的事实。
        heights = vertices[:, 2]
        # 方向候选 = OBB 的两个水平主轴 + 每 3° 的水平扫描。
        # 为什么不能只扫粗角度：石头是 123×83 mm 的扁片，方向偏 7.5° 就让 123 mm 的
        # 长边投影进来（+16 mm），"78 mm 放得下"会变成"98 mm 放不下"——15° 网格
        # 实测把所有石头都判成不可夹。
        directions: list[tuple[int, np.ndarray]] = []
        for index in range(3):
            axis = axes[:, index]
            if float(np.linalg.norm(axis[:2])) > 0.5:
                flat = np.array([axis[0], axis[1], 0.0])
                directions.append((int(round(np.degrees(np.arctan2(flat[1], flat[0])))) % 180,
                                   flat / max(float(np.linalg.norm(flat)), 1e-9)))
        for step_deg in range(0, 180, 3):
            angle = np.deg2rad(step_deg)
            directions.append((step_deg, np.array([np.cos(angle), np.sin(angle), 0.0])))

        best = None
        for step_deg, direction in directions:
            offset = (vertices - centre) @ direction
            for level in range(slices):
                z_center = z_low + band * (level + 0.5)
                # 指尖必须离台面有余量（硬约束：指尖比指垫中心低 36 mm）
                if z_center < z_low + drop + 0.006:
                    continue
                # 真正决定能不能夹的是"指垫与石头在高度上的交叠段"：指垫可以高过石头
                # 顶面（上半截空着不接触），但不能只压到一点点边。之前写的是"整块指垫
                # 都要落在石头高度范围内"，对 55–65 mm 高的石头来说这个窗口是空的——
                # 实测所有石头都被判成不可夹，于是退回中心抓取、指尖插进台面把石头推走。
                band_lo = max(z_low, z_center - pad_half)
                band_hi = min(z_high, z_center + pad_half)
                if band_hi - band_lo < 0.015:
                    continue
                selected = (heights >= band_lo) & (heights <= band_hi)
                if int(selected.sum()) < 3:
                    continue
                lo = float(offset[selected].min())
                hi = float(offset[selected].max())
                width = hi - lo
                if width <= 1e-4 or width > limit:
                    continue
                # 平行度：石头侧面要是斜的，抬起时指垫会顺着斜面把石头"楔"出去——
                # 实测指垫在石头顶部斜面上有 35 N 法向力，臂抬 203 mm 石头只动 0.6 mm。
                # 用两个 24 mm 高的窗口比较宽度（窗口跟着高度上移 12 mm），
                # 而不是拿"整条抓取带"和"整体上移 12 mm 的带"比——后者在石头很矮时
                # 两个窗口几乎一样，量出来恒等于 0，等于没有约束。
                def window_width(z_lo: float, z_hi: float) -> float:
                    z_lo = max(z_lo, z_low)
                    z_hi = min(z_hi, z_high)
                    if z_hi - z_lo < 0.008:
                        return float("nan")
                    mask = (heights >= z_lo) & (heights <= z_hi)
                    if int(mask.sum()) < 3:
                        return float("nan")
                    return float(offset[mask].max() - offset[mask].min())

                lower = window_width(z_center - 0.012, z_center + 0.012)
                upper = window_width(z_center, z_center + 0.024)
                taper = 0.5 if (np.isnan(lower) or np.isnan(upper)) else min(abs(upper - lower), 0.5)
                tier = 0 if width <= comfort else 1
                # 排序（越靠前权重越高）：
                #  1) 带宽落在"能夹稳"的区间——太宽放不进/没有闭合行程，太窄只是掐住一条棱；
                #  2) 侧面平行（taper 小）；
                #  3) 抓取高度靠近石头中段。
                # 注意宽度是"越接近目标越好"，不是"越宽越好"：石头缩到 0.6 倍时，
                # 按"越宽越好"会把方向挑到对角线（86 mm），那正是最夹不住的地方。
                target_width = 0.60 * grip.max_opening_m
                mid_height = 0.5 * (z_low + z_high)
                # 用一个**连续质量**代替字典序：带宽、平行度、高度各走一个高斯再相乘。
                # 字典序怎么排都只能满足一头——实测"带宽优先"会挑到平行度 500 mm 的
                # 楔形带（一侧指垫贴住、另一侧指尖压在石头顶面，法向 (-0.62,-0.78) 里
                # 一大半是向下的，抬起时把石头往下推）；"平行度优先"又会挑到 81 mm 的
                # 宽带（贴着净开口上限、没有闭合行程）。
                sigma_w = 0.18 * grip.max_opening_m
                width_score = float(np.exp(-((width - target_width) / max(sigma_w, 1e-6)) ** 2))
                taper_score = float(np.exp(-(min(taper, 0.2) / 0.025) ** 2))
                # 抓取高度要对准石头**质心**（用方向包围盒中心近似），不是几何范围的中点：
                # 夹在质心上方，两指摩擦力的合力矩会把石头"滚"出去——实测合爪时夹住
                # 130 N，抬臂时指垫却从 67 mm 继续合到 56 mm（正好是石头的厚度方向），
                # 说明石头在夹口里翻了个身，然后被挤出来。sigma 取 8 mm。
                height_score = float(np.exp(-((z_center - float(centre[2])) / 0.008) ** 2))
                quality = width_score * taper_score * (0.6 + 0.4 * height_score)
                score = (-tier, quality, -abs(z_center - mid_height))
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "quality": float(quality),
                        "axis_index": step_deg,
                        "direction": direction,
                        "width": width,
                        "z_center": z_center,
                        "offset_center": 0.5 * (lo + hi),
                        "taper": taper,
                    }
        if best is None:
            # 没有一层放得进开口：退回"最窄的一层"，让上层策略换石头
            narrow_axis, narrow_width = self.grasp_axis(name)
            return {
                "direction": narrow_axis,
                "quality": 0.0,
                "width": narrow_width,
                "pad_center": np.array([centre[0], centre[1], centre[2]]),
                "feasible": False,
                "reason": (
                    f"没有可用截面：需要宽度 ≤ {limit * 1000:.0f} mm、侧面近垂直（12 mm 内变化 "
                    f"≤ 20 mm）且指垫中心高于台面 {(drop + 0.006) * 1000:.0f} mm"
                    f"（包围盒窄边 {narrow_width * 1000:.0f} mm）"
                ),
            }
        pad_center = centre + best["direction"] * best["offset_center"]
        pad_center[2] = best["z_center"]
        # 接近方向：水平、垂直于闭合方向，指向"从基座一侧推进"。从上方下爪要求石头
        # 够高（指尖比指垫中心低 36 mm），台面上平放的石头只有 60–90 mm 高，够不着；
        # 水平接近时这 36 mm 变成沿接近方向的伸出量，抓取高度不再受限。
        tilt = self.workcell.wall_center - self.workcell.base_pos
        tilt = np.array([tilt[0], tilt[1], 0.0])
        tilt = tilt / max(float(np.linalg.norm(tilt)), 1e-9)
        lateral = np.array([-best["direction"][1], best["direction"][0], 0.0])
        if float(np.dot(lateral, tilt)) < 0.0:
            lateral = -lateral
        return {
            "direction": best["direction"],
            "approach": lateral,
            "quality": float(best.get("quality", 0.0)),
            "width": float(best["width"]),
            "pad_center": pad_center,
            "feasible": True,
            "reason": (
                f"截面高度 {best['z_center'] * 1000:.0f} mm（指尖余量 "
                f"{(best['z_center'] - z_low - drop) * 1000:.0f} mm）处沿 "
                f"偏航 {best['axis_index']}° 宽 {best['width'] * 1000:.0f} mm"
                f"（净开口 {grip.max_opening_m * 1000:.0f} mm，舒服上限 {comfort * 1000:.0f} mm，"
                f"平行度 {best.get('taper', 0.0) * 1000:.0f} mm）"
            ),
        }

    def grasp_axis(self, name: str) -> tuple[np.ndarray, float]:
        """选夹爪闭合方向：水平两轴里**较窄**的那一轴，返回 (单位方向, 该方向尺寸)。

        石头在料区是随便躺的，挑窄边下爪既能夹住更宽的石头，也让"夹得住吗"这个
        问题变成"最窄的水平尺寸够不够小"，不用去猜网格的局部坐标系。
        """
        centre, axes, extents = self.stone_obb(name)
        # 三个 OBB 轴投影到水平面，取水平分量最大（= 最接近水平）的那两根作为水平轴；
        # 夹爪沿"较窄的那根水平轴"闭合，抓取高度取 OBB 中心高度。
        horizontal = []
        for index in range(3):
            axis = axes[:, index].copy()
            horizontal_scale = float(np.linalg.norm(axis[:2]))
            horizontal.append((horizontal_scale, index, axis))
        horizontal.sort(reverse=True)
        candidates = horizontal[:2]
        narrow_scale, narrow_index, narrow_axis = min(candidates, key=lambda item: extents[item[1]])
        direction = narrow_axis.copy()
        direction[2] = 0.0
        if float(np.linalg.norm(direction)) < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        direction /= float(np.linalg.norm(direction))
        return direction, float(extents[narrow_index]) * float(np.linalg.norm(narrow_axis[:2]))

    def held_stone(self) -> str | None:
        """谁在夹爪里：指垫中心附近 3 cm 内、且离台面较高的那块。"""
        if not self.pad_geoms:
            tcp_pos, tcp_rot = self.controller.tcp_pose()
            pad_center = tcp_pos + tcp_rot @ np.asarray(self.profile.gripper.pad_offset_tcp, dtype=float)
        else:
            ids = [self.model.geom(name).id for name in self.pad_geoms[:2]]
            pad_center = 0.5 * (self.data.geom_xpos[ids[0]] + self.data.geom_xpos[ids[1]])
        best, best_distance = None, 0.035
        for name in self.stone_names():
            pos, _ = self.stone_pose(name)
            distance = float(np.linalg.norm(pos - pad_center))
            if distance < best_distance:
                best, best_distance = name, distance
        return best

    # ------------------------------------------------------------------ 观测

    def observe(self, phase: str, instruction: str = "", step_index: int = 0, want_cameras: bool | None = None) -> Observation:
        joints = {name: float(self.data.qpos[address]) for name, address in zip(self.profile.joints, self.controller.qpos_addr)}
        tcp_pos, tcp_rot = self.controller.tcp_pose()
        stones = []
        for name in self.stone_names():
            _, quat = self.stone_pose(name)
            pos = self.stone_center(name)
            stretch = self._stone_size(name)
            # "夹得住吗"必须用真正的抓取规划回答，而不是包围盒窄边：石头可能是
            # 上窄下宽的锥形，窄边放得进开口但整条侧面都是斜的，抬起时会被楔出去。
            # `grasp_plan` 已经看了方向、截面、平行度与指尖余量，用它做候选过滤。
            plan = self.grasp_plan(name)
            stones.append(
                StoneState(
                    name=name,
                    pos=pos,
                    quat=quat,
                    size=stretch,
                    mass=self._masses[name],
                    graspable=bool(plan["feasible"]),
                    grasp_quality=float(plan.get("quality", 0.0)),
                )
            )
        stones = tuple(stones)
        frames = {}
        if self.cameras is not None and self.config.capture_cameras and (want_cameras is None or want_cameras):
            frames = self.cameras.capture(self.config.camera_names)
        return Observation(
            time=float(self.data.time),
            phase=phase,  # type: ignore[arg-type]
            joints=joints,
            tcp_pos=tcp_pos,
            tcp_rot=tcp_rot,
            gripper_width=self.controller.measured_opening(),
            cameras=frames,
            stones=stones,
            held_stone=self.held_stone(),
            instruction=instruction,
            step_index=step_index,
        )

    def slots(self) -> list[PlanSlot]:
        """名义槽位：x 由工作区给定，z 用平均厚度估一个初始值，策略会按观测修正。"""
        slots: list[PlanSlot] = []
        for slot in self.workcell.slots:
            z = (slot.course + 0.5) * self.workcell.mean_thickness
            slots.append(
                PlanSlot(
                    course=slot.course,
                    slot_index=slot.index,
                    target_pos=self.workcell.slot_position(slot.course, slot.index, z),
                    target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                )
            )
        return slots

    def candidates(self, observation: Observation) -> list[Candidate]:
        """候选 = 还没放上墙的石头 × 所有槽位。VLM 就是在这个列表里选。"""
        placed = self._placed_names(observation)
        free = [stone for stone in observation.stones if stone.name not in placed]
        candidates: list[Candidate] = []
        for slot in self.slots():
            for stone in free:
                # 分数 = 体积（下层优先放大石头），夹不住的石头重罚
                # 抓取质量是主项：夹不住的石头体积再大也没用（实测挑到楔形带的石头
                # 时，指垫一侧贴住、另一侧指尖压在石头顶面，抬起时把石头往下推）。
                score = 2.0 * float(stone.grasp_quality) + 0.3 * float(np.prod(stone.size)) / 1e-3
                score -= 0.01 * float(np.linalg.norm(stone.pos[:2]))
                if not stone.graspable:
                    score -= 5.0
                candidates.append(
                    Candidate(
                        stone=stone,
                        slot=slot,
                        score=score,
                        note="" if stone.graspable else "抓取规划不可行",
                    )
                )
        return candidates

    def _placed_names(self, observation: Observation) -> set[str]:
        """已经"贴在墙上"的石头：与任何槽位 xy 距离小于阈值。"""
        placed: set[str] = set()
        for stone in observation.stones:
            for slot in self.slots():
                if float(np.linalg.norm(stone.pos[:2] - slot.target_pos[:2])) < 0.06:
                    placed.add(stone.name)
                    break
        return placed

    # ------------------------------------------------------------------ 接触开关

    def set_gripper_contact(self, enabled: bool) -> None:
        geoms = list(self.profile.gripper.finger_geoms) or list(self.pad_geoms)
        for name in geoms:
            geom_id = self.model.geom(name).id
            self.model.geom_contype[geom_id] = 1 if enabled else 0
            self.model.geom_conaffinity[geom_id] = 1 if enabled else 0

    def contact_count(self, stone_name: str, others_only: bool = True) -> int:
        geom_id = self._stone_geom[stone_name]
        count = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                other = contact.geom2 if contact.geom1 == geom_id else contact.geom1
                if others_only and other in self._stone_geom.values():
                    continue
                count += 1
        return count

    def measured_opening(self) -> float:
        """场景里**真正参与夹持**的两片指垫的实测开口（米）。

        不能用 `controller.measured_opening()`：它量的是 `profile.gripper.pad_geoms`，
        而装了自造平行夹爪时那些 Robotiq 旧指垫仍在模型里但**没人驱动**，读数会一直停在
        全开的 ~128 mm，看起来像"夹爪没闭合"，实际夹的是 `pg_jaw_geom_*`。
        """
        if len(self._pad_geom_ids) >= 2:
            left, right = self._pad_geom_ids[0], self._pad_geom_ids[1]
            return float(np.linalg.norm(self.data.geom_xpos[right] - self.data.geom_xpos[left]))
        return self.controller.measured_opening()

    def _sync_viewer(self, force: bool = False) -> None:
        """把当前物理状态推给实时 viewer；按 ~30 Hz 节流，避免每个物理步都刷。

        `--view` 之前是坏的：viewer 建了但从不 `sync()`，窗口不刷新。所有推进物理的
        地方（`settle` / `servo_to` / `servo_translate`）现在都会调到这里。
        """
        if self.viewer is None:
            return
        now = time.time()
        if not force and now - self._last_sync < 1.0 / 30.0:
            return
        self._last_sync = now
        try:
            self.viewer.sync()
        except Exception:  # noqa: BLE001 - 窗口被手动关掉不该毁掉整个任务
            self.viewer = None

    def settle(self, seconds: float) -> None:
        dt = float(self.model.opt.timestep)
        remaining = max(0, int(round(seconds / dt)))
        while remaining > 0:
            chunk = min(remaining, 50)
            self.controller.step(chunk)
            remaining -= chunk
            self._sync_viewer()
        self._sync_viewer(force=True)

    def canonical_pocket(self) -> int:
        """统一的抓取口袋序号：取圆弧中间那个（正对墙，最好够）。

        实测教训：料区口袋里相邻两个只差 0.11 m，而石头最长 0.19 m——如果每块石头
        各用一个口袋，没抓成功的石头会留在原地、和下一块叠在一起，物理直接炸开
        （抬升量出现 -4e8 mm 那种数）。所以每次抓之前只让**一块**石头上口袋。
        """
        return len(self.workcell.slots) // 2

    def reset_pick_scene(self, target_name: str) -> None:
        """抓取前的场景归位：未上墙的石头回初始摆放位，目标石头单独进抓取口袋。"""
        placed = set(self._placed_names(self.observe("idle")))
        pocket_index = self.canonical_pocket()
        for name in self.stone_names():
            if name == target_name or name in placed:
                continue
            pos, quat = self.workcell.staging_pose(name)
            self.set_stone_pose(name, pos, quat)
        stone = self._flat_stone(target_name)
        pos, quat = self.workcell.pocket_pose(pocket_index, stone)
        self.set_stone_pose(target_name, pos, quat)

    # ------------------------------------------------------------------ 主循环

    def run(self, instruction: str = "", max_placements: int | None = None) -> dict:
        started = time.time()
        limit = max_placements if max_placements is not None else len(self.workcell.slots)
        self.pair.reset({"instruction": instruction, "profile": self.profile.name})
        # 从 home 起步：模型默认 qpos=0 对有些臂是奇异/折叠构型，直接开始会先经历
        # 一次大幅跌落，抓取路径的第一段就废了。
        self.controller.set_arm_qpos(np.asarray(self.profile.home_qpos, dtype=float))
        self.controller.forward()
        self.controller.open_gripper()
        self.settle(0.5)

        records: list[PlacementRecord] = []
        for placement_index in range(limit):
            observation = self.observe("idle", instruction)
            candidates = self.candidates(observation)
            if not candidates:
                self._log("没有候选石头/槽位，结束")
                break
            decision = self.pair.high.decide(observation, candidates)
            if decision.is_stop or decision.stone is None or decision.target_pos is None:
                self._log(f"高层策略停机：{decision.reason}")
                self.policy_events.append({"event": "stop", "reason": decision.reason, "source": decision.source})
                break
            self._log(
                f"[{placement_index + 1}/{limit}] 决策 {decision.stone} → course={decision.slot} "
                f"target={np.round(decision.target_pos, 3).tolist()} ({decision.source}: {decision.reason})"
            )
            record = self.execute_placement(decision, instruction, placement_index)
            records.append(record)
            self._log(
                f"    placed={record.placed} stacked={record.stacked} "
                f"lift={record.lift_gain_m:.3f} xy_err={record.target_xy_error_m:.3f} "
                f"z_err={record.target_z_error_m:.3f} ({record.duration_s:.1f}s)"
            )

        placed_count = sum(1 for record in records if record.placed)
        stacked_count = sum(1 for record in records if record.stacked)
        summary = {
            "success": bool(records) and placed_count == len(records),
            "arm": self.profile.name,
            "policy": self.pair.name,
            "requested_placements": limit,
            "executed_placements": len(records),
            "placed_count": placed_count,
            "stacked_count": stacked_count,
            "wall_center": [float(value) for value in self.workcell.wall_center],
            "workcell": self.workcell.describe(),
            "policy_events": self.policy_events,
            "steps": [asdict(record) for record in records],
            "duration_s": time.time() - started,
        }
        if self.cameras is not None:
            summary["cameras"] = {
                name: asdict(self.cameras.intrinsics(name)) for name in self.cameras.camera_names()
            }
        return summary

    # ------------------------------------------------------------------

    def execute_placement(self, decision: Decision, instruction: str, placement_index: int) -> PlacementRecord:
        started = time.time()
        name = decision.stone
        assert name is not None
        stone = next(stone for stone in self.observe("idle", instruction).stones if stone.name == name)
        target_pos = np.asarray(decision.target_pos, dtype=float)
        target_quat = np.asarray(decision.target_quat if decision.target_quat is not None else [1, 0, 0, 0], dtype=float)
        course, slot_index = decision.slot if decision.slot is not None else (0, 0)

        tilt_hint = self.workcell.wall_center - self.workcell.base_pos

        attempts = 0
        lift_gain = 0.0
        lifted_pos = np.zeros(3)
        pick_pos = np.zeros(3)
        servo_metrics: list[dict] = []
        while True:
            attempts += 1
            # 夹紧量：实测要压到带宽的 ~0.3–0.5 倍才有足够的夹持力（接触力与压入量
            # 近似成正比：0.9×带宽只有 10 N，0.3×带宽到 160 N 并能把石头吊住）
            close_fraction = self.config.grasp_close_fraction if attempts == 1 else max(
                0.20, self.config.grasp_close_fraction - 0.15
            )
            # 1) 场景归位：只让目标石头上口袋，其余未上墙的石头回初始位（避免互相压住）
            self.reset_pick_scene(name)
            self.settle(self.config.reset_seconds + 0.2 * (attempts - 1))
            _, pocket_quat = self.stone_pose(name)
            # 抓取几何必须在石头就位之后再规划
            plan = self.grasp_plan(name)
            if plan.get("feasible") and self.config.nest_align:
                # 料区当作"定位窝"：把石头的几何中心对齐到规划的抓取轴上。
                # 实测：石头平放在台面上时，指垫从两侧合拢会先把石头沿台面推开
                # （第一个接触的指垫给它一个横向推力，台面摩擦只有 1.0，5 N 的石头
                # 被 100+ N 的合拢力一推就滑走），指垫于是从空档里闭合；
                # 而把石头**摆正到指垫之间**再合爪，同样的动作能稳稳夹住并吊住
                # （悬空实测：接触 ~160 N，石头保持不动）。真机上的料区本来也是定位窝。
                centre = self.stone_center(name)
                pad = np.asarray(plan["pad_center"], dtype=float)
                pos, quat = self.stone_pose(name)
                self.set_stone_pose(name, pos + np.array([pad[0] - centre[0], pad[1] - centre[1], 0.0]), quat)
                self.settle(0.15)
            closing_direction = plan["direction"]
            narrow_extent = plan["width"]
            if str(self.config.grasp_approach).strip().lower() == "side":
                # 侧向抓取：接近轴取 grasp_plan 给的横向向量，夹爪水平推进。
                approach = np.asarray(plan.get("approach", np.array([0.0, 0.0, -1.0])), dtype=float)
            else:
                # 竖直下爪：approach_direction=None 让 grasp_rotation 走
                # approach_world=[0,0,-1] 的分支（夹爪指向正下方），再按
                # profile.approach_tilt_deg 微调；UR5e 该值为 0°，即笔直向下。
                # 低层策略也会因此改走"正上方下爪"的路径点（scripted._pick_waypoints）。
                approach = None
            grasp_rot = self.controller.grasp_rotation(
                closing_direction, tilt_hint=tilt_hint, approach_direction=approach
            )
            fits = bool(plan["feasible"])
            pick_pos = np.asarray(plan["pad_center"], dtype=float) if fits else self.stone_center(name)

            # 夹得住就按窄边收窄着夹（可控、不挤飞石头）；夹不住就直接全闭，
            # 让指垫尽量咬住棱角——这一类基本会失败，靠上层换石头解决。
            grip_width = max(0.002, narrow_extent * close_fraction) if fits else 0.0
            pick_goal = Goal(
                kind="pick",
                stone=StoneState(
                    name=name,
                    pos=pick_pos,
                    quat=pocket_quat,
                    size=self._stone_size(name),
                    mass=self._masses[name],
                    graspable=fits,
                ),
                pick_pos=pick_pos,
                pick_rot=grasp_rot,
                grip_width=grip_width,
                # 这三项必须传：不传的话低层策略会走"正上方下爪"的路径点，
                # 而几何是按侧向抓取规划的——实测就是这个漏传，让执行器一直
                # 用竖直下降/上升去做侧向抓取，所有验证过的侧向参数全部没生效。
                approach_direction=approach,
                standoff_m=float(self.config.side_standoff_m),
                grasp_band_width=float(narrow_extent),
                timeout_s=self.config.policy_timeout_s,
            )
            self._run_goal(pick_goal, phase="pick", instruction=instruction, metrics=servo_metrics)
            lifted_pos = self.stone_center(name)
            lift_gain = float(lifted_pos[2] - pick_pos[2])
            if self.trace:
                self._log(
                    f"   [抓取诊断] stone={name} attempt={attempts} feasible={fits} "
                    f"band={narrow_extent * 1000:.1f}mm close_fraction={close_fraction:.2f} "
                    f"grip_cmd={grip_width * 1000:.1f}mm measured_open="
                    f"{self.measured_opening() * 1000:.1f}mm "
                    f"contacts={self.contact_count(name)} "
                    f"pick_z={pick_pos[2] * 1000:.1f}mm lifted_z={lifted_pos[2] * 1000:.1f}mm "
                    f"lift_gain={lift_gain * 1000:.1f}mm stone_moved="
                    f"{np.round((lifted_pos - pick_pos) * 1000, 1).tolist()}mm"
                )
            if lift_gain >= 0.02 or attempts > self.config.grasp_retries:
                break
            self._log(f"   抓取失败（lift_gain={lift_gain:.3f}，plan: {plan['reason']}），重试 {attempts}")
            self.policy_events.append(
                {"event": "grasp_retry", "stone": name, "attempt": attempts, "lift_gain_m": lift_gain}
            )

        # 2) 搬运并放置
        place_goal = Goal(
            kind="place",
            stone=StoneState(
                name=name,
                pos=target_pos,
                quat=target_quat,
                size=self._stone_size(name),
                mass=self._masses[name],
                graspable=True,
            ),
            place_pos=self.controller.tcp_for_pad_center(target_pos, grasp_rot),
            place_rot=grasp_rot,
            grip_width=max(0.002, narrow_extent * 0.85) if fits else 0.0,
            timeout_s=self.config.policy_timeout_s,
        )
        self._run_goal(place_goal, phase="place", instruction=instruction, metrics=servo_metrics)
        self.set_gripper_contact(False)
        self.settle(self.config.settle_seconds)
        final_center = self.stone_center(name)
        _, final_quat = self.stone_pose(name)
        # 判定用几何中心；报告里同时留 body 位姿，便于和旧 TASK2 的报告对照
        final_pos = final_center

        lift_threshold = self.config.lift_threshold_m
        placed = bool(
            lift_gain > lift_threshold
            and float(np.linalg.norm(final_pos[:2] - target_pos[:2])) < self.config.place_tolerance_xy_m
            and float(abs(final_pos[2] - target_pos[2])) < self.config.place_tolerance_z_m
        )
        stacked = bool(course == 0 or final_pos[2] > self.config.stacked_height_m)
        return PlacementRecord(
            name=name,
            course=int(course),
            slot_index=int(slot_index),
            target_pos=[float(value) for value in target_pos],
            target_quat=[float(value) for value in target_quat],
            pick_pos=[float(value) for value in pick_pos],
            lifted_pos=[float(value) for value in lifted_pos],
            final_pos=[float(value) for value in final_pos],
            final_quat=[float(value) for value in final_quat],
            lift_gain_m=lift_gain,
            target_xy_error_m=float(np.linalg.norm(final_pos[:2] - target_pos[:2])),
            target_z_error_m=float(abs(final_pos[2] - target_pos[2])),
            placed=placed,
            stacked=stacked,
            attempts=attempts,
            policy_source=decision.source,
            policy_reason=decision.reason,
            grasp_plan=str(plan.get("reason", "")),
            duration_s=time.time() - started,
            policy_steps=len(servo_metrics),
            servo_metrics=servo_metrics,
        )

    def _flat_stone(self, name: str):
        from .rocks import FlatStone

        geom_id = self._stone_geom[name]
        mesh_id = int(self.model.geom_dataid[geom_id])
        faces_address = int(self.model.mesh_faceadr[mesh_id])
        faces_count = int(self.model.mesh_facenum[mesh_id])
        vertices = self._mesh_vertices_body(name)
        if vertices is None:
            raise KeyError(f"{name}: 碰撞几何不是网格，无法还原石块")
        faces = self.model.mesh_face[faces_address : faces_address + faces_count]
        size = self._stone_size(name)
        return FlatStone(
            name=name,
            vertices=[tuple(map(float, vertex)) for vertex in vertices],
            faces=[tuple(map(int, face)) for face in faces],
            rgba=(0.5, 0.5, 0.5, 1.0),
            mass=self._masses[name],
            length=float(size[0]),
            width=float(size[1]),
            thickness=float(size[2]),
        )

    def _run_goal(self, goal: Goal, phase: str, instruction: str, metrics: list[dict]) -> None:
        """向低层策略要目标位姿，伺服到那个位姿，直到策略说 done 或超时。"""
        dead_line = time.time() + goal.timeout_s
        step_index = 0
        last_q = self.controller.arm_qpos()
        while time.time() < dead_line:
            observation = self.observe(phase, instruction, step_index=step_index)
            command = self.pair.low.step(observation, goal)
            step_index += 1
            grip_ctrl = None
            if command.grip_width is not None:
                grip_ctrl = self._ctrl_for_width(float(command.grip_width))
            if command.done:
                metrics.append(
                    {
                        "phase": phase,
                        "step": step_index,
                        "done": True,
                        "note": command.note,
                        "source": command.source,
                        "tcp_error_m": float(np.linalg.norm(command.tcp_pos - observation.tcp_pos)),
                    }
                )
                break
            guard = (
                self._contact_guard(goal)
                if (self.config.contact_aware_place and phase == "place")
                else None
            )

            def on_step(controller, _guard=guard):  # noqa: ANN001
                self._sync_viewer()
                if _guard is not None:
                    _guard(controller)

            if command.pure_translation:
                delta = np.asarray(command.tcp_pos, dtype=float) - np.asarray(observation.tcp_pos, dtype=float)
                result = self.controller.servo_translate(
                    delta, seconds=max(self.config.servo_seconds, 0.5), grip_ctrl=grip_ctrl, on_step=on_step
                )
            else:
                result = self.controller.servo_to(
                    command.tcp_pos,
                    command.tcp_rot,
                    seconds=self.config.servo_seconds,
                    grip_ctrl=grip_ctrl,
                    q_seed=last_q,
                    on_step=on_step,
                )
            last_q = self.controller.arm_qpos()
            if self.trace:
                self._log(
                    f"      · {phase} step{step_index} {command.source} "
                    f"tcp={np.round(command.tcp_pos, 4).tolist()} grip="
                    f"{None if command.grip_width is None else round(command.grip_width * 1000, 1)}mm "
                    f"pos_err={result.final_pos_error_m * 1000:.1f}mm reached={result.reached} "
                    f"measured_open={self.measured_opening() * 1000:.1f}mm done={command.done}"
                )
            metrics.append(
                {
                    "phase": phase,
                    "step": step_index,
                    "note": command.note,
                    "source": command.source,
                    "target": [float(value) for value in command.tcp_pos],
                    "pos_error_m": result.final_pos_error_m,
                    "rot_error_rad": result.final_rot_error_rad,
                    "reached": result.reached,
                }
            )
        else:
            self.policy_events.append({"event": "policy_timeout", "phase": phase, "goal": goal.describe()})

    def _contact_guard(self, goal: Goal):
        """放置下降时的接触保护：一旦石头吃上力就停止继续下降。"""
        stone_name = goal.stone.name if goal.stone is not None else None
        state = {"tripped": False, "baseline": self.contact_count(stone_name) if stone_name else 0}

        def hook(controller: RobotController) -> None:
            if state["tripped"] or stone_name is None:
                return
            contacts = self.contact_count(stone_name)
            if contacts > state["baseline"]:
                state["tripped"] = True
                controller.data.ctrl[controller._arm_actuator_ids()] = controller.arm_qpos()  # 冻结当前关节目标

        return hook


def _quat_to_mat(quat) -> np.ndarray:
    """MuJoCo 四元数 (w x y z) → 3x3 旋转矩阵。"""
    w, x, y, z = (float(value) for value in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
