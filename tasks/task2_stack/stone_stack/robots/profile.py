"""机械臂 profile：一套臂 + 夹爪的可切换描述。

为什么需要它
------------
原 TASK2 把 UR5e + Robotiq 2F-140 的假设写死在执行器里（关节名、`grip_site`、
`data.ctrl[:6]`、两只反向手指、夹爪几何名字）。要让同一套堆叠逻辑跑在别的臂上，
必须把这些假设收进一个显式的数据结构，而不是散在控制代码里。

一个 profile 描述四类东西
--------------------------
1. **装配**：臂的 MJCF、夹爪是自带还是外挂、外挂时插进哪根 body、基座装在哪。
2. **控制**：被位置伺服的关节名、执行器增益（沿用源文件还是自己建）、home 位姿。
3. **夹爪几何**：TCP 在哪、开口方向 / 接近方向在 TCP 系里是哪个轴、开口宽度与
   控制量的关系、能夹多宽、哪些 geom 是指垫。
4. **工作区**：基座位姿、石块缩放、抓取宽度上限。

这里所有"实测"数字都来自本机对编译后模型的测量（`tools/probe_arm_geometry.py`
与本轮 UR5e 夹爪曲线），不是抄注释。测量方法与复现命令见
`docs/ARM_GEOMETRY.md`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: MoonSim 仓库根目录：<root>/tasks/task2_stack/stone_stack/robots/profile.py
REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS = REPO_ROOT / "assets" / "robots"
ROBOSUITE_ASSETS = ASSETS / "robosuite" / "models" / "assets"


# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ActuatorSpec:
    """臂关节的伺服配置。

    `mode="source"` 表示直接用源 MJCF 里的 `<actuator>`（Menagerie Panda、Piper 自带
    位置伺服）；`mode="position"` 表示丢掉源执行器（robosuite UR5e 自带的是一阶
    `<motor>`），按这里给的增益重建位置伺服。
    """

    mode: str = "source"
    kp: float = 650.0
    kv: float = 55.0
    forcerange: tuple[float, float] = (-280.0, 280.0)
    ctrlrange: dict[str, tuple[float, float]] = field(default_factory=dict)
    name_template: str = "pos_{joint}"


@dataclass(frozen=True)
class GripperSpec:
    """夹爪几何与命令约定。角度/向量都在 **TCP site 系** 里。"""

    kind: str
    #: 参与夹爪控制的执行器名（顺序即命令顺序）
    actuator_names: tuple[str, ...]
    #: 完全张开 / 完全闭合的命令
    open_ctrl: tuple[float, ...]
    close_ctrl: tuple[float, ...]
    #: **指垫中心距** = intercept + slope * ctrl（用第一个执行器控制量做自变量）
    width_intercept: float
    width_slope: float
    #: 指垫厚度（两片指垫内侧到中心的距离之和）：可用净开口 = 指垫中心距 − 这个值。
    #: 这个数不能省——实测 UR5e 指垫中心距 128.4 mm 看着能夹 110 mm 的石头，
    #: 但每片指垫厚 16.7 mm，真正能塞进去的净开口只有 95 mm；按中心距判"能夹"
    #: 会让指垫在下爪时直接撞在石头上，把石头撞飞 26 cm（实测）。
    pad_thickness_m: float
    #: **可用净开口**（米）：判定"夹得住吗"用这个，不是指垫中心距
    max_opening_m: float
    min_opening_m: float
    #: 张开方向（两指分离方向）与接近方向（从 TCP 指向被夹物体）在 TCP 系下的单位向量
    open_axis_tcp: tuple[float, float, float]
    approach_axis_tcp: tuple[float, float, float]
    #: 指垫中心相对 TCP 原点的偏移（TCP 系）
    pad_offset_tcp: tuple[float, float, float]
    #: 指垫 / 指尖的碰撞 geom 名，放置后要临时关掉它们避免把石头带倒
    pad_geoms: tuple[str, ...] = ()
    #: 除了指垫之外、随夹爪一起运动的碰撞 geom（整只夹爪），用于更彻底的接触开关
    finger_geoms: tuple[str, ...] = ()
    #: 两根手指的 body 名（没有独立指垫 geom 时用它测开口）
    finger_bodies: tuple[str, str] | None = None
    #: 指关节位置伺服的增益。上游 robosuite 是 kp=20，旧 TASK2 执行器用 kp=150。
    #: 实测夹持负载下手指会退让（150 N·m/rad 时 65 N 切向力就能顶开），
    #: 夹持阶段需要更硬的伺服，否则抬起时石头从指间滑下去。
    actuator_kp: float = 150.0
    actuator_kv: float = 5.0
    #: 命令量到"每侧手指位移"的换算是否需要取反（UR/2F-140 第二路是反向的）
    mirrored: bool = False

    def width_from_ctrl(self, ctrl: float) -> float:
        return self.width_intercept + self.width_slope * ctrl

    def ctrl_for_width(self, width: float) -> tuple[float, ...]:
        """把目标**净开口**换成命令。会夹在完全张开/闭合之间。"""
        pad_center = width + self.pad_thickness_m
        raw = (pad_center - self.width_intercept) / self.width_slope if self.width_slope else 0.0
        lo = min(self.open_ctrl[0], self.close_ctrl[0])
        hi = max(self.open_ctrl[0], self.close_ctrl[0])
        value = float(np.clip(raw, lo, hi))
        if len(self.actuator_names) == 1:
            return (value,)
        if self.mirrored:
            return (value, -value)
        return tuple([value] * len(self.actuator_names))

    def graspable(self, width: float, margin: float = 0.95) -> bool:
        """还能夹得住吗？`margin=0.95` 是"要留一点指垫间隙"，不是保守系数：
        实测 UR5e 夹爪最大开口 128.4 mm，夹住过 118 mm 的石头，而 126 mm 的开始打滑。"""
        return width <= self.max_opening_m * margin


@dataclass(frozen=True)
class CameraMount:
    """相机安装：`in_tcp=True` 表示位姿是相对 TCP 的（腕部相机），否则是世界系。"""

    name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    fovy_deg: float
    in_tcp: bool = False
    #: 腕部相机要挂到 TCP 所在 body 上；None 表示挂世界
    parent_body: str | None = None


@dataclass(frozen=True)
class ArmProfile:
    name: str
    description: str
    arm_xml: Path
    base_body: str
    joints: tuple[str, ...]
    home_qpos: np.ndarray
    base_pos: tuple[float, float, float]
    actuator: ActuatorSpec
    gripper: GripperSpec
    #: TCP：给现有 site 名，或者给要在 `tcp_parent_body` 上现建的 site
    tcp_site: str
    tcp_parent_body: str | None = None
    tcp_site_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_site_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #: 外挂夹爪：夹爪 MJCF 与插入目标
    gripper_xml: Path | None = None
    gripper_attach_body: str | None = None
    gripper_root_body: str | None = None
    #: 关掉机器人自碰撞（robosuite UR5e 的显示/碰撞几何本来就不自碰）
    disable_self_collision: bool = False
    #: 是否给世界加默认的 joint damping/armature（源文件自带 `<default>` 时不要再加）
    scene_defaults: bool = True
    #: 石块尺寸缩放：夹爪开口小的臂用小石头
    stone_scale: float = 1.0
    #: 工作区缩放：基座到墙/料区的距离按臂展缩放
    work_scale: float = 1.0
    #: 抓取时接近轴相对竖直方向的倾角（度）。
    #: 短臂/腕部限位紧的臂够不到"笔直向下"的位姿（Piper 实测：0° 时上层槽位差 65 mm，
    #: 倾 10° 后 0.08 mm），所以倾角是 profile 的一部分，而不是执行器的常数。
    approach_tilt_deg: float = 0.0
    #: 标称臂展（米），只用于文档与工作区缩放
    reach_m: float = 0.85
    #: 最小可达半径（米）：太靠近基座的位姿反而解不出来（肘部折不过来）
    min_reach_m: float = 0.20
    #: 相机（腕部相机由装配阶段按 TCP 位姿自动求安装点）
    cameras: tuple[CameraMount, ...] = ()
    #: 夹爪完全闭合时两指之间的最小开口，用于判定"夹住了"（米）
    note: str = ""


# --------------------------------------------------------------------------------------
# UR5e + Robotiq 2F-140（原 TASK2 基线，robosuite 资产，装配方式与旧执行器一致）
# --------------------------------------------------------------------------------------

UR_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

UR5E = ArmProfile(
    name="ur5e",
    description="robosuite UR5e + Robotiq 2F-140（TASK2 原基线）",
    arm_xml=ROBOSUITE_ASSETS / "robots" / "ur5e" / "robot.xml",
    base_body="base",
    joints=UR_JOINTS,
    # 旧执行器用的 elbow-up home 位姿，实测 TCP 在基座前方 0.46 m、离台面 0.36 m
    home_qpos=np.array([0.74, -1.30, 1.50, -1.76, -1.57, -0.83], dtype=float),
    base_pos=(-0.30, -0.35, 0.0),
    actuator=ActuatorSpec(
        mode="position",
        kp=650.0,
        kv=55.0,
        forcerange=(-280.0, 280.0),
        ctrlrange={name: (-6.28319, 6.28319) for name in UR_JOINTS} | {"elbow_joint": (-3.14159, 3.14159)},
        name_template="ur_pos_{joint}",
    ),
    gripper=GripperSpec(
        kind="robotiq_2f140",
        actuator_names=("finger_1", "finger_2"),
        open_ctrl=(0.0, 0.0),
        close_ctrl=(0.7, -0.7),
        # 实测（tools/probe_arm_geometry.py）：指垫中心距 = -0.277955·ctrl + 0.126593，
        # 可用 ctrl ∈ [0, 0.4375]，之后指垫互相顶住、中心距反而回升。
        width_intercept=0.126593,
        width_slope=-0.277955,
        # 实测（把夹爪完全闭合去夹一块已知宽度的石头）：石头宽 65.5 mm 时指垫中心距
        # 停在 83.2 mm -> 每片指垫有效厚度 8.8 mm、合计 17.6 mm。
        # 之前抄了"ctrl=0.3 时指垫中心距 42.0 / 内表面间距 8.6"推出的 33.4 mm，
        # 那是**另一个开度**下的值（2F-140 是四连杆，指垫朝向随开度变），
        # 结果每条合拢指令都恰好落在"刚贴上、零夹紧力"的位置——石头纹丝不动。
        pad_thickness_m=0.0176,
        # 净开口 = 128.4 − 17.6 = 110.8 mm
        max_opening_m=0.1108,
        min_opening_m=0.0011,
        # 开合轴 = grip_site 的 +x（实测 +1.000）；接近轴 = grip_site 的 +z：
        # 实测 legacy home 位姿下 flange z=0.450 → 指垫 0.178 → site 0.146，
        # 也就是 site 在指垫**再往前 32 mm**，+z 指向下压方向。写反会把臂压到台面下面。
        open_axis_tcp=(1.0, 0.0, 0.0),
        approach_axis_tcp=(0.0, 0.0, 1.0),
        pad_offset_tcp=(0.0, 0.0, -0.0323),
        pad_geoms=(
            "left_fingerpad_collision",
            "right_fingerpad_collision",
            "left_fingertip_collision",
            "right_fingertip_collision",
        ),
        finger_geoms=(
            "left_outer_finger_collision",
            "left_inner_finger_collision",
            "left_fingertip_collision",
            "left_fingerpad_collision",
            "right_outer_finger_collision",
            "right_inner_finger_collision",
            "right_fingertip_collision",
            "right_fingerpad_collision",
        ),
        mirrored=True,
    ),
    tcp_site="grip_site",
    gripper_xml=ROBOSUITE_ASSETS / "grippers" / "robotiq_gripper_140.xml",
    gripper_attach_body="right_hand",
    gripper_root_body="right_gripper",
    disable_self_collision=True,
    # 0.60：料区把石头立起来放（见 Workcell.supply_orientation + 托架），石头高度
    # 变成 0.6×118 ≈ 71 mm，指垫中心放在 50 mm 时指尖离台面 14 mm；截面带宽
    # 56 mm 对净开口 95 mm，留出 39 mm 闭合行程——这是"能真正夹紧"的量级。
    # 原来的 1.0 会让带宽 85–100 mm 顶到净开口上限，指垫贴上去就没有夹紧力。
    stone_scale=0.60,
    work_scale=1.0,
    reach_m=0.85,
    min_reach_m=0.20,
    cameras=(
        CameraMount(name="wrist", pos=(-0.045, 0.0, 0.03), quat=(1.0, 0.0, 0.0, 0.0), fovy_deg=75.0, in_tcp=True),
    ),
    note="原 10 块墙实测 8/10 成功；夹爪开口最大 128 mm，石块不缩放",
)


# --------------------------------------------------------------------------------------
# Franka Panda（MuJoCo Menagerie 原样搬运：7 轴 + 自带平行夹爪）
# --------------------------------------------------------------------------------------


PANDA = ArmProfile(
    name="panda",
    description="MuJoCo Menagerie Franka Panda（7 轴 + 自带平行夹爪，开口 77 mm）",
    arm_xml=ASSETS / "franka" / "panda.xml",
    base_body="link0",
    joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
    home_qpos=np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853], dtype=float),
    base_pos=(-0.30, -0.35, 0.0),
    actuator=ActuatorSpec(mode="source"),
    gripper=GripperSpec(
        kind="panda",
        actuator_names=("actuator8",),
        open_ctrl=(255.0,),
        close_ctrl=(0.0,),
        # 实测指垫中心距 = 3.13725e-4·ctrl + 0.0054（R²=1.0）；内表面净开口 = 0.08·ctrl/255
        width_intercept=0.0054,
        width_slope=0.000313725,
        pad_thickness_m=0.0054,
        max_opening_m=0.0800,
        min_opening_m=0.0,
        open_axis_tcp=(0.0, -1.0, 0.0),
        approach_axis_tcp=(0.0, 0.0, 1.0),
        # TCP 取在指垫内表面中点（hand 系 +z 0.1034），指垫中心几乎与之重合
        pad_offset_tcp=(0.0, 0.0, -0.0005),
        pad_geoms=(),
        finger_geoms=(),
        finger_bodies=("left_finger", "right_finger"),
    ),
    tcp_site="tcp",
    tcp_parent_body="hand",
    # Menagerie 的 hand body 在 link7 前方 0.107；指垫内表面在 hand 系 +z 0.1034（实测）
    tcp_site_pos=(0.0, 0.0, 0.1034),
    tcp_site_quat=(1.0, 0.0, 0.0, 0.0),
    scene_defaults=False,
    stone_scale=0.60,
    work_scale=1.0,
    reach_m=0.85,
    min_reach_m=0.20,
    cameras=(
        CameraMount(name="wrist", pos=(0.0, -0.055, -0.04), quat=(0.9238795, 0.0, 0.0, -0.3826834), fovy_deg=80.0, in_tcp=True),
    ),
    note="7 轴冗余臂：IK 用关节限位裁剪的阻尼最小二乘，抓取宽度上限 0.9*77=69 mm",
)


# --------------------------------------------------------------------------------------
# Piper（MoonSim 自有资产：6 轴 + 平行夹爪，开口 70 mm）
# --------------------------------------------------------------------------------------


PIPER = ArmProfile(
    name="piper",
    description="Piper 6 轴 + 平行夹爪（TASK1 同款资产，开口 70 mm，负载最小）",
    arm_xml=ASSETS / "piper" / "arm.xml",
    base_body="arm_base",
    joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    home_qpos=np.array([0.0, 1.10, -1.40, 0.0, 0.30, 0.0], dtype=float),
    base_pos=(-0.24, -0.28, 0.0),
    actuator=ActuatorSpec(mode="source"),
    gripper=GripperSpec(
        kind="piper",
        actuator_names=("gripper",),
        # 注意方向：ctrl=0 是**闭合**（指垫相碰），ctrl=0.035 才是全开。反过来写
        # 会让"张开"变成"夹紧"，石头根本放不下去。
        open_ctrl=(0.035,),
        close_ctrl=(0.0,),
        # 实测指垫中心距 = 1.997291·ctrl + 0.016738（ctrl∈[0,0.035]，rms 1.5e-5）
        width_intercept=0.016738,
        width_slope=1.997291,
        pad_thickness_m=0.0174,
        # 全开时指垫内表面净开口 69.3 mm（README 的 6.996 cm 说的是指垫最小间距）
        max_opening_m=0.0693,
        min_opening_m=0.0,
        open_axis_tcp=(0.0, 1.0, 0.0),
        approach_axis_tcp=(1.0, 0.0, 0.0),
        # tool_tip 在指垫前方 33.4 mm（实测）
        pad_offset_tcp=(-0.033446, 0.0, 0.0),
        pad_geoms=(),
        finger_geoms=(),
        finger_bodies=("link7", "link8"),
    ),
    tcp_site="tool_tip",
    scene_defaults=True,
    stone_scale=0.55,
    work_scale=0.80,
    # 实测扫过 10/15/20/25/30/35°：15° 时 20 个槽位 + 10 个料区口袋全部解到 0.1 mm
    # 以内；10° 时顶层槽位卡在 joint5 上限（差 18 mm），25° 以上料区口袋开始变差。
    approach_tilt_deg=15.0,
    reach_m=0.61,
    min_reach_m=0.22,
    cameras=(
        CameraMount(name="wrist", pos=(0.0, 0.0, -0.05), quat=(1.0, 0.0, 0.0, 0.0), fovy_deg=85.0, in_tcp=True),
    ),
    note="开口最小：石块按 0.55 缩放，工作区按 0.72 靠近基座",
)


ARM_PROFILES: dict[str, ArmProfile] = {profile.name: profile for profile in (UR5E, PANDA, PIPER)}


def get_profile(name: str) -> ArmProfile:
    key = name.strip().lower()
    if key not in ARM_PROFILES:
        raise KeyError(f"未知机械臂 {name!r}；可用：{', '.join(sorted(ARM_PROFILES))}")
    return ARM_PROFILES[key]


def list_profiles() -> list[ArmProfile]:
    return [ARM_PROFILES[key] for key in sorted(ARM_PROFILES)]
