"""按 profile 装配 MuJoCo 场景：臂 + 夹爪 + 台面 + 石块 + 三路相机。

装配流程（`build_scene`）
------------------------
1. 解析臂（和外挂夹爪）的 MJCF，把 `mesh file` 换成绝对路径；
2. 复制 `default / asset / tendon / equality / contact / sensor` 段（不同来源的
   `<default>` 会合并进同一棵树，否则 Panda 的 class 默认值会失效）；
3. 世界：灯光、台面、世界系相机（top / front / overview）、按 profile 位姿摆放的机械臂；
4. 第一遍编译模型，量出 TCP 所在 body 的实际位姿，据此把腕部相机装到正确的位置——
   腕部相机的**语义**是"相对 TCP"（`CameraMount.in_tcp`），装到 XML 里必须换成
   相对父 body 的静态位姿，硬编码这些数字会在换臂时全部失效，所以这里实测；
5. 第二遍带上腕部相机重新编译，返回可直接仿真的 `BuiltScene`。

工作区（`Workcell`）是算出来的，不是抄来的：墙固定在世界的原点附近（这样所有臂
的墙在同一位置、世界相机取景一致、效果可比），机械臂基座按 profile 摆在墙前，
料区（supply pockets）在以基座为圆心的一段圆弧上，半径按臂展缩放。
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..rocks import FlatStone
from .profile import ArmProfile

#: 旧执行器里的场景装配与石块视觉工具，直接复用，避免两套实现漂移
from scripts.run_official_ur5e_robotiq_grasp_test import (  # noqa: E402
    _disable_body_collisions,
    _find_body,
    _fmt,
    _retune_gripper_for_stones,
    apply_clean_ur5e_visual,
)
from scripts.run_official_ur5e_robotiq_wall_stack import (  # noqa: E402
    quat_to_mat,
    stone_body,
    stone_mesh_asset,
    stone_texture_asset,
    stone_visual_material_asset,
    stone_visual_mesh_asset,
)

#: 墙中心相对基座的方向：沿用旧 TASK2 的 (-0.30,-0.35) → (0,0)，即"前右 45°"
WALL_DIRECTION = np.array([0.30, 0.35, 0.0], dtype=float)
WALL_DIRECTION /= np.linalg.norm(WALL_DIRECTION)
#: 旧场景里基座到墙中心的距离
WALL_DISTANCE = float(np.linalg.norm([0.30, 0.35]))
#: 料区圆弧：以"基座→墙"方位角为中心，两侧各展开这么多度
POCKET_ANGLE_SPREAD_DEG = 48.0
#: 台面半尺寸（米）的**下限**。实际台面按布局自动放大，见 `table_half_extents`。
TABLE_HALF = np.array([0.90, 0.75, 0.025], dtype=float)
#: 自动台面相对石头外沿再留的余量（米）
TABLE_MARGIN_M = 0.10


def table_half_extents(workcell: "Workcell", stones, margin: float = TABLE_MARGIN_M) -> np.ndarray:
    """按实际布局算台面半尺寸，保证**没有石头落在台面之外**。

    为什么不能只用一个常数：台面是有限的 `box` geom（不是无限 plane），
    而供料网格最后一行由 `pocket_radius + 0.16 + 3·spacing` 决定——UR5e 上实测
    `y=0.884`，超过原来的 `TABLE_HALF[1]=0.75`。那 4 块石头下面没有地面，
    会**直接掉进虚空**（实测 1.5 s 后 z=-2.73 m 且仍在加速），表现为"石头悬空不落地"。

    这里把每块石头的投影外沿（`|pos| + 半个最长边 + margin`）以及基座、墙心都算进去，
    再取与 `TABLE_HALF` 的逐轴最大值，所以只会变大、不会缩小既有场景。
    """
    half_x = float(TABLE_HALF[0])
    half_y = float(TABLE_HALF[1])
    sizes = {stone.name: stone for stone in stones}
    for name, (pos, _quat) in workcell.staging_poses.items():
        stone = sizes.get(name)
        reach = 0.5 * max(stone.length, stone.width) if stone is not None else 0.06
        half_x = max(half_x, abs(float(pos[0])) + reach + margin)
        half_y = max(half_y, abs(float(pos[1])) + reach + margin)
    for anchor in (np.asarray(workcell.base_pos, dtype=float), np.asarray(workcell.wall_center, dtype=float)):
        half_x = max(half_x, abs(float(anchor[0])) + margin)
        half_y = max(half_y, abs(float(anchor[1])) + margin)
    return np.array([half_x, half_y, float(TABLE_HALF[2])], dtype=float)


def look_at(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """返回 MuJoCo 相机的 `xyaxes` 字符串：相机沿自身 -z 看向 target。"""
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - pos
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    z_axis = -forward
    up = np.asarray(up, dtype=float)
    x_axis = np.cross(up, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:  # 正对上下时换一个参考轴
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    return _fmt(np.concatenate([x_axis, y_axis]))


@dataclass(frozen=True)
class SceneOptions:
    """视觉与分辨率选项。默认值与旧 TASK2 执行器一致，方便逐帧对照。"""

    stone_visual_style: str = "paper"
    stone_visual_roughness: float = 0.0025
    stone_visual_subdivisions: int = 2
    stone_grain_texture: bool = False
    stone_grain_strength: float = 0.38
    stone_grain_particles: int = 140
    robot_visual: str = "clean"
    camera_width: int = 640
    camera_height: int = 480
    timestep: float = 0.0015
    #: 世界相机
    top_camera_height_m: float = 1.15
    front_camera_offset: tuple[float, float, float] = (0.62, -0.62, 0.46)
    overview_offset: tuple[float, float, float] = (0.55, -1.02, 0.62)


@dataclass(frozen=True)
class Slot:
    course: int
    index: int
    x: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.course, self.index)


@dataclass
class Workcell:
    """算好的工作区：墙位、每一层的槽位、料区口袋、初始摆放位。"""

    profile: ArmProfile
    base_pos: np.ndarray
    wall_center: np.ndarray
    wall_axis: np.ndarray
    courses: tuple[int, ...]
    slots: tuple[Slot, ...]
    slot_spacing: float
    stone_scale: float
    mean_thickness: float
    mean_length: float
    pocket_radius: float
    staging_poses: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    #: 实际使用的层块数（可能因为够不到而被削减）
    requested_courses: tuple[int, ...] = ()
    adjustments: list[str] = field(default_factory=list)
    wall_distance_m: float = 0.0
    wall_azimuth_rad: float = 0.0
    #: 每块石头把最低顶点抬到台面所需的 z 偏移（编译后实测，见 `measure_ground_offsets`）
    ground_offsets: dict[str, float] = field(default_factory=dict)
    #: 抓取前石头的朝向。
    #: "flat" = 平放（默认）：改成侧向抓取后，指尖那 36 mm 的偏移不再是"往台面里插"，
    #:        石头多矮都能夹中段，所以平放（最稳）即可，也不需要托架。
    #: "edge" = 立起来：只在上方下爪时才需要（要求石头够高），配合托架使用。
    supply_orientation: str = "flat"

    def slot_position(self, course: int, index: int, z: float) -> np.ndarray:
        slot = next(s for s in self.slots if s.course == course and s.index == index)
        return np.array([slot.x, self.wall_center[1], z], dtype=float)

    def slot_yaw(self, course: int, index: int) -> float:
        return 0.0

    def pocket_pose(self, index: int, stone: FlatStone) -> tuple[np.ndarray, np.ndarray]:
        """料区口袋（抓取前把石头重置到这里）。

        口袋在以基座为圆心、朝"基座→墙"方位角两侧各 48° 的圆弧上均分——固定用世界
        ±x 方位会在机械臂侧后方放出口袋，那里往往正好是工作空间的空洞。
        """
        count = max(len(self.slots), 1)
        spread = np.deg2rad(POCKET_ANGLE_SPREAD_DEG)
        if count == 1:
            angles = [self.wall_azimuth_rad]
        else:
            step = 2.0 * spread / (count - 1)
            angles = [self.wall_azimuth_rad - spread + step * i for i in range(count)]
        angle = angles[index % count]
        radius = self.pocket_radius
        pos = self.base_pos + np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self.supply_orientation == "edge":
            # 立起来（绕石头长轴转 90°）：平放时石头只有 ~78 mm 高，而指垫中心必须
            # 高于台面 42 mm 指尖才不撞台面，能夹到的只有顶部斜面——抬起时会被楔出去
            # （实测 35 N 夹持力下臂抬 203 mm、石头只动 0.6 mm）。立起来后高度变成
            # 石头宽度（~118 mm），指垫可以夹在近垂直的中段侧面上。
            quat = np.array([0.70710678, 0.70710678, 0.0, 0.0])
        z = self.ground_offsets.get(stone.name)
        if z is None:
            z = -float(np.asarray(stone.vertices, dtype=float)[:, 2].min()) + 0.004
        pos = np.array([pos[0], pos[1], float(z)], dtype=float)
        return pos, quat

    def staging_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.staging_poses[name]

    def describe(self) -> dict:
        return {
            "arm": self.profile.name,
            "base_pos": [round(float(v), 4) for v in self.base_pos],
            "wall_center": [round(float(v), 4) for v in self.wall_center],
            "courses": list(self.courses),
            "slot_count": len(self.slots),
            "slot_spacing_m": round(self.slot_spacing, 4),
            "stone_scale": self.stone_scale,
            "mean_thickness_m": round(self.mean_thickness, 4),
            "mean_length_m": round(self.mean_length, 4),
            "pocket_radius_m": round(self.pocket_radius, 4),
            "wall_distance_m": round(self.wall_distance_m, 4),
            "requested_courses": list(self.requested_courses),
            "adjustments": list(self.adjustments),
        }


def _supply_quat(workcell: "Workcell") -> np.ndarray:
    if workcell.supply_orientation == "edge":
        return np.array([0.70710678, 0.70710678, 0.0, 0.0])
    return np.array([1.0, 0.0, 0.0, 0.0])


def measure_ground_offsets(model, stone_names, quat=None) -> dict[str, float]:
    """量出每块石头"底面贴台面"所需的 body 高度。

    为什么要实测而不是用生成器给的顶点：编译后几何用的是
    `body ∘ geom_pos ∘ geom_quat`（不含 mesh_quat），和生成器输出的局部坐标不是同一个
    坐标系。实测按生成器顶点摆石头会整块陷进台面约 5 cm；按这里的实测偏移摆，石块
    底面正好贴在台面上（drop test 里 geom 的世界 z 范围是 -0.003..+0.087，即平躺）。
    """
    offsets: dict[str, float] = {}
    for name in stone_names:
        geom_id = int(model.geom(f"{name}_geom").id)
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            offsets[name] = 0.004
            continue
        address = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[address : address + count], dtype=float)
        rotation = _quat_to_matrix(model.geom_quat[geom_id])
        body_vertices = vertices @ rotation.T + np.asarray(model.geom_pos[geom_id], dtype=float)
        if quat is not None:
            body_vertices = body_vertices @ _quat_to_matrix(quat).T
        offsets[name] = -float(body_vertices[:, 2].min()) + 0.004
    return offsets


def _quat_to_matrix(quat) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def scale_stones(stones: list[FlatStone], scale: float) -> list[FlatStone]:
    """等比缩放石块（顶点、名义尺寸、质量）。

    为什么不在生成时就缩放：`_align_to_target_obb` 的目标尺寸和
    `ROCK_WALL_TEMPLATES` 是两套风格共用的先验，缩放放在后处理里可以让
    "石头长什么样"和"这台臂能夹多大"彻底解耦。
    """
    if abs(scale - 1.0) < 1e-9:
        return stones
    scaled: list[FlatStone] = []
    for stone in stones:
        vertices = np.asarray(stone.vertices, dtype=float) * scale
        scaled.append(
            FlatStone(
                name=stone.name,
                vertices=[tuple(map(float, vertex)) for vertex in vertices],
                faces=list(stone.faces),
                rgba=stone.rgba,
                mass=float(stone.mass) * scale**3,
                length=float(stone.length) * scale,
                width=float(stone.width) * scale,
                thickness=float(stone.thickness) * scale,
            )
        )
    return scaled


def build_workcell(
    profile: ArmProfile,
    stones: list[FlatStone],
    courses: tuple[int, ...],
    wall_distance_scale: float = 1.0,
) -> Workcell:
    """算出工作区，并把"这台臂够不够得着"显式算清楚。

    墙中心 = 基座 + 墙方向 × 距离。距离不能随便定：
    - 太远 → 最外侧槽位的 IK 解不出来；
    - 太近 → 最内侧槽位落在基座附近，肘部折不过来（Piper 尤其明显）。
    墙的长轴是 X，它在"基座→墙"方向上的投影是 |0.651|，所以近端/远端分别是
    `d ∓ 0.651·span/2`。可行区间取 `[min_reach + 0.651·span/2, 0.88·reach − 0.651·span/2]`；
    区间为空就说明这块墙对这台臂太宽了——自动减掉最上层的一块，并把调整记录进
    `Workcell.adjustments`（不是静默改配置）。
    """
    base_pos = np.asarray(profile.base_pos, dtype=float)
    lengths = np.array([float(stone.length) for stone in stones])
    thicknesses = np.array([float(stone.thickness) for stone in stones])
    mean_length = float(lengths.mean())
    mean_thickness = float(thicknesses.mean())
    slot_spacing = mean_length * 1.06

    requested = tuple(int(count) for count in courses)
    effective = list(requested)
    adjustments: list[str] = []
    projection = abs(float(WALL_DIRECTION[0]))  # 墙长轴在基座→墙方向上的投影系数

    def feasible_interval(counts: list[int]) -> tuple[float, float] | None:
        span = (max(counts) - 1) * slot_spacing
        half = projection * 0.5 * span
        low = profile.min_reach_m + half
        high = 0.88 * profile.reach_m - half
        return (low, high) if low <= high else None

    interval = feasible_interval(effective)
    while interval is None and max(effective) > 1:
        # 从块数最多的那一层减一块，直到装得下
        widest = max(range(len(effective)), key=lambda index: effective[index])
        effective[widest] -= 1
        adjustments.append(
            f"层 {widest} 从 {requested[widest]} 块减到 {effective[widest]} 块：这台臂够不到那么宽的墙"
        )
        interval = feasible_interval(effective)
    if interval is None:
        raise ValueError(f"{profile.name}: 连一块石头的墙都放不下，检查 reach_m / min_reach_m 配置")

    desired = WALL_DISTANCE * profile.work_scale * wall_distance_scale
    distance = float(np.clip(desired, interval[0] + 1e-6, interval[1] - 1e-6))
    wall_center = base_pos + WALL_DIRECTION * distance
    wall_center[2] = 0.0

    slots: list[Slot] = []
    for course_index, count in enumerate(effective):
        offsets = (np.arange(count) - 0.5 * (count - 1)) * slot_spacing
        for slot_index, offset in enumerate(offsets):
            slots.append(Slot(course=course_index, index=slot_index, x=float(wall_center[0] + offset)))

    # 料区在"基座→墙"方位角两侧展开，半径不超过臂展的 72%
    wall_azimuth = float(np.arctan2(WALL_DIRECTION[1], WALL_DIRECTION[0]))
    pocket_radius = 0.72 * profile.reach_m
    workcell = Workcell(
        profile=profile,
        base_pos=base_pos,
        wall_center=wall_center,
        wall_axis=np.array([1.0, 0.0, 0.0]),
        courses=tuple(effective),
        slots=tuple(slots),
        slot_spacing=slot_spacing,
        stone_scale=profile.stone_scale,
        mean_thickness=mean_thickness,
        mean_length=mean_length,
        pocket_radius=pocket_radius,
    )
    workcell.requested_courses = requested
    workcell.adjustments = adjustments
    workcell.wall_distance_m = distance
    workcell.wall_azimuth_rad = wall_azimuth

    # 初始摆放：料区外侧一网格，互不重叠即可（执行器抓之前会把石头重置到口袋）
    staging: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    columns = 4
    spacing = max(0.11, float(np.max(lengths)) * 1.35)
    origin = base_pos + np.array([0.0, pocket_radius + 0.16, 0.0])
    for index, stone in enumerate(stones):
        row, column = divmod(index, columns)
        pos = origin + np.array([
            (column - 0.5 * (columns - 1)) * spacing,
            row * spacing,
            -float(np.asarray(stone.vertices, dtype=float)[:, 2].min()) + 0.004,
        ])
        staging[stone.name] = (pos, np.array([1.0, 0.0, 0.0, 0.0]))
    workcell.staging_poses = staging
    return workcell


# --------------------------------------------------------------------------------------
# MJCF 装配
# --------------------------------------------------------------------------------------


def _absolutize_assets(root: ET.Element, base_dir: Path) -> None:
    """把 `<asset>` 里的相对路径换成绝对路径，**遵守源文件的 meshdir/texturedir**。

    旧执行器的 `_absolutize_mesh_files` 直接按 XML 所在目录拼路径，对 robosuite 资产
    没问题（它不用 meshdir），但 Piper 的 `arm.xml` 写的是 `meshdir="assets"`，
    按 XML 目录拼会去找 `piper/link4.STL`（实际在 `piper/assets/link4.STL`）。
    """
    compiler = root.find("compiler")
    mesh_dir = base_dir
    texture_dir = base_dir
    if compiler is not None:
        if compiler.get("meshdir"):
            mesh_dir = base_dir / compiler.get("meshdir")
        if compiler.get("texturedir"):
            texture_dir = base_dir / compiler.get("texturedir")
    for element in root.iter():
        file_name = element.get("file")
        if not file_name or Path(file_name).is_absolute():
            continue
        if element.tag == "mesh":
            element.set("file", str((mesh_dir / file_name).resolve()))
        elif element.tag == "texture":
            element.set("file", str((texture_dir / file_name).resolve()))
        else:
            element.set("file", str((base_dir / file_name).resolve()))


def _merge_defaults(target_root: ET.Element, source_root: ET.Element) -> None:
    """把来源的 `<default>` 合并进目标。

    MuJoCo 只允许一个顶层 `<default>`。Panda 的 class 默认值必须保留，否则
    `class="panda"` 的执行器和几何会全部丢掉范围与增益。
    """
    source = source_root.find("default")
    if source is None:
        return
    target = target_root.find("default")
    if target is None:
        target_root.insert(0, copy.deepcopy(source))
        return
    for child in list(source):
        target.append(copy.deepcopy(child))


def _copy_section(target_root: ET.Element, source_root: ET.Element, tag: str) -> None:
    section = source_root.find(tag)
    if section is None:
        return
    existing = target_root.find(tag)
    if existing is None:
        # 位置：asset 必须在 worldbody 之前、actuator 在最后，交给调用方用 insert 顺序控制
        target_root.append(copy.deepcopy(section))
        return
    for child in list(section):
        existing.append(copy.deepcopy(child))


def _soften_pad_contacts(gripper_body: ET.Element) -> None:
    """把指垫接触改软一点。

    上游 `_retune_gripper_for_stones` 给指垫设的是 `solref 0.004 1` /
    `solimp 0.96 0.995 0.0005`——非常硬。位置伺服的指垫以这个刚度撞上石头时，
    接触冲量会把只有 0.5 kg 的石头弹出去（实测：合爪时夹住 130 N，一抬臂石头就被
    挤出去，指垫继续合拢到 56 mm 而石头还留在台面上）。把时间常数放到 20 ms、
    阻尼放宽，接触建立变"软着陆"。
    """
    for geom in gripper_body.iter("geom"):
        name = geom.get("name", "")
        if "fingerpad_collision" in name or "fingertip_collision" in name:
            geom.set("solref", "0.02 1")
            geom.set("solimp", "0.90 0.95 0.001")


def _add_synthetic_gripper(root, worldbody, robot_body, profile, spec: dict) -> None:
    """在夹爪根 body 上挂两根 slide 驱动的平板手指，并停用原指垫/指尖的碰撞。

    命令约定与 Piper 一致：`ctrl` 是单侧手指位移，净开口 = 2·ctrl。
    """
    parent = _find_body(robot_body, spec["parent_body"])
    axis = np.asarray(spec["axis_local"], dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    up = np.asarray(spec["up_local"], dtype=float)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    # 平板朝向：面法向 = 滑轨轴（两片面正对），高度方向取 up
    z_axis = axis
    x_axis = np.cross(up, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    basis = np.column_stack([x_axis, y_axis, z_axis])
    half = 0.5 * float(np.trace(basis))  # 仅用于占位，真正的四元数在下面
    trace = float(np.trace(basis))
    if trace > 0.0:
        s_ = np.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * s_, (basis[2, 1] - basis[1, 2]) / s_,
                (basis[0, 2] - basis[2, 0]) / s_, (basis[1, 0] - basis[0, 1]) / s_]
    else:
        idx = int(np.argmax(np.diag(basis)))
        if idx == 0:
            s_ = np.sqrt(1.0 + basis[0, 0] - basis[1, 1] - basis[2, 2]) * 2.0
            quat = [(basis[2, 1] - basis[1, 2]) / s_, 0.25 * s_,
                    (basis[0, 1] + basis[1, 0]) / s_, (basis[0, 2] + basis[2, 0]) / s_]
        elif idx == 1:
            s_ = np.sqrt(1.0 + basis[1, 1] - basis[0, 0] - basis[2, 2]) * 2.0
            quat = [(basis[0, 2] - basis[2, 0]) / s_, (basis[0, 1] + basis[1, 0]) / s_,
                    0.25 * s_, (basis[1, 2] + basis[2, 1]) / s_]
        else:
            s_ = np.sqrt(1.0 + basis[2, 2] - basis[0, 0] - basis[1, 1]) * 2.0
            quat = [(basis[1, 0] - basis[0, 1]) / s_, (basis[0, 2] + basis[2, 0]) / s_,
                    (basis[1, 2] + basis[2, 1]) / s_, 0.25 * s_]
    quat = np.asarray(quat, dtype=float)
    quat /= max(float(np.linalg.norm(quat)), 1e-9)

    # 原指垫/指尖停碰撞
    for name in list(profile.gripper.pad_geoms) + list(profile.gripper.finger_geoms):
        for candidate in robot_body.iter("geom"):
            if candidate.get("name") == name:
                candidate.set("contype", "0")
                candidate.set("conaffinity", "0")

    for index, key in enumerate(("left_local", "right_local")):
        centre = np.asarray(spec[key], dtype=float)
        sign = 1.0 if index == 0 else -1.0
        body = ET.SubElement(parent, "body", {"name": f"pg_jaw_{index}", "pos": _fmt(centre)})
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"pg_jaw_slide_{index}",
                "type": "slide",
                "axis": _fmt(axis * sign),
                "range": f"0 {spec['half_travel']:.4f}",
                "damping": "2",
                "armature": "0.001",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"pg_jaw_geom_{index}",
                "type": "box",
                "size": "0.030 0.020 0.004",
                "quat": _fmt(quat),
                "rgba": "0.05 0.08 0.11 1",
                "friction": "12.0 0.5 0.05",
                "condim": "6",
                "solref": "0.02 1",
                "solimp": "0.90 0.95 0.001",
            },
        )
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for index in range(2):
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"pg_jaw_act_{index}",
                "joint": f"pg_jaw_slide_{index}",
                "kp": "3000",
                "kv": "60",
                "ctrlrange": f"0 {spec['half_travel']:.4f}",
                "forcerange": "-400 400",
            },
        )


def _asset_name(element: ET.Element) -> str:
    """asset 的有效名字：显式 `name`，否则 MuJoCo 会取文件名（去扩展名）当名字。

    Panda 的 `<mesh file="link0_0.obj"/>` 就属于后者——按 `name` 判重会把它们
    全部当成同名丢掉，然后报 "mesh 'link0_1' not found"。
    """
    name = element.get("name")
    if name:
        return name
    file_name = element.get("file")
    if file_name:
        return Path(file_name).stem
    return ""


def _dedupe_assets(asset: ET.Element) -> list[str]:
    """去掉重名 asset（臂和夹爪可能都叫 `base` 之类），保留第一个。"""
    seen: set[tuple[str, str]] = set()
    dropped: list[str] = []
    for child in list(asset):
        key = (child.tag, _asset_name(child))
        if key in seen:
            asset.remove(child)
            dropped.append(f"{child.tag}:{key[1]}")
        else:
            seen.add(key)
    return dropped


def _merge_assets(target_root: ET.Element, source_root: ET.Element) -> None:
    asset = target_root.find("asset")
    if asset is None:
        asset = ET.SubElement(target_root, "asset")
    source = source_root.find("asset")
    if source is None:
        return
    for child in list(source):
        asset.append(copy.deepcopy(child))


def _ordering_key(tag: str) -> int:
    """MuJoCo 对顶层元素的顺序有要求：compiler/option/size/visual/default → asset →
    worldbody → tendon → equality → actuator → sensor → keyframe → contact。"""
    order = [
        "compiler", "option", "size", "visual", "statistic", "default",
        "asset", "worldbody", "tendon", "equality", "actuator", "sensor", "keyframe", "contact",
    ]
    return order.index(tag) if tag in order else len(order)


def _reorder(root: ET.Element) -> None:
    children = sorted(list(root), key=lambda element: _ordering_key(element.tag))
    for child in children:
        root.remove(child)
    for child in children:
        root.append(child)


def measure_parallel_jaws(model, data, pad_geoms, reference_width: float) -> list[tuple[str, tuple, tuple]]:
    """在两片指垫当前位置上，算出"面互相平行"的替换夹爪安装点。

    为什么要替换：robosuite 的 2F-140 指垫是四连杆带出来的，合拢时两片面**不平行**
    （实测左片面法向偏离闭合轴 15°、右面 8°）。于是合拢时两片法向力不共线，
    产生约 27 N 的净横向力——0.5 kg 的石头（5 N）直接被这个力挤出去，
    而横向恰好是指垫面内方向，摩擦帮不上忙（实测起动阈值只有 1 N）。
    换着力/摩擦参数、加软接触、分段合拢全都无效，因为问题在几何。

    返回 [(body_name, pos_local, quat_local), ...]：在两片指垫的位置上各放一块
    与闭合轴垂直的平板，指垫面因此严格平行。
    """
    import mujoco  # noqa: F401

    ids = [model.geom(name).id for name in pad_geoms[:2]]
    bodies = [int(model.geom_bodyid[gid]) for gid in ids]
    # 必须在**抓取时的大致开度**下测量：2F-140 是四连杆，手指越合越"外八"，
    # 在满开状态算出来的平板方向，到实际抓取开度时又偏了十几度（实测净横向力
    # 只从 27 N 降到 10 N，就是因为量错了开度）。
    finger_actuators = [
        actuator_id for actuator_id in range(model.nu) if model.actuator(actuator_id).name.startswith("finger")
    ]
    if finger_actuators:
        # 指垫中心距 ↔ ctrl 的线性标定：0.126593 − 0.277955·ctrl（见 profile）
        target_center = reference_width + 0.0176
        ctrl = float(np.clip((target_center - 0.126593) / -0.277955, 0.0, 0.7))
        for actuator_id in finger_actuators:
            sign = 1.0 if model.actuator(actuator_id).name.endswith("1") else -1.0
            data.ctrl[actuator_id] = sign * ctrl
        for _ in range(1500):
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    pos = [data.geom_xpos[gid].copy() for gid in ids]
    closing = pos[1] - pos[0]
    closing /= max(float(np.linalg.norm(closing)), 1e-9)
    mounts = []
    for index, (gid, body_id) in enumerate(zip(ids, bodies)):
        body_rot = data.xmat[body_id].reshape(3, 3)
        body_pos = data.xpos[body_id]
        # 左指垫朝 +closing（指向对面），右指垫朝 -closing
        normal_world = closing if index == 0 else -closing
        local_pos = body_rot.T @ (pos[index] - body_pos)
        local_normal = body_rot.T @ normal_world
        local_normal /= max(float(np.linalg.norm(local_normal)), 1e-9)
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z_axis, local_normal)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis /= norm
            angle = float(np.arccos(np.clip(np.dot(z_axis, local_normal), -1.0, 1.0)))
            half = 0.5 * angle
            quat = np.concatenate([[np.cos(half)], axis * np.sin(half)])
        mounts.append((model.body(body_id).name, tuple(float(v) for v in local_pos), tuple(float(v) for v in quat)))
    return mounts


def measure_synthetic_gripper(model, data, profile, reference_width: float):
    """算出一套**自造平行夹爪**的安装参数：两根滑轨手指 + 平板，面永远平行。

    为什么必须自造：robosuite 的 2F-140 是四连杆，指垫装在会转动的手指 body 上，
    实测两片面在所有开度下都不平行（89.7 mm 时 7.6°、38.9 mm 时 27°、20.7 mm 时 40°）。
    合拢时两片法向力因此不共线，530 N 的法向力里有一大截变成方向相反、互相抵消的
    切向分量——静态能扛 59 N 拉力，一动却带不动石头。
    换成 slide 关节驱动的平板后，面在**任何开度**都严格平行。
    """
    import mujoco  # noqa: F401

    pad_geoms = profile.gripper.pad_geoms[:2]
    ids = [model.geom(name).id for name in pad_geoms]
    bodies = [int(model.geom_bodyid[gid]) for gid in ids]
    # 先把夹爪合到参考开度再量（四连杆的指垫朝向随开度变）
    finger_actuators = [
        a for a in range(model.nu) if model.actuator(a).name.startswith("finger")
    ]
    if finger_actuators:
        target_center = reference_width + 0.0176
        ctrl = float(np.clip((target_center - 0.126593) / -0.277955, 0.0, 0.7))
        for a in finger_actuators:
            data.ctrl[a] = ctrl if model.actuator(a).name.endswith("1") else -ctrl
        for _ in range(1500):
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    pos = [data.geom_xpos[gid].copy() for gid in ids]
    axis_world = pos[1] - pos[0]
    axis_world /= max(float(np.linalg.norm(axis_world)), 1e-9)
    # 挂在夹爪根 body 上：这样它不随手指转动
    root_body_id = bodies[0]
    root_name = model.body(root_body_id).name
    root_pos = data.xpos[root_body_id].copy()
    root_rot = data.xmat[root_body_id].reshape(3, 3)
    # 平板的"高度方向"取世界竖直在 root 系里的方向
    up_local = root_rot.T @ np.array([0.0, 0.0, 1.0])
    axis_local = root_rot.T @ axis_world
    return {
        "parent_body": root_name,
        "axis_local": axis_local,
        "up_local": up_local,
        "left_local": root_rot.T @ (pos[0] - root_pos),
        "right_local": root_rot.T @ (pos[1] - root_pos),
        "half_travel": 0.5 * float(np.linalg.norm(pos[1] - pos[0])),
    }


def build_scene_xml(
    profile: ArmProfile,
    stones: list[FlatStone],
    workcell: Workcell,
    options: SceneOptions,
    wrist_mount: tuple[str, tuple[float, float, float], tuple[float, float, float, float]] | None = None,
    jaw_mounts: list[tuple[str, tuple, tuple]] | None = None,
    synthetic_gripper: dict | None = None,
) -> str:
    """拼出完整场景 XML。`wrist_mount` 为第二遍传入的腕部相机安装点。"""
    for path in (profile.arm_xml, profile.gripper_xml):
        if path is not None and not Path(path).exists():
            raise SystemExit(f"缺少机械臂资产：{path}")

    arm_root = ET.parse(profile.arm_xml).getroot()
    _absolutize_assets(arm_root, profile.arm_xml.parent)
    if options.robot_visual == "clean" and profile.name == "ur5e":
        apply_clean_ur5e_visual(arm_root, profile.arm_xml.parent)

    gripper_root = None
    if profile.gripper_xml is not None:
        gripper_root = ET.parse(profile.gripper_xml).getroot()
        _absolutize_assets(gripper_root, profile.gripper_xml.parent)

    root = ET.Element("mujoco", {"model": f"task2_{profile.name}_stone_stack"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{options.timestep:g}",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "gravity": "0 0 -9.81",
            "iterations": "170",
        },
    )
    ET.SubElement(root, "size", {"nconmax": "3200", "njmax": "6400"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": str(options.camera_width), "offheight": str(options.camera_height)})
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": "0.30 0.30 0.30", "diffuse": "0.70 0.70 0.68", "specular": "0.12 0.12 0.12"},
    )
    if profile.scene_defaults:
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "joint", {"damping": "1.2", "armature": "0.01"})
        ET.SubElement(default, "geom", {"solref": "0.006 1", "solimp": "0.92 0.99 0.001"})

    _merge_defaults(root, arm_root)
    if gripper_root is not None:
        _merge_defaults(root, gripper_root)
    _merge_assets(root, arm_root)
    if gripper_root is not None:
        _merge_assets(root, gripper_root)
    dropped = _dedupe_assets(root.find("asset"))

    # 台面贴图 / 材质（与旧执行器一致）。texrepeat 跟着台面尺寸走，保持格子物理边长不变。
    table_half = table_half_extents(workcell, stones)
    tex_x = max(1, int(round(2.0 * table_half[0] / 0.36)))
    tex_y = max(1, int(round(2.0 * table_half[1] / 0.375)))
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "table_grid",
            "type": "2d",
            "builtin": "checker",
            "width": "256",
            "height": "256",
            "rgb1": "0.56 0.56 0.52",
            "rgb2": "0.42 0.42 0.39",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "table_mat", "texture": "table_grid", "texrepeat": f"{tex_x} {tex_y}", "reflectance": "0.025"},
    )

    for stone in stones:
        asset.append(stone_mesh_asset(stone))
        if options.stone_visual_roughness > 0.0:
            use_grain = options.stone_grain_texture and options.stone_grain_strength > 0.0
            if use_grain:
                asset.append(stone_texture_asset(stone, options.stone_visual_style, options.stone_grain_strength))
            asset.append(stone_visual_material_asset(stone, options.stone_visual_style, use_grain))
            asset.append(
                stone_visual_mesh_asset(
                    stone, options.stone_visual_roughness, options.stone_visual_subdivisions, options.stone_visual_style
                )
            )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "-0.6 -0.8 1.6", "dir": "0 0 -1", "diffuse": "0.95 0.95 0.90"})
    ET.SubElement(worldbody, "light", {"name": "fill", "pos": "0.6 0.7 1.1", "dir": "0 0 -1", "diffuse": "0.45 0.45 0.42"})

    wall = workcell.wall_center
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": _fmt(np.asarray(options.overview_offset) + np.array([0.0, 0.0, 0.0])),
            "xyaxes": look_at(options.overview_offset, wall + np.array([0.0, 0.0, 0.06])),
            "fovy": "45",
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "top",
            "pos": _fmt(wall + np.array([0.0, 0.0, options.top_camera_height_m])),
            "xyaxes": "1 0 0 0 1 0",
            "fovy": "58",
        },
    )
    front_pos = wall + np.asarray(options.front_camera_offset, dtype=float)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "front",
            "pos": _fmt(front_pos),
            "xyaxes": look_at(front_pos, wall + np.array([0.0, 0.0, 0.05])),
            "fovy": "45",
        },
    )
    # 料区托架：两条沿 X 的矮脊，把立起来的石头夹在中间。
    # 为什么需要它：石头立起来（绕长轴 90°）后，X 方向靠 176 mm 的底边很稳，
    # 但 Y 方向只有 78 mm 底边配 118 mm 高，落地就会倒回平放——一平放，指垫中心
    # 又只能落在顶部斜面（抬起时被楔出去）。矮脊只有 15 mm 高，指垫在 42 mm 以上
    # 作业，不会和它打架。
    pocket_index = max(len(workcell.slots) // 2, 0)
    reference = stones[0] if stones else None
    if reference is not None and workcell.supply_orientation == "edge":
        pocket_pos, _ = workcell.pocket_pose(pocket_index, reference)
        half_length = 0.5 * max(float(stone.length) for stone in stones) * 1.1
        half_thickness = 0.5 * max(float(stone.thickness) for stone in stones)
        for name, offset in (("cradle_neg", -1.0), ("cradle_pos", 1.0)):
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": name,
                    "type": "box",
                    "pos": _fmt([pocket_pos[0], pocket_pos[1] + offset * (half_thickness + 0.010), 0.0075]),
                    "size": _fmt([half_length, 0.008, 0.0075]),
                    "rgba": "0.32 0.30 0.28 1",
                    "friction": "1.0 0.02 0.001",
                    "condim": "4",
                },
            )

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "table",
            "type": "box",
            "pos": _fmt([0.0, 0.0, -table_half[2]]),
            "size": _fmt(table_half),
            "material": "table_mat",
            "friction": "1.0 0.02 0.001",
            "condim": "4",
        },
    )

    robot_body = copy.deepcopy(_find_body(arm_root.find("worldbody"), profile.base_body))
    robot_body.set("pos", _fmt(np.asarray(profile.base_pos, dtype=float)))
    if profile.disable_self_collision:
        _disable_body_collisions(robot_body)

    if profile.tcp_parent_body is not None:
        parent = _find_body(robot_body, profile.tcp_parent_body)
        ET.SubElement(
            parent,
            "site",
            {
                "name": profile.tcp_site,
                "pos": _fmt(profile.tcp_site_pos),
                "quat": _fmt(profile.tcp_site_quat),
                "size": "0.008",
                "rgba": "1 0.1 0.1 1",
            },
        )
    if profile.gripper_xml is not None:
        if profile.gripper_attach_body is None or profile.gripper_root_body is None:
            raise ValueError(f"{profile.name}: 外挂夹爪必须给 gripper_attach_body 与 gripper_root_body")
        gripper_body = copy.deepcopy(_find_body(gripper_root.find("worldbody"), profile.gripper_root_body))
        if profile.name == "ur5e":
            _retune_gripper_for_stones(gripper_body)
            _soften_pad_contacts(gripper_body)
        _find_body(robot_body, profile.gripper.attach_body if False else profile.gripper_attach_body).append(gripper_body)

    if synthetic_gripper is not None:
        _add_synthetic_gripper(root, worldbody, robot_body, profile, synthetic_gripper)
    elif jaw_mounts:
        # 用平行的平板替换原指垫的碰撞：指垫仍然显示，但不再参与接触
        for name in profile.gripper.pad_geoms:
            geom = None
            for candidate in robot_body.iter("geom"):
                if candidate.get("name") == name:
                    geom = candidate
                    break
            if geom is not None:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        for index, (body_name, pos_local, quat_local) in enumerate(jaw_mounts):
            body = _find_body(robot_body, body_name)
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"parallel_jaw_{index}",
                    "type": "box",
                    "pos": _fmt(pos_local),
                    "quat": _fmt(quat_local),
                    "size": "0.030 0.020 0.004",
                    "rgba": "0.05 0.08 0.11 1",
                    # 摩擦给足：实测有效摩擦只有配置值的 ~1/30（0.07 对 3.6），
                    # 夹持力 59 N 时竖向摩擦容量只有约 4 N，刚好小于石头自重 5 N，
                    # 于是石头"慢慢滑下去"。这里把系数拉高，把余量做出来。
                    "friction": "12.0 0.5 0.05",
                    "condim": "6",
                    "solref": "0.006 1",
                    "solimp": "0.92 0.99 0.001",
                },
            )

    if wrist_mount is not None:
        parent_name, cam_pos, cam_quat = wrist_mount
        parent = _find_body(robot_body, parent_name)
        ET.SubElement(
            parent,
            "camera",
            {
                "name": "wrist",
                "pos": _fmt(cam_pos),
                "quat": _fmt(cam_quat),
                "fovy": f"{profile.cameras[0].fovy_deg:g}" if profile.cameras else "75",
            },
        )
    worldbody.append(robot_body)

    for stone in stones:
        pos, quat = workcell.staging_pose(stone.name)
        body = stone_body(
            stone,
            pos,
            quat,
            options.stone_visual_roughness,
            options.stone_visual_style,
            options.stone_grain_strength,
            options.stone_grain_particles,
        )
        # 石头这一侧的接触也要放软：上游给的是 solref 0.005 / solimp 0.92 0.99 0.001，
        # 配上 1.5 ms 步长，切向（摩擦）约束解不充分——实测有效摩擦只有配置值的
        # 1/30（0.07 对 3.6），夹持 83 N 时竖向摩擦容量约 5.8 N，恰好等于石头自重，
        # 于是"夹着但抬不动"。两侧都放软之后切向约束才能被解出来。
        for geom in body.iter("geom"):
            if geom.get("name", "").endswith("_geom"):
                geom.set("solref", "0.02 1")
                geom.set("solimp", "0.90 0.95 0.001")
        worldbody.append(body)

    # 执行器
    actuator = ET.SubElement(root, "actuator")
    if profile.actuator.mode == "source":
        for source_root in (arm_root, gripper_root):
            section = None if source_root is None else source_root.find("actuator")
            if section is None:
                continue
            for child in list(section):
                actuator.append(copy.deepcopy(child))
    else:
        for joint_name in profile.joints:
            low, high = profile.actuator.ctrlrange.get(joint_name, (-3.14159, 3.14159))
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": profile.actuator.name_template.format(joint=joint_name),
                    "joint": joint_name,
                    "kp": f"{profile.actuator.kp:g}",
                    "kv": f"{profile.actuator.kv:g}",
                    "ctrlrange": _fmt([low, high]),
                    "forcerange": _fmt(profile.actuator.forcerange),
                },
            )
        if gripper_root is not None:
            section = gripper_root.find("actuator")
            if section is not None:
                for child in list(section):
                    copied = copy.deepcopy(child)
                    copied.set("kp", f"{profile.gripper.actuator_kp:g}")
                    copied.set("kv", f"{profile.gripper.actuator_kv:g}")
                    copied.set("forcerange", "-220 220")
                    actuator.append(copied)

    _copy_section(root, arm_root, "tendon")
    if gripper_root is not None:
        _copy_section(root, gripper_root, "tendon")
    _copy_section(root, arm_root, "equality")
    if gripper_root is not None:
        _copy_section(root, gripper_root, "equality")
    _copy_section(root, arm_root, "sensor")
    if gripper_root is not None:
        _copy_section(root, gripper_root, "sensor")
    _copy_section(root, arm_root, "contact")
    if gripper_root is not None:
        _copy_section(root, gripper_root, "contact")

    if dropped:
        root.set("model", f"{root.get('model')}_dedup{len(dropped)}")
    _reorder(root)
    return ET.tostring(root, encoding="unicode")


def tcp_mount_in_parent(model, data, tcp_site: str, camera_pos, camera_quat):
    """把"相对 TCP"的相机位姿换算成"相对 TCP 父 body"的静态位姿。

    `camera_pos/quat` 是相机在 TCP site 系里的位姿；返回值可直接写进 XML。
    """
    import mujoco

    site_id = model.site(tcp_site).id
    body_id = int(model.site_bodyid[site_id])
    mujoco.mj_forward(model, data)
    site_pos = data.site_xpos[site_id].copy()
    site_rot = data.site_xmat[site_id].reshape(3, 3).copy()
    body_pos = data.xpos[body_id].copy()
    body_rot = data.xmat[body_id].reshape(3, 3).copy()

    cam_pos = np.asarray(camera_pos, dtype=float)
    cam_quat = np.asarray(camera_quat, dtype=float)
    world_pos = site_pos + site_rot @ cam_pos
    world_rot = site_rot @ quat_to_mat(cam_quat)

    local_pos = body_rot.T @ (world_pos - body_pos)
    local_rot = body_rot.T @ world_rot
    # 旋转矩阵 → 四元数（w x y z），与 MuJoCo 约定一致
    trace = float(np.trace(local_rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (local_rot[2, 1] - local_rot[1, 2]) / s,
                         (local_rot[0, 2] - local_rot[2, 0]) / s, (local_rot[1, 0] - local_rot[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(local_rot)))
        if index == 0:
            s = np.sqrt(1.0 + local_rot[0, 0] - local_rot[1, 1] - local_rot[2, 2]) * 2.0
            quat = np.array([(local_rot[2, 1] - local_rot[1, 2]) / s, 0.25 * s,
                             (local_rot[0, 1] + local_rot[1, 0]) / s, (local_rot[0, 2] + local_rot[2, 0]) / s])
        elif index == 1:
            s = np.sqrt(1.0 + local_rot[1, 1] - local_rot[0, 0] - local_rot[2, 2]) * 2.0
            quat = np.array([(local_rot[0, 2] - local_rot[2, 0]) / s, (local_rot[0, 1] + local_rot[1, 0]) / s,
                             0.25 * s, (local_rot[1, 2] + local_rot[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + local_rot[2, 2] - local_rot[0, 0] - local_rot[1, 1]) * 2.0
            quat = np.array([(local_rot[1, 0] - local_rot[0, 1]) / s, (local_rot[0, 2] + local_rot[2, 0]) / s,
                             (local_rot[1, 2] + local_rot[2, 1]) / s, 0.25 * s])
    quat = quat / max(float(np.linalg.norm(quat)), 1e-9)
    return model.body(body_id).name, tuple(float(v) for v in local_pos), tuple(float(v) for v in quat)


@dataclass
class BuiltScene:
    model: object
    data: object
    workcell: Workcell
    options: SceneOptions
    xml: str
    wrist_mount: tuple[str, tuple[float, float, float], tuple[float, float, float, float]] | None
    wrist_in_tcp: tuple[np.ndarray, np.ndarray] | None
    #: 实际参与接触的指垫 geom（装了平行夹爪时是 parallel_jaw_*）
    pad_geoms: tuple[str, ...] = ()
    #: 指垫有效厚度（米）：净开口 = 指垫中心距 − 这个值
    pad_thickness_m: float = 0.0
    #: 用了自造平行夹爪时，夹爪命令的换算方式（开口 = 2·ctrl）
    synthetic_gripper: dict | None = None


def build_scene(profile: ArmProfile, stones: list[FlatStone], workcell: Workcell, options: SceneOptions) -> BuiltScene:
    """两遍装配：先编译一次量出腕部相机安装点，再带上相机编译最终模型。"""
    import mujoco

    os.environ.setdefault("MUJOCO_GL", "egl")
    plain_xml = build_scene_xml(profile, stones, workcell, options, wrist_mount=None)
    probe_model = mujoco.MjModel.from_xml_string(plain_xml)
    probe_data = mujoco.MjData(probe_model)

    jaw_mounts = []
    synthetic = None
    if profile.name == "ur5e" and profile.gripper.pad_geoms:
        expected_block = 0.60 * profile.gripper.max_opening_m
        synthetic = measure_synthetic_gripper(probe_model, probe_data, profile, expected_block)

    wrist_mount = None
    wrist_in_tcp = None
    if profile.cameras:
        mount = next((camera for camera in profile.cameras if camera.name == "wrist" and camera.in_tcp), None)
        if mount is not None:
            wrist_mount = tcp_mount_in_parent(probe_model, probe_data, profile.tcp_site, mount.pos, mount.quat)
            wrist_in_tcp = (np.asarray(mount.pos, dtype=float), np.asarray(mount.quat, dtype=float))

    xml = build_scene_xml(
        profile,
        stones,
        workcell,
        options,
        wrist_mount=wrist_mount,
        jaw_mounts=jaw_mounts or None,
        synthetic_gripper=synthetic,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 编译后按真实几何把石头摆到台面上（见 measure_ground_offsets 的说明）
    workcell.ground_offsets = measure_ground_offsets(
        model, [stone.name for stone in stones], quat=_supply_quat(workcell)
    )
    for stone in stones:
        joint_id = model.joint(f"{stone.name}_free").id
        address = int(model.jnt_qposadr[joint_id])
        pos, quat = workcell.staging_pose(stone.name)
        data.qpos[address : address + 3] = pos
        data.qpos[address + 3 : address + 7] = quat
    mujoco.mj_forward(model, data)
    if synthetic is not None:
        pad_geoms = ("pg_jaw_geom_0", "pg_jaw_geom_1")
        pad_thickness = 0.008
    elif jaw_mounts:
        pad_geoms = tuple(f"parallel_jaw_{index}" for index in range(len(jaw_mounts)))
        pad_thickness = 0.008  # 平板厚 8 mm：两片内表面相距 = 中心距 − 8 mm
    else:
        pad_geoms = tuple(profile.gripper.pad_geoms)
        pad_thickness = profile.gripper.pad_thickness_m
    return BuiltScene(
        model=model,
        data=data,
        workcell=workcell,
        options=options,
        xml=xml,
        wrist_mount=wrist_mount,
        wrist_in_tcp=wrist_in_tcp,
        pad_geoms=pad_geoms,
        pad_thickness_m=pad_thickness,
        synthetic_gripper=synthetic,
    )
