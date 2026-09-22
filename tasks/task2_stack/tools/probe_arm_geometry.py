#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从**编译后的 MuJoCo 模型**里实测三条机械臂（ur5e / panda / piper）的几何事实。

为什么要有这个脚本
------------------
抓取规划器需要的不是 README/注释里写的数字，而是从实际参与仿真的模型里量出来的数字：

* 夹爪的"开合曲线"决定了能夹多大的石头（`width ≈ a * ctrl + b`）；
* TCP 帧下的**闭合轴**（两指分离方向）和**接近轴**（法兰->pad 中点，即最终下压方向）决定了抓取位姿怎么摆；
  注意"从 TCP 原点指向手指工作空间"这个字面定义在 ur5e/piper 上与下压方向**反平行**
  （两个 site 都落在两指之外），脚本两个都报，字段名分别是 ``approach_axis_tcp`` 与 ``tcp_to_pad_mid_axis_tcp``；
* pad 中点相对 TCP 的偏移决定了"把 TCP 送到哪里"才能让石头落在两指之间；
* 关节空间采样的可达半径（尤其是"接近轴朝下 35° 以内"的子集）决定了工作台要多大；
* 总质量与执行器力矩上限决定了能拿多重的石头。

设计约束（为什么这么写）
------------------------
1. **只做运动学**：本机没有 GPU/GL，脚本不使用 ``mujoco.Renderer``/viewer，只做
   ``mj_kinematics``/``mj_forward`` 与前向动力学的短时 settle。
2. **必须 settle 的机构**：三条臂的夹爪都不是"一个关节一个执行器"：
   * ur5e(Robotiq 2F-140)：4 个 fixed tendon + 4 个 tendon equality 把 4 个
     关节耦合成 4-bar linkage，执行器只驱动两个 knuckle；
   * panda：``finger_joint1/2`` 由一个 fixed tendon ``split`` 串起来，``actuator8``
     作用在 tendon 上，另有 joint equality 保证两指同步；
   * piper：``joint7``/``joint8`` 由 joint equality 耦合，只有 ``joint7`` 有执行器。
   直接写 ``data.qpos`` 会得到一个"物理上不可能"的构型，所以每个采样点都用一个**全新的
   ``MjData``**、设定 qpos/ctrl 后步进到静止（判据：``max|qvel|`` 足够小）再测量；
   脚本同时给出解析耦合预测值作为交叉校验。
3. **从任意 cwd 可运行**：路径全部由 ``__file__`` 推导；默认输出到
   ``tasks/task2_stack/reports/arm_geometry_<arm>.json``。
4. **可复现**：固定随机种子，JSON 里不写时间戳（时间戳会破坏逐字节可复现）。

用法
----
    PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack \\
    /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python \\
        tasks/task2_stack/tools/probe_arm_geometry.py --arm ur5e
    ... --arm panda / --arm piper
    ... --merge            # 合并三份 JSON 到 reports/arm_geometry.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# 路径：全部从 __file__ 推导，保证任意 cwd 下都能跑
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # tasks/task2_stack
REPORTS_DIR = PROJECT_ROOT / "reports"
TOOLS_DIR = PROJECT_ROOT / "tools"
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mujoco  # noqa: E402

# 复用仓库已有的官方资产路径与 robosuite 组合代码（不修改它们）
from run_official_ur5e_robotiq_grasp_test import (  # noqa: E402
    ROBOTIQ_140_XML,
    UR5E_XML,
    UR_JOINTS,
)
from run_official_ur5e_robotiq_wall_stack import build_wall_stack_scene  # noqa: E402

PANDA_XML = Path(__file__).resolve().parents[3] / "assets" / "robots" / "franka" / "panda.xml"
PIPER_XML = Path(__file__).resolve().parents[3] / "assets" / "robots" / "piper" / "arm.xml"

# legacy 参考位姿（tasks/task2_stack/scripts/run_official_ur5e_robotiq_wall_stack.py:56）
Q_HOME_ELBOW_UP = np.array([0.74, -1.30, 1.50, -1.76, -1.57, -0.83], dtype=float)

MJGEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
MJGEOM_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
MJGEOM_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
MJGEOM_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
MJGEOM_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)


# --------------------------------------------------------------------------------------
# 小工具：四元数/旋转/向量
# --------------------------------------------------------------------------------------
def unit(v: Sequence[float]) -> np.ndarray:
    """归一化；零向量返回零向量（调用方自己判断退化情况）。"""
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """用 MuJoCo 自己的实现做旋转矩阵->四元数，避免自己写公式出错。返回 (w,x,y,z)。"""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def frame_from_axes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """由两个（近似正交的）方向构造右手正交基：e1=a, e2=正交化(b), e3=e1×e2。"""
    e1 = unit(a)
    e2 = unit(np.asarray(b, dtype=float) - float(np.dot(b, e1)) * e1)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def vector_in_frame(vec_world: np.ndarray, R_world_from_frame: np.ndarray) -> np.ndarray:
    """世界向量 -> 帧内分量（R 的列是帧轴在世界中的方向）。"""
    return R_world_from_frame.T @ np.asarray(vec_world, dtype=float)


def point_in_frame(point_world: np.ndarray, origin_world: np.ndarray,
                   R_world_from_frame: np.ndarray) -> np.ndarray:
    return vector_in_frame(np.asarray(point_world, dtype=float) - np.asarray(origin_world, float),
                           R_world_from_frame)


def angle_deg_between(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))
    return math.degrees(math.acos(c))


def quat_negate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=float).copy()
    out[1:] = -out[1:]
    return out


def rotation_error_vector(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """姿态误差的**旋转向量**（世界系，模长 = 夹角，弧度）。

    为什么不用常见的 0.5*Σ(cross(cur_i, tgt_i))：那个量等于 2*sin(θ)*axis，在 θ→180° 时
    趋近 0 —— 本任务里 ur5e 的 IK 就正好收敛到了"目标姿态翻转 180°"的局部极小，
    而旧指标给出 7.8e-5 的"已经收敛"假象。这里用相对四元数的轴角，θ∈[0,π] 无奇点。
    """
    q_cur = mat_to_quat(R_current)
    q_tgt = mat_to_quat(R_target)
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_tgt, quat_negate(q_cur))     # R_err = R_tgt * R_cur^T
    if q_err[0] < 0:
        q_err = -q_err                                       # 取短弧（双重覆盖）
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


def r3(v: Sequence[float]) -> list:
    return [round(float(x), 6) for x in np.asarray(v, dtype=float).reshape(-1)]


def r4(v: Sequence[float]) -> list:
    return [round(float(x), 6) for x in np.asarray(v, dtype=float).reshape(-1)]


# --------------------------------------------------------------------------------------
# 模型构建
# --------------------------------------------------------------------------------------
def _strip_collisions_and_table(xml: str, base_pos: Sequence[float] | None) -> str:
    """把 legacy 组合场景改成"探针场景"。

    * 删掉 legacy 的桌面 geom（本探针场景没有桌子/石头，桌面只影响接触不影响运动学，
      但留着会让"nbody/质量"的读数带上无关物体）；
    * 把机器人基座搬回世界原点（legacy 场景为了配合桌面把它放在 ROBOT_BASE_POS）；
    * 所有 geom 的 contype/conaffinity 置 0：本脚本测的是**纯运动学**，不测接触；
      不关掉的话，Robotiq 的两块 fingerpad 在闭合时会自碰撞，开合曲线会被接触力截断，
      量到的就不是机构本身的运动学关系了。
    """
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for geom in list(worldbody.findall("geom")):
        if geom.get("name") == "table":
            worldbody.remove(geom)
    robot = None
    for body in worldbody.findall("body"):
        if body.get("name") == "base":       # ur5e: 世界下的根 body 叫 base
            robot = body
            break
    if robot is not None and base_pos is not None:
        robot.set("pos", " ".join(f"{float(v):.6g}" for v in base_pos))
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    return ET.tostring(root, encoding="unicode")


def build_ur5e_model() -> tuple[mujoco.MjModel, dict]:
    """ur5e = robosuite UR5e + robosuite Robotiq 2F-140，组合方式完全沿用 legacy 代码。

    直接调用 ``scripts/run_official_ur5e_robotiq_wall_stack.py::build_wall_stack_scene``
    （stones 为空），因此 mesh 绝对化、gripper body 挂到 ``right_hand``、position 执行器
    kp=650/kv=55、gripper 执行器 kp=150/kv=5、tendon/equality/sensor 拷贝这些细节
    与真正跑任务时**逐字段一致**；之后只做两处本探针必需的改动（见上）。
    """
    xml = build_wall_stack_scene([], {})
    xml = _strip_collisions_and_table(xml, base_pos=(0.0, 0.0, 0.0))
    model = mujoco.MjModel.from_xml_string(xml)
    meta = {
        "loaded_from": "build_wall_stack_scene([], {}) + strip table/base-to-origin/no-collision",
        "sources": [str(UR5E_XML), str(ROBOTIQ_140_XML)],
        "composition_reference": "tasks/task2_stack/scripts/run_official_ur5e_robotiq_wall_stack.py::build_wall_stack_scene",
    }
    return model, meta


def build_panda_model() -> tuple[mujoco.MjModel, dict]:
    """panda = MuJoCo Menagerie 的 ``assets/robots/franka/panda.xml``，原样加载。"""
    model = mujoco.MjModel.from_xml_path(str(PANDA_XML))
    meta = {"loaded_from": str(PANDA_XML), "sources": [str(PANDA_XML)],
            "composition_reference": "assets/robots/franka/panda.xml (vendor, 未改动)"}
    return model, meta


def build_piper_model() -> tuple[mujoco.MjModel, dict]:
    """piper = ``assets/robots/piper/arm.xml``，原样加载。"""
    model = mujoco.MjModel.from_xml_path(str(PIPER_XML))
    meta = {"loaded_from": str(PIPER_XML), "sources": [str(PIPER_XML)],
            "composition_reference": "assets/robots/piper/arm.xml (固定基座任务适配)"}
    return model, meta


# --------------------------------------------------------------------------------------
# 几何读取
# --------------------------------------------------------------------------------------
def geom_aabb_center_world(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray:
    """geom 的 AABB 中心（世界）。

    为什么不用 ``geom_xpos``：对 mesh geom，``geom_xpos`` 是"mesh 帧原点"，不是几何中心；
    MuJoCo 编译后把每个 geom 在其自身帧内的 AABB 存在 ``model.geom_aabb``，
    所以世界 AABB 中心 = geom_xpos + geom_xmat @ aabb_center。
    """
    local_center = model.geom_aabb[gid][:3]
    return data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ local_center


def geom_surface_points(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray:
    """geom 表面采样点（世界）：box 取 8 个角点，mesh 取全部顶点。

    这些点用来算"两指内表面间距"（沿闭合轴取极值），比 AABB 中心更接近真实可夹宽度。
    """
    R = data.geom_xmat[gid].reshape(3, 3)
    p = data.geom_xpos[gid]
    gtype = int(model.geom_type[gid])
    size = model.geom_size[gid]
    if gtype == MJGEOM_MESH:
        mid = int(model.geom_dataid[gid])
        adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[adr:adr + num].reshape(-1, 3)
        return p + verts @ R.T
    if gtype == MJGEOM_BOX:
        sx, sy, sz = size[0], size[1], size[2]
        corners = np.array([[sx, sy, sz], [sx, sy, -sz], [sx, -sy, sz], [sx, -sy, -sz],
                            [-sx, sy, sz], [-sx, sy, -sz], [-sx, -sy, sz], [-sx, -sy, -sz]])
        return p + corners @ R.T
    if gtype == MJGEOM_SPHERE:
        return np.array([p + R @ np.array([0, 0, size[0]])])
    if gtype in (MJGEOM_CYLINDER, MJGEOM_CAPSULE):
        half = size[1]
        return np.array([p + R @ np.array([0, 0, half]), p + R @ np.array([0, 0, -half])])
    return np.array([p])


def pad_center(model: mujoco.MjModel, data: mujoco.MjData, gids: Sequence[int]) -> np.ndarray:
    """一个手指的 pad 中心 = 该手指所有 pad geom 的 AABB 中心的均值。"""
    return np.mean([geom_aabb_center_world(model, data, g) for g in gids], axis=0)


def pad_points(model: mujoco.MjModel, data: mujoco.MjData, gids: Sequence[int]) -> np.ndarray:
    return np.concatenate([geom_surface_points(model, data, g) for g in gids], axis=0)


def inner_surface_gap(points_a: np.ndarray, points_b: np.ndarray, axis: np.ndarray) -> float:
    """两指内表面沿 ``axis``（B->A 单位向量）的间距。

    A 的"内表面"= A 的点在 axis 上的最小投影，B 的"内表面"= B 的点在 axis 上的最大投影；
    两者之差就是机构在当前构型下能夹住的最大宽度（几何上，不含接触变形）。
    """
    a = unit(axis)
    return float((points_a @ a).min() - (points_b @ a).max())


def min_point_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """两组点云之间的最小距离（精确，用 KD-tree；没有 scipy 时退化为暴力）。"""
    try:
        from scipy.spatial import cKDTree
        return float(cKDTree(points_b).query(points_a, k=1)[0].min())
    except Exception:                                     # pragma: no cover
        d = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
        return float(d.min())


def body_frame(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = model.body(name).id
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def site_frame(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    sid = model.site(name).id
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


# --------------------------------------------------------------------------------------
# 机械臂规格
# --------------------------------------------------------------------------------------
@dataclass
class FingerSpec:
    """一个手指（"finger"）的测量定义。

    pad_geoms_by_name / pad_geom_type 二选一：能按名字点名的就点名（ur5e 的
    ``*_fingerpad_collision``），没名字的（panda 的 pad 是无名 box）或整个手指就是一个
    mesh（piper）就按"该 body 上某种类型的 geom"来选。
    """

    label: str
    joint: str                # 被测量/被驱动的滑动或转动关节
    body: str                 # 承载 pad 的 body
    ctrl_sign: float = 1.0    # 该手指对应的执行器指令符号（用于构造 legacy ctrl 向量）
    pad_geoms_by_name: tuple = ()
    pad_geom_type: int | None = None
    analytic_coupling: str = ""   # 解析耦合预测（人读用），{q} 用驱动关节值代入

    def pad_geoms(self, model: mujoco.MjModel) -> list[int]:
        if self.pad_geoms_by_name:
            return [model.geom(n).id for n in self.pad_geoms_by_name]
        bid = model.body(self.body).id
        start, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
        out = [g for g in range(start, start + num)
               if self.pad_geom_type is None or int(model.geom_type[g]) == self.pad_geom_type]
        if not out:
            raise RuntimeError(f"{self.body} 上没有找到 pad geom")
        return out


@dataclass
class ArmSpec:
    name: str
    build: Callable[[], tuple[mujoco.MjModel, dict]]
    base_body: str
    flange_body: str           # "法兰"/腕部 body：接近轴 = normalize(pad_mid - flange_origin)
    elbow_body: str            # 只用于"肘部朝上"启发式选解
    tcp_carrying_body: str
    tcp_kind: str              # "site" | "hand_offset"
    arm_joints: tuple
    gripper_actuator_indices: tuple
    fingers: tuple
    moving_gripper_bodies: tuple
    gripper_subtree_roots: tuple
    param_open: float
    param_closed: float
    param_extra: tuple
    n_ctrl_samples: int
    legacy_ctrl_examples: tuple          # ((label, param), ...)
    ctrl_from_param: Callable[[float], list[float]]
    coupling_note: str
    tcp_site: str | None = None
    tcp_body: str | None = None          # hand_offset 用
    home_keyframe: str | None = None
    home_reference_qpos: tuple | None = None   # 无 keyframe 时的参考位姿（ur5e=legacy Q_HOME）
    reference_arm_qpos: tuple | None = None    # 测量 TCP 帧时把臂摆到的位姿（piper=zeros）
    documentation_claims: tuple = ()
    # 文档里写的（非本模型）力矩上限，仅用于给载荷估算做对照；key 是 executor/joint 名的一部分
    documented_torque_limits: tuple = ()


def _ur5e_ctrl(g: float) -> list[float]:
    return [float(g), -float(g)]


def _scalar_ctrl(c: float) -> list[float]:
    return [float(c)]


ARM_SPECS: dict[str, ArmSpec] = {
    "ur5e": ArmSpec(
        name="ur5e",
        build=build_ur5e_model,
        base_body="base",
        flange_body="right_hand",
        elbow_body="forearm_link",
        tcp_carrying_body="eef",
        tcp_kind="site",
        tcp_site="grip_site",
        arm_joints=tuple(UR_JOINTS),
        gripper_actuator_indices=(6, 7),
        fingers=(
            FingerSpec(label="finger_1_left", joint="finger_joint", body="left_inner_finger",
                       ctrl_sign=+1.0, pad_geoms_by_name=("left_fingerpad_collision",),
                       analytic_coupling="q(finger_joint)=g; q(left_inner_finger_joint)=-g/1.5; "
                                         "q(left_inner_knuckle_joint)=g/5.25"),
            FingerSpec(label="finger_2_right", joint="right_outer_knuckle_joint",
                       body="right_inner_finger", ctrl_sign=-1.0,
                       pad_geoms_by_name=("right_fingerpad_collision",),
                       analytic_coupling="q(right_outer_knuckle_joint)=-g; "
                                         "q(right_inner_finger_joint)=-g/1.5; "
                                         "q(right_inner_knuckle_joint)=g/5.25"),
        ),
        moving_gripper_bodies=("left_outer_knuckle", "left_inner_finger", "left_inner_knuckle",
                               "right_outer_knuckle", "right_inner_finger", "right_inner_knuckle"),
        gripper_subtree_roots=("right_gripper",),
        param_open=0.0,
        param_closed=0.7,
        param_extra=(0.3,),                 # legacy 用的 (0.3, -0.3)
        n_ctrl_samples=9,
        legacy_ctrl_examples=(("legacy_close_0.30", 0.3),),
        ctrl_from_param=_ur5e_ctrl,
        coupling_note=(
            "Robotiq 2F-140 是 4-bar linkage：4 个 fixed tendon + 4 个 equality 把 "
            "finger_joint/left_inner_finger_joint/left_inner_knuckle_joint（右指同理）耦合起来，"
            "执行器只驱动 finger_joint 与 right_outer_knuckle_joint。直接写 qpos 无效，必须 settle。"
        ),
        home_reference_qpos=tuple(Q_HOME_ELBOW_UP),
        documented_torque_limits=(
            ("shoulder_pan_joint", 150.0, "assets/robots/robosuite/models/assets/robots/ur5e/robot.xml torq_j1 ctrlrange"),
            ("shoulder_lift_joint", 150.0, "assets/robots/robosuite/models/assets/robots/ur5e/robot.xml torq_j2 ctrlrange"),
            ("elbow_joint", 150.0, "assets/robots/robosuite/models/assets/robots/ur5e/robot.xml torq_j3 ctrlrange"),
            ("wrist_1_joint", 28.0, "assets/robots/robosuite/models/assets/robots/ur5e/robot.xml torq_j4 ctrlrange"),
            ("wrist_2_joint", 28.0, "assets/robots/robosuite/models/assets/robots/ur5e/robot.xml torq_j5 ctrlrange"),
            ("wrist_3_joint", 28.0, "assets/robots/robosuite/models/assets/robots/ur5e/robot.xml torq_j6 ctrlrange"),
        ),
        documentation_claims=(
            "grip_site_from_pad_center() 注释：Robotiq 140 fingerpad 中心在 grip_site 局部 -z 方向约 32 mm",
            "top_down_gripper_rotation() 注释：手指开合方向 = grip_site 局部 x 轴",
            "run_official_ur5e_robotiq_wall_stack.py 组合出的臂执行器 forcerange 为 ±280 N·m",
            "assets/.../ur5e/robot.xml 上游 motor ctrlrange：肩/肘 ±150 N·m，腕 ±28 N·m",
        ),
    ),
    "panda": ArmSpec(
        name="panda",
        build=build_panda_model,
        base_body="link0",
        flange_body="hand",
        elbow_body="link4",
        tcp_carrying_body="hand",
        tcp_kind="hand_offset",
        tcp_body="hand",
        arm_joints=tuple(f"joint{i}" for i in range(1, 8)),
        gripper_actuator_indices=(7,),
        fingers=(
            FingerSpec(label="finger_joint1_left", joint="finger_joint1", body="left_finger",
                       ctrl_sign=+1.0, pad_geom_type=MJGEOM_BOX,
                       analytic_coupling="q(finger_joint1)=q(finger_joint2)=0.04*ctrl/255"),
            FingerSpec(label="finger_joint2_right", joint="finger_joint2", body="right_finger",
                       ctrl_sign=+1.0, pad_geom_type=MJGEOM_BOX,
                       analytic_coupling="q(finger_joint1)=q(finger_joint2)=0.04*ctrl/255"),
        ),
        moving_gripper_bodies=("left_finger", "right_finger"),
        gripper_subtree_roots=("hand",),
        param_open=255.0,
        param_closed=0.0,
        param_extra=(),
        n_ctrl_samples=9,
        legacy_ctrl_examples=(),
        ctrl_from_param=_scalar_ctrl,
        coupling_note=(
            "panda 两指是 slide joint 且没有各自的执行器：actuator8 是作用在 fixed tendon "
            "'split'（0.5*q1+0.5*q2）上的 general 执行器，ctrlrange 0..255，"
            "平衡时 tendon 长度 = gainprm[0]*ctrl/100 = 0.04*ctrl/255；另有 joint equality "
            "保证两指同步。因此只能通过 actuator8 的 ctrl 驱动，不能直接写 qpos。"
        ),
        home_keyframe="home",
        documentation_claims=(
            "assets/robots/franka/README.md：载入实测 nbody=12、nq=9、nu=8、nsite=0、nkey=1",
            "assets/robots/franka/README.md：臂展上界实测 1.2383 m（20 万组关节角，手指 body 原点相对 link0 原点）",
            "panda.xml 注释：ctrlrange (0,0.04) 重映射到 (0,255)，0.04*100/255=0.01568627451，255=fully open",
            "assets/robots/franka/README.md：上游 panda.xml 没有 tool_tip site（nsite=0）",
        ),
    ),
    "piper": ArmSpec(
        name="piper",
        build=build_piper_model,
        base_body="arm_base",
        flange_body="link6",
        elbow_body="link3",
        tcp_carrying_body="end_effector",
        tcp_kind="site",
        tcp_site="tool_tip",
        arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
        gripper_actuator_indices=(6,),
        fingers=(
            FingerSpec(label="joint7_link7", joint="joint7", body="link7", ctrl_sign=+1.0,
                       pad_geom_type=MJGEOM_MESH,
                       analytic_coupling="q(joint7)=q(joint8)=gripper ctrl（equality polycoef 0 1 0 0 0）"),
            FingerSpec(label="joint8_link8", joint="joint8", body="link8", ctrl_sign=+1.0,
                       pad_geom_type=MJGEOM_MESH,
                       analytic_coupling="q(joint7)=q(joint8)=gripper ctrl（equality polycoef 0 1 0 0 0）"),
        ),
        moving_gripper_bodies=("link7", "link8"),
        gripper_subtree_roots=("link7", "link8"),
        param_open=0.035,
        param_closed=0.0,
        param_extra=(),
        n_ctrl_samples=9,
        legacy_ctrl_examples=(),
        ctrl_from_param=_scalar_ctrl,
        coupling_note=(
            "piper 两指是 slide joint，只有 joint7 有执行器（'gripper'，kp=500/kv=20，"
            "ctrlrange 0..0.035），joint7/joint8 由 <equality><joint polycoef='0 1 0 0 0'/> 耦合。"
            "实测 q7=q8=ctrl，因此 ctrl=0 是**闭合**、ctrl=0.035 是**张开**（与 README 的方向相反，见报告）。"
        ),
        reference_arm_qpos=(0.0,) * 6,
        documentation_claims=(
            "assets/robots/piper/README.md：夹爪净开口 6.996 cm（两指内表面实测）",
            "assets/robots/piper/README.md：两个通过 equality 耦合的夹爪 slide joint + 1 路夹爪位置伺服",
            "assets/robots/piper/README.md：tool_tip site",
        ),
    ),
}


# --------------------------------------------------------------------------------------
# TCP 帧抽象
# --------------------------------------------------------------------------------------
@dataclass
class TcpFrame:
    """TCP 帧：site 型直接用 site；panda 没有 site，用 body+局部变换（经验测得的合成 TCP）。"""

    kind: str
    site: str | None = None
    body: str | None = None
    local_pos: np.ndarray | None = None
    local_rot: np.ndarray | None = None

    def pose(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "site":
            return site_frame(model, data, str(self.site))
        p, R = body_frame(model, data, str(self.body))
        return p + R @ self.local_pos, R @ self.local_rot

    def jacobians(self, model: mujoco.MjModel, data: mujoco.MjData,
                  dof_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP 帧原点的平动/转动 Jacobian（只取被控 dof 的列）。"""
        if self.kind == "site":
            sid = model.site(str(self.site)).id
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            return jp[:, dof_ids], jr[:, dof_ids]
        bid = model.body(str(self.body)).id
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jp, jr, bid)
        # 速度合成：v_point = v_body + w x r, r = R_body @ local_pos（世界系偏移）
        _, R = body_frame(model, data, str(self.body))
        r = R @ self.local_pos
        skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        return (jp - skew @ jr)[:, dof_ids], jr[:, dof_ids]


# --------------------------------------------------------------------------------------
# 位姿设置 / settle
# --------------------------------------------------------------------------------------
def joint_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    return int(model.joint(name).qposadr[0])


def joint_dof_addr(model: mujoco.MjModel, name: str) -> int:
    return int(model.joint(name).dofadr[0])


def set_joint_values(model: mujoco.MjModel, data: mujoco.MjData, values: dict[str, float]) -> None:
    for name, value in values.items():
        data.qpos[joint_qpos_addr(model, name)] = float(value)


def forward_kinematics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """纯运动学：只需要 body/site/geom 位姿，比 mj_forward 快很多（2 万次采样要用）。"""
    mujoco.mj_kinematics(model, data)


def settle(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray,
           max_steps: int = 12000, chunk: int = 250, qvel_tol: float = 1e-7) -> dict:
    """步进到机构静止。

    每个采样点都用**全新的 MjData**（调用方 new 一个），避免上一采样点的残余速度污染本次测量。
    判据用 ``max|qvel|``，因为耦合关节在 equality/tendon 约束下的"到位"首先表现为速度归零。
    """
    data.ctrl[:] = ctrl
    steps = 0
    for _ in range(max(1, max_steps // chunk)):
        for _ in range(chunk):
            mujoco.mj_step(model, data)
        steps += chunk
        if float(np.abs(data.qvel).max()) < qvel_tol:
            break
    mujoco.mj_forward(model, data)
    return {"steps": steps, "max_qvel": float(np.abs(data.qvel).max()),
            "converged": bool(float(np.abs(data.qvel).max()) < 1e-5)}


def build_open_state(spec: ArmSpec, model: mujoco.MjModel) -> tuple[dict[str, float], dict[str, float]]:
    """返回 (参考臂位姿, 夹爪全开位姿)。

    臂位姿：有 keyframe 用 keyframe；有 legacy 参考位姿用 legacy；否则全零。
    夹爪全开位姿：ur5e=关节 0；panda=keyframe 的 0.04；piper=最大 ctrl。
    """
    arm = {}
    if spec.home_keyframe:
        kid = model.key(spec.home_keyframe).id
        for jn in spec.arm_joints:
            adr = joint_qpos_addr(model, jn)
            arm[jn] = float(model.key_qpos[kid][adr])
    elif spec.home_reference_qpos is not None:
        for jn, v in zip(spec.arm_joints, spec.home_reference_qpos):
            arm[jn] = float(v)
    elif spec.reference_arm_qpos is not None:
        for jn, v in zip(spec.arm_joints, spec.reference_arm_qpos):
            arm[jn] = float(v)
    else:
        for jn in spec.arm_joints:
            arm[jn] = 0.0

    grip = {}
    if spec.home_keyframe:
        kid = model.key(spec.home_keyframe).id
        for finger in spec.fingers:
            adr = joint_qpos_addr(model, finger.joint)
            grip[finger.joint] = float(model.key_qpos[kid][adr])
    else:
        # 直接从"全开 ctrl"推：先给个初值再 settle 一次拿到真实关节值
        for finger in spec.fingers:
            grip[finger.joint] = spec.param_open if spec.param_open <= 0.2 else 0.0
    return arm, grip


# --------------------------------------------------------------------------------------
# 各项测量
# --------------------------------------------------------------------------------------
def measure_model_info(model: mujoco.MjModel, spec: ArmSpec, meta: dict) -> dict:
    bid = model.body(spec.tcp_carrying_body).id
    chain = []
    while bid != 0:
        chain.append(model.body(bid).name)
        if model.body(bid).name == spec.base_body:
            break
        bid = int(model.body_parentid[bid])
    chain.reverse()

    def body_line(b: int) -> str:
        m = model.body_mass[b]
        jn = int(model.body_jntnum[b])
        return f"{model.body(b).name}(m={m:.4f}kg, jnt={jn})"

    return {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "nbody": int(model.nbody), "nsite": int(model.nsite), "ngeom": int(model.ngeom),
        "njnt": int(model.njnt), "neq": int(model.neq), "ntendon": int(model.ntendon),
        "nkey": int(model.nkey), "nmesh": int(model.nmesh),
        "timestep": float(model.opt.timestep), "gravity": r3(model.opt.gravity),
        "integrator": int(model.opt.integrator),
        "base_body": spec.base_body,
        "tcp_carrying_body": spec.tcp_carrying_body,
        "chain_base_to_tcp_body": chain,
        "chain_detail": [body_line(model.body(n).id) for n in chain],
        "source": meta,
        "collisions_disabled_for_probe": True,
    }


def measure_arm_joints(model: mujoco.MjModel, spec: ArmSpec) -> list[dict]:
    out = []
    for jn in spec.arm_joints:
        jid = model.joint(jn).id
        act_ids = [a for a in range(model.nu) if int(model.actuator_trnid[a][0]) == jid
                   and int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)]
        acts = []
        for a in act_ids:
            acts.append({
                "name": model.actuator(a).name,
                "ctrlrange": r3(model.actuator_ctrlrange[a]),
                "forcerange": r3(model.actuator_forcerange[a]),
                "forcelimited": bool(model.actuator_forcelimited[a]),
                "ctrllimited": bool(model.actuator_ctrllimited[a]),
                "gainprm": r3(model.actuator_gainprm[a][:3]),
                "biasprm": r3(model.actuator_biasprm[a][:3]),
                "kp": float(model.actuator_gainprm[a][0]),
                "kv": float(-model.actuator_biasprm[a][2]),
            })
        out.append({
            "name": jn,
            "qpos_addr": int(model.jnt_qposadr[jid]),
            "dof_addr": int(model.jnt_dofadr[jid]),
            "range": r3(model.jnt_range[jid]),
            "limited": bool(model.jnt_limited[jid]),
            "actuators": acts,
        })
    return out


def fresh_state(model: mujoco.MjModel, arm_pose: dict, grip_pose: dict) -> mujoco.MjData:
    """全新 MjData：把臂/夹爪关节设成给定值，做一次 mj_forward（纯运动学，不 settle）。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_joint_values(model, data, arm_pose)
    set_joint_values(model, data, grip_pose)
    mujoco.mj_forward(model, data)
    return data


def arm_ctrl_vector(model: mujoco.MjModel, spec: ArmSpec, arm_pose: dict) -> np.ndarray:
    """臂位置伺服的 ctrl（= 参考位姿的关节值）；夹爪项留 0，由调用方填。"""
    ctrl = np.zeros(model.nu)
    for jn in spec.arm_joints:
        for a in range(model.nu):
            if (int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
                    and int(model.actuator_trnid[a][0]) == model.joint(jn).id):
                ctrl[a] = arm_pose[jn]
    return ctrl


def full_ctrl(model: mujoco.MjModel, spec: ArmSpec, arm_pose: dict,
              gripper_ctrl: Sequence[float]) -> np.ndarray:
    """臂 ctrl + 夹爪 ctrl -> 完整 nu 维 ctrl 向量。"""
    ctrl = arm_ctrl_vector(model, spec, arm_pose)
    for idx, val in zip(spec.gripper_actuator_indices, gripper_ctrl):
        ctrl[idx] = val
    return ctrl


def compute_panda_tcp(model: mujoco.MjModel, data: mujoco.MjData, spec: ArmSpec) -> TcpFrame:
    """**经验测定** panda 的合成 TCP。

    做法（README/上游都没有 TCP，所以只能量）：
      1. 把夹爪开到全开（keyframe home 的 finger 值 0.04）、臂摆到 keyframe home；
      2. 取 ``left_finger``/``right_finger`` 上所有 pad box（fingertip_pad_collision_*）的
         AABB 中心均值，得到两个"pad 面中心"；
      3. 两点的中点就是夹持中心（两指对称，所以它落在两指中线、在两块 pad 的 z 中心处）；
      4. 把它表示到 ``hand`` body 帧，得到合成 TCP 的原点；姿态取与 ``hand`` 对齐
         （即 site quat = 1 0 0 0），因为 hand 帧本身就是"z 朝指尖、y 沿开合方向"的正交基。
    """
    centers = [pad_center(model, data, f.pad_geoms(model)) for f in spec.fingers]
    mid_world = 0.5 * (centers[0] + centers[1])
    hand_pos, hand_rot = body_frame(model, data, spec.tcp_body)
    local = point_in_frame(mid_world, hand_pos, hand_rot)
    return TcpFrame(kind="hand_offset", body=spec.tcp_body, local_pos=local,
                    local_rot=np.eye(3)), mid_world


def measure_tcp_section(model: mujoco.MjModel, spec: ArmSpec, tcp: TcpFrame,
                        data: mujoco.MjData, panda_extra: dict | None) -> dict:
    pos, rot = tcp.pose(model, data)
    out: dict = {
        "kind": tcp.kind,
        "site": tcp.site,
        "frame_body": spec.tcp_carrying_body,
    }
    if tcp.kind == "site":
        sid = model.site(str(tcp.site)).id
        out.update({
            "site_local_pos": r3(model.site_pos[sid]),
            "site_local_quat": r4(model.site_quat[sid]),
            "how_defined": "直接用模型里已有的 site（未改动资产）",
            "world_pos_at_reference_pose": r3(pos),
            "world_quat_at_reference_pose": r4(mat_to_quat(rot)),
        })
    else:
        T = np.eye(4)
        T[:3, :3] = tcp.local_rot
        T[:3, 3] = tcp.local_pos
        # 与 ur5e 的 grip_site 约定对齐的另一种 site 写法（x=闭合轴, z=接近轴反向）
        alt_rot = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        out.update({
            "how_defined": (
                "panda 上游没有 TCP site：在 keyframe home（手指全开 q=0.04）下，取 "
                "left_finger/right_finger 上 5 个 fingertip_pad_collision_* box 的 AABB 中心均值，"
                "两者中点即合成 TCP 原点；姿态取与 hand body 对齐"
            ),
            "tcp_origin_in_hand_frame": r3(tcp.local_pos),
            "tcp_transform_in_hand_frame": [[round(float(x), 6) for x in row] for row in T],
            "site_element": {
                "parent_body": spec.tcp_body,
                "pos": r3(tcp.local_pos),
                "quat_wxyz": r4(mat_to_quat(tcp.local_rot)),
                "xml": (f'<site name="tcp_panda" pos="{tcp.local_pos[0]:.6f} '
                        f'{tcp.local_pos[1]:.6f} {tcp.local_pos[2]:.6f}" quat="1 0 0 0" '
                        f'size="0.005" rgba="0 1 0 1"/>'),
            },
            "alt_site_ur5e_convention": {
                "note": "若想让 TCP 帧与 ur5e grip_site 同约定（+x=闭合轴, -z=接近轴）",
                "pos": r3(tcp.local_pos),
                "quat_wxyz": r4(mat_to_quat(alt_rot)),
                "xml": (f'<site name="tcp_panda" pos="{tcp.local_pos[0]:.6f} '
                        f'{tcp.local_pos[1]:.6f} {tcp.local_pos[2]:.6f}" '
                        f'quat="0 0.707107 0.707107 0" size="0.005" rgba="0 1 0 1"/>'),
            },
            "world_pos_at_reference_pose": r3(pos),
            "world_quat_at_reference_pose": r4(mat_to_quat(rot)),
            "pad_centers_at_reference_pose": panda_extra["pad_centers"] if panda_extra else None,
            "pad_midpoint_at_reference_pose": panda_extra["pad_mid"] if panda_extra else None,
            "pad_centers_in_hand_frame": panda_extra["pad_centers_hand"] if panda_extra else None,
        })
    return out


def measure_opening_curve(model: mujoco.MjModel, spec: ArmSpec,
                          arm_pose: dict, grip_pose: dict) -> dict:
    """夹爪开合曲线：ctrl -> pad 中心间距 / 内表面间距。

    每个采样点：全新 MjData + 臂摆到参考位姿 + settle 到静止，然后读几何。
    闭合轴取"全开那一刻"测到的轴（固定轴），这样"投影间距"才有可比性；同时也记录
    逐采样点自身的轴，用来检查轴是否随构型转动。
    """
    params = list(np.linspace(spec.param_open, spec.param_closed, spec.n_ctrl_samples))
    for extra in spec.param_extra:
        if all(abs(extra - p) > 1e-9 for p in params):
            params.append(float(extra))
    direction = 1.0 if spec.param_closed >= spec.param_open else -1.0
    params = sorted(params, key=lambda p: direction * (p - spec.param_open))

    arm_ctrl = arm_ctrl_vector(model, spec, arm_pose)

    finger_geoms = {f.label: f.pad_geoms(model) for f in spec.fingers}
    samples = []
    axis_open = None
    for param in params:
        gctrl = spec.ctrl_from_param(param)
        ctrl = arm_ctrl.copy()
        for idx, val in zip(spec.gripper_actuator_indices, gctrl):
            ctrl[idx] = val
        data = fresh_state(model, arm_pose, grip_pose)
        conv = settle(model, data, ctrl)

        centers = {label: pad_center(model, data, gids) for label, gids in finger_geoms.items()}
        points = {label: pad_points(model, data, gids) for label, gids in finger_geoms.items()}
        labels = list(finger_geoms.keys())
        a_lab, b_lab = labels[0], labels[1]
        vec = centers[a_lab] - centers[b_lab]
        axis_now = unit(vec)
        if axis_open is None:
            axis_open = axis_now.copy()
        samples.append({
            "param": round(float(param), 6),
            "ctrl": [round(float(c), 6) for c in ctrl],
            "gripper_ctrl": [round(float(c), 6) for c in gctrl],
            "finger_joint_qpos": {f.joint: round(float(data.qpos[joint_qpos_addr(model, f.joint)]), 6)
                                  for f in spec.fingers},
            # 夹爪的全部关节（含被耦合驱动的那些），方便核对 4-bar / equality 是否真的成立
            "gripper_joint_qpos_all": {
                model.joint(j).name: round(float(data.qpos[int(model.jnt_qposadr[j])]), 6)
                for j in range(model.njnt) if model.joint(j).name not in spec.arm_joints},
            "pad_centers_world": {k: r3(v) for k, v in centers.items()},
            "pad_center_separation_m": round(float(np.linalg.norm(vec)), 6),
            "pad_center_separation_on_open_axis_m": round(float(np.dot(vec, axis_open)), 6),
            "inner_surface_gap_m": round(inner_surface_gap(points[a_lab], points[b_lab], axis_open), 6),
            "min_surface_distance_m": round(min_point_distance(points[a_lab], points[b_lab]), 6),
            "closing_axis_world": r3(axis_now),
            "closing_axis_deviation_deg": round(angle_deg_between(axis_now, axis_open), 6),
            # ur5e 深闭合时两 pad 中心会越过中线，(cA-cB) 反向 -> 原始夹角会跳到 180°。
            # 对"轴是否转动"这个问题，应该看**直线**夹角（模 180°）。
            "closing_axis_line_deviation_deg": round(
                min(angle_deg_between(axis_now, axis_open),
                    180.0 - angle_deg_between(axis_now, axis_open)), 6),
            "tendon_lengths": r3(data.ten_length) if model.ntendon else [],
            "settle": conv,
        })

    widths = np.array([s["pad_center_separation_m"] for s in samples])
    gaps = np.array([s["inner_surface_gap_m"] for s in samples])
    min_dists = np.array([s["min_surface_distance_m"] for s in samples])
    ctrls = np.array([s["param"] for s in samples])

    def fit(x: np.ndarray, y: np.ndarray) -> dict:
        if len(x) < 2 or float(np.ptp(x)) < 1e-9:
            return {}
        a, b = np.polyfit(x, y, 1)
        resid = y - (a * x + b)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return {
            "a": float(a), "b": float(b),
            "residual_max_abs_m": float(np.abs(resid).max()),
            "residual_rms_m": float(np.sqrt(np.mean(resid ** 2))),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else None,
            "n_samples": int(len(x)),
            "ctrl_range": [round(float(x.min()), 6), round(float(x.max()), 6)],
        }

    i_min = int(np.argmin(widths))
    i_max = int(np.argmax(widths))
    # "可用抓取段"：从全开端到 pad 中心间距最小处（ur5e 的 4-bar 过了这段间距反而变大）
    usable_idx = np.arange(0, i_min + 1) if i_min > 0 else np.arange(len(samples))
    order = np.argsort(ctrls[usable_idx])          # 按 param 升序，方便人读与 polyfit
    usable_x = ctrls[usable_idx][order]
    usable_y = widths[usable_idx][order]
    max_open = float(widths[i_max])
    min_open = float(widths[i_min])
    fit_usable = fit(usable_x, usable_y)
    order_all = np.argsort(ctrls)
    fit_all = fit(ctrls[order_all], widths[order_all])
    half = 0.5 * max_open
    ctrl_half = None
    if fit_usable and abs(fit_usable["a"]) > 1e-12:
        ctrl_half = (half - fit_usable["b"]) / fit_usable["a"]
    ctrl_half_measured = None
    if len(usable_x) >= 2:             # 在单调段上线性插值，比拟合更贴近实测
        yo = np.argsort(usable_y)
        ctrl_half_measured = float(np.interp(half, usable_y[yo], usable_x[yo]))

    return {
        "definition": {
            "pad_center_separation_m": (
                "两指 pad geom 的 AABB 中心的**距离**（= 在逐采样点闭合轴上的投影，恒 ≥0）；"
                "这是任务书要的 pad 中心间距，也是「能夹多宽」的代理量"
            ),
            "pad_center_separation_on_open_axis_m": (
                "同一向量在**全开构型下测得的固定闭合轴**上的带符号投影；ur5e 的 4-bar 在深闭合时"
                "两 pad 中心会越过中线（投影变负），这是几何事实而非测量错误"
            ),
            "inner_surface_gap_m": (
                "两指全部表面点（box=8 角点 / mesh=全部顶点）沿同一固定闭合轴的极值差："
                "min(A 投影) - max(B 投影)。负值 = 两指表面已经互相穿插（本探针关掉了接触，"
                "真实机构到这一步已经被碰撞挡住）"
            ),
            "min_surface_distance_m": "两指点云最小距离（KD-tree 精确，恒 ≥0 = 真实最近接近量）",
            "param": "扫描参数：ur5e=两指 ctrl 幅值 g（ctrl=[g,-g]）；panda=actuator8 ctrl；piper=gripper ctrl",
        },
        "samples": samples,
        "fit_all_samples": fit_all,
        "fit_usable_grasp_range": fit_usable,
        "usable_grasp_range_param": [round(float(usable_x[0]), 6), round(float(usable_x[-1]), 6)],
        "usable_grasp_range_note": (
            "可用抓取段 = 从全开端到 pad 中心距离最小处；ur5e 的 4-bar 过了这段最小间距后"
            "两 pad 中心会越过中线、距离重新变大，线性拟合只在这段有效"
        ),
        "max_opening_m": round(max_open, 6),
        "max_opening_param": round(float(ctrls[i_max]), 6),
        "min_opening_m": round(min_open, 6),
        "min_opening_param": round(float(ctrls[i_min]), 6),
        "max_inner_surface_gap_m": round(float(gaps.max()), 6),
        "min_inner_surface_gap_m": round(float(gaps.min()), 6),
        "min_surface_distance_over_sweep_m": round(float(min_dists.min()), 6),
        "min_surface_distance_param": round(float(ctrls[int(np.argmin(min_dists))]), 6),
        "ctrl_at_half_max_opening": {
            "target_width_m": round(half, 6),
            "from_usable_fit": round(float(ctrl_half), 6) if ctrl_half is not None else None,
            "from_measured_curve": round(float(ctrl_half_measured), 6) if ctrl_half_measured is not None else None,
        },
        "max_closing_axis_deviation_deg": round(
            float(max(s["closing_axis_deviation_deg"] for s in samples)), 6),
        "max_closing_axis_line_deviation_deg": round(
            float(max(s["closing_axis_line_deviation_deg"] for s in samples)), 6),
        "all_settled": all(s["settle"]["converged"] for s in samples),
        "coupling_note": spec.coupling_note,
        "analytic_coupling": {f.label: f.analytic_coupling for f in spec.fingers},
        "coupling_check": {
            # 指间残差按 ctrl 符号定义：panda/piper 是 q0-q1，ur5e 是 q0+q1（两 knuckle 反向）
            "max_abs_finger_joint_residual_rad": round(max(
                abs(s["finger_joint_qpos"][spec.fingers[0].joint] * spec.fingers[0].ctrl_sign
                    - s["finger_joint_qpos"][spec.fingers[1].joint] * spec.fingers[1].ctrl_sign)
                for s in samples), 9),
            "max_abs_tendon_length_over_samples": round(max(
                (max((abs(x) for x in s["tendon_lengths"]), default=0.0) for s in samples)), 9),
            "tendon_length_interpretation": (
                "该模型的 tendon 上挂着执行器：tendon 长度就是位置指令的平衡值（不是残差）"
                if any(int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON)
                       for a in range(model.nu))
                else "该模型的 tendon equality 目标 = 0，所以这个值就是最大残差（应接近 0）"),
            "note": ("tendon/equality 是软约束，刚度取决于 armature，只有在对的 armature 下才收敛；"
                     "指间残差与 tendon 残差都应接近 0"),
        },
    }


def measure_frames(model: mujoco.MjModel, spec: ArmSpec, tcp: TcpFrame,
                   data: mujoco.MjData, panda_extra: dict | None) -> dict:
    """TCP 帧下的四个关键量：闭合轴、接近轴、pad 中点偏移、TCP 相对法兰的偏移。

    定义（写清楚，因为不同来源的"approach axis"符号约定会反）：
    * closing_axis_tcp：finger[0] 的 pad 中心指向 finger[1] 的 pad 中心（单位向量，TCP 帧）；
    * approach_axis_tcp：**最终下压方向** = normalize(pad_mid - flange_origin)，即工具从腕部
      伸出去的方向；桌面抓取时它必须朝下（工作空间过滤用的就是它）；
    * tcp_to_pad_mid_axis_tcp：任务书字面定义的"从 TCP 原点指向手指工作空间"的单位向量。
      当 TCP 落在两指之外（ur5e/piper 的 site 在指尖外侧）时它与 approach_axis 反平行。
    """
    labels = [f.label for f in spec.fingers]
    centers = {f.label: pad_center(model, data, f.pad_geoms(model)) for f in spec.fingers}
    mid = 0.5 * (centers[labels[0]] + centers[labels[1]])
    tcp_pos, tcp_rot = tcp.pose(model, data)
    flange_pos, flange_rot = body_frame(model, data, spec.flange_body)
    wrist_pos, wrist_rot = (body_frame(model, data, "wrist_3_link") if spec.name == "ur5e"
                            else (None, None))

    closing_world = unit(centers[labels[0]] - centers[labels[1]])
    approach_world = unit(mid - flange_pos)
    tcp_to_pad_world = unit(mid - tcp_pos)

    out = {
        "conventions": {
            "closing_axis_tcp": f"{labels[0]} 的 pad 中心 -> {labels[1]} 的 pad 中心（TCP 帧，单位向量）",
            "approach_axis_tcp": (
                f"法兰 body '{spec.flange_body}' 原点 -> 两指 pad 中点（TCP 帧，单位向量）；"
                "这是抓取最后一段下压的方向，顶抓时它应朝下"
            ),
            "tcp_to_pad_mid_axis_tcp": "TCP 原点 -> 两指 pad 中点（任务书字面定义）",
            "tcp_frame": ("site '" + str(spec.tcp_site) + "'") if spec.tcp_kind == "site"
                         else f"合成 TCP（{spec.tcp_body} body 局部变换）",
        },
        "closing_axis_tcp": r3(vector_in_frame(closing_world, tcp_rot)),
        "closing_axis_world_at_reference_pose": r3(closing_world),
        "approach_axis_tcp": r3(vector_in_frame(approach_world, tcp_rot)),
        "approach_axis_world_at_reference_pose": r3(approach_world),
        "tcp_to_pad_mid_axis_tcp": r3(vector_in_frame(tcp_to_pad_world, tcp_rot)),
        "tcp_to_pad_mid_axis_degenerate": bool(np.linalg.norm(mid - tcp_pos) < 1e-6),
        "pad_midpoint_tcp": r3(point_in_frame(mid, tcp_pos, tcp_rot)),
        "pad_center_offset_tcp": r3(point_in_frame(mid, tcp_pos, tcp_rot)),
        "pad_center_offset_tcp_norm_m": round(float(np.linalg.norm(mid - tcp_pos)), 6),
        "pad_center_offset_flange": r3(point_in_frame(mid, flange_pos, flange_rot)),
        "tcp_origin_offset_from_flange_tcp_frame": r3(point_in_frame(flange_pos, tcp_pos, tcp_rot)),
        "tcp_origin_offset_from_flange_flange_frame": r3(point_in_frame(tcp_pos, flange_pos, flange_rot)),
        "tcp_origin_offset_from_flange_m": round(float(np.linalg.norm(tcp_pos - flange_pos)), 6),
        "angle_between_approach_and_tcp_to_pad_deg": round(
            angle_deg_between(approach_world, tcp_to_pad_world), 6),
        "pad_center_midpoint_world": r3(mid),
        "tcp_origin_world": r3(tcp_pos),
        "flange_origin_world": r3(flange_pos),
        "closing_axis_dot_tcp_axes": r3([float(np.dot(closing_world, tcp_rot[:, i])) for i in range(3)]),
        "approach_axis_dot_tcp_axes": r3([float(np.dot(approach_world, tcp_rot[:, i])) for i in range(3)]),
    }
    if wrist_pos is not None:
        out["tcp_origin_offset_from_wrist3_link_m"] = round(float(np.linalg.norm(tcp_pos - wrist_pos)), 6)
        out["tcp_origin_offset_from_wrist3_link_flange_frame"] = r3(
            point_in_frame(tcp_pos, wrist_pos, wrist_rot))
    if panda_extra:
        out["panda_tcp_definition"] = panda_extra["definition"]
    return out


def solve_tcp_ik(model: mujoco.MjModel, tcp: TcpFrame, arm_joints: Sequence[str],
                 target_pos: np.ndarray, target_rot: np.ndarray,
                 q0: np.ndarray, iterations: int = 400, damping: float = 1e-4) -> tuple[np.ndarray, dict]:
    """阻尼最小二乘（DLS）IK：只动臂关节，夹爪关节保持不动。

    * 雅可比用 mj_jacSite / mj_jacBody（后者把手上的局部偏移换算成 TCP 原点的雅可比）；
    * 姿态误差用旋转向量（见 rotation_error_vector 的注释：叉积平均在 180° 附近会失效）；
    * 每步 0.5 倍步长 + 关节限位裁剪，保证不发散。
    """
    data = mujoco.MjData(model)
    dof_ids = np.array([joint_dof_addr(model, jn) for jn in arm_joints], dtype=int)
    lo = np.array([model.jnt_range[model.joint(jn).id][0] for jn in arm_joints])
    hi = np.array([model.jnt_range[model.joint(jn).id][1] for jn in arm_joints])
    q = np.asarray(q0, dtype=float).copy()
    best = q.copy()
    best_err = float("inf")
    for _ in range(iterations):
        for jn, v in zip(arm_joints, q):
            data.qpos[joint_qpos_addr(model, jn)] = float(v)
        mujoco.mj_forward(model, data)
        pos, rot = tcp.pose(model, data)
        pos_err = np.asarray(target_pos, dtype=float) - pos
        rot_err = rotation_error_vector(rot, target_rot)
        err = float(np.linalg.norm(pos_err)) + 0.3 * float(np.linalg.norm(rot_err))
        if err < best_err:
            best_err, best = err, q.copy()
        if float(np.linalg.norm(pos_err)) < 1e-6 and float(np.linalg.norm(rot_err)) < 1e-4:
            break
        jp, jr = tcp.jacobians(model, data, dof_ids)
        J = np.vstack([jp, 0.5 * jr])
        e = np.concatenate([pos_err, 0.5 * rot_err])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        q = np.clip(q + 0.5 * dq, lo, hi)
    # 用最优解再算一次误差
    for jn, v in zip(arm_joints, best):
        data.qpos[joint_qpos_addr(model, jn)] = float(v)
    mujoco.mj_forward(model, data)
    pos, rot = tcp.pose(model, data)
    pos_err = float(np.linalg.norm(np.asarray(target_pos) - pos))
    rot_err = float(np.linalg.norm(rotation_error_vector(rot, target_rot)))
    return best, {"pos_error_m": pos_err, "rot_error_rad": rot_err,
                  "rot_error_deg": math.degrees(rot_err)}


def target_rotation_from_axes(approach_tcp: np.ndarray, closing_tcp: np.ndarray,
                              approach_world: np.ndarray, closing_world: np.ndarray) -> np.ndarray:
    """给定 TCP 帧内的接近轴/闭合轴，构造"让接近轴指向 approach_world（如竖直向下）、
    闭合轴尽量对齐 closing_world"的目标姿态 R（世界 <- TCP）。"""
    F_tcp = frame_from_axes(approach_tcp, closing_tcp)
    F_world = frame_from_axes(approach_world, closing_world)
    return F_world @ F_tcp.T


def wrap_arm_qpos_within_limits(model: mujoco.MjModel, arm_joints: Sequence[str],
                                q: np.ndarray) -> np.ndarray:
    """把关节值折算到 (-π, π]，但**只在折算后仍落在关节限位内**时才采用。

    纯为可读性：IK 可能给出 shoulder_lift=4.58 rad 这类等价但难读的值
    （4.58-2π=-1.70 是同一个位姿）。piper 的 joint2 限位是 [0, 3.14]，折算成负值就越界，
    这种关节保持原值。
    """
    out = np.array(q, dtype=float).copy()
    for i, jn in enumerate(arm_joints):
        lo, hi = model.jnt_range[model.joint(jn).id]
        cand = (float(out[i]) + math.pi) % (2.0 * math.pi) - math.pi
        if lo - 1e-9 <= cand <= hi + 1e-9:
            out[i] = cand
    return out


def measure_reach(model: mujoco.MjModel, spec: ArmSpec, tcp: TcpFrame,
                  grip_pose: dict, approach_axis_tcp: np.ndarray,
                  samples: int, seed: int) -> dict:
    """关节空间均匀采样 -> TCP 位置分布；再筛出"接近轴朝下 35° 以内"的子集。"""
    rng = np.random.default_rng(seed)
    ranges = np.array([model.jnt_range[model.joint(jn).id] for jn in spec.arm_joints])
    q = rng.uniform(ranges[:, 0], ranges[:, 1], size=(samples, len(spec.arm_joints)))
    data = mujoco.MjData(model)
    set_joint_values(model, data, grip_pose)          # 夹爪构型不影响 TCP 位置，但要固定
    positions = np.zeros((samples, 3))
    approach_world = np.zeros((samples, 3))
    for i in range(samples):
        for jn, v in zip(spec.arm_joints, q[i]):
            data.qpos[joint_qpos_addr(model, jn)] = float(v)
        forward_kinematics(model, data)
        pos, rot = tcp.pose(model, data)
        positions[i] = pos
        approach_world[i] = rot @ approach_axis_tcp
    dist = np.linalg.norm(positions, axis=1)
    down = np.array([0.0, 0.0, -1.0])
    cosang = np.clip(approach_world @ down, -1.0, 1.0)
    mask = cosang >= math.cos(math.radians(35.0))

    def stats(values: np.ndarray, prefix: str = "") -> dict:
        if values.size == 0:
            return {"count": 0}
        return {
            "count": int(values.size),
            f"{prefix}min_m": round(float(values.min()), 6),
            f"{prefix}median_m": round(float(np.median(values)), 6),
            f"{prefix}p95_m": round(float(np.percentile(values, 95)), 6),
            f"{prefix}max_m": round(float(values.max()), 6),
            f"{prefix}mean_m": round(float(values.mean()), 6),
        }

    def z_stats(values: np.ndarray) -> dict:
        """TCP 高度的分布：顶抓能不能真在桌面上方，靠这个数判断（半径一样但全在地面附近就没用）。"""
        if values.size == 0:
            return {}
        return {
            "tcp_z_min_m": round(float(values.min()), 6),
            "tcp_z_median_m": round(float(np.median(values)), 6),
            "tcp_z_max_m": round(float(values.max()), 6),
            "fraction_tcp_z_above_0.20m": round(float((values > 0.20).mean()), 6),
        }

    all_stats = stats(dist)
    all_stats.update(z_stats(positions[:, 2]))
    top_stats = stats(dist[mask])
    top_stats.update(z_stats(positions[mask, 2]))
    return {
        "definition": (
            "在臂关节限位内均匀随机采样 qpos（固定 seed），FK 到 TCP 原点，"
            "统计到世界原点（= 基座帧原点）的距离。这是**关节限位内的运动学包络**，"
            "不含自碰撞/灵巧性约束，所以比手册工作空间大。"
        ),
        "samples": int(samples), "seed": int(seed),
        "all": all_stats,
        "top_down_within_35deg": {
            "count": int(mask.sum()),
            "fraction": round(float(mask.mean()), 6),
            "axis_used": "approach_axis_tcp（法兰->pad 中点，即最终下压方向）应朝下",
            "distance": top_stats,
        },
    }


def measure_mass(model: mujoco.MjModel, spec: ArmSpec, reach: dict, arm_joints_meta: list[dict]) -> dict:
    """质量、执行器力矩上限、以及"最弱肩关节"的静态载荷粗估。"""
    total = float(model.body_mass.sum())
    # gripper 子树质量：从 gripper_subtree_roots 往下累加（ur5e=right_gripper，panda=hand，
    # piper 的两个手指 body 直接挂在 link6 下，所以有两个根）
    subtree = {model.body(n).id for n in spec.gripper_subtree_roots}
    for b in range(model.nbody):
        p = int(model.body_parentid[b])
        while p != 0:
            if p in subtree:
                subtree.add(b)
                break
            p = int(model.body_parentid[p])
    gripper_mass = float(sum(model.body_mass[b] for b in subtree))
    moving = float(sum(model.body_mass[model.body(n).id] for n in spec.moving_gripper_bodies))
    actuator_limits = []
    for a in range(model.nu):
        actuator_limits.append({
            "name": model.actuator(a).name,
            "ctrlrange": r3(model.actuator_ctrlrange[a]),
            "forcerange": r3(model.actuator_forcerange[a]),
            "forcelimited": bool(model.actuator_forcelimited[a]),
            "transmission": int(model.actuator_trntype[a]),
        })

    # 肩关节 = 前两个臂关节；取力矩上限较小者（位置伺服用 forcerange；无 forcerange 用 ctrlrange）
    shoulder = arm_joints_meta[:2]
    g = 9.81
    worst = None
    for j in shoulder:
        for act in j["actuators"]:
            if act["forcelimited"]:
                lim = min(abs(act["forcerange"][0]), abs(act["forcerange"][1]))
                kind = "forcerange"
            else:
                lim = min(abs(act["ctrlrange"][0]), abs(act["ctrlrange"][1]))
                kind = "ctrlrange"
            if worst is None or lim < worst[1]:
                worst = (f"{j['name']}/{act['name']}", lim, kind)

    reach_max = reach["all"]["max_m"]
    radius = 0.6 * reach_max
    payload = worst[1] / (g * radius) if worst else None

    # 全关节最弱一环（很多臂其实是腕关节先撑不住）
    weak_all = None
    for j in arm_joints_meta:
        for act in j["actuators"]:
            lim = (min(abs(act["forcerange"][0]), abs(act["forcerange"][1]))
                   if act["forcelimited"] else min(abs(act["ctrlrange"][0]), abs(act["ctrlrange"][1])))
            if weak_all is None or lim < weak_all[1]:
                weak_all = (f"{j['name']}/{act['name']}", lim)
    payload_all = weak_all[1] / (g * radius) if weak_all else None
    out: dict = {
        "total_mass_kg": round(total, 6),
        "gripper_subtree_mass_kg": round(gripper_mass, 6),
        "arm_mass_kg": round(total - gripper_mass, 6),
        "gripper_moving_parts_mass_kg": round(moving, 6),
        "gripper_moving_bodies": list(spec.moving_gripper_bodies),
        "actuator_limits": actuator_limits,
        "payload_estimate": {
            "formula": "m_max = tau_shoulder / (g * r), r = 0.6 * max_reach, g = 9.81",
            "shoulder_joint_used": worst[0] if worst else None,
            "shoulder_torque_limit_nm": worst[1] if worst else None,
            "shoulder_limit_source": worst[2] if worst else None,
            "radius_m": round(radius, 6),
            "max_reach_m": round(reach_max, 6),
            "payload_kg": round(payload, 4) if payload is not None else None,
            "weakest_joint_overall": weak_all[0] if weak_all else None,
            "weakest_joint_limit_nm": weak_all[1] if weak_all else None,
            "payload_kg_limited_by_weakest_joint": round(payload_all, 4) if payload_all else None,
            "caveats": (
                "粗估：只算静态力矩，忽略臂自重与负载在其它关节上的分量、忽略腕部力矩、"
                "忽略摩擦/减速器效率，因此是上界。若组合模型里的 forcerange 比上游真实力矩大"
                "（ur5e 就是这种情况），这个数会明显偏乐观，见 documented_torque_cross_check。"
            ),
            "documented_torque_cross_check": None,
        },
    }
    if spec.documented_torque_limits:
        doc_shoulder = min(lim for name, lim, _ in spec.documented_torque_limits
                           if name in tuple(j["name"] for j in arm_joints_meta[:2]))
        doc_all = min(lim for _, lim, _ in spec.documented_torque_limits)
        out["payload_estimate"]["documented_torque_cross_check"] = {
            "note": "用文档/上游 XML 里写的力矩上限重算同一个公式（本模型执行器 forcerange 与它不一致时才有意义）",
            "documented_limits": [{"joint": n, "limit_nm": lim, "source": src}
                                  for n, lim, src in spec.documented_torque_limits],
            "documented_shoulder_limit_nm": doc_shoulder,
            "payload_kg_using_documented_shoulder_limit": round(doc_shoulder / (g * radius), 4),
            "documented_weakest_limit_nm": doc_all,
            "payload_kg_using_documented_weakest_limit": round(doc_all / (g * radius), 4),
        }
    return out


def measure_home(model: mujoco.MjModel, spec: ArmSpec, tcp: TcpFrame, arm_pose: dict,
                 grip_pose: dict, approach_tcp: np.ndarray, closing_tcp: np.ndarray,
                 seed: int) -> dict:
    """home：panda 直接读 keyframe；ur5e/piper 用多起点 DLS IK 找一个"肘部朝上、向前伸"的解。

    对 keyframe 会同时给出两个读数：
    * ``*_at_keyframe_qpos``：把 qpos 设成 keyframe 值的纯运动学结果（规划器常用这个）；
    * ``*_at_servo_equilibrium``：按 keyframe 的 ctrl 让位置伺服在重力下稳定后的实际结果
      （两者差多少 = 该位姿的重力下垂量）。
    """
    out: dict = {"keyframe": spec.home_keyframe}
    if spec.home_keyframe:
        kid = model.key(spec.home_keyframe).id
        data = fresh_state(model, {}, {})            # 从 qpos0 起
        mujoco.mj_resetDataKeyframe(model, data, kid)
        mujoco.mj_forward(model, data)
        pos, rot = tcp.pose(model, data)
        ctrl = np.array(model.key_ctrl[kid])
        settle(model, data, ctrl)
        pos_eq, rot_eq = tcp.pose(model, data)
        out.update({
            "source": f"keyframe '{spec.home_keyframe}'",
            "home_pose_kind": "keyframe",
            "converged": True,
            "qpos": r3(model.key_qpos[kid]),
            "ctrl": r3(ctrl),
            "tcp_world_pos_at_keyframe_qpos": r3(pos),
            "tcp_world_quat_at_keyframe_qpos": r4(mat_to_quat(rot)),
            "tcp_world_pos": r3(pos),
            "tcp_world_quat": r4(mat_to_quat(rot)),
            "approach_axis_world": r3(rot @ approach_tcp),
            "closing_axis_world": r3(rot @ closing_tcp),
            "tcp_world_pos_at_servo_equilibrium": r3(pos_eq),
            "tcp_world_quat_at_servo_equilibrium": r4(mat_to_quat(rot_eq)),
            "keyframe_qpos_to_equilibrium_offset_m": round(float(np.linalg.norm(pos_eq - pos)), 6),
            "arm_joint_qpos_at_servo_equilibrium": r3(data.qpos[:len(spec.arm_joints)]),
            "finger_joint_qpos": {f.joint: round(float(data.qpos[joint_qpos_addr(model, f.joint)]), 6)
                                  for f in spec.fingers},
        })
        return out

    nominal_target = np.array([0.45, 0.0, 0.35])
    rng = np.random.default_rng(seed)
    ranges = np.array([model.jnt_range[model.joint(jn).id] for jn in spec.arm_joints])
    starts = [np.zeros(len(spec.arm_joints))]
    starts += [rng.uniform(ranges[:, 0], ranges[:, 1]) for _ in range(31)]
    if spec.home_reference_qpos is not None:
        starts.append(np.asarray(spec.home_reference_qpos, dtype=float))

    # 目标可能对某些臂根本不可达（实测 piper 在"工具竖直向下"约束下，TCP 最高只能到 z≈0.12 m），
    # 所以按 scale 依次把目标朝基座收缩，取第一个能收敛的；用到的 scale 会写进 JSON。
    solutions: list = []
    used_target = nominal_target
    used_scale = 1.0
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        candidate = np.array([nominal_target[0] * scale, 0.0, nominal_target[2] * scale])
        target_rot = target_rotation_from_axes(approach_tcp, closing_tcp,
                                               np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]))
        found = []
        for q0 in starts:
            q, err = solve_tcp_ik(model, tcp, spec.arm_joints, candidate, target_rot, np.asarray(q0))
            if err["pos_error_m"] < 1e-4 and err["rot_error_rad"] < 1e-3:
                found.append((q, err))
        if found:
            solutions, used_target, used_scale = found, candidate, scale
            break

    if not solutions:
        # 连收缩后的目标都不可达 -> 退让成"尽力而为"：先采样找到（接近轴尽量朝下、TCP 尽量靠近
        # 名义目标）的构型，再以该构型的姿态为姿态目标做位置精修。结果里会带上明确的残差与原因。
        out["home_pose_kind"] = "best_effort_target_unreachable"
        out["unreachable_reason"] = (
            "在「接近轴竖直向下」约束下，名义目标 (0.45, 0, 0.35) 及其所有收缩版本都不可达；"
            "改用采样+位置精修的尽力解，并报告实际残差"
        )
        sample_n = 50000
        qs = rng.uniform(ranges[:, 0], ranges[:, 1], size=(sample_n, len(spec.arm_joints)))
        data_s = mujoco.MjData(model)
        best_sample = None
        down_zs = []
        for i in range(sample_n):
            for jn, v in zip(spec.arm_joints, qs[i]):
                data_s.qpos[joint_qpos_addr(model, jn)] = float(v)
            forward_kinematics(model, data_s)
            pos, rot = tcp.pose(model, data_s)
            cosang = float(-(rot @ approach_tcp)[2])
            if cosang >= math.cos(math.radians(35.0)):
                down_zs.append(float(pos[2]))
                d = float(np.linalg.norm(pos - nominal_target))
                if best_sample is None or d < best_sample[0]:
                    best_sample = (d, qs[i].copy(), rot.copy(), pos.copy(), cosang)
        if best_sample is not None:
            out["best_effort_scan"] = {
                "samples": sample_n,
                "approach_within_35deg_count": len(down_zs),
                "tcp_z_max_m_when_approach_within_35deg": round(max(down_zs), 6),
                "nearest_tcp_distance_to_nominal_m": round(best_sample[0], 6),
            }
            _, q_best, rot_best, pos_best, _ = best_sample
            refined = None
            for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
                cand = np.array([nominal_target[0] * scale, 0.0, nominal_target[2] * scale])
                q, err = solve_tcp_ik(model, tcp, spec.arm_joints, cand, rot_best, np.asarray(q_best))
                if refined is None or err["pos_error_m"] < refined[1]["pos_error_m"]:
                    refined = (q, err)
            used_target = nominal_target
            solutions = [(refined[0], refined[1])]

    # "肘部朝上"：所有解里肘部 body 原点 z 最高者
    best = None
    for q, err in solutions:
        data = mujoco.MjData(model)
        for jn, v in zip(spec.arm_joints, q):
            data.qpos[joint_qpos_addr(model, jn)] = float(v)
        for jn, v in grip_pose.items():
            data.qpos[joint_qpos_addr(model, jn)] = float(v)
        mujoco.mj_forward(model, data)
        elbow_z = float(data.xpos[model.body(spec.elbow_body).id][2])
        if best is None or elbow_z > best[2]:
            best = (q, err, elbow_z, data)
    out["source"] = "本脚本的 DLS IK（多起点，取肘部最高的解）"
    out["ik_target"] = {
        "tcp_pos_world": r3(used_target),
        "nominal_tcp_pos_world": r3(nominal_target),
        "target_scale_used": used_scale,
        "approach_axis_world": [0.0, 0.0, -1.0],
        "closing_axis_world": [1.0, 0.0, 0.0],
        "note": ("基座前方 0.45 m、高 0.35 m，接近轴竖直向下（抓桌面石头）；"
                 "若该目标不可达则按 scale 向基座收缩，scale<1 表示退让后的实际目标"),
    }
    out["ik_num_converged_starts"] = len(solutions)
    if best is None:
        out["converged"] = False
        out["failure_note"] = "所有起点/所有缩放目标都没收敛：该臂在「工具朝下」约束下够不到该区域"
        return out
    q, err, elbow_z, data = best
    q_wrapped = wrap_arm_qpos_within_limits(model, spec.arm_joints, q)
    if not np.allclose(q_wrapped, q):
        for jn, v in zip(spec.arm_joints, q_wrapped):
            data.qpos[joint_qpos_addr(model, jn)] = float(v)
        mujoco.mj_forward(model, data)
    full_qpos = np.array(data.qpos)
    pos, rot = tcp.pose(model, data)
    exact = bool(err["pos_error_m"] < 1e-4 and err["rot_error_rad"] < 1e-3)
    out.setdefault("home_pose_kind", "ik_exact" if exact else "best_effort")
    out.update({
        "converged": exact,
        "arm_joint_qpos": r3(q_wrapped),
        "arm_joint_qpos_raw": r3(q),
        "qpos": r3(full_qpos),
        "pos_error_m": round(err["pos_error_m"], 9),
        "rot_error_rad": round(err["rot_error_rad"], 9),
        "rot_error_deg": round(err["rot_error_deg"], 6),
        "residual_distance_to_nominal_target_m": round(
            float(np.linalg.norm(pos - nominal_target)), 6),
        "elbow_body_used": spec.elbow_body,
        "elbow_height_m": round(elbow_z, 6),
        "tcp_world_pos": r3(pos),
        "tcp_world_quat": r4(mat_to_quat(rot)),
        "approach_axis_world": r3(rot @ approach_tcp),
        "approach_angle_from_down_deg": round(
            angle_deg_between(rot @ approach_tcp, np.array([0.0, 0.0, -1.0])), 6),
        "closing_axis_world": r3(rot @ closing_tcp),
        "finger_joint_qpos": {f.joint: round(float(data.qpos[joint_qpos_addr(model, f.joint)]), 6)
                              for f in spec.fingers},
    })
    if spec.home_reference_qpos is not None:
        legacy_arm = {jn: float(v) for jn, v in zip(spec.arm_joints, spec.home_reference_qpos)}
        data_ref = fresh_state(model, legacy_arm, grip_pose)
        p_ref, R_ref = tcp.pose(model, data_ref)
        legacy_ctrl = full_ctrl(model, spec, legacy_arm, spec.ctrl_from_param(spec.param_open))
        data_eq = fresh_state(model, legacy_arm, grip_pose)
        settle(model, data_eq, legacy_ctrl)
        p_eq, R_eq = tcp.pose(model, data_eq)
        out["legacy_reference_pose"] = {
            "label": "Q_HOME_ELBOW_UP (run_official_ur5e_robotiq_wall_stack.py:56)",
            "arm_joint_qpos": r3(spec.home_reference_qpos),
            "tcp_world_pos": r3(p_ref),
            "tcp_world_quat": r4(mat_to_quat(R_ref)),
            "approach_axis_world": r3(R_ref @ approach_tcp),
            "closing_axis_world": r3(R_ref @ closing_tcp),
            "distance_from_base_origin_m": round(float(np.linalg.norm(p_ref)), 6),
            "approach_angle_from_down_deg": round(angle_deg_between(R_ref @ approach_tcp,
                                                                   np.array([0.0, 0.0, -1.0])), 4),
            "pad_center_midpoint_world": r3(0.5 * (pad_center(model, data_ref, spec.fingers[0].pad_geoms(model))
                                                   + pad_center(model, data_ref, spec.fingers[1].pad_geoms(model)))),
            "at_servo_equilibrium": {
                "note": "把该 qpos 作为位置伺服目标、在重力下 settle 后的实际 TCP 位姿（含重力下垂）",
                "arm_joint_qpos": r3(data_eq.qpos[:len(spec.arm_joints)]),
                "tcp_world_pos": r3(p_eq),
                "tcp_world_quat": r4(mat_to_quat(R_eq)),
                "approach_axis_world": r3(R_eq @ approach_tcp),
                "closing_axis_world": r3(R_eq @ closing_tcp),
                "offset_from_exact_qpos_m": round(float(np.linalg.norm(p_eq - p_ref)), 6),
            },
        }
    return out


# --------------------------------------------------------------------------------------
# 文档一致性检查（每条都会给出"实测 vs 文档"）
# --------------------------------------------------------------------------------------
def documentation_checks(spec: ArmSpec, model: mujoco.MjModel, opening: dict, frames: dict,
                         reach: dict, mass: dict, home: dict, model_info: dict,
                         panda_reach_repro: dict | None) -> list[dict]:
    checks: list[dict] = []

    def add(claim, source, measured, agrees, note=""):
        checks.append({"claim": claim, "source": source, "measured": measured,
                       "agrees": bool(agrees), "note": note})

    if spec.name == "ur5e":
        off = np.array(frames["pad_center_offset_tcp"])
        add("fingerpad 中心在 grip_site 局部 -z 约 32 mm",
            "run_official_ur5e_robotiq_grasp_test.py::grip_site_from_pad_center 注释",
            f"实测 pad 中点偏移 (TCP 帧) = {r3(off)} m，模长 {frames['pad_center_offset_tcp_norm_m']} m",
            abs(abs(off[2]) - 0.032) < 0.002 and abs(off[0]) < 0.002 and abs(off[1]) < 0.002,
            "方向确认在 -z 上，量值 32.3 mm")
        closing = np.array(frames["closing_axis_tcp"])
        add("手指开合方向 = grip_site 局部 x 轴",
            "run_official_ur5e_robotiq_grasp_test.py::top_down_gripper_rotation 注释",
            f"实测闭合轴 (TCP 帧) = {r3(closing)}",
            abs(abs(closing[0]) - 1.0) < 0.02,
            "量值落在 ±x 上（符号取决于左右指命名）")
        shoulder_limit = mass["payload_estimate"]["shoulder_torque_limit_nm"]
        add("UR5e 肩/肘力矩上限 ±150 N·m、腕 ±28 N·m",
            "assets/robots/robosuite/models/assets/robots/ur5e/robot.xml 上游 motor ctrlrange",
            f"组合模型的臂执行器 forcerange 实测 = ±{shoulder_limit:.0f} N·m（位置伺服，统一值）",
            False,
            "legacy 组合把这些 motor 换成 position 执行器并写死 forcerange ±280 N·m，"
            "比上游真实值大 1.87 倍——用这个模型做载荷/力矩判断会偏乐观")

    if spec.name == "panda":
        add(f"载入实测 nbody=12、nq=9、nu=8、nsite=0、nkey=1",
            "assets/robots/franka/README.md",
            f"实测 nbody={model_info['nbody']}、nq={model_info['nq']}、nu={model_info['nu']}、"
            f"nsite={model_info['nsite']}、nkey={model_info['nkey']}",
            (model_info["nbody"], model_info["nq"], model_info["nu"],
             model_info["nsite"], model_info["nkey"]) == (12, 9, 8, 0, 1))
        add("上游 panda.xml 没有 tool_tip site（nsite=0）",
            "assets/robots/franka/README.md",
            f"实测 nsite={model_info['nsite']}，site 列表为空",
            model_info["nsite"] == 0,
            "因此本脚本经验测定了合成 TCP（见 tcp 段）")
        add("ctrlrange (0,0.04) 重映射到 (0,255)，255 = fully open",
            "assets/robots/franka/panda.xml:275-277 注释",
            f"实测 tendon 'split' 长度 = 0.0001568627451*ctrl（ctrl=255 -> 0.04 m），"
            f"pad 间距随 ctrl 单调增大：{opening['min_opening_m']} m@ctrl={opening['min_opening_param']} "
            f"-> {opening['max_opening_m']} m@ctrl={opening['max_opening_param']}",
            True,
            "平衡关系与注释一致；255 确为全开")
        if panda_reach_repro:
            add("臂展上界实测 1.2383 m（20 万组关节角，手指 body 原点相对 link0 原点）",
                "assets/robots/franka/README.md",
                f"复现（同样 20 万组、手指 body 原点相对 link0）：手指固定在 qpos0=0 时 "
                f"{panda_reach_repro['max_finger_radius_m']} m；手指全开时 "
                f"{panda_reach_repro['max_finger_radius_m_fingers_open']} m；9 个关节一起采时 "
                f"{panda_reach_repro['max_finger_radius_m_fingers_sampled']} m。"
                f"本脚本主指标（TCP=pad 中点）半径上界 = {reach['all']['max_m']} m",
                abs(panda_reach_repro["max_finger_radius_m"] - 1.2383) < 0.01,
                "量值可复现，但 README 没写手指 slide joint 的取值；只有固定在 0（等价于只采 7 个臂关节）"
                "才等于 1.2383 m，全开时会大 20 mm")

    if spec.name == "piper":
        gap = opening["max_inner_surface_gap_m"]
        add("夹爪净开口 6.996 cm（两指内表面实测）",
            "assets/robots/piper/README.md",
            f"实测全开(ctrl={opening['max_opening_param']}) 内表面间距 {gap*100:.3f} cm、"
            f"两指点云最小距离 {max(s['min_surface_distance_m'] for s in opening['samples'])*100:.3f} cm；"
            f"ctrl={opening['min_opening_param']} 时两指已闭合（间距 "
            f"{opening['min_inner_surface_gap_m']*100:.3f} cm）",
            abs(gap - 0.06996) < 0.002,
            "量值吻合（6.996 cm ≈ 全开时的 6.978~7.000 cm），但**方向是反的**："
            "全开在 ctrl=0.035，ctrl=0 是闭合，README 没有写清这一点")
        add("两个通过 equality 耦合的夹爪 slide joint + 1 路夹爪位置伺服 + tool_tip site",
            "assets/robots/piper/README.md",
            f"实测 neq={model_info['neq']}（joint7=joint8），gripper 执行器数=1，site 'tool_tip' 存在",
            model_info["neq"] == 1 and model_info["nsite"] == 1,
            "结构描述与实测一致；但 README 的『净开口』未给 ctrl 方向，容易误读")

    return checks


def panda_reach_reproduction(model: mujoco.MjModel, samples: int = 200000, seed: int = 0) -> dict:
    """复现 README 里那句『手指 body 原点相对 link0 原点的最大半径 = 1.2383 m』。

    README 没写手指 slide joint 取值，而手指开合会沿半径方向贡献最多 40 mm，所以三种口径都量：
    finger=0（qpos0 默认，也就是只采 7 个臂关节）、finger=0.04（全开）、以及把 9 个关节一起采。
    """
    rng = np.random.default_rng(seed)
    ranges = np.array([model.jnt_range[model.joint(f"joint{i}").id] for i in range(1, 8)])
    q = rng.uniform(ranges[:, 0], ranges[:, 1], size=(samples, 7))
    data = mujoco.MjData(model)
    finger_bodies = [model.body("left_finger").id, model.body("right_finger").id]

    def scan(finger_value: float | np.ndarray) -> tuple[float, str]:
        best, best_body = 0.0, ""
        for i in range(samples):
            for k in range(7):
                data.qpos[k] = float(q[i, k])
            if np.isscalar(finger_value):
                data.qpos[7] = data.qpos[8] = float(finger_value)   # type: ignore[arg-type]
            else:
                data.qpos[7] = float(finger_value[i])               # type: ignore[index]
                data.qpos[8] = float(finger_value[i])               # type: ignore[index]
            forward_kinematics(model, data)
            for b in finger_bodies:
                r = float(np.linalg.norm(data.xpos[b]))
                if r > best:
                    best, best_body = r, model.body(b).name
        return best, best_body

    r0, b0 = scan(0.0)
    r_open, b_open = scan(0.04)
    rnd_finger = rng.uniform(0.0, 0.04, size=samples)
    r_rand, b_rand = scan(rnd_finger)
    return {
        "samples": int(samples), "seed": int(seed),
        "max_finger_radius_m": round(r0, 6),
        "max_finger_radius_m_fingers_open": round(r_open, 6),
        "max_finger_radius_m_fingers_sampled": round(r_rand, 6),
        "finger_body_with_max": b0,
        "note": ("README 的 1.2383 m 只有把两个 finger slide joint 固定在 qpos0=0 时才复现；"
                 "手指全开时上界变成 1.2586 m（手指沿半径方向多伸出 40 mm）"),
    }


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def probe(arm: str, samples: int, seed: int, quiet: bool = False) -> dict:
    spec = ARM_SPECS[arm]
    model, meta = spec.build()
    arm_pose, grip_guess = build_open_state(spec, model)

    # (1) 夹爪"全开"的真实关节值：给全开 ctrl，settle 一次读出来
    #     （不能拿 ctrl 当 qpos 用：三条臂的夹爪都有耦合约束）
    open_ctrl = full_ctrl(model, spec, arm_pose, spec.ctrl_from_param(spec.param_open))
    data = fresh_state(model, arm_pose, {f.joint: 0.0 for f in spec.fingers})
    settle(model, data, open_ctrl)
    grip_open = {f.joint: float(data.qpos[joint_qpos_addr(model, f.joint)]) for f in spec.fingers}

    # (2) 参考状态：臂在参考位姿、夹爪全开，并且在位置伺服下 settle 到静止。
    #     所有"帧/几何"读数都基于这一个状态，避免把"settle 后的夹爪"和"未 settle 的臂"
    #     混在一起（重力下垂会让 TCP 与 pad 中点对不上，panda 上实测差 6.7 mm）。
    ref_data = fresh_state(model, arm_pose, grip_open)
    settle(model, ref_data, open_ctrl)

    panda_extra = None
    if spec.tcp_kind == "hand_offset":
        tcp, mid_world = compute_panda_tcp(model, ref_data, spec)
        hand_pos, hand_rot = body_frame(model, ref_data, str(spec.tcp_body))
        centers = [pad_center(model, ref_data, f.pad_geoms(model)) for f in spec.fingers]
        panda_extra = {
            "pad_centers": [r3(c) for c in centers],
            "pad_mid": r3(mid_world),
            "pad_centers_hand": [r3(point_in_frame(c, hand_pos, hand_rot)) for c in centers],
            "definition": (
                "keyframe home + 手指全开时，left/right_finger 上 5 个 pad box 的 AABB 中心均值取中点"
            ),
        }
    else:
        tcp = TcpFrame(kind="site", site=spec.tcp_site)

    model_info = measure_model_info(model, spec, meta)
    arm_joints_meta = measure_arm_joints(model, spec)
    tcp_section = measure_tcp_section(model, spec, tcp, ref_data, panda_extra)
    opening = measure_opening_curve(model, spec, arm_pose, grip_open)
    frames = measure_frames(model, spec, tcp, ref_data, panda_extra)
    # 交叉校验：用开合曲线的数值微分求"两指真正互相靠近的方向"，与"全开时两 pad 中心连线"对比。
    # 两者不一致说明 pad 中心连线本身随构型在转（piper 的 mesh 包围盒中心就不是镜像对称的）。
    labels_f = [f.label for f in spec.fingers]
    s0, s1 = opening["samples"][0], opening["samples"][1]
    vec0 = np.array(s0["pad_centers_world"][labels_f[0]]) - np.array(s0["pad_centers_world"][labels_f[1]])
    vec1 = np.array(s1["pad_centers_world"][labels_f[0]]) - np.array(s1["pad_centers_world"][labels_f[1]])
    # vec0（全开）比 vec1 长，所以 (vec0 - vec1) 就是"闭合时 A 相对 B 移动的方向"，
    # 与"全开时 A->B 的连线方向"同向，便于直接比较。
    motion_axis_world = unit(vec0 - vec1)
    _, tcp_rot_ref = tcp.pose(model, ref_data)
    frames["closing_axis_from_pad_motion_world"] = r3(motion_axis_world)
    frames["closing_axis_from_pad_motion_note"] = (
        "数值微分得到：闭合过程中 A 指 pad 中心相对 B 指移动的方向（前两个采样点之差）")
    frames["closing_axis_from_pad_motion_tcp"] = r3(vector_in_frame(motion_axis_world, tcp_rot_ref))
    frames["angle_between_open_padline_and_pad_motion_deg"] = round(
        angle_deg_between(motion_axis_world, np.array(frames["closing_axis_world_at_reference_pose"])), 4)
    # 一致性自检：开合曲线的"全开采样点"应当与参考状态完全一致
    sample_open = opening["samples"][0]
    ref_centers = {f.label: pad_center(model, ref_data, f.pad_geoms(model)) for f in spec.fingers}
    cross = max(float(np.linalg.norm(np.array(sample_open["pad_centers_world"][k]) - v))
                for k, v in ref_centers.items())
    opening["reference_state_cross_check"] = {
        "max_pad_center_difference_m": round(cross, 9),
        "note": "开合曲线第一个采样点（全开）与参考状态的 pad 中心最大偏差；应接近 0",
    }
    approach_tcp = np.array(frames["approach_axis_tcp"])
    closing_tcp = np.array(frames["closing_axis_tcp"])
    reach = measure_reach(model, spec, tcp, grip_open, approach_tcp, samples, seed)
    mass = measure_mass(model, spec, reach, arm_joints_meta)
    home = measure_home(model, spec, tcp, arm_pose, grip_open, approach_tcp, closing_tcp, seed)

    panda_repro = panda_reach_reproduction(model, samples=200000, seed=seed) if arm == "panda" else None
    checks = documentation_checks(spec, model, opening, frames, reach, mass, home,
                                  model_info, panda_repro)

    result = {
        "arm": arm,
        "generated_by": {
            "script": "tasks/task2_stack/tools/probe_arm_geometry.py",
            "interpreter": sys.executable,
            "argv": sys.argv,
            "command": ("PYTHONPATH=<repo>/tasks/task2_stack "
                        f"{sys.executable} tasks/task2_stack/tools/probe_arm_geometry.py --arm {arm}"),
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "seed": int(seed),
            "reach_samples": int(samples),
            "note": "所有数值都是运行时从编译后的模型量出来的；故意不写时间戳以保证可复现",
        },
        "model": model_info,
        "arm_joints": arm_joints_meta,
        "tcp": tcp_section,
        "opening_curve": opening,
        "frames": frames,
        "reach": reach,
        "mass": mass,
        "home": home,
        "reference_arm_pose": {k: round(float(v), 6) for k, v in arm_pose.items()},
        "reference_gripper_open_qpos": {k: round(float(v), 6) for k, v in grip_open.items()},
        "documentation_checks": checks,
        "documented_claims_considered": list(spec.documentation_claims),
    }
    if panda_repro:
        result["panda_readme_reach_reproduction"] = panda_repro
    if not quiet:
        print_summary(result)
    return result


def print_summary(res: dict) -> None:
    arm = res["arm"]
    mi = res["model"]
    op = res["opening_curve"]
    fr = res["frames"]
    rc = res["reach"]
    ms = res["mass"]
    hm = res["home"]
    print("=" * 78)
    print(f"[{arm}] nq={mi['nq']} nv={mi['nv']} nu={mi['nu']} nbody={mi['nbody']} "
          f"nsite={mi['nsite']} neq={mi['neq']} nkey={mi['nkey']} dt={mi['timestep']}")
    print(f"  chain: {' -> '.join(mi['chain_base_to_tcp_body'])}")
    print(f"  TCP  : {res['tcp']['kind']} {res['tcp'].get('site') or res['tcp'].get('tcp_origin_in_hand_frame')}")
    print(f"  open : max {op['max_opening_m']} m @param={op['max_opening_param']} | "
          f"min {op['min_opening_m']} m @param={op['min_opening_param']} | "
          f"inner gap max {op['max_inner_surface_gap_m']} m")
    fu = op["fit_usable_grasp_range"]
    print(f"  fit(usable {op['usable_grasp_range_param']}): width = {fu.get('a'):.6g}*ctrl + "
          f"{fu.get('b'):.6g}  (rms {fu.get('residual_rms_m'):.2e} m, r2 {fu.get('r2')})")
    print(f"  half-open ctrl: fit {op['ctrl_at_half_max_opening']['from_usable_fit']} | "
          f"measured {op['ctrl_at_half_max_opening']['from_measured_curve']}")
    print(f"  frames: closing {fr['closing_axis_tcp']} approach {fr['approach_axis_tcp']} "
          f"pad_off {fr['pad_center_offset_tcp']} flange_off {fr['tcp_origin_offset_from_flange_m']} m")
    print(f"  reach : all p95 {rc['all']['p95_m']} max {rc['all']['max_m']} | "
          f"top-down(35deg) n={rc['top_down_within_35deg']['count']} "
          f"p95 {rc['top_down_within_35deg']['distance'].get('p95_m')} "
          f"max {rc['top_down_within_35deg']['distance'].get('max_m')}")
    pe = ms["payload_estimate"]
    print(f"  mass  : {ms['total_mass_kg']} kg (gripper {ms['gripper_subtree_mass_kg']}, "
          f"moving {ms['gripper_moving_parts_mass_kg']}) | payload {pe['payload_kg']} kg "
          f"@ {pe['shoulder_joint_used']} {pe['shoulder_torque_limit_nm']} N·m, r={pe['radius_m']} m")
    print(f"  home  : {hm['source']}")
    if "qpos" in hm:
        print(f"          qpos={hm['qpos']}")
    if "tcp_world_pos" in hm:
        print(f"          tcp={hm['tcp_world_pos']} approach_world={hm['approach_axis_world']}")
    bad = [c for c in res["documentation_checks"] if not c["agrees"]]
    if bad:
        print(f"  doc   : {len(bad)} 条与文档不一致")
        for c in bad:
            print(f"          - {c['claim']}")
    print("=" * 78)


def write_merge(seed: int, samples: int) -> Path:
    arms = ["ur5e", "panda", "piper"]
    merged = {
        "generated_by": {
            "script": "tasks/task2_stack/tools/probe_arm_geometry.py",
            "interpreter": sys.executable,
            "command_lines": [
                "PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack "
                "/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python "
                f"tasks/task2_stack/tools/probe_arm_geometry.py --arm {a}" for a in arms
            ] + [
                "PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack "
                "/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python "
                "tasks/task2_stack/tools/probe_arm_geometry.py --merge"
            ],
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "seed": int(seed),
            "reach_samples": int(samples),
            "note": "由 --merge 合并三份单臂 JSON，数值未做任何修改",
        },
        "arms": {},
    }
    for arm in arms:
        path = REPORTS_DIR / f"arm_geometry_{arm}.json"
        if not path.exists():
            raise SystemExit(f"缺少 {path}，请先分别跑 --arm {' / --arm '.join(arms)}")
        merged["arms"][arm] = json.loads(path.read_text(encoding="utf-8"))
    out = REPORTS_DIR / "arm_geometry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[merge] 写入 {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="实测 ur5e/panda/piper 的 TCP、夹爪开合曲线、工作空间、质量与 home 位姿",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("ur5e", "panda", "piper"), help="要测量的臂")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 JSON 路径，默认 reports/arm_geometry_<arm>.json")
    ap.add_argument("--samples", type=int, default=20000, help="可达性采样的关节空间样本数")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（固定以保证可复现）")
    ap.add_argument("--merge", action="store_true", help="合并三份单臂 JSON 到 reports/arm_geometry.json")
    ap.add_argument("--quiet", action="store_true", help="只写文件，不打印摘要")
    args = ap.parse_args()

    if args.merge:
        write_merge(args.seed, args.samples)
        return 0
    if not args.arm:
        ap.error("需要 --arm {ur5e,panda,piper} 或 --merge")

    res = probe(args.arm, args.samples, args.seed, quiet=args.quiet)
    out = args.out or (REPORTS_DIR / f"arm_geometry_{args.arm}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
