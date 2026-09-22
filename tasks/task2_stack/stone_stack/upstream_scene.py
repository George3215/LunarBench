"""复用上游 UR5e 与石头，提供可达料区并增加观测相机。

上游版本与差异记录见 docs/UPSTREAM_DIRECT.md。默认生成十块 paper 石头；
提供 upstream_report 时复用该报告的十块石头及初始 yaw，但不把放置计划交给策略。
"""
from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts import run_official_ur5e_robotiq_wall_stack as upstream
from .robots.profile import get_profile
from .robots.scene import BuiltScene, SceneOptions, Slot, Workcell
from .tools.stones import load_stones, load_report_stones


def build_upstream(config):
    robot, scene = config["robot"], config["scene"]
    if robot["arm"] != "ur5e" or robot["courses"] != [4, 3, 2, 1] or robot["stones"] != 10:
        raise ValueError("upstream scene requires ur5e, 10 stones and courses [4,3,2,1]")
    if robot["stone_scale"] not in (None, 1.0):
        raise ValueError("upstream layout preserves original stone scale (1.0)")
    report_path = scene["upstream_report"]
    if report_path:
        stones, entries = load_report_stones(report_path)
        if len(stones) != 10 or [sum(e["course"] == i for e in entries) for i in range(4)] != [4, 3, 2, 1]:
            raise ValueError("upstream_report must contain a complete ten-stone 4+3+2+1 plan")
    else:
        stones = load_stones(config, scale=1.0, count=10)
        entries = [dict(name=s.name, quat=[1, 0, 0, 0]) for s in stones]
    poses = {s.name: upstream.initial_supply_pose(i, entry, s)
             for i, (s, entry) in enumerate(zip(stones, entries))}
    if scene["supply_layout"] == "reachable":
        if len(scene["supply_positions_xy"]) != 10:
            raise ValueError("supply_positions_xy must contain exactly ten positions")
        for stone, xy in zip(stones, scene["supply_positions_xy"]):
            poses[stone.name][0][:2] = xy
    profile = replace(get_profile("ur5e"), home_qpos=upstream.Q_HOME_ELBOW_UP.copy(),
                      base_pos=tuple(upstream.ROBOT_BASE_POS), stone_scale=1.0)
    # 不调用 build_workcell：它会重排料区、缩放工作区或削减层数。
    slots = tuple(Slot(course, i, (i - (n - 1) / 2) * 0.20)
                  for course, n in enumerate([4, 3, 2, 1]) for i in range(n))
    workcell = Workcell(profile=profile, base_pos=upstream.ROBOT_BASE_POS.copy(),
                       wall_center=np.zeros(3), wall_axis=np.array([1., 0, 0]),
                       courses=(4, 3, 2, 1), requested_courses=(4, 3, 2, 1),
                       slots=slots, slot_spacing=0.20, stone_scale=1.0,
                       mean_thickness=float(np.mean([s.thickness for s in stones])),
                       mean_length=float(np.mean([s.length for s in stones])),
                       pocket_radius=0.0, staging_poses=poses)
    source_xml = upstream.build_wall_stack_scene(stones, poses)
    root = ET.fromstring(source_xml)
    # 相机节点不参与动力学；保持上游机器人、夹爪、石头、接触与步长原样。
    world = root.find("worldbody")
    ET.SubElement(world, "camera", name="top", pos="0 0.18 1.65", xyaxes="1 0 0 0 1 0", fovy="58")
    # 正前方：相机位于世界 -Y 侧，水平沿 +Y 看向墙面；画面右为 +X、上为 +Z。
    # 光轴与墙面 XZ 垂直（90°），不再复制斜向 overview。位置在 YAML 中修改。
    front_pos = workcell.wall_center + np.asarray(scene["front_camera_offset"])
    ET.SubElement(world, "camera", name="front", pos=upstream._fmt(front_pos),
                  xyaxes="1 0 0 0 0 1", fovy="58")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    options = SceneOptions(camera_width=scene["camera_width"], camera_height=scene["camera_height"],
                           timestep=float(model.opt.timestep))
    built = BuiltScene(model=model, data=data, workcell=workcell, options=options, xml=xml,
                       wrist_mount=None, wrist_in_tcp=None,
                       pad_geoms=profile.gripper.pad_geoms, pad_thickness_m=profile.gripper.pad_thickness_m)
    return built, workcell, profile
