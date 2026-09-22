"""TASK2 共用物理成功判据；策略和奖励都不能替代这里的最终检查。"""
import numpy as np


def support_layers(names, grounded, supports):
    """地面为第 0 层；上层必须接触已有下层。无支撑的石头不分配层。"""
    layers = {name: 0 for name in grounded}
    for _ in names:
        for name in names:
            lower = supports[name]
            if name not in layers and lower and all(s in layers for s in lower):
                layers[name] = 1 + max(layers[s] for s in lower)
    return layers


def evaluate_wall(executor):
    e = executor
    names = e.stone_names()
    body_names = {e.model.body(name).id: name for name in names}
    centers = {name: e.stone_center(name) for name in names}
    grounded, robot_contact = set(), set()
    supports = {name: set() for name in names}
    for contact in e.data.contact[:e.data.ncon]:
        if contact.dist > 0.001:
            continue
        a, b = (int(e.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        for stone_id, other_id in ((a, b), (b, a)):
            if stone_id not in body_names:
                continue
            name = body_names[stone_id]
            if other_id == 0:
                grounded.add(name)
            elif other_id in body_names:
                other = body_names[other_id]
                # 过滤同层横向接触；要求接触法向有明显竖直分量且支撑物较低。
                if centers[name][2] > centers[other][2] + 0.01 and abs(contact.frame[2]) > 0.3:
                    supports[name].add(other)
            else:
                robot_contact.add(name)
    layers = support_layers(names, grounded, supports)
    counts = [sum(level == i for level in layers.values()) for i in range(4)]
    wall = e.workcell.wall_center
    outside = [n for n in names if abs(centers[n][0] - wall[0]) > 0.45 or abs(centers[n][1] - wall[1]) > 0.16]
    moving = []
    for name in names:
        dof = int(e.model.jnt_dofadr[e.model.joint(f"{name}_free").id])
        velocity = e.data.qvel[dof:dof + 6]
        if np.linalg.norm(velocity[:3]) > 0.015 or np.linalg.norm(velocity[3:]) > 0.15:
            moving.append(name)
    # 上层应跨接两块下层，避免把多个独立单柱误判为干砌墙。
    unbridged = [n for n, level in layers.items() if level > 0 and
                 len([s for s in supports[n] if layers.get(s) == level - 1]) < 2]
    success = (len(names) == 10 and len(layers) == 10 and counts == [4, 3, 2, 1]
               and not outside and not moving and not robot_contact and not unbridged)
    return dict(success=bool(success), criterion="task2_contact_wall_v1",
                expected_courses=[4, 3, 2, 1], observed_courses=counts,
                stone_count=len(names), unsupported=sorted(set(names) - layers.keys()),
                outside_wall=outside, moving=moving, touching_robot=sorted(robot_contact),
                unbridged=unbridged, layers=layers)
