"""奖励只读取真实物理状态，不移动石头，不生成机器人动作。

权重放在 policy.rl.reward；目标位置只是固定任务的奖励锚点，不是执行路径。
用势函数变化奖励进展，原地重复保持不能反复领取抓取/放置分数。
"""
import numpy as np


def wall_targets(courses, spacing, thickness):
    return np.asarray([[(i - (count - 1) / 2) * spacing, 0, (level + .5) * thickness]
                       for level, count in enumerate(courses) for i in range(count)])


def dense_terms(executor, targets, initial_z, evaluation, goal_tolerance):
    e = executor
    names = e.stone_names()
    centers = np.asarray([e.stone_center(n) for n in names])
    pad_center = np.mean(e.data.geom_xpos[e._pad_geom_ids[:2]], axis=0)
    # 两侧真实指垫都接触同一石块，才计为夹持；夹爪命令本身不算。
    contacts = [set() for _ in names]
    stone_ids = {e.model.body(n).id: i for i, n in enumerate(names)}
    pads = {g: side for side, g in enumerate(e._pad_geom_ids[:2])}
    for contact in e.data.contact[:e.data.ncon]:
        if contact.dist > .001:
            continue
        for pad, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
            body = int(e.model.geom_bodyid[other])
            if pad in pads and body in stone_ids:
                contacts[stone_ids[body]].add(pads[pad])
    grasped = np.array([len(sides) == 2 for sides in contacts], dtype=float)
    distances = np.linalg.norm(centers - targets, axis=1)
    closeness = np.exp(-distances / .10)
    expected = np.repeat(np.arange(4), [4, 3, 2, 1])
    supported = np.array([evaluation['layers'].get(n, -1) == level and
                          n not in evaluation['unbridged'] for n, level in zip(names, expected)])
    released = np.array([n not in evaluation['touching_robot'] for n in names])
    still = np.array([n not in evaluation['moving'] for n in names])
    placed = supported & released & still & (distances < goal_tolerance)
    remaining = ~placed
    reach = np.exp(-np.linalg.norm(centers - pad_center, axis=1) / .15)
    lifted = np.clip((centers[:, 2] - initial_z) / .08, 0, 1)
    return dict(reach=float(np.max(reach * remaining)),
                grasp=float(np.max(grasped * remaining)),
                lift=float(np.max(grasped * lifted * remaining)),
                transport=float(np.max(grasped * lifted * closeness * remaining)),
                place=float(np.sum(closeness * supported * released * still)),
                placed_count=int(placed.sum()))


def potential(terms, weights):
    return sum(weights[key] * terms[key] for key in ('reach', 'grasp', 'lift', 'transport', 'place'))
