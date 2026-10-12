"""稠密势函数与可修改的阶段奖励；只读取物理，不替策略选择动作。

接触不是抓取成功。抓取/抬升要求同一石头双指受力、脱离外部支撑且保持；
每块石头的阶段奖金每回合只发一次。课程成功与完整十石墙成功分别记录。
"""
import mujoco
import numpy as np

STAGES = {'grasp': 0, 'lift': 0, 'place': 1, 'stack4': 4,
          'stack7': 7, 'stack9': 9, 'stack10': 10}


def wall_targets(courses, spacing, thickness):
    return np.asarray([[(i - (count - 1) / 2) * spacing, 0, (level + .5) * thickness]
                       for level, count in enumerate(courses) for i in range(count)])


def stone_signals(e, targets, initial_z, evaluation, tolerance, contact_force):
    names = e.stone_names()
    centers = np.asarray([e.stone_center(n) for n in names])
    pads = e._pad_geom_ids[:2]
    pad_positions = e.data.geom_xpos[pads]
    pad_center = pad_positions.mean(axis=0)
    bodies = {e.model.body(n).id: i for i, n in enumerate(names)}
    pad_sides = {geom: side for side, geom in enumerate(pads)}
    forces = np.zeros((len(names), 2))
    resting = np.zeros(len(names), dtype=bool)
    force = np.zeros(6)
    for cid, contact in enumerate(e.data.contact[:e.data.ncon]):
        if contact.dist > .001:
            continue
        mujoco.mj_contactForce(e.model, e.data, cid, force)
        if force[0] < contact_force:
            continue
        for stone_geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
            body = int(e.model.geom_bodyid[stone_geom])
            if body not in bodies:
                continue
            index = bodies[body]
            other_body = int(e.model.geom_bodyid[other])
            if other in pad_sides:
                forces[index, pad_sides[other]] += force[0]
            if other_body == 0 or other_body in bodies:
                resting[index] = True
    touches = forces >= contact_force
    grasped = touches.all(axis=1)
    distances = np.linalg.norm(centers - targets, axis=1)
    expected = np.repeat(np.arange(4), [4, 3, 2, 1])
    supported = np.array([evaluation['layers'].get(n, -1) == level and n not in evaluation['unbridged']
                          for n, level in zip(names, expected)])
    released = np.array([n not in evaluation['touching_robot'] for n in names])
    still = np.array([n not in evaluation['moving'] for n in names])
    placed = supported & released & still & (distances < tolerance)
    # 居中反馈先于真实接触出现：夹爪接近石头，且两侧指垫到石心距离相近。
    pad_distances = np.linalg.norm(centers[:, None] - pad_positions[None], axis=2)
    reach = np.exp(-np.linalg.norm(centers - pad_center, axis=1) / .15)
    centered = reach * np.exp(-np.abs(pad_distances[:, 0] - pad_distances[:, 1]) / .025)
    tcp_pos, tcp_rot = e.controller.tcp_pose()
    relative = (centers - tcp_pos) @ tcp_rot
    return dict(centers=centers, forces=forces, grasped=grasped, resting=resting,
                touch=touches.mean(axis=1), reach=reach, center=centered,
                height=centers[:, 2] - initial_z, relative=relative,
                closeness=np.exp(-distances / .10), placed=placed,
                place=np.exp(-distances / .10) * supported * released * still)


class CurriculumReward:
    def __init__(self, config, count):
        self.cfg = config
        self.stage = config['stage']
        self.required = STAGES[self.stage]
        self.grasp_time = np.zeros(count)
        self.lift_time = np.zeros(count)
        self.place_time = np.zeros(count)
        self.credited = {key: np.zeros(count, dtype=bool) for key in ('grasp', 'lift', 'place')}
        self.previous_relative = None

    def update(self, signals, seconds):
        c, s = self.cfg, signals
        stable_relative = np.ones(len(self.grasp_time), dtype=bool)
        if self.previous_relative is not None:
            stable_relative = np.linalg.norm(s['relative'] - self.previous_relative, axis=1) <= c['relative_tolerance_m']
        self.previous_relative = s['relative'].copy()
        retained = s['grasped'] & ~s['resting'] & stable_relative
        grasp_ok = retained & (s['height'] >= c['grasp_height_m'])
        lift_ok = retained & (s['height'] >= c['lift_height_m'])
        self.grasp_time = np.where(grasp_ok, self.grasp_time + seconds, 0.)
        self.lift_time = np.where(lift_ok, self.lift_time + seconds, 0.)
        placed = s['placed'] & self.credited['grasp']
        self.place_time = np.where(placed, self.place_time + seconds, 0.)
        achieved = dict(grasp=self.grasp_time >= c['grasp_hold_seconds'] - 1e-9,
                        lift=self.lift_time >= c['lift_hold_seconds'] - 1e-9,
                        place=self.place_time >= c['place_hold_seconds'] - 1e-9)
        # 抓取课不奖励搬运/放置；放置及堆叠课要求该石头本回合确实被抓起过。
        achieved['place'] &= self.credited['grasp'] | achieved['grasp']
        bonus, events = 0., {}
        for key in ('grasp', 'lift', 'place'):
            enabled = key == 'grasp' or (key == 'lift' and self.stage != 'grasp') or self.required > 0
            fresh = achieved[key] & ~self.credited[key]
            events[key] = np.flatnonzero(fresh).tolist() if enabled else []
            if enabled:
                bonus += len(events[key]) * c[key + '_bonus']
                self.credited[key] |= achieved[key]
        if self.stage == 'grasp':
            stage_ok = bool(achieved['grasp'].any())
        elif self.stage == 'lift':
            stage_ok = bool(achieved['lift'].any())
        elif self.stage == 'place':
            stage_ok = bool(achieved['place'][:4].any())
        else:
            stage_ok = bool(achieved['place'][:self.required].all())
        # 完整十石墙仍以共用物理 evaluator 的稳定判据为准，由 env 决定。
        active = ~placed
        if self.required > 0:
            limit = 4 if self.stage == 'place' else self.required
            active[np.arange(len(active)) >= limit] = False
            # 先完成当前层，再给上层搬运反馈，避免奖励从悬空目标直接跳级。
            level_ends = (4, 7, 9, 10)
            for end in level_ends:
                if not placed[:min(end, limit)].all():
                    active[np.arange(len(active)) >= end] = False
                    break
        if self.stage in ('grasp', 'lift'):
            active = np.ones(len(active), dtype=bool)
        held = np.clip(self.grasp_time / c['grasp_hold_seconds'], 0., 1.)
        lift = np.clip(s['height'] / c['lift_height_m'], 0., 1.) * retained
        terms = dict(reach=float(np.max(s['reach'] * active)),
                     center=float(np.max(s['center'] * active)),
                     touch=float(np.max(s['touch'] * active)),
                     grasp=float(np.max(s['grasped'] * active)),
                     hold=float(np.max(held * active)),
                     lift=float(np.max(lift * active)),
                     transport=float(np.max(lift * s['closeness'] * active)) if self.required else 0.,
                     place=float(np.sum((s['place'] * self.credited['grasp'])[:(4 if self.stage == 'place' else self.required)])) if self.required else 0.,
                     placed_count=int(placed.sum()), geometric_placed_count=int(s['placed'].sum()))
        return terms, bonus, stage_ok, events


def potential(terms, weights):
    return sum(weights[key] * terms[key] for key in ('reach', 'center', 'touch', 'grasp', 'hold', 'lift', 'transport', 'place'))
