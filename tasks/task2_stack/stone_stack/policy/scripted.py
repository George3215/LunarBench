"""脚本策略：不依赖任何模型，纯几何 + 真值的确定性基线。

高层（`ScriptedHighLevel`）
--------------------------
回答"下一块放哪"。规则：
- 槽位按层、按序号顺序填（先铺满第一层再上第二层）；
- 每块石头放在当前槽位上方已有石头的顶面（用真值观测算支撑高度，不是查计划表）；
- 石头按体积从大到小优先用于低层（下层承重），体积相同则取离基座近的。

这条规则**不比旧 TASK2 的稳定性搜索规划器**：它不做支撑面积/扰动存活评估，
只保证"下层先放、大石头先放"。它的价值是确定性、快、以及给 VLM 策略提供候选与兜底。

低层（`ScriptedLowLevel`）
-------------------------
回答"末端下一步去哪"。每个 Goal 展开成一串 TCP 路径点（接近 → 下降 → 夹 → 抬 →
转运 → 下降 → 放 → 退），策略只在"路径点是否到达"这一层闭环：
执行器每次问它要下一个 TCP 目标，它按实测 TCP 误差决定是推进到下一个点还是继续当前点。
这样 IK、接触、碰撞开关仍全部留在执行器里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .base import (
    Candidate,
    Command,
    Decision,
    Goal,
    Observation,
    PlanSlot,
    StoneState,
)


# --------------------------------------------------------------------------------------
# 高层
# --------------------------------------------------------------------------------------


@dataclass
class ScriptedHighLevel:
    """先铺满低层、再往上层放；大石头优先放低层。"""

    name: str = "scripted"
    #: 已有石头与槽位的 x 距离小于这个值就算"这个槽被占了"
    slot_tolerance_m: float = 0.06
    #: 石头在下层时的最大长度倍数（超过槽间距就换一块）
    max_length_ratio: float = 1.15
    placed_positions: list[np.ndarray] = field(default_factory=list)
    placed_sizes: list[np.ndarray] = field(default_factory=list)

    def reset(self, context: dict[str, Any]) -> None:
        self.placed_positions = []
        self.placed_sizes = []
        for stone in context.get("placed_stones", ()):  # 续跑时把已放好的石头带进来
            self.placed_positions.append(np.asarray(stone["pos"], dtype=float))
            self.placed_sizes.append(np.asarray(stone["size"], dtype=float))

    def decide(self, observation: Observation, candidates: Sequence[Candidate]) -> Decision:
        if not candidates:
            return Decision(action="stop", source=self.name, reason="没有候选（石头或槽位用尽）")

        # 1. 选槽位：层序 → 序号；跳过已经被占的
        ordered_slots = sorted({candidate.slot.key: candidate.slot for candidate in candidates}.values(),
                               key=lambda slot: (slot.course, slot.slot_index))
        chosen_slot = None
        for slot in ordered_slots:
            if self._slot_is_free(slot, observation):
                chosen_slot = slot
                break
        if chosen_slot is None:
            return Decision(action="stop", source=self.name, reason="所有槽位都已占用")

        # 2. 选石头：该槽位可用石头里，长度放得下、体积最大、离基座最近
        usable = [c for c in candidates if c.slot.key == chosen_slot.key and c.stone.graspable]
        if not usable:
            usable = [c for c in candidates if c.slot.key == chosen_slot.key]
        if not usable:
            return Decision(action="stop", source=self.name, reason=f"槽位 {chosen_slot.key} 没有可用石头")
        spacing = self._slot_spacing(usable)
        fitting = [c for c in usable if float(np.max(c.stone.size)) <= spacing * self.max_length_ratio] or usable
        # 先看抓取质量（规划给出的连续分数），再看体积（下层放大石头），最后看距离。
        # 只按体积挑会挑到"带宽刚好但侧面是楔形"的石头——那正是夹不住的那一类。
        chosen = max(
            fitting,
            key=lambda c: (
                round(float(c.stone.grasp_quality), 2),
                float(np.prod(c.stone.size)),
                -float(np.linalg.norm(c.stone.pos[:2])),
            ),
        )

        # 3. 目标高度：该槽位下面已有的石头顶面；没有支撑就落在台面上
        support_top = self._support_top(chosen_slot, chosen.stone, observation)
        target_pos = np.array(
            [chosen.slot.target_pos[0], chosen.slot.target_pos[1], support_top + 0.5 * float(chosen.stone.size[2])],
            dtype=float,
        )
        target_quat = np.asarray(chosen.slot.target_quat, dtype=float)
        return Decision(
            action="place",
            stone=chosen.stone.name,
            slot=chosen.slot.key,
            target_pos=target_pos,
            target_quat=target_quat,
            source=self.name,
            reason=(
                f"course={chosen.slot.course} slot={chosen.slot.slot_index} "
                f"support_top={support_top:.3f} volume={float(np.prod(chosen.stone.size)) * 1e6:.0f}cm3"
            ),
        )

    # ------------------------------------------------------------------ 内部

    def _slot_spacing(self, candidates: Sequence[Candidate]) -> float:
        xs = sorted({float(c.slot.target_pos[0]) for c in candidates})
        if len(xs) < 2:
            return 0.2
        diffs = np.diff(xs)
        return float(np.min(diffs[diffs > 1e-6])) if np.any(diffs > 1e-6) else 0.2

    def _slot_is_free(self, slot: PlanSlot, observation: Observation) -> bool:
        for stone in observation.stones:
            if float(np.linalg.norm(stone.pos[:2] - np.asarray(slot.target_pos[:2]))) < self.slot_tolerance_m:
                return False
        return True

    def _support_top(self, slot: PlanSlot, stone: StoneState, observation: Observation) -> float:
        """槽位下方石头的最高顶面（世界 z）。没有支撑返回 0（台面）。"""
        top = 0.0
        radius = 0.5 * float(max(stone.size)) + 0.02
        for other in observation.stones:
            if other.name == stone.name:
                continue
            if float(np.linalg.norm(other.pos[:2] - np.asarray(slot.target_pos[:2]))) > radius:
                continue
            top = max(top, float(other.pos[2] + 0.5 * float(other.size[2])))
        return top


# --------------------------------------------------------------------------------------
# 低层
# --------------------------------------------------------------------------------------


@dataclass
class _Waypoint:
    pos: np.ndarray
    rot: np.ndarray
    grip_width: float | None
    label: str
    #: 只做纯平移（见 Command.pure_translation）
    pure_translation: bool = False
    #: 到位判据放宽到 8 mm：位置伺服在重力/接触下的稳态误差就是 5–12 mm，
    #: 卡 2 mm 会让状态机永远停在"下降"这一步（配合 stall_patience 兜底）。
    pos_tol: float = 0.008
    rot_tol: float = 0.08


@dataclass
class ScriptedLowLevel:
    """把 Goal 展开成 TCP 路径点，逐个闭环推进。"""

    name: str = "scripted"
    approach_height_m: float = 0.16
    travel_height_m: float = 0.30
    place_clearance_m: float = 0.012
    retreat_height_m: float = 0.16
    #: 夹住石头时命令的开口宽度 = 石头宽度 × 这个比例（<1 才会压紧）
    close_fraction: float = 0.85
    #: 张开时用到的"安全宽度"：比石块宽，避免提起时蹭到
    open_extra_m: float = 0.03
    #: 关节角变化小于这个值就认为伺服稳定（配合执行器的固定伺服时长使用）
    joint_tol_rad: float = 1.0e-3
    #: 合拢分几步走（步进式建立接触，避免一步压到位把石头弹出去）
    close_ramp_steps: int = 4
    #: 抬升的第一段高度（米）：先小幅抬起让接触适应，再继续抬
    lift_break_height_m: float = 0.02
    #: 侧向抓取时"预合拢"多留的余量（米）：先在站位收到石头宽度 + 这个值，
    #: 推进到位后再补到真正的夹紧量，避免接触瞬间把石头推走。
    pre_close_margin_m: float = 0.004
    #: 卡住判据：误差已经不大但连续几步都不再改善，就当作"到了"往下走。
    #: 没有这一条，下降段一旦被石头/台面顶住（位置误差停在 5–10 mm）就会永远停在
    #: "下降"这一步，夹爪永远不闭合——实测就是这个问题。
    stall_tolerance_m: float = 0.015
    stall_patience: int = 2
    stall_improvement_m: float = 0.001

    _waypoints: list[_Waypoint] = field(default_factory=list)
    _index: int = 0
    _key: str = ""
    _last_qpos: np.ndarray | None = None
    _settle_steps: int = 0
    _best_error: float = float("inf")
    _stall_steps: int = 0

    def reset(self, context: dict[str, Any]) -> None:
        self._waypoints = []
        self._index = 0
        self._key = ""
        self._last_qpos = None
        self._settle_steps = 0

    # ------------------------------------------------------------------

    def step(self, observation: Observation, goal: Goal) -> Command:
        key = f"{goal.kind}:{goal.stone.name if goal.stone else '-'}"
        if key != self._key:
            self._key = key
            self._waypoints = self._plan(observation, goal)
            self._index = 0
            self._settle_steps = 0
            self._best_error = float("inf")
            self._stall_steps = 0
        if not self._waypoints:
            return Command(
                tcp_pos=np.asarray(observation.tcp_pos, dtype=float),
                tcp_rot=np.asarray(observation.tcp_rot, dtype=float),
                done=True,
                note="no waypoints",
                source=self.name,
            )

        # 到点了就推进；都走完则报 done
        while self._index < len(self._waypoints):
            waypoint = self._waypoints[self._index]
            pos_error = float(np.linalg.norm(waypoint.pos - np.asarray(observation.tcp_pos, dtype=float)))
            rot = np.asarray(observation.tcp_rot, dtype=float)
            rot_error = float(
                np.linalg.norm(
                    0.5
                    * (
                        np.cross(rot[:, 0], waypoint.rot[:, 0])
                        + np.cross(rot[:, 1], waypoint.rot[:, 1])
                        + np.cross(rot[:, 2], waypoint.rot[:, 2])
                    )
                )
            )
            stalled = False
            if pos_error <= waypoint.pos_tol and rot_error <= waypoint.rot_tol:
                self._settle_steps += 1
                if self._settle_steps >= 2:  # 连续两次到位，避免抖动误判
                    self._index += 1
                    self._settle_steps = 0
                    self._best_error = float("inf")
                    self._stall_steps = 0
                    continue
            else:
                self._settle_steps = 0
                if pos_error < self.stall_tolerance_m:
                    if pos_error < self._best_error - self.stall_improvement_m:
                        self._best_error = pos_error
                        self._stall_steps = 0
                    else:
                        self._stall_steps += 1
                        stalled = self._stall_steps >= self.stall_patience
                if stalled:
                    self._index += 1
                    self._best_error = float("inf")
                    self._stall_steps = 0
                    continue
            break

        if self._index >= len(self._waypoints):
            last = self._waypoints[-1]
            return Command(
                tcp_pos=last.pos,
                tcp_rot=last.rot,
                grip_width=last.grip_width,
                done=True,
                note=f"finished {len(self._waypoints)} waypoints",
                source=self.name,
            )
        waypoint = self._waypoints[self._index]
        return Command(
            tcp_pos=waypoint.pos,
            tcp_rot=waypoint.rot,
            grip_width=waypoint.grip_width,
            done=False,
            pure_translation=waypoint.pure_translation,
            note=f"waypoint {self._index + 1}/{len(self._waypoints)} {waypoint.label}",
            source=self.name,
        )

    # ------------------------------------------------------------------

    def _plan(self, observation: Observation, goal: Goal) -> list[_Waypoint]:
        if goal.kind == "pick" and goal.stone is not None and goal.pick_pos is not None:
            return self._pick_waypoints(goal)
        if goal.kind == "place" and goal.place_pos is not None:
            return self._place_waypoints(goal)
        if goal.kind == "retreat":
            rot = np.asarray(goal.place_rot if goal.place_rot is not None else observation.tcp_rot, dtype=float)
            pos = np.asarray(observation.tcp_pos, dtype=float) + np.array([0.0, 0.0, self.retreat_height_m])
            return [_Waypoint(pos, rot, None, "retreat")]
        if goal.kind == "home" and goal.place_pos is not None:
            return [_Waypoint(np.asarray(goal.place_pos, dtype=float), np.asarray(goal.place_rot, dtype=float), None, "home")]
        return []

    def _pick_waypoints(self, goal: Goal) -> list[_Waypoint]:
        stone = goal.stone
        rot = np.asarray(goal.pick_rot, dtype=float)
        pads = np.asarray(goal.pick_pos, dtype=float)
        # 合拢量优先用执行器给的 Goal.grip_width（它按抓取规划的截面带宽算）；
        # 策略自己的 close_fraction 只是没给时的兜底。之前这里无条件覆盖 Goal，
        # 结果执行器怎么调合拢量都没用——实测四组不同 close_fraction 给出完全一样的抬升量。
        close_width = (
            float(goal.grip_width)
            if goal.grip_width is not None
            else max(0.002, float(np.min(stone.size)) * self.close_fraction)
        )
        if goal.approach_direction is not None:
            # 侧向抓取：先退到侧向站位，再水平推进到抓取位姿，合拢后竖直抬起。
            # 抬起必须竖直——侧向拔会把石头从指垫间拖出去。
            direction = np.asarray(goal.approach_direction, dtype=float)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            standoff = pads - direction * max(goal.standoff_m, 0.02)
            lift = pads + np.array([0.0, 0.0, self.approach_height_m])
            # 两段合拢：先在站位把指垫收到"石头宽度 + 余量"，再水平推进，
            # 最后才补上夹紧量。一次合到底会让指垫在接触瞬间把石头顶走——
            # 实测几何完全跨得住（内表面 ±47.5 mm vs 石头 ±33 mm）、法向力也有
            # 11 N×2，但石头被推出指垫之间，最后指垫在空档里合到 66 mm 才停。
            # 预合拢 = 截面带宽 + 余量（**必须比石头宽**，让指垫能跨住石头推进），
            # 然后再分步收到真正的夹紧量。之前写成"最终夹紧量 + 4 mm"，
            # 结果指垫在站位上就收到 29 mm、推进时直接撞在 65 mm 宽的石头侧面把它推走。
            band = goal.grasp_band_width if goal.grasp_band_width is not None else None
            if band is not None:
                pre_close = band + self.pre_close_margin_m
            elif goal.grip_width is not None:
                pre_close = goal.grip_width + 0.02
            else:
                pre_close = None
            waypoints = [
                _Waypoint(standoff, rot, pre_close, "side-standoff"),
                _Waypoint(pads, rot, pre_close, "advance", pos_tol=0.006),
            ]
            # 合拢分 4 段走（而不是一步到位）：位置伺服一步压到位时，接触冲量会把
            # 只有 0.5 kg 的石头弹出去。分段的每一步只压一点点，接触是"逐步建立"的。
            start = pre_close if pre_close is not None else close_width + 0.02
            for index in range(1, self.close_ramp_steps + 1):
                blend = index / self.close_ramp_steps
                width = start + (close_width - start) * blend
                waypoints.append(_Waypoint(pads, rot, width, f"close-{index}/{self.close_ramp_steps}"))
            # 驻留一拍：接触建立后让接触力稳定下来再抬
            waypoints.append(_Waypoint(pads, rot, close_width, "close-dwell"))
            # 抬起分两段：先抬 2 cm 让接触适应（这一段最容易把石头挤出去），
            # 停一拍，再抬到目标高度。实测一次性抬 16 cm 时，指垫在抬升途中
            # 从 74 mm 继续合到 38 mm——石头被落下了。
            lift_low = pads + np.array([0.0, 0.0, self.lift_break_height_m])
            waypoints.append(_Waypoint(lift_low, rot, close_width, "lift-break", pure_translation=True))
            waypoints.append(_Waypoint(lift_low, rot, close_width, "lift-break-dwell", pure_translation=True))
            waypoints.append(_Waypoint(lift, rot, close_width, "lift", pure_translation=True))
            return waypoints
        approach = pads + np.array([0.0, 0.0, self.approach_height_m])
        lift = pads + np.array([0.0, 0.0, self.approach_height_m])
        return [
            _Waypoint(approach, rot, None, "above-pick"),
            _Waypoint(pads, rot, None, "descend", pos_tol=0.006),
            _Waypoint(pads, rot, close_width, "close"),
            _Waypoint(lift, rot, close_width, "lift"),
        ]

    def _place_waypoints(self, goal: Goal) -> list[_Waypoint]:
        rot = np.asarray(goal.place_rot, dtype=float)
        pads = np.asarray(goal.place_pos, dtype=float)
        above = pads + np.array([0.0, 0.0, self.travel_height_m])
        released = pads + np.array([0.0, 0.0, self.place_clearance_m])
        retreat = pads + np.array([0.0, 0.0, self.retreat_height_m])
        hold_width = goal.grip_width if goal.grip_width is not None else 0.0
        return [
            _Waypoint(above, rot, hold_width, "above-place"),
            _Waypoint(released, rot, hold_width, "place-descent", pos_tol=0.006),
            _Waypoint(released, rot, None, "open"),
            _Waypoint(retreat, rot, None, "retreat"),
        ]
