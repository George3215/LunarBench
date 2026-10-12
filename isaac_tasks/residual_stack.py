"""完整堆叠的局部残差策略。

程序负责路点和夹爪；6D 策略只在石头/放置目标附近修正位置与旋转。
不吸附、不瞬移石头，阶段切换必须由真实位姿和接触确认。
"""

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
)
from task2 import META, Task2Cfg, Task2Env
from torch.nn.functional import one_hot

CLEAR, APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, PLACE, RELEASE, RETRACT, SETTLE = (
    range(10)
)
PHASES = (
    "clear",
    "approach",
    "descend",
    "grasp",
    "lift",
    "transport",
    "place",
    "release",
    "retract",
    "settle",
)


def bounded(vector, limit):
    """限制向量范数，不分别裁剪三个轴而扩大球形动作范围。"""
    return vector * (limit / vector.norm(dim=-1, keepdim=True).clamp_min(limit))


def rotation_delta(vector):
    angle = vector.norm(dim=-1)
    return quat_from_angle_axis(angle, vector / angle[:, None].clamp_min(1e-8))


@configclass
class ResidualStackCfg(Task2Cfg):
    stage = "stack10"
    action_space = 6
    observation_space = 210
    # 程序搬运每动作最多 4 cm，仍小于 5 cm 硬门控；局部残差单独限为 1 cm。
    max_delta_m = 0.04
    residual = {  # noqa: RUF012 - configclass 为每个配置实例复制字典。
        "local_radius_m": 0.05,
        "position_residual_m": 0.01,
        "rotation_residual_rad": 0.15,
        "hover_m": 0.12,
        "grasp_z_offset_m": 0.012,
        "support_preload_m": 0.003,
        "travel_z_m": 0.45,
        "waypoint_tolerance_m": 0.015,
        "close_tolerance_m": 0.015,
        "contact_hold_s": 0.48,
        "release_distance_m": 0.025,
        "phase_steps": 120,
        "max_retries": 3,
        "progress_weight": 5.0,
        "residual_cost": 0.002,
        "retry_penalty": 2.0,
    }


class ResidualStackEnv(Task2Env):
    def __init__(self, cfg, **kwargs):
        p = cfg.residual
        if not (0 < p["position_residual_m"] <= p["local_radius_m"] <= 0.05):
            raise ValueError(
                "Residual position and activation radius must be within 5 cm"
            )
        if not (0 < p["rotation_residual_rad"] <= 0.35):
            raise ValueError("Residual rotation must be within 0.35 rad")
        super().__init__(cfg, **kwargs)
        self.rows = torch.arange(self.num_envs, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stone_index = torch.zeros_like(self.phase)
        self.phase_steps = torch.zeros_like(self.phase)
        self.retries = torch.zeros_like(self.phase)
        self.contact_time = torch.zeros(self.num_envs, device=self.device)
        self.completed = torch.zeros(
            (self.num_envs, 10), device=self.device, dtype=torch.bool
        )
        self.nominal_quat = (
            torch.tensor([0.0, 0.70710678, 0.70710678, 0.0], device=self.device)
            .expand(self.num_envs, -1)
            .clone()
        )
        self.grip_quat = self.nominal_quat.clone()
        self.object_in_tcp = torch.zeros((self.num_envs, 3), device=self.device)
        self.pick_center = torch.zeros_like(self.object_in_tcp)
        self.load_mass = torch.zeros(self.num_envs, device=self.device)
        self.stone_mass = torch.tensor(
            [s["mass"] for s in META["stones"]], device=self.device
        )
        self.support_indices = torch.tensor(
            [[0, 0]] * 4 + [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8]],
            device=self.device,
        )
        self.anchor = torch.zeros_like(self.object_in_tcp)
        self.anchor_quat = self.nominal_quat.clone()
        self.local_active = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.residual_action = torch.zeros((self.num_envs, 6), device=self.device)
        self.close_gripper = self.local_active.clone()
        self.local_phi = torch.zeros(self.num_envs, device=self.device)
        self.failure = self.local_active.clone()
        # 全程计数不随单个回合重置，报告区分真实阶段覆盖与训练更新成功。
        self.phase_visits = torch.zeros(
            len(PHASES), device=self.device, dtype=torch.long
        )
        self.completed_peak = torch.zeros_like(self.phase)
        self.episode_counts = torch.zeros(2, device=self.device, dtype=torch.long)

    def selected(self, values):
        return values[self.rows, self.stone_index]

    def placement_target(self):
        # 槽位保留原 x/y；高度按石头实际尺寸和下层当前中心估计。
        # 统一平均厚度会让薄石悬停数毫米，接触只剩稀疏信号。
        lower = self.support_indices[self.stone_index]
        centers = self.stone_centers()
        top = (
            (centers[self.rows[:, None], lower, 2] + self.initial_z[lower])
            .max(-1)
            .values
        )
        top = torch.where(self.stone_index < 4, 0.0, top)
        target = self.targets[self.stone_index].clone()
        target[:, 2] = (
            top
            + self.initial_z[self.stone_index]
            - self.cfg.residual["support_preload_m"]
        )
        return target

    def geometry(self):
        tcp, quat = self.tcp_pose()
        tcp = tcp - self.scene.env_origins
        center = self.selected(self.stone_centers())
        target = self.placement_target()
        pad = self.pad_positions().mean(1) - self.scene.env_origins
        # TCP 并非指垫中点；使用当前机械结构偏置构造抓取路点。
        pad_offset = quat_apply(quat_inv(quat), pad - tcp)
        return tcp, quat, center, target, pad, pad_offset

    def waypoint(self):
        tcp, _quat, center, target, pad, offset = self.geometry()
        p = self.cfg.residual
        orientation = torch.where(
            (self.phase >= LIFT)[:, None], self.grip_quat, self.nominal_quat
        )
        grasp_tcp = center - quat_apply(orientation, offset)
        grasp_tcp[:, 2] += p["grasp_z_offset_m"]
        place_tcp = target - quat_apply(orientation, self.object_in_tcp)
        goal = grasp_tcp.clone()
        goal[:, 2] = p["travel_z_m"]
        clear = self.phase == CLEAR
        goal[clear] = tcp[clear]
        goal[clear, 2] = p["travel_z_m"]
        descend = self.phase == DESCEND
        goal[descend] = grasp_tcp[descend]
        goal[descend, 2] += 0.035
        grasp = self.phase == GRASP
        goal[grasp] = grasp_tcp[grasp]
        lifting = self.phase == LIFT
        lift_goal = self.pick_center - quat_apply(orientation, self.object_in_tcp)
        lift_goal[:, 2] += p["travel_z_m"] - self.pick_center[:, 2]
        goal[lifting] = lift_goal[lifting]
        transport = self.phase == TRANSPORT
        goal[transport] = place_tcp[transport]
        goal[transport, 2] += p["travel_z_m"] - target[transport, 2]
        placing = (self.phase == PLACE) | (self.phase == RELEASE)
        goal[placing] = place_tcp[placing]
        retract = self.phase >= RETRACT
        goal[retract] = place_tcp[retract]
        goal[retract, 2] += p["hover_m"]
        # PLACE 的远段下降仍由程序执行，进入 5 cm 球内才接通策略。
        local = (grasp & ((pad - center).norm(dim=-1) <= p["local_radius_m"])) | (
            (self.phase == PLACE)
            & ((center - target).norm(dim=-1) <= p["local_radius_m"])
        )
        return goal, orientation, local

    def _pre_physics_step(self, actions):
        self.anchor, self.anchor_quat, self.local_active = self.waypoint()
        self.residual_action = actions.clamp(-1, 1) * self.local_active[:, None]
        p = self.cfg.residual
        dp = bounded(self.residual_action[:, :3], 1.0) * p["position_residual_m"]
        dr = bounded(self.residual_action[:, 3:], 1.0) * p["rotation_residual_rad"]
        desired_quat = quat_mul(rotation_delta(dr), self.anchor_quat)
        goal = self.anchor + dp
        tcp, quat, center, _target, pad, _ = self.geometry()
        # 同一抓取点闭合后保持夹爪，不能因一次残差偏离而反复开合。
        grasp_center = center.clone()
        grasp_center[:, 2] += p["grasp_z_offset_m"]
        near = (pad - grasp_center).norm(dim=-1) <= p["close_tolerance_m"]
        self.close_gripper |= (self.phase == GRASP) & near
        self.close_gripper &= (self.phase >= GRASP) & (self.phase <= PLACE)
        # 复用原有每动作 EEF 门控和局部 DLS；程序路点也不能绕过门控。
        command = torch.zeros((self.num_envs, 7), device=self.device)
        command[:, :3] = (
            bounded(goal - tcp, self.cfg.max_delta_m) / self.cfg.max_delta_m
        )
        turn = axis_angle_from_quat(quat_mul(desired_quat, quat_inv(quat)))
        command[:, 3:6] = (
            bounded(turn, self.cfg.max_rotation_rad) / self.cfg.max_rotation_rad
        )
        command[:, 6] = torch.where(self.close_gripper, -1.0, 1.0)
        super()._pre_physics_step(command)

    def arm_feedforward(self, jacobian):
        # 原小步 DLS 的位置误差产生的力矩不足以托起 2–4 kg 石头。
        # 抓持确认后加 J_COM^T * (-m*g)，仅补偿重力，不施加吸附力。
        offset = quat_apply(self.tcp_pose()[1], self.object_in_tcp)
        linear = jacobian[:, :3] + torch.cross(
            jacobian[:, 3:].transpose(1, 2), offset[:, None].expand(-1, 6, -1), dim=-1
        ).transpose(1, 2)
        force = (
            -torch.tensor(self.cfg.sim.gravity, device=self.device)
            * self.load_mass[:, None]
        )
        load = (linear.transpose(1, 2) @ force[:, :, None]).squeeze(-1)
        return super().arm_feedforward(jacobian) + load

    def _signals(self):
        # 公共物理判据提供接触、支撑层、稳定性和完整墙成功；不改动 success 定义。
        _, _, _, native = super()._signals()
        s = self.signals
        p = self.cfg.residual
        tcp, quat, center, target, pad, _ = self.geometry()
        old_phase = self.phase.clone()
        self.phase_visits += torch.bincount(old_phase, minlength=len(PHASES))
        self.phase_steps += 1
        bilateral = self.selected(s["bilateral"])
        stable = self.selected(s["stable"])
        self.contact_time = torch.where(
            bilateral & stable & self.close_gripper,
            self.contact_time + self.step_dt,
            0.0,
        )
        confirmed = self.contact_time >= p["contact_hold_s"] - 1e-7
        distance = (center - target).norm(dim=-1)
        angle = axis_angle_from_quat(quat_mul(self.anchor_quat, quat_inv(quat))).norm(
            dim=-1
        )
        grasp_center = center.clone()
        grasp_center[:, 2] += p["grasp_z_offset_m"]
        local_alignment = torch.exp(-(pad - grasp_center).norm(dim=-1) / 0.025)
        contact = self.selected(s["pad_touch"]).float().mean(-1)
        grip_phi = 2 * local_alignment + contact + 0.5 * torch.exp(-angle / 0.2)
        grip_phi += (self.contact_time / p["contact_hold_s"]).clamp(0, 1)
        place_phi = (
            3 * torch.exp(-distance / 0.025)
            + 2 * torch.exp(-(center[:, 2] - target[:, 2]).abs() / 0.01)
            + self.selected(s["layer_ok"]).float()
            + 0.5 * torch.exp(-angle / 0.2)
            + self.selected(s["slow"]).float() * torch.exp(-distance / 0.025)
        )
        # 各阶段的势函数变化引导整条任务；阶段切换重置基线，不发切换虚奖。
        route_phi = torch.exp(-(tcp - self.anchor).norm(dim=-1) / 0.15)
        height = center[:, 2] - self.initial_z[self.stone_index]
        lift_phi = (
            2 * (height / self.cfg.curriculum["lift_height_m"]).clamp(0, 1) * bilateral
        )
        transport_phi = (
            2
            * torch.exp(-(center[:, :2] - target[:, :2]).norm(dim=-1) / 0.15)
            * bilateral
        )
        phi = torch.where(old_phase == GRASP, grip_phi, route_phi)
        phi = torch.where(old_phase == LIFT, lift_phi, phi)
        phi = torch.where(old_phase == TRANSPORT, transport_phi, phi)
        phi = torch.where((old_phase >= PLACE) & (old_phase <= SETTLE), place_phi, phi)
        progress = torch.where(self.local_phi == 0, 0.0, phi - self.local_phi)
        self.local_phi = phi
        near_goal = (tcp - self.anchor).norm(dim=-1) < p["waypoint_tolerance_m"]
        aligned = angle < 0.15
        transitions = [
            (CLEAR, APPROACH, near_goal & aligned),
            (APPROACH, DESCEND, near_goal & aligned),
            (DESCEND, GRASP, near_goal & aligned),
            (GRASP, LIFT, confirmed),
            (
                LIFT,
                TRANSPORT,
                self.selected(s["achieved"])[:, 1]
                & (center[:, 2] >= p["travel_z_m"] - 0.02),
            ),
            (TRANSPORT, PLACE, near_goal & bilateral),
            (
                PLACE,
                RELEASE,
                (distance < p["release_distance_m"])
                & self.selected(s["layer_ok"])
                & self.selected(s["slow"]),
            ),
            (RELEASE, RETRACT, (self.phase_steps >= 2) & self.selected(s["released"])),
            (RETRACT, SETTLE, near_goal),
        ]
        for source, destination, condition in transitions:
            self.phase[(old_phase == source) & condition] = destination
        grabbed = (old_phase == GRASP) & (self.phase == LIFT)
        self.object_in_tcp[grabbed] = quat_apply(
            quat_inv(quat[grabbed]), center[grabbed] - tcp[grabbed]
        )
        self.grip_quat[grabbed] = quat[grabbed]
        self.pick_center[grabbed] = center[grabbed]
        loaded = (self.phase >= LIFT) & (self.phase <= PLACE) & bilateral
        self.load_mass = torch.where(loaded, self.stone_mass[self.stone_index], 0.0)
        placed = self.selected(s["achieved"])[:, 2]
        finished = (old_phase == SETTLE) & placed
        self.completed[self.rows[finished], self.stone_index[finished]] = True
        self.completed_peak = torch.maximum(self.completed_peak, self.completed.sum(-1))
        next_stone = (~self.completed).long().argmax(-1)
        advance = finished & ~self.completed.all(-1)
        self.stone_index[advance] = next_stone[advance]
        self.phase[advance] = CLEAR
        self.retries[advance] = 0
        # 掉抓和阶段卡住只重试当前石头，不能重置仿真或伪造成功。
        carrying = (old_phase == LIFT) | (old_phase == TRANSPORT)
        lost = carrying & ~bilateral & (self.phase_steps > 3)
        stalled = (self.phase_steps >= p["phase_steps"]) & ~native
        retry = (lost | stalled) & ~finished
        self.retries += retry.long()
        self.phase[retry] = CLEAR
        self.close_gripper[retry] = False
        self.load_mass[retry] = 0.0
        self.contact_time[retry] = 0.0
        self.local_phi[retry] = 0.0
        changed = self.phase != old_phase
        self.phase_steps[changed | retry] = 0
        self.local_phi[changed] = 0.0
        # 已完成层倒塌不算“仍然完成”；终止本次尝试，完整 success 仍由物理检查决定。
        collapsed = (self.completed & ~s["placed"] & (s["distance"] > 0.06)).any(-1)
        self.failure = s["fallen"] | (self.retries >= p["max_retries"]) | collapsed
        success = native & ~self.failure
        done = success | self.failure
        reward = (
            p["progress_weight"] * progress
            + s["bonus"]
            + self.cfg.reward_weights["success_bonus"] * success
            - self.cfg.reward_weights["step_cost"]
            - p["residual_cost"] * self.residual_action.square().mean(-1)
            - p["retry_penalty"] * retry
            - self.cfg.reward_weights["fall_penalty"] * self.failure
        )
        self.extras["log"] = {
            "reward/mean": reward.mean(),
            "reward/dense_progress": progress.mean(),
            "success/full_wall": success.float().mean(),
            "residual/active_fraction": self.local_active.float().mean(),
            "residual/completed_stones": self.completed.sum(-1).float().mean(),
            "residual/retry": retry.float().mean(),
            "grasp/contacts": bilateral.float().mean(),
        }
        for i, name in enumerate(PHASES):
            self.extras["log"]["phase/" + name] = (self.phase == i).float().mean()
        return reward, done, success, success

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        self.episode_counts[0] += (terminated | truncated).sum()
        self.episode_counts[1] += self._result[2].sum()
        return terminated, truncated

    def training_metrics(self):
        counts = self.episode_counts.tolist()
        return {
            "phase_action_counts": dict(zip(PHASES, self.phase_visits.tolist())),
            "max_completed_stones": int(self.completed_peak.max()),
            "completed_episodes": counts[0],
            "full_wall_successes": counts[1],
            "full_wall_success_rate": counts[1] / counts[0] if counts[0] else None,
        }

    def _get_observations(self):
        obs = super()._get_observations()
        anchor, rotation, active = self.waypoint()
        center = self.selected(self.stone_centers())
        context = torch.cat(
            [
                center,
                self.placement_target(),
                anchor,
                rotation,
                one_hot(self.phase, len(PHASES)).float(),
                self.stone_index[:, None].float() / 9.0,
                active[:, None].float(),
            ],
            dim=-1,
        )
        obs["policy"] = torch.cat([obs["policy"], context], dim=-1)
        if "critic" in obs:
            obs["critic"] = torch.cat([obs["critic"], context], dim=-1)
        return obs

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.rows
        super()._reset_idx(env_ids)
        self.phase[env_ids] = CLEAR
        self.stone_index[env_ids] = 0
        self.phase_steps[env_ids] = 0
        self.retries[env_ids] = 0
        self.completed[env_ids] = False
        self.contact_time[env_ids] = 0.0
        self.local_phi[env_ids] = 0.0
        self.close_gripper[env_ids] = False
        self.failure[env_ids] = False
        self.residual_action[env_ids] = 0.0
        self.object_in_tcp[env_ids] = 0.0
        self.pick_center[env_ids] = 0.0
        self.load_mass[env_ids] = 0.0
        self.grip_quat[env_ids] = self.nominal_quat[env_ids]
