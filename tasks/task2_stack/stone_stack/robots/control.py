"""与机械臂无关的控制层：关节寻址、阻尼最小二乘 IK、伺服、夹爪开关。

旧执行器把 `UR_JOINTS`、`grip_site`、`data.ctrl[:6]`、两只反向手指写死在
`drive_segment` 里；这里全部改成由 `ArmProfile` 驱动，因此 6 轴 / 7 轴、单执行器 /
双执行器夹爪走同一条代码路径。

IK 是纯 Python 阻尼最小二乘（沿用旧实现），只在 TCP 位姿上做 6 维误差，关节被裁剪到
各自限位。它不是运动规划：不避障、不做可达性搜索，这些由调用方用位姿序列规避。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profile import ArmProfile


@dataclass
class ServoResult:
    """一次伺服的实测结果，用于报告与判定，不用来"相信"成功。"""

    steps: int
    final_pos_error_m: float
    final_rot_error_rad: float
    reached: bool


class RobotController:
    """把 profile 变成可执行的控制对象。"""

    def __init__(self, model, data, profile: ArmProfile, ik_data=None):
        self.model = model
        self.data = data
        self.profile = profile
        self.ik_data = data if ik_data is None else ik_data
        self.qpos_addr, self.dof_addr, self.lower, self.upper = self._joint_addresses()
        self._tcp_site_id = model.site(profile.tcp_site).id
        self._pad_geom_ids = [
            model.geom(name).id for name in profile.gripper.pad_geoms if model.geom(name).id >= 0
        ]
        self._finger_body_ids = [
            model.body(name).id for name in (profile.gripper.finger_bodies or ())
        ]
        self._gripper_actuator_ids = [
            model.actuator(name).id for name in profile.gripper.actuator_names
        ]

    # ---------------------------------------------------------------- 关节

    def _joint_addresses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qpos, dof, lower, upper = [], [], [], []
        for name in self.profile.joints:
            joint = self.model.joint(name)
            if joint.id < 0:
                raise KeyError(f"{self.profile.name}: 场景里没有关节 {name!r}")
            qpos.append(int(self.model.jnt_qposadr[joint.id]))
            dof.append(int(self.model.jnt_dofadr[joint.id]))
            lower.append(float(self.model.jnt_range[joint.id, 0]))
            upper.append(float(self.model.jnt_range[joint.id, 1]))
        return np.asarray(qpos), np.asarray(dof), np.asarray(lower), np.asarray(upper)

    @property
    def dof_count(self) -> int:
        return len(self.profile.joints)

    def arm_qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.qpos_addr], dtype=float).copy()

    def set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_addr] = np.asarray(q, dtype=float)

    def set_arm_ctrl(self, q: np.ndarray) -> None:
        for actuator_id, value in zip(self._arm_actuator_ids(), q):
            self.data.ctrl[actuator_id] = value

    def _arm_actuator_ids(self) -> list[int]:
        """按关节顺序找对应的执行器：源执行器可能叫任何名字，按 `trnid` 反查最稳。"""
        ids: list[int] = []
        joint_ids = [self.model.joint(name).id for name in self.profile.joints]
        for joint_id in joint_ids:
            found = -1
            for actuator_id in range(self.model.nu):
                if self.model.actuator_trntype[actuator_id] != 0:  # 只认 joint 传动
                    continue
                if int(self.model.actuator_trnid[actuator_id, 0]) == int(joint_id):
                    found = actuator_id
                    break
            if found < 0:
                raise KeyError(f"{self.profile.name}: 关节 {self.model.joint(joint_id).name} 没有执行器")
            ids.append(found)
        return ids

    # ---------------------------------------------------------------- TCP

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self._tcp_site_id].copy(),
            self.data.site_xmat[self._tcp_site_id].reshape(3, 3).copy(),
        )

    def tcp_pose_for(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """给定关节角算 TCP 位姿，在**独立的 MjData** 上跑，不改变仿真状态。

        注意：不能用 `self.ik_data`——`__init__` 在未显式传入时把 `ik_data` 设为
        `self.data` 本身（task2 的调用方都没传），那样会直接改写正在仿真的 qpos。
        这里单独维护一份只读 FK scratch。

        用途：对一段关节轨迹做**运动学前瞻**（`kinematics.preview_trajectory`），
        以及在不推进物理的前提下检查目标是否可达。
        """
        import mujoco

        scratch = getattr(self, "_fk_data", None)
        if scratch is None:
            scratch = mujoco.MjData(self.model)
            self._fk_data = scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.qpos_addr] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self.model, scratch)
        return (
            scratch.site_xpos[self._tcp_site_id].copy(),
            scratch.site_xmat[self._tcp_site_id].reshape(3, 3).copy(),
        )

    def forward(self) -> None:
        import mujoco

        mujoco.mj_forward(self.model, self.data)

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q0: np.ndarray | None = None,
        iterations: int = 220,
        orientation_weight: float = 0.35,
        tolerance_pos: float = 1.0e-4,
        tolerance_rot: float = 1.0e-3,
    ) -> np.ndarray:
        """阻尼最小二乘 IK。返回裁剪到关节限位后的关节角。"""
        import mujoco

        data = self.ik_data
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)
        q = (self.arm_qpos() if q0 is None else np.asarray(q0, dtype=float)).copy()
        # IK 在独立的 MjData 上跑：不污染正在仿真的那份状态
        data.qpos[:] = self.data.qpos
        for _ in range(iterations):
            data.qpos[self.qpos_addr] = q
            mujoco.mj_forward(self.model, data)
            current_pos = data.site_xpos[self._tcp_site_id].copy()
            current_rot = data.site_xmat[self._tcp_site_id].reshape(3, 3).copy()
            pos_error = target_pos - current_pos
            rot_error = 0.5 * (
                np.cross(current_rot[:, 0], target_rot[:, 0])
                + np.cross(current_rot[:, 1], target_rot[:, 1])
                + np.cross(current_rot[:, 2], target_rot[:, 2])
            )
            if np.linalg.norm(pos_error) < tolerance_pos and np.linalg.norm(rot_error) < tolerance_rot:
                break
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, self._tcp_site_id)
            jacobian = np.vstack([jac_pos[:, self.dof_addr], orientation_weight * jac_rot[:, self.dof_addr]])
            error = np.concatenate([pos_error, orientation_weight * rot_error])
            damping = 1.0e-3
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            q = np.clip(q + 0.7 * delta, self.lower, self.upper)
        return self.unwrap_to_seed(q, (self.arm_qpos() if q0 is None else np.asarray(q0, dtype=float)))

    def unwrap_to_seed(self, q: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """把 IK 解拉回离初值最近的等价角（连续关节按 2π 取模）。

        没有这一步，位置伺服会"绕远路"：UR5e 的 5 个关节范围是 ±2π，IK 可能给出
        q6 = -6.10（等价于 +0.18），伺服就真的把腕子倒转 350°——实测表现为机械臂
        一直在走却永远到不了目标（292 个策略步、117 秒仿真都没收敛）。
        """
        q = np.asarray(q, dtype=float).copy()
        seed = np.asarray(seed, dtype=float)
        for index in range(len(q)):
            span = float(self.upper[index] - self.lower[index])
            if span >= 2.0 * np.pi - 1e-6:
                q[index] = seed[index] + float(np.angle(np.exp(1j * (q[index] - seed[index]))))
                q[index] = float(np.clip(q[index], self.lower[index], self.upper[index]))
        return q

    def solve_ik_best(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q0: np.ndarray | None = None,
        iterations: int = 220,
    ) -> np.ndarray:
        """多初值 IK：单一初值会掉进局部极小（同一目标点，肘上/肘下只差几度就解不出来）。

        初值取 当前位姿 → home → 限位中点，取残差最小的解。实测这是"同一个槽位
        偶尔抓不到"和"稳定抓得到"的区别。
        """
        seeds = []
        if q0 is not None:
            seeds.append(np.asarray(q0, dtype=float))
        seeds.append(np.asarray(self.profile.home_qpos, dtype=float))
        seeds.append(0.5 * (self.lower + self.upper))
        best_q, best_error = None, None
        for seed in seeds:
            q = self.solve_ik(target_pos, target_rot, q0=seed, iterations=iterations)
            saved = self.arm_qpos()
            self.set_arm_qpos(q)
            self.forward()
            pos_error, rot_error = self.tcp_error(target_pos, target_rot)
            self.set_arm_qpos(saved)
            error = pos_error + 0.05 * rot_error
            if best_error is None or error < best_error:
                best_q, best_error = q, error
            if pos_error < 1.0e-3 and rot_error < 1.0e-2:
                break
        self.forward()
        assert best_q is not None
        return best_q

    def tcp_error(self, target_pos: np.ndarray, target_rot: np.ndarray) -> tuple[float, float]:
        pos, rot = self.tcp_pose()
        pos_error = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - pos))
        rot_error = float(
            np.linalg.norm(
                0.5
                * (
                    np.cross(rot[:, 0], np.asarray(target_rot)[:, 0])
                    + np.cross(rot[:, 1], np.asarray(target_rot)[:, 1])
                    + np.cross(rot[:, 2], np.asarray(target_rot)[:, 2])
                )
            )
        )
        return pos_error, rot_error

    # ---------------------------------------------------------------- 夹爪

    def use_gripper_actuators(self, names: tuple[str, ...]) -> None:
        """改由另一组执行器驱动夹爪（装了自造平行夹爪时用）。"""
        self._gripper_actuator_ids = [int(self.model.actuator(name).id) for name in names]

    def set_gripper_ctrl(self, ctrl) -> None:
        values = np.asarray(ctrl, dtype=float).reshape(-1)
        if values.size != len(self._gripper_actuator_ids):
            raise ValueError(f"{self.profile.name}: 夹爪需要 {len(self._gripper_actuator_ids)} 个控制量，给了 {values.size}")
        for actuator_id, value in zip(self._gripper_actuator_ids, values):
            self.data.ctrl[actuator_id] = float(value)
        self.gripper_command = values.copy()

    def open_gripper(self) -> None:
        self.set_gripper_ctrl(self.profile.gripper.open_ctrl)

    def close_gripper(self, width: float | None = None) -> None:
        gripper = self.profile.gripper
        if width is None:
            self.set_gripper_ctrl(gripper.close_ctrl)
        else:
            self.set_gripper_ctrl(gripper.ctrl_for_width(width))

    def measured_opening(self) -> float:
        """实测开口（米）：优先用指垫 geom 中心距，否则用两指 body 原点距。"""
        if len(self._pad_geom_ids) >= 2:
            left, right = self._pad_geom_ids[0], self._pad_geom_ids[1]
            return float(np.linalg.norm(self.data.geom_xpos[right] - self.data.geom_xpos[left]))
        if len(self._finger_body_ids) >= 2:
            left, right = self._finger_body_ids
            return float(np.linalg.norm(self.data.xpos[right] - self.data.xpos[left]))
        # 没有可测几何时退回控制量换算（会在报告里标注来源）
        return self.profile.gripper.width_from_ctrl(float(np.asarray(self.gripper_command).reshape(-1)[0]))

    # ---------------------------------------------------------------- 抓取几何

    def grasp_rotation(
        self,
        closing_direction: np.ndarray,
        tilt_hint: np.ndarray | None = None,
        approach_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """求 TCP 朝向：接近轴基本向下，张开轴指向给定的水平方向。

        这是旧 `top_down_gripper_rotation()` 的夹爪无关版本：旧实现把"开口轴 =
        grip_site 的 x 轴、接近轴 = -z"写死，这里从 profile 的实测轴向量构造正交基。

        `approach_tilt_deg > 0` 时把整个抓取基座绕**俯仰轴**（approach × closing）
        转一个角度：接近轴和张开轴一起倾斜。倾角方向由 `tilt_hint` 定（一般给
        "基座→墙"方向，取倾斜后接近轴水平分量与该方向同号的那个解）。绕张开轴转
        （只摆接近轴、张开轴保持水平）对 Piper 实测没用——上层槽位照样差几十毫米。
        """
        gripper = self.profile.gripper
        approach_tcp = _unit(gripper.approach_axis_tcp)
        open_tcp = _unit(gripper.open_axis_tcp)
        # 两个实测轴理论上正交，实际有 1e-2 量级偏差，这里正交化
        open_tcp = _unit(open_tcp - approach_tcp * float(np.dot(open_tcp, approach_tcp)))
        third_tcp = np.cross(approach_tcp, open_tcp)

        if approach_direction is None:
            approach_world = np.array([0.0, 0.0, -1.0])
            open_world = np.asarray(closing_direction, dtype=float).copy()
            open_world[2] = 0.0
            open_world = _unit(open_world)
        else:
            # 侧向抓取：接近轴水平，张开轴垂直于它（也在水平面内），
            # 两指的高度方向就是竖直方向——指尖的 36 mm 偏移变成沿接近方向的伸出量，
            # 不再受"离台面多高"约束，平放在台面上的矮石头也能抓。
            approach_world = _unit(np.asarray(approach_direction, dtype=float))
            open_world = np.asarray(closing_direction, dtype=float).copy()
            open_world = _unit(open_world - approach_world * float(np.dot(open_world, approach_world)))
        third_world = np.cross(approach_world, open_world)

        tilt = float(self.profile.approach_tilt_deg) if approach_direction is None else 0.0
        if abs(tilt) > 1e-9:
            pitch_axis = _unit(np.cross(approach_world, open_world))
            hint = None
            if tilt_hint is not None:
                hint = np.asarray(tilt_hint, dtype=float).copy()
                hint[2] = 0.0
                if float(np.linalg.norm(hint)) > 1e-9:
                    hint = _unit(hint)
            theta = np.deg2rad(tilt)
            candidates = []
            for sign in (1.0, -1.0):
                rotation = _rotation_about(pitch_axis, sign * theta)
                approach = rotation @ approach_world
                score = 0.0 if hint is None else float(np.dot(_unit(approach - np.array([0.0, 0.0, approach[2]])), hint))
                if hint is None:
                    score = sign
                candidates.append((score, rotation))
            rotation_pitch = max(candidates, key=lambda item: item[0])[1]
            approach_world = rotation_pitch @ approach_world
            open_world = rotation_pitch @ open_world
            third_world = rotation_pitch @ third_world

        basis_tcp = np.column_stack([open_tcp, third_tcp, approach_tcp])
        basis_world = np.column_stack([open_world, third_world, approach_world])
        rotation = basis_world @ basis_tcp.T
        return rotation

    def tcp_for_pad_center(self, pad_center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """把"指垫中心要到的位置"换成 TCP 目标位置。"""
        offset = np.asarray(self.profile.gripper.pad_offset_tcp, dtype=float)
        return np.asarray(pad_center, dtype=float) - np.asarray(rotation, dtype=float) @ offset

    # ---------------------------------------------------------------- 伺服

    def servo_to(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        seconds: float,
        grip_ctrl=None,
        q_seed: np.ndarray | None = None,
        ik_iterations: int = 180,
        tolerance_pos: float = 8.0e-3,
        tolerance_rot: float = 5.0e-2,
        chunk_steps: int = 80,
        hold_chunks: int = 2,
        correction_gain: float = 0.5,
        max_correction_pos: float = 0.10,
        max_correction_rot: float = 0.5,
        on_step=None,
        preserve_hold_ctrl: bool = False,
    ) -> ServoResult:
        """把 TCP 闭环伺服到目标位姿。

        为什么不是"解一次 IK 然后按住"：这些臂用的是位置伺服（UR5e kp=650），带着
        几十毫米的静差（实测纯 IK 命令在 0.6 m 臂展处差 6–27 mm）。所以这里每个 chunk
        重新量 TCP 误差，用**积分**修正目标再解 IK。

        修正必须带增益和饱和：直接令 `目标 = 名义目标 + 误差`（增益 1）会在滞后对象上
        正反馈振荡——实测同样三个位姿从 6–27 mm 恶化到 155–320 mm。这里增益 0.5、
        位置修正饱和 ±0.10 m、姿态 ±0.5 rad。
        """
        import mujoco

        model, data = self.model, self.data
        dt = float(model.opt.timestep)
        total_steps = max(1, int(round(seconds / dt)))
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)
        q = self.arm_qpos() if q_seed is None else np.asarray(q_seed, dtype=float).copy()
        # 直接策略的保持动作延续已有控制目标，避免每轮把重力静差重新写成关节目标。
        if preserve_hold_ctrl and q_seed is None:
            q = data.ctrl[self._arm_actuator_ids()].copy()
        done_steps = 0
        stable = 0
        final_pos_error = 0.0
        final_rot_error = 0.0
        correction = np.zeros(3)
        rotation_correction = np.eye(3)
        while done_steps < total_steps:
            measured_pos, measured_rot = self.tcp_pose()
            pos_error_vec = target_pos - measured_pos
            rot_error_vec = 0.5 * (
                np.cross(measured_rot[:, 0], target_rot[:, 0])
                + np.cross(measured_rot[:, 1], target_rot[:, 1])
                + np.cross(measured_rot[:, 2], target_rot[:, 2])
            )
            final_pos_error = float(np.linalg.norm(pos_error_vec))
            final_rot_error = float(np.linalg.norm(rot_error_vec))
            if final_pos_error < tolerance_pos and final_rot_error < tolerance_rot:
                stable += 1
                if stable >= hold_chunks:
                    break
            else:
                stable = 0
                norm = float(np.linalg.norm(pos_error_vec))
                if norm > max_correction_pos:
                    pos_error_vec = pos_error_vec * (max_correction_pos / norm)
                correction = np.clip(
                    correction + correction_gain * pos_error_vec, -max_correction_pos, max_correction_pos
                )
                rot_norm = float(np.linalg.norm(rot_error_vec))
                if rot_norm > 1e-9:
                    step = min(1.0, max_correction_rot / max(rot_norm, 1e-9)) * correction_gain
                    rotation_correction = _rotation_about(rot_error_vec / rot_norm, step * rot_norm) @ rotation_correction
                q = self.solve_ik_best(
                    target_pos + correction,
                    rotation_correction @ target_rot,
                    q0=q,
                    iterations=ik_iterations,
                )
            for _ in range(chunk_steps):
                self.set_arm_ctrl(q)
                if grip_ctrl is not None:
                    self.set_gripper_ctrl(grip_ctrl)
                mujoco.mj_step(model, data)
                done_steps += 1
                if on_step is not None:
                    on_step(self)
                if done_steps >= total_steps:
                    break
        measured_pos, measured_rot = self.tcp_pose()
        pos_error, rot_error = self.tcp_error(target_pos, target_rot)
        return ServoResult(
            steps=done_steps,
            final_pos_error_m=pos_error,
            final_rot_error_rad=rot_error,
            # 1 cm 级别的到位精度是这些位置伺服臂在重力+接触下的实际水平；抓取和码放
            # 的容差都比它大（石头窄边判据留 15% 余量，码放判定容差 13 cm）。
            reached=pos_error < max(tolerance_pos, 1.5e-2) and rot_error < 6.0e-2,
        )

    def servo_translate(
        self,
        delta: np.ndarray,
        seconds: float,
        grip_ctrl=None,
        chunk_steps: int = 60,
        on_step=None,
    ) -> ServoResult:
        """只做平移的伺服：用雅可比位置解把 TCP 挪动 `delta`，姿态保持不变。

        为什么不复用 `servo_to`：那个每次重新解 IK，抬起过程中末端会横向漂几毫米，
        夹在指间的石头会被拖出去。实测同一个位姿，用雅可比纯平移抬 120 mm 石头跟着走
        （+75.7 mm 后被记录），用 IK 重解抬 160 mm 石头原地不动。
        """
        import mujoco

        model, data = self.model, self.data
        ik = self.ik_data
        dt = float(model.opt.timestep)
        total_steps = max(1, int(round(seconds / dt)))
        start_pos, start_rot = self.tcp_pose()
        target_pos = np.asarray(start_pos, dtype=float) + np.asarray(delta, dtype=float)
        q = self.arm_qpos()
        done_steps = 0
        while done_steps < total_steps:
            # 关键：误差要对**命令位姿**算，不是对实测位姿算。
            # 位置伺服带着几十毫米静差，用实测误差做增量的话每个 chunk 只补 4 mm，
            # 而静差一直把那 4 mm 吃掉——实测 0.5 s 只升 0.8 mm，看起来像"抬不动"。
            # 改成"命令位姿追目标"（增量累积在命令上），物理臂带着静差跟随，
            # 整段位移就能完整走完。
            ik.qpos[:] = data.qpos
            ik.qpos[self.qpos_addr] = q
            mujoco.mj_forward(model, ik)
            command_pos = ik.site_xpos[self._tcp_site_id].copy()
            error = target_pos - command_pos
            if float(np.linalg.norm(error)) < 2.0e-3:
                break
            ik.qpos[self.qpos_addr] = q
            mujoco.mj_forward(model, ik)
            jac_pos = np.zeros((3, model.nv))
            jac_rot = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, ik, jac_pos, jac_rot, self._tcp_site_id)
            step = np.clip(error, -0.008, 0.008)
            dq = np.linalg.pinv(jac_pos[:, self.dof_addr]) @ step
            q = np.clip(q + dq, self.lower, self.upper)
            for _ in range(chunk_steps):
                self.set_arm_ctrl(q)
                if grip_ctrl is not None:
                    self.set_gripper_ctrl(grip_ctrl)
                mujoco.mj_step(model, data)
                done_steps += 1
                if on_step is not None:
                    on_step(self)
                if done_steps >= total_steps:
                    break
        pos_error, rot_error = self.tcp_error(target_pos, start_rot)
        return ServoResult(
            steps=done_steps,
            final_pos_error_m=pos_error,
            final_rot_error_rad=rot_error,
            reached=pos_error < 1.5e-2,
        )

    def step(self, count: int = 1, grip_ctrl=None, q=None) -> None:
        import mujoco

        if q is not None:
            self.set_arm_ctrl(np.asarray(q, dtype=float))
        if grip_ctrl is not None:
            self.set_gripper_ctrl(grip_ctrl)
        for _ in range(max(0, int(count))):
            mujoco.mj_step(self.model, self.data)


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """罗德里格斯公式：绕单位轴 `axis` 转 `angle`。"""
    axis = _unit(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=float
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
