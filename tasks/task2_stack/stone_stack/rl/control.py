"""RL 的局部末端控制：不搜索 home/其他 IK 分支，不生成抓取路径。

每 8 个真实物理步更新一次小幅关节目标。模型仍决定完整的末端目标与夹爪开合。
只修改 ctrl；运动由 MuJoCo 积分，ik_data 只用于当前命令附近的运动学计算。
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..robots.control import ServoResult


def bounded(vector, maximum):
    return vector * min(1., maximum / max(float(np.linalg.norm(vector)), 1e-12))


def servo_local(controller, target_pos, target_rot, seconds, grip_ctrl, on_step):
    c = controller
    model, data, ik = c.model, c.data, c.ik_data
    if ik is data:
        raise ValueError('Local RL IK requires separate scratch MjData')
    steps = max(1, round(seconds / model.opt.timestep))
    q = data.ctrl[c._arm_actuator_ids()].copy()
    q = np.clip(q, c.arm_qpos() - .15, c.arm_qpos() + .15)
    ik.qpos[:] = data.qpos
    ik.qpos[c.qpos_addr] = q
    mujoco.mj_forward(model, ik)
    start_pos, start_rot = c.tcp_pose()
    # 保留已有伺服静差：零增量应延续支撑重力的命令，不能每次把测量下垂写回目标。
    position_bias = bounded(ik.site_xpos[c._tcp_site_id] - start_pos, .05)
    rotation_bias = Rotation.from_rotvec(bounded(Rotation.from_matrix(
        ik.site_xmat[c._tcp_site_id].reshape(3, 3) @ start_rot.T).as_rotvec(), .2)).as_matrix()
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for step in range(steps):
        if step % 8 == 0:
            measured_pos, measured_rot = c.tcp_pose()
            measured_q = c.arm_qpos()
            q = np.clip(q, measured_q - .15, measured_q + .15)
            ik.qpos[:] = data.qpos
            ik.qpos[c.qpos_addr] = q
            mujoco.mj_forward(model, ik)
            # 小幅实测误差补偿静差；总反馈被限制，避免持续接触时积分失控。
            pos_goal = target_pos + position_bias + bounded(.5 * (target_pos - measured_pos), .01)
            rot_feedback = bounded(.5 * Rotation.from_matrix(target_rot @ measured_rot.T).as_rotvec(), .05)
            rot_goal = Rotation.from_rotvec(rot_feedback).as_matrix() @ rotation_bias @ target_rot
            dp = bounded(pos_goal - ik.site_xpos[c._tcp_site_id], .01)
            dr = bounded(Rotation.from_matrix(rot_goal @ ik.site_xmat[c._tcp_site_id].reshape(3, 3).T).as_rotvec(), .05)
            mujoco.mj_jacSite(model, ik, jac_pos, jac_rot, c._tcp_site_id)
            jac = np.vstack([jac_pos[:, c.dof_addr], jac_rot[:, c.dof_addr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + .03 ** 2 * np.eye(6), np.concatenate([dp, dr]))
            q = np.clip(q + np.clip(dq, -.03, .03), c.lower, c.upper)
            q = np.clip(q, measured_q - .15, measured_q + .15)
            c.set_arm_ctrl(q)
            c.set_gripper_ctrl(grip_ctrl)
        mujoco.mj_step(model, data)
        on_step(c)
    pos, rot = c.tcp_pose()
    error = float(np.linalg.norm(target_pos - pos))
    angle = float(Rotation.from_matrix(target_rot @ rot.T).magnitude())
    return ServoResult(steps, error, angle, error < .002 and angle < .02)
