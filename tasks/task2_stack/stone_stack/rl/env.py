"""固定十块石头 4+3+2+1 环境。只有 reset 恢复场景，step 仅执行机器人动作。

pixels 给 DrQ-v2：三视角按通道拼接，uint8 CHW；state 给 SAC：显式仿真特权状态。
稠密奖励可读取物体真值，但 pixels 策略观测不包含这些真值。
"""
from collections import deque
import threading
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..upstream_scene import build_upstream
from ..execution import Executor
from ..evaluation import evaluate_wall
from .reward import wall_targets, dense_terms, potential


class StoneStackEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 10}

    def __init__(self, config, render_mode=None):
        super().__init__()
        self.config = config
        self.cfg = config['policy']['rl']
        if config['robot']['courses'] != [4, 3, 2, 1] or config['robot']['stones'] != 10:
            raise ValueError('The first RL task requires exactly ten stones and courses 4+3+2+1')
        if not 0 < self.cfg['max_delta_m'] <= .05 or not 0 < self.cfg['max_rotation_rad'] <= .35:
            raise ValueError('EEF safeguards are at most 0.05 m / 0.35 rad')
        self.render_mode = render_mode
        self.built, self.workcell, self.profile = build_upstream(config)
        self.e = Executor(self.built, self.profile, None)
        self.model, self.data = self.e.model, self.e.data
        self.names = self.e.stone_names()
        self.targets = wall_targets([4, 3, 2, 1], self.cfg['slot_spacing_m'], self.workcell.mean_thickness)
        self.e.controller.set_arm_qpos(np.asarray(self.profile.home_qpos))
        self.e.controller.forward()
        self.e.controller.set_arm_ctrl(self.e.controller.arm_qpos())
        self.e.controller.set_gripper_ctrl(self.profile.gripper.open_ctrl)
        self.e.settle(.65)
        self.initial_z = np.array([self.e.stone_center(n)[2] for n in self.names])
        self.state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
        self.initial = np.empty(mujoco.mj_stateSize(self.model, self.state_kind))
        mujoco.mj_getState(self.model, self.data, self.initial, self.state_kind)
        self.action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self.renderer = None
        self.viewer = None
        self.frames = deque(maxlen=self.cfg['frame_stack'])
        if self.cfg['observation'] == 'pixels':
            channels = 3 * len(self.cfg['cameras']) * self.cfg['frame_stack']
            self.observation_space = spaces.Box(0, 255, shape=(channels, 84, 84), dtype=np.uint8)
        elif self.cfg['observation'] == 'state':
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=self._state().shape, dtype=np.float32)
        else:
            raise ValueError('observation must be pixels or state')

    def _state(self):
        c = self.e.controller
        pos, rot = c.tcp_pose()
        values = [c.arm_qpos(), self.data.qvel[c.dof_addr], pos, rot.ravel(),
                  [self.e.measured_opening() - self.e.pad_thickness_m], self.targets.ravel()]
        for name in self.names:
            body = self.model.body(name).id
            dof = int(self.model.jnt_dofadr[self.model.joint(name + '_free').id])
            values.extend([self.e.stone_center(name), self.data.xquat[body], self.data.qvel[dof:dof + 6]])
        return np.concatenate(values).astype(np.float32)

    def _pixels(self):
        # 首次取图再创建 GL 上下文，让训练器先初始化 PyTorch/Triton。
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 84, 84)
        images = []
        for name in self.cfg['cameras']:
            self.renderer.update_scene(self.data, camera=name)
            images.append(self.renderer.render().copy().transpose(2, 0, 1))
        return np.concatenate(images, axis=0)

    def _observation(self, reset=False):
        if self.cfg['observation'] == 'state':
            return self._state()
        frame = self._pixels()
        if reset:
            self.frames.clear()
            for _ in range(self.cfg['frame_stack']):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
        return np.concatenate(self.frames, axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.render_mode == 'human' and self.viewer is None:
            # EGL 离屏上下文先于 GLFW viewer，二者共享同一个模型。
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 84, 84)
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.e.viewer = self.viewer

        mujoco.mj_setState(self.model, self.data, self.initial, self.state_kind)
        mujoco.mj_forward(self.model, self.data)
        self.step_id, self.stable_seconds = 0, 0.0
        self.finished = False
        evaluation = evaluate_wall(self.e)
        terms = dense_terms(self.e, self.targets, self.initial_z, evaluation, self.cfg['goal_tolerance_m'])
        self.previous_potential = potential(terms, self.cfg['reward'])
        self.e._sync_viewer(force=True)
        return self._observation(reset=True), {'is_success': False, 'reward_terms': terms}

    def step(self, action):
        if self.finished:
            raise RuntimeError('Episode ended; call reset before step')
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Action must be seven finite numbers in [-1, 1]')
        # Box -> 单位球映射；角落动作也不会超过平移/旋转范数上限。
        dp = action[:3].astype(float) / max(1., float(np.linalg.norm(action[:3].astype(float)))) * self.cfg['max_delta_m']
        dr = action[3:6].astype(float) / max(1., float(np.linalg.norm(action[3:6].astype(float)))) * self.cfg['max_rotation_rad']
        pos, rot = self.e.controller.tcp_pose()
        opening = float((action[6] + 1) / 2 * self.profile.gripper.max_opening_m)
        before_time = float(self.data.time)
        servo = self.e.controller.servo_to(
            pos + dp, Rotation.from_rotvec(dr).as_matrix() @ rot,
            seconds=self.cfg['control_seconds'], grip_ctrl=self.e._ctrl_for_width(opening),
            preserve_hold_ctrl=True, tolerance_pos=.002, tolerance_rot=.02,
            hold_chunks=int(np.ceil(self.cfg["control_seconds"] / self.model.opt.timestep)) + 1, on_step=lambda _: self.e._sync_viewer())
        self.step_id += 1
        evaluation = evaluate_wall(self.e)
        self.stable_seconds = self.stable_seconds + float(self.data.time) - before_time if evaluation['success'] else 0.
        success = evaluation['success'] and self.stable_seconds >= self.cfg['success_hold_seconds']
        centers = np.array([self.e.stone_center(n) for n in self.names])
        fallen = bool(np.any(centers[:, 2] < -.10))
        terminated = bool(success or fallen)
        truncated = bool(self.step_id >= self.cfg['episode_steps'] and not terminated)
        terms = dense_terms(self.e, self.targets, self.initial_z, evaluation, self.cfg['goal_tolerance_m'])
        phi = potential(terms, self.cfg['reward'])
        # 真终止时吸收状态势能为 0；time limit 则保留 bootstrap 的下一观测。
        next_phi = 0. if terminated else phi
        shaping = self.cfg['gamma'] * next_phi - self.previous_potential
        self.previous_potential = phi
        weights = self.cfg['reward']
        reward = shaping - weights['step_cost'] - weights['action_cost'] * float(np.mean(action ** 2))
        reward += weights['success_bonus'] * success - weights['fall_penalty'] * fallen
        self.finished = terminated or truncated
        info = dict(is_success=bool(success), termination_reason='success' if success else
                    'fallen' if fallen else 'time_limit' if truncated else None,
                    reward_terms=terms, potential=phi, shaping=shaping, evaluation=evaluation,
                    step_id=self.step_id, stable_seconds=self.stable_seconds,
                    target_translation_m=float(np.linalg.norm(dp)), target_rotation_rad=float(np.linalg.norm(dr)),
                    tcp_error_m=float(servo.final_pos_error_m))
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 84, 84)
        self.renderer.update_scene(self.data, camera='front')
        return self.renderer.render().copy()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            for thread in threading.enumerate():
                if thread.name.endswith('(_launch_internal)'):
                    thread.join()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
