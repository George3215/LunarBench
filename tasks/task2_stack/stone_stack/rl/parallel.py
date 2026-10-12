"""CPU MuJoCo 多进程采样；GPU 只负责共享策略和各进程的 EGL 渲染。

使用 spawn，避免继承父进程的 CUDA/EGL 上下文。禁用自动重置，先保存
真正的末帧，再仅重置结束的环境，保证时间截断仍可正确 bootstrap。
"""
from functools import partial
import os

import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from .env import StoneStackEnv


class PausableEnv(gym.Wrapper):
    """最后不足一批时，None 只返回缓存观测，不推进物理、不计步。"""
    def reset(self, **kwargs):
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info

    def step(self, action):
        if action is None:
            return self.last_obs, 0., False, False, {}
        result = self.env.step(action)
        self.last_obs = result[0]
        return result


def make_worker(config):
    return PausableEnv(StoneStackEnv(config))


def unpack_info(info, index):
    result = {}
    for key, value in info.items():
        if key.startswith('_') or not info['_' + key][index]:
            continue
        item = unpack_info(value, index) if isinstance(value, dict) else value[index]
        result[key] = item.item() if isinstance(item, np.generic) else item
    return result


class Collector:
    def __init__(self, env, config):
        self.env = env
        self.count = config['policy']['rl']['num_envs']
        self.vector = None
        if self.count > 1:
            # 子进程只做物理与渲染，限制 BLAS 线程，避免每个 worker 再占满 CPU。
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            self.vector = AsyncVectorEnv([partial(make_worker, config)] * self.count,
                                         context='spawn', autoreset_mode=AutoresetMode.DISABLED)

    def reset(self, seed):
        if self.vector is not None:
            return self.vector.reset(seed=seed)[0]
        return self.env.reset(seed=seed)[0][None]

    def step(self, actions):
        if self.vector is not None:
            active = len(actions)
            padded = list(actions) + [None] * (self.count - active)
            obs, rewards, terminated, truncated, info = self.vector.step(padded)
            return obs[:active], rewards[:active], terminated[:active], truncated[:active], [unpack_info(info, i) for i in range(active)]
        obs, reward, terminated, truncated, info = self.env.step(actions[0])
        return obs[None], np.array([reward]), np.array([terminated]), np.array([truncated]), [info]

    def reset_finished(self, observations, done):
        if not done.any():
            return observations
        if self.vector is not None:
            return self.vector.reset(options={'reset_mask': done})[0]
        return self.env.reset()[0][None]

    def close(self):
        if self.vector is not None:
            self.vector.close()
