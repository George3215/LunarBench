"""小型 n-step replay；截断保留 bootstrap，真实终止清零，绝不跨 reset 拼接。"""
from collections import deque
import numpy as np


class Replay:
    def __init__(self, capacity, batch_size, nstep, gamma, seed):
        self.items = deque(maxlen=capacity)
        self.pending = deque()
        self.streams = {0: self.pending}
        self.batch_size, self.nstep, self.gamma = batch_size, nstep, gamma
        self.rng = np.random.default_rng(seed)

    def _commit(self, pending):
        reward, discount = 0., 1.
        for transition in list(pending)[:self.nstep]:
            obs, action, r, next_obs, terminated = transition
            reward += discount * r
            discount *= self.gamma * (not terminated)
            if terminated:
                break
        first_obs, first_action = pending[0][:2]
        self.items.append((first_obs, first_action, [reward], [discount], next_obs))
        pending.popleft()

    def add(self, obs, action, reward, next_obs, terminated, truncated, env_id=0):
        # 各环境独立累积 n-step；共享采样池，但不能拼接不同环境的轨迹。
        pending = self.streams.setdefault(env_id, deque())
        pending.append((obs, action.copy(), reward, next_obs, terminated))
        if len(pending) >= self.nstep:
            self._commit(pending)
        if terminated or truncated:
            while pending:
                self._commit(pending)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self

    def __next__(self):
        indices = self.rng.integers(len(self.items), size=self.batch_size)
        samples = [self.items[int(i)] for i in indices]
        obs, action, reward, discount, next_obs = zip(*samples)
        return (np.stack(obs), np.asarray(action, np.float32), np.asarray(reward, np.float32),
                np.asarray(discount, np.float32), np.stack(next_obs))
