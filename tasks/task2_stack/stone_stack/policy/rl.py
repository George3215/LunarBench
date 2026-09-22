"""RL 策略注册描述；训练和评估由 stone_stack.rl 直接驱动 Gymnasium 环境。"""
from dataclasses import dataclass


@dataclass
class RLPolicy:
    config: dict
    name: str = 'rl'
    description: str = 'DrQ-v2 图像策略 / SAC 状态策略，直接控制 TCP 与夹爪'

    @property
    def needs_cameras(self):
        return tuple(self.config['cameras']) if self.config['observation'] == 'pixels' else ()
