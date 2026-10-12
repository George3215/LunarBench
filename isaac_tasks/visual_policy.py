"""非对称视觉 PPO：Actor 读取三路 RGB 与配置的状态输入，Critic 读取完整训练状态。
直接控制使用本体；残差控制还明确提供石头中心、目标与阶段。

三台相机共享 CNN，按固定视角顺序拼接特征；图像不进入状态归一化器。
原始 uint8 RGB 存入 rollout，PPO 更新时重新编码，使 CNN 参与反向传播。
"""

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization
from torch import nn


class VisualActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, local_action_mask=False, **kwargs):
        views = obs["rgb"].shape[1]
        proprio_dim = obs["policy"].shape[-1]
        features = 128
        # 父类负责分布、Critic、PPO 接口；只替换 Actor 的观测编码。
        example = obs.clone(False)
        example["policy"] = torch.zeros(
            obs.batch_size[0], proprio_dim + views * features, device=obs.device
        )
        kwargs["actor_obs_normalization"] = False
        super().__init__(example, obs_groups, num_actions, **kwargs)
        self.local_action_mask = local_action_mask
        self.proprio_normalizer = EmpiricalNormalization(proprio_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, features),
            nn.ELU(),
        )

    def get_actor_obs(self, obs):
        rgb = obs["rgb"]  # [batch, view, height, width, channel], uint8
        batch, views, height, width, channels = rgb.shape
        images = rgb.reshape(batch * views, height, width, channels).permute(0, 3, 1, 2)
        encoded = self.encoder(images.float() / 255.0 - 0.5).reshape(batch, -1)
        return torch.cat([self.proprio_normalizer(obs["policy"]), encoded], dim=-1)

    def update_normalization(self, obs):
        self.proprio_normalizer.update(obs["policy"])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def act(self, obs, **kwargs):
        actions = super().act(obs, **kwargs)
        if self.local_action_mask:
            # context 最后一维是局部控制开关。程序阶段输出严格为零，
            # 用固定分布屏蔽这些样本的 Actor/探索梯度，Critic 仍学习完整回报。
            active = obs["policy"][:, -1:]
            self.distribution = torch.distributions.Normal(
                self.distribution.mean * active,
                torch.where(active.bool(), self.distribution.stddev, 1.0),
            )
            actions = actions * active
        return actions

    def act_inference(self, obs):
        actions = super().act_inference(obs)
        if self.local_action_mask:
            actions = actions * obs["policy"][:, -1:]
        return actions
