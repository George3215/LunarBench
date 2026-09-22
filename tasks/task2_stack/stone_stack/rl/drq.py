"""直接复用 facebookresearch/drqv2 的网络、增强与更新，不改上游算法文件。"""
from pathlib import Path
import sys
import torch

UPSTREAM = Path(__file__).resolve().parents[2] / 'policy/drqv2'
sys.path.insert(0, str(UPSTREAM))
from drqv2 import DrQV2Agent
import utils


@torch.no_grad()
def act_batch(agent, observations, step):
    """与上游 act 相同的分布，一次 GPU 前向为所有环境分别采样动作。"""
    features = agent.encoder(torch.as_tensor(observations, device=agent.device))
    dist = agent.actor(features, utils.schedule(agent.stddev_schedule, step))
    actions = dist.sample(clip=None)
    if step < agent.num_expl_steps:
        actions.uniform_(-1., 1.)
    return actions.cpu().numpy()


def make_agent(env, cfg):
    if cfg['observation'] != 'pixels':
        raise ValueError('DrQ-v2 requires pixels; use SAC for state observations')
    return DrQV2Agent(obs_shape=env.observation_space.shape, action_shape=env.action_space.shape,
                     device=cfg['device'], lr=cfg['learning_rate'], feature_dim=cfg['feature_dim'],
                     hidden_dim=cfg['hidden_dim'], critic_target_tau=.01,
                     num_expl_steps=cfg['learning_starts'], update_every_steps=cfg['update_every'],
                     stddev_schedule=cfg['stddev_schedule'], stddev_clip=.3, use_tb=True)


def save_agent(agent, path, step, config):
    # 只存 state_dict，避免依赖整个 Python 对象的 pickle 布局。
    names = ('encoder', 'actor', 'critic', 'critic_target', 'encoder_opt', 'actor_opt', 'critic_opt')
    torch.save(dict(step=step, config=config, **{n: getattr(agent, n).state_dict() for n in names}), path)


def load_agent(agent, path):
    checkpoint = torch.load(path, map_location=agent.device, weights_only=True)
    for name in ('encoder', 'actor', 'critic', 'critic_target', 'encoder_opt', 'actor_opt', 'critic_opt'):
        getattr(agent, name).load_state_dict(checkpoint[name])
    return checkpoint
