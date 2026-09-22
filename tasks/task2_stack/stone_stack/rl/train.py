"""训练/加载评估入口。输出配置、逐步奖励、损失、模型与环境原生判定。"""
from datetime import datetime
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from PIL import Image

from .env import StoneStackEnv
from .replay import Replay

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def evaluate(env, predict, episodes, output):
    results = []
    with (output / 'evaluation_steps.jsonl').open('w', buffering=1) as trace:
        for episode in range(episodes):
            obs, _ = env.reset()
            total = 0.
            while True:
                action = np.asarray(predict(obs), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                trace.write(json.dumps(dict(episode=episode, action=action.tolist(), reward=reward, **info)) + '\n')
                if terminated or truncated:
                    break
            Image.fromarray(env.render()).save(output / f'eval_{episode:03d}_front.png')
            results.append(dict(episode=episode, reward=total, steps=info['step_id'],
                                success=info['is_success'], reason=info['termination_reason'],
                                evaluation=info['evaluation']))
    report = dict(episodes=results, success_rate=float(np.mean([r['success'] for r in results])))
    write_json(output / 'evaluation.json', report)
    return report


def train_drq(env, config, output, tracking=None):
    from time import perf_counter
    from .drq import make_agent, save_agent, load_agent, act_batch
    from .parallel import Collector
    cfg = config['policy']['rl']
    count = cfg['num_envs']
    if count < 1 or cfg['total_steps'] % count:
        raise ValueError('num_envs must be positive and divide total_steps exactly')
    agent = make_agent(env, cfg)
    start = 0
    if cfg['checkpoint']:
        start = load_agent(agent, cfg['checkpoint'])['step']
    replay = Replay(cfg['replay_capacity'], cfg['batch_size'], cfg['nstep'], cfg['gamma'], config['seed'])
    updates, episodes = 0, 0
    episode_returns = np.zeros(count)
    last_metrics = {}
    first_parameters = [p.detach().clone() for p in agent.actor.parameters()]
    collector = Collector(env, config)
    try:
        obs = collector.reset(config['seed'])
        started = perf_counter()
        collect_seconds, update_seconds = 0., 0.
        LOGGER.info('DrQ-v2 training device=%s num_envs=%d total_transitions=%d', cfg['device'], count, cfg['total_steps'])
        with (output / 'train.jsonl').open('w', buffering=1) as trace:
            for base_step in range(start, start + cfg['total_steps'], count):
                tick = perf_counter()
                actions = act_batch(agent, obs, base_step)
                next_obs, rewards, terminated, truncated, infos = collector.step(actions)
                collect_seconds += perf_counter() - tick
                # 每条真实 transition 对应一个全局步，保持原有每两步一次更新的比例。
                # 不能把一次向量 step 当一次样本，否则并行越多，训练更新反而越少。
                for i in range(count):
                    step = base_step + i
                    reward, info = float(rewards[i]), infos[i]
                    replay.add(obs[i], actions[i], reward, next_obs[i], bool(terminated[i]), bool(truncated[i]), env_id=i)
                    tick = perf_counter()
                    metrics = {}
                    if len(replay) >= cfg['batch_size'] and step >= cfg['learning_starts']:
                        metrics = agent.update(iter(replay), step)
                        updates += bool(metrics)
                        if metrics:
                            last_metrics = metrics
                    update_seconds += perf_counter() - tick
                    episode_returns[i] += reward
                    trace.write(json.dumps(dict(step=step + 1, env_id=i, reward=reward,
                                                action=actions[i].tolist(), **info, **metrics)) + '\n')
                    done = bool(terminated[i] or truncated[i])
                    if tracking is not None and ((step + 1) % cfg['log_every'] == 0 or done):
                        record = {'train/reward': reward, 'train/potential': info['potential'],
                                  'train/updates': updates, 'train/replay_size': len(replay),
                                  'perf/transitions_per_second': (step + 1 - start) / (perf_counter() - started),
                                  'perf/collect_seconds': collect_seconds, 'perf/update_seconds': update_seconds}
                        record.update({'loss/' + key: value for key, value in last_metrics.items()})
                        record.update({'reward/' + key: value for key, value in info['reward_terms'].items()})
                        if done:
                            record.update({'episode/return': float(episode_returns[i]), 'episode/length': info['step_id'],
                                           'episode/env_id': i, 'episode/success': int(info['is_success']),
                                           'episode/fallen': int(terminated[i] and not info['is_success']),
                                           'episode/time_limit': int(truncated[i])})
                        tracking.log(record, step=step + 1)
                    if done:
                        episodes += 1
                        LOGGER.info('RL episode=%d env=%d steps=%d return=%.3f success=%s',
                                    episodes, i, info['step_id'], episode_returns[i], info['is_success'])
                        episode_returns[i] = 0.
                    if (step + 1) % cfg['log_every'] == 0:
                        LOGGER.info('DrQ-v2 step=%d updates=%d replay=%d critic_loss=%s samples/s=%.2f',
                                    step + 1, updates, len(replay), last_metrics.get('critic_loss'),
                                    (step + 1 - start) / (perf_counter() - started))
                # 已执行完整批次后再存盘，checkpoint 步数与真实采样量一致。
                end_step = base_step + count
                if end_step // cfg['save_every'] > base_step // cfg['save_every']:
                    save_agent(agent, output / 'checkpoint.pt', end_step, config)
                obs = collector.reset_finished(next_obs, terminated | truncated)
        elapsed = perf_counter() - started
    finally:
        collector.close()
    save_agent(agent, output / 'checkpoint.pt', start + cfg['total_steps'], config)
    changed = any(not torch.equal(a, b) for a, b in zip(first_parameters, agent.actor.parameters()))
    summary = dict(algorithm='drqv2', environment_steps=cfg['total_steps'], num_envs=count,
                   device=cfg['device'], updates=updates, elapsed_seconds=elapsed,
                   transitions_per_second=cfg['total_steps'] / elapsed,
                   collect_seconds=collect_seconds, update_seconds=update_seconds,
                   completed_episodes=episodes, actor_parameters_changed=changed,
                   checkpoint=str((output / 'checkpoint.pt').resolve()),
                   resume_note='Weights/optimizers resume; replay is freshly collected.')
    write_json(output / 'training.json', summary)
    restored = make_agent(env, cfg)
    load_agent(restored, output / 'checkpoint.pt')
    def predict(observation):
        with torch.no_grad():
            return restored.act(observation, start + cfg['total_steps'], eval_mode=True)
    summary['evaluation'] = evaluate(env, predict, cfg['eval_episodes'], output)
    return summary


def train_sac(env, config, output):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    cfg = config['policy']['rl']
    if cfg['observation'] != 'state':
        raise ValueError('This SAC baseline uses state observations; DrQ-v2 uses pixels')
    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % cfg['log_every'] == 0:
                LOGGER.info('SAC step=%d reward=%.3f', self.num_timesteps, self.locals['rewards'][0])
            return True
    if cfg['checkpoint']:
        model = SAC.load(cfg['checkpoint'], env=env, device=cfg['device'])
    else:
        model = SAC('MlpPolicy', env, learning_rate=cfg['learning_rate'], gamma=cfg['gamma'],
                    buffer_size=cfg['replay_capacity'], learning_starts=cfg['learning_starts'],
                    batch_size=cfg['batch_size'], train_freq=1, gradient_steps=1,
                    policy_kwargs=dict(net_arch=[cfg['hidden_dim'], cfg['hidden_dim']]),
                    device=cfg['device'], seed=config['seed'], verbose=0)
    model.set_logger(configure(str(output), ['csv']))
    initial = [p.detach().clone() for p in model.actor.parameters()]
    model.learn(total_timesteps=cfg['total_steps'], callback=Progress(), reset_num_timesteps=not bool(cfg['checkpoint']))
    model.save(output / 'checkpoint')
    changed = any(not torch.equal(a, b) for a, b in zip(initial, model.actor.parameters()))
    restored = SAC.load(output / 'checkpoint.zip', device=cfg['device'])
    summary = dict(algorithm='sac', environment_steps=cfg['total_steps'], updates=model._n_updates,
                   actor_parameters_changed=changed, checkpoint=str((output / 'checkpoint.zip').resolve()))
    write_json(output / 'training.json', summary)
    summary['evaluation'] = evaluate(env, lambda obs: restored.predict(obs, deterministic=True)[0],
                                     cfg['eval_episodes'], output)
    return summary


def run_rl(config, mode, view=False):
    cfg = config['policy']['rl']
    if cfg['checkpoint']:
        cfg['checkpoint'] = str((ROOT / cfg['checkpoint']).resolve())
    if mode not in ('train', 'eval', 'run'):
        raise ValueError('RL modes: train, eval/run')
    if mode == 'train' and cfg['num_envs'] > 1 and (view or cfg['algorithm'] != 'drqv2'):
        raise ValueError('Parallel collection currently uses DrQ-v2 without --view; view saved policies with --mode eval')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    output = ROOT / 'outputs' / ('rl_' + cfg['algorithm'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
    tracking = None
    if mode == 'train' and cfg['wandb']['enabled']:
        import wandb
        tracking = wandb.init(project=cfg['wandb']['project'], name=output.name,
                              config=config, dir=str(output))
    env = None
    try:
        env = StoneStackEnv(config, render_mode='human' if view else None)
        (output / 'scene.xml').write_text(env.built.xml)
        write_json(output / 'task.json', dict(task='stack10_4321', courses=[4, 3, 2, 1],
                   stone_names=env.names, targets_m=env.targets.tolist(), observation=cfg['observation'],
                   action='world translation xyz, world rotation vector xyz, opening; normalized [-1,1]',
                   success_criterion='task2_contact_wall_v1 plus stability hold'))
        if mode == 'train':
            if cfg['algorithm'] == 'drqv2':
                summary = train_drq(env, config, output, tracking)
            else:
                summary = train_sac(env, config, output)
        else:
            if not cfg['checkpoint']:
                raise ValueError('Set policy.rl.checkpoint to a trained model for eval/run')
            if cfg['algorithm'] == 'drqv2':
                from .drq import make_agent, load_agent
                agent = make_agent(env, cfg)
                checkpoint = load_agent(agent, cfg['checkpoint'])
                def predict(obs):
                    with torch.no_grad():
                        return agent.act(obs, checkpoint['step'], eval_mode=True)
            elif cfg['algorithm'] == 'sac':
                from stable_baselines3 import SAC
                agent = SAC.load(cfg['checkpoint'], device=cfg['device'])
                predict = lambda obs: agent.predict(obs, deterministic=True)[0]
            else:
                raise ValueError('algorithm must be drqv2 or sac')
            summary = evaluate(env, predict, cfg['eval_episodes'], output)
        write_json(output / 'summary.json', summary)
        if tracking is not None:
            tracking.summary.update(summary)
        LOGGER.info('RL 完成：%s', output)
        LOGGER.info('结果：%s', summary)
    except BaseException:
        if tracking is not None:
            tracking.finish(exit_code=1)
            tracking = None
        raise
    finally:
        if env is not None:
            env.close()
        if tracking is not None:
            tracking.finish()
    return 0
