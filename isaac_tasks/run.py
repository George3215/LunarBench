"""三个入口参数；环境条件、训练规模和检查点统一在 YAML 中修改。"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "configs/task2.yaml"))
parser.add_argument("--mode", choices=["train", "eval"], default="train")
parser.add_argument("--view", action="store_true")
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text())
app = AppLauncher(
    headless=not args.view, enable_cameras=config["cameras"], device="cuda:0"
).app
logging.basicConfig(level=logging.INFO, force=True)
try:
    import torch
    from task2 import Task2Cfg, Task2Env

    residual = config.get("control", "direct") == "residual"
    if residual:
        from residual_stack import ResidualStackCfg, ResidualStackEnv

        cfg = ResidualStackCfg()
        cfg.residual.update(config.get("residual", {}))
    else:
        cfg = Task2Cfg()
    cfg.scene.num_envs = config["num_envs"]
    cfg.stage = config["stage"]
    cfg.reward_weights.update(config.get("reward", {}))
    cfg.curriculum.update(config.get("curriculum", {}))
    cfg.cameras = config["cameras"]
    visual = config.get("observation", "state") == "visual"
    cfg.observation_mode = "visual" if visual else "state"
    cfg.camera_width = config.get("camera_width", 640)
    cfg.camera_height = config.get("camera_height", 480)
    if visual:
        cfg.observation_space = 50 if residual else 25
        cfg.state_space = 210 if residual else 185
    cfg.sim.dt = config["physics_dt"]
    cfg.decimation = config["control_steps"]
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = config.get("episode_steps", 1000) * (
        cfg.sim.dt * cfg.decimation
    )
    env = ResidualStackEnv(cfg) if residual else Task2Env(cfg)
    logging.basicConfig(level=logging.INFO, force=True)

    output = ROOT / "outputs" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(output))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        if visual:
            # RSL-RL 3.1 的扩展类解析使用 runner 命名空间，不修改上游安装文件。
            import rsl_rl.runners.on_policy_runner as runner_module
            from visual_policy import VisualActorCritic

            runner_module.VisualActorCritic = VisualActorCritic
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        train_cfg = {
            "num_steps_per_env": config.get("rollout_steps", 24),
            "save_interval": 50,
            "obs_groups": {
                "policy": ["policy"],
                "critic": ["critic" if visual else "policy"],
            },
            "policy": {
                "class_name": "VisualActorCritic" if visual else "ActorCritic",
                "init_noise_std": config.get("init_noise_std", 0.5),
                "noise_std_type": "log",
                "actor_hidden_dims": [256, 128, 64],
                "critic_hidden_dims": [256, 128, 64],
                "activation": "elu",
                "actor_obs_normalization": True,
                "critic_obs_normalization": True,
            },
            "algorithm": {
                "class_name": "PPO",
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "num_learning_epochs": 4,
                # 增大此值可降低视觉 CNN 单次反向传播的显存峰值。
                "num_mini_batches": config.get("num_mini_batches", 4),
                "learning_rate": 0.0003,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            },
            "logger": config.get("logger", "tensorboard"),
            "wandb_project": "task2-isaac-stack",
            "run_name": output.name,
        }
        if visual:
            train_cfg["policy"]["local_action_mask"] = residual
        runner = OnPolicyRunner(
            wrapped, train_cfg, log_dir=str(output), device="cuda:0"
        )
        if visual:
            # RSL-RL 默认按 float32 分配所有观测；图像缓存改为 uint8，减少 75% 容量。
            storage = runner.alg.storage.observations
            storage["rgb"] = storage["rgb"].to(torch.uint8)
        if config["checkpoint"]:
            runner.load(config["checkpoint"], load_optimizer=args.mode == "train")
        elif residual:
            # 未训练策略从零残差开始；训练探索仍受局部区域和残差幅度限制。
            torch.nn.init.zeros_(runner.alg.policy.actor[-1].weight)
            torch.nn.init.zeros_(runner.alg.policy.actor[-1].bias)
        if args.mode == "train":
            before = [p.detach().clone() for p in runner.alg.policy.parameters()]
            encoder_before = (
                [p.detach().clone() for p in runner.alg.policy.encoder.parameters()]
                if visual
                else []
            )
            tick = time.perf_counter()
            runner.learn(
                num_learning_iterations=config["iterations"],
                init_at_random_ep_len=False,
            )
            elapsed = time.perf_counter() - tick
            changed = any(
                not torch.equal(a, b)
                for a, b in zip(before, runner.alg.policy.parameters())
            )
            finite = all(
                bool(torch.isfinite(p).all()) for p in runner.alg.policy.parameters()
            )
            runner.save(str(output / "model.pt"))
            transitions = (
                config["iterations"] * train_cfg["num_steps_per_env"] * env.num_envs
            )
            report = {
                "mode": "train",
                "num_envs": env.num_envs,
                "iterations": config["iterations"],
                "transitions": transitions,
                "elapsed_s": elapsed,
                "transitions_per_second": transitions / elapsed,
                "actor_parameters_changed": changed,
                "parameters_finite": finite,
                "observation": cfg.observation_mode,
                "encoder_parameters_changed": any(
                    not torch.equal(a, b)
                    for a, b in zip(
                        encoder_before, runner.alg.policy.encoder.parameters()
                    )
                )
                if visual
                else None,
            }
        else:
            if not config["checkpoint"]:
                raise ValueError("eval requires checkpoint in YAML")
            policy = runner.get_inference_policy(device=env.device)
            observations = wrapped.get_observations()
            episodes = successes = native = timeouts = 0
            returns = torch.zeros(env.num_envs, device=env.device)
            completed_returns = []
            with torch.inference_mode():
                for step in range(config["steps"]):
                    observations, reward, done, info = wrapped.step(
                        policy(observations)
                    )
                    returns += reward
                    ended = done.bool()
                    episodes += int(ended.sum())
                    successes += int((env._result[2] & ended).sum())
                    native += int((env._result[3] & ended).sum())
                    timeouts += int((info["time_outs"].bool() & ended).sum())
                    completed_returns.extend(returns[ended].tolist())
                    returns[ended] = 0
            report = {
                "mode": "eval",
                "checkpoint": config["checkpoint"],
                "stage": config["stage"],
                "num_envs": env.num_envs,
                "steps": config["steps"],
                "completed_episodes": episodes,
                "stage_successes": successes,
                "full_wall_successes": native,
                "timeouts": timeouts,
                "stage_success_rate": successes / episodes if episodes else None,
                "full_wall_success_rate": native / episodes if episodes else None,
                "mean_return": sum(completed_returns) / episodes if episodes else None,
                "scope": "fixed initial scene; unfinished episodes excluded",
            }
        if env.cameras:
            from PIL import Image

            for name, camera in env.cameras.items():
                rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
                Image.fromarray(rgb).save(output / (name + ".png"))
        if residual:
            report["residual"] = env.training_metrics()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.info("TASK2 ISAAC RESULT %s output=%s", report, output)
        if args.mode == "train" and train_cfg["logger"] == "wandb":
            runner.writer.stop()
    finally:
        env.close()
except Exception:
    logger.exception("Isaac TASK2 failed")
    raise
finally:
    app.close()
