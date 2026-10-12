"""显式离线验收入口。训练和评估不导入此文件，也不执行相机自检。"""

import argparse
import json
import logging
import time
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "configs/view.yaml"))
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text())
app = AppLauncher(headless=True, enable_cameras=config["cameras"], device="cuda:0").app
try:
    import torch
    from task2 import META, Task2Cfg, Task2Env

    cfg = Task2Cfg()
    cfg.scene.num_envs = config["num_envs"]
    cfg.stage = config["stage"]
    cfg.cameras = config["cameras"]
    cfg.camera_width = config.get("camera_width", 640)
    cfg.camera_height = config.get("camera_height", 480)
    cfg.sim.dt = config["physics_dt"]
    cfg.decimation = config["control_steps"]
    cfg.sim.render_interval = cfg.decimation
    env = Task2Env(cfg)
    logging.basicConfig(level=logging.INFO, force=True)
    output = ROOT / "outputs" / ("validation_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    try:
        observations, _ = env.reset()
        start = (env.tcp_pose()[0] - env.scene.env_origins).clone()
        expected = torch.tensor(META["initial_tcp_position"], device=env.device)
        initial_error = torch.linalg.vector_norm(start - expected, dim=-1)
        source_positions = torch.tensor(
            [s["position"] for s in META["stones"]], device=env.device
        )
        positions = (
            torch.stack([s.data.root_pos_w for s in env.stones], dim=1)
            - env.scene.env_origins[:, None]
        )
        stone_error = torch.linalg.vector_norm(
            positions - source_positions, dim=-1
        ).max()
        masses = torch.stack(
            [s.root_physx_view.get_masses()[0, 0].cpu() for s in env.stones]
        )
        source_masses = torch.tensor([s["mass"] for s in META["stones"]])
        mass_error = float((masses - source_masses).abs().max())
        camera_errors = {}
        for name, camera in env.cameras.items():
            expected_position = torch.tensor(
                META["cameras"][name]["initial_position"], device=env.device
            )
            camera_errors[name] = float(
                (camera.data.pos_w - env.scene.env_origins - expected_position)
                .norm(dim=-1)
                .max()
            )
        tick = time.perf_counter()
        maximum_speed = 0.0
        for step in range(config["steps"]):
            actions = torch.zeros((env.num_envs, 7), device=env.device)
            actions[:, 6] = 1.0
            if step >= 2:
                actions[:, 2] = 0.1
                if env.num_envs > 1:
                    actions[1, :6] = 0.0  # 第二个环境保持不动，检查环境间隔离。
            if config.get("random_actions", False):
                actions.uniform_(-1, 1)
            observations, reward, done, truncated, info = env.step(actions)
            assert torch.isfinite(observations["policy"]).all(), "Non-finite state"
            assert torch.isfinite(reward).all(), "Non-finite reward"
            maximum_speed = max(
                maximum_speed, float(env.robot.data.joint_vel.abs().max())
            )
        elapsed = time.perf_counter() - tick
        displacement = env.tcp_pose()[0] - env.scene.env_origins - start
        ground_forces = torch.stack(
            [s.data.force_matrix_w[:, 0, 10] for s in env.contacts], dim=1
        )
        mimic_error = []
        for pair in META["mimic"]:
            ids = env.robot.find_joints(
                [pair["joint"], pair["reference"]], preserve_order=True
            )[0]
            q = env.robot.data.joint_pos[:, ids]
            mimic_error.append(float((q[:, 0] + pair["gearing"] * q[:, 1]).abs().max()))
        report = {
            "num_envs": env.num_envs,
            "device": env.device,
            "steps": config["steps"],
            "transitions_per_second": env.num_envs * config["steps"] / elapsed,
            "physics_dt": cfg.sim.dt,
            "control_steps": cfg.decimation,
            "initial_tcp_error_max_m": float(initial_error.max()),
            "initial_stone_error_max_m": float(stone_error),
            "stone_mass_error_max_kg": mass_error,
            "initial_camera_position_errors_m": camera_errors,
            "tcp_displacement_first_env_m": displacement[0].tolist(),
            "stationary_env_drift_m": float(displacement[1].norm())
            if env.num_envs > 1
            else None,
            "max_joint_velocity_rad_s": maximum_speed,
            "mimic_error_rad": mimic_error,
            "mean_ground_supported_stones": float(
                (ground_forces.norm(dim=-1) > 0.05).float().sum(-1).mean()
            ),
            "observation_shape": list(observations["policy"].shape),
            "parameters": config,
        }
        if env.cameras:
            from PIL import Image

            for name, camera in env.cameras.items():
                rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
                Image.fromarray(rgb).save(output / (name + ".png"))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.info("ISAAC VALIDATION %s output=%s", report, output)
    finally:
        env.close()
except Exception:
    logging.basicConfig(level=logging.INFO, force=True)
    logger.exception("Isaac validation failed")
    raise
finally:
    app.close()
