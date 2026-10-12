"""显式运行零残差基线，检查真实运动、阶段切换和动作门控。正常训练不导入。"""

import json
import logging
import time
from pathlib import Path

from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)

app = AppLauncher(headless=True, enable_cameras=False, device="cuda:0").app
try:
    import torch
    from residual_stack import PHASES, ResidualStackCfg, ResidualStackEnv

    cfg = ResidualStackCfg()
    cfg.scene.num_envs = 2
    env = ResidualStackEnv(cfg)
    logging.basicConfig(level=logging.INFO, force=True)
    out = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / ("residual_" + time.strftime("%Y%m%d_%H%M%S"))
    )
    out.mkdir(parents=True)
    traces = []
    try:
        obs, _ = env.reset()
        for step in range(240):
            before, rot = env.tcp_pose()
            obs, reward, done, truncated, info = env.step(
                torch.zeros((env.num_envs, 6), device=env.device)
            )
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            assert obs["policy"].shape == (2, 210)
            assert (env.residual_action[~env.local_active] == 0).all()
            s = env.signals
            row = {
                "step": step,
                "phase": PHASES[int(env.phase[0])],
                "stone": int(env.stone_index[0]),
                "tcp": (env.tcp_pose()[0][0] - env.scene.env_origins[0]).tolist(),
                "pad_delta": (
                    env.pad_positions()[0, 1] - env.pad_positions()[0, 0]
                ).tolist(),
                "pad": (
                    env.pad_positions()[0].mean(0) - env.scene.env_origins[0]
                ).tolist(),
                "center": env.selected(env.stone_centers())[0].tolist(),
                "anchor": env.anchor[0].tolist(),
                "bilateral": bool(env.selected(s["bilateral"])[0]),
                "finger_forces": env.contacts[int(env.stone_index[0])]
                .data.force_matrix_w[0, 0, 11:13]
                .tolist(),
                "pad_forces": env.selected(env.pad_contacts(env.pad_positions()))[
                    0
                ].tolist(),
                "closing": bool(env.close_gripper[0]),
                "joints": env.robot.data.joint_pos[0].tolist(),
                "reward": float(reward[0]),
                "done": bool(done[0]),
                "completed": int(env.completed[0].sum()),
            }
            traces.append(row)
            if step % 10 == 0 or done.any():
                logger.info("RESIDUAL %s", row)
        (out / "trace.json").write_text(json.dumps(traces, indent=2) + "\n")
        logger.info("RESIDUAL TRACE %s", out)
    finally:
        env.close()
finally:
    app.close()
