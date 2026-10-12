"""离线检查图像对动作的影响与特权状态隔离；不启动仿真，不参与训练。"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict
from visual_policy import VisualActorCritic

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)[
    "model_state_dict"
]
# 用本轮保存的真实 RGB 和已学习的本体均值构造可重复输入。
images = np.stack(
    [
        np.asarray(Image.open(args.checkpoint.parent / (name + ".png")).convert("RGB"))
        for name in ("eye_in_hand", "top", "front")
    ]
)
obs = TensorDict(
    {
        "policy": weights["proprio_normalizer._mean"].clone(),
        "critic": weights["critic_obs_normalizer._mean"].clone(),
        "rgb": torch.from_numpy(images.copy()).unsqueeze(0),
    },
    batch_size=[1],
)
num_actions = weights["log_std"].numel()
residual = num_actions == 6
if residual:
    obs["policy"][:, -1] = 1.0
policy = VisualActorCritic(
    obs,
    {"policy": ["policy"], "critic": ["critic"]},
    num_actions,
    local_action_mask=residual,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    critic_obs_normalization=True,
    noise_std_type="log",
)
policy.load_state_dict(weights)
policy.eval()
with torch.no_grad():
    original = policy.act_inference(obs)
    hidden = obs.clone()
    hidden["rgb"].zero_()
    image_effect = float((original - policy.act_inference(hidden)).abs().max())
    privileged = obs.clone()
    privileged["critic"] += 10.0
    state_leak = float((original - policy.act_inference(privileged)).abs().max())
    assert image_effect > 1e-7, "RGB did not affect the actor"
    assert state_leak == 0.0, "Actor read privileged critic state"
# 训练时 CNN 必须可微，不能把预计算特征当作冻结观测。
policy.zero_grad()
policy.act_inference(obs).square().sum().backward()
gradient = sum(float(p.grad.abs().sum()) for p in policy.encoder.parameters())
assert gradient > 0.0, "No policy gradient reached the image encoder"
report = {
    "image_ablation_action_max_difference": image_effect,
    "critic_state_change_actor_max_difference": state_leak,
    "encoder_gradient_l1": gradient,
    "rgb_shape": list(obs["rgb"].shape),
}
if residual:
    inactive = obs.clone()
    inactive["policy"][:, -1] = 0.0
    policy.zero_grad()
    actions = policy.act(inactive)
    assert (actions == 0).all(), "Program phase produced nonzero residual"
    loss = policy.get_actions_log_prob(actions).sum() + policy.entropy.sum()
    loss.backward()
    actor_gradient = sum(
        float(p.grad.abs().sum())
        for p in policy.actor.parameters()
        if p.grad is not None
    )
    assert actor_gradient == 0.0, "Program phase updated residual actor"
    report["program_phase_actor_gradient_l1"] = actor_gradient
    report["program_phase_actions_zero"] = True
(args.checkpoint.parent / "visual_contract.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
logger.info("Visual actor contract: %s", report)
