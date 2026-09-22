"""Policy 注册表：按名字装配"高层 + 低层"。

配置 `policy.name` 选的就是这里的东西。VLM 相关实现是延迟 import 的：
本机没有 GPU / 没有 vLLM 服务时，`scripted` 必须照常可用。
"""

from __future__ import annotations

from typing import Any

from .base import (
    Candidate,
    Command,
    Decision,
    FallbackLowLevel,
    Goal,
    HighLevelPolicy,
    LowLevelPolicy,
    Observation,
    PolicyPair,
    StoneState,
)
from .scripted import ScriptedHighLevel, ScriptedLowLevel

#: 所有策略名与说明（`run.py --mode list` 用）
POLICY_NAMES: tuple[str, ...] = ("qwen-direct", "rl", "scripted", "vlm", "vlm-high", "vlm-low", "qwen-agent")


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """取 `policy.<key>` 段；兼容直接传 `{<key>: {...}}` 的旧调用。"""
    return dict(config.get("policy", config).get(key, {}))


def build_policy(
    name: str,
    scripted_high: ScriptedHighLevel,
    scripted_low: ScriptedLowLevel,
    config: dict[str, Any] | None = None,
):
    """装配策略包。`scripted_high/low` 由调用方按场景参数构造，VLM 用它们做兜底。"""
    config = config or {}
    key = name.strip().lower()
    if key == "rl":
        from .rl import RLPolicy
        return RLPolicy(_section(config, "rl"))
    if key == "qwen-direct":
        from .direct import DirectPolicy
        return DirectPolicy(_section(config, "direct"))
    if key == "scripted":
        return PolicyPair(
            name="scripted",
            high=scripted_high,
            low=scripted_low,
            description="纯几何脚本策略：先铺满低层，大石头优先放低层",
            needs_cameras=(),
        )

    from .vlm import VLMHighLevel, VLMLowLevel  # 延迟 import：没装 requests/没服务时不影响 scripted

    vlm_config = _section(config, "vlm")
    if key in ("vlm", "vlm-high", "vlm-low"):
        high: HighLevelPolicy = (
            VLMHighLevel(vlm_config, fallback=scripted_high) if key in ("vlm", "vlm-high") else scripted_high
        )
        low: LowLevelPolicy = scripted_low
        if key in ("vlm", "vlm-low"):
            low = FallbackLowLevel(VLMLowLevel(vlm_config, fallback=scripted_low), scripted_low)
        cameras = tuple(vlm_config.get("cameras", ("wrist", "top", "front")))
        return PolicyPair(
            name=key,
            high=high,
            low=low,
            description=f"Qwen3-VL({vlm_config.get('model', 'Qwen3-VL-8B')}) @ {vlm_config.get('base_url', 'http://127.0.0.1:3001/v1')} + 脚本兜底",
            needs_cameras=cameras,
            requires_network=True,
        )

    if key == "qwen-agent":
        # Codex(QwenVL) 只做高层；低层沿用脚本，避免每控制周期起子进程。
        from .agent import CodexAgentHighLevel

        agent_config = _section(config, "agent")
        cameras = tuple(agent_config.get("cameras", ("wrist", "top", "front")))
        return PolicyPair(
            name=key,
            high=CodexAgentHighLevel(agent_config, fallback=scripted_high),
            low=scripted_low,
            description=(
                f"Codex agent + Qwen3-VL({agent_config.get('model', 'Qwen3-VL-8B')}) @ {agent_config.get('base_url', 'http://127.0.0.1:3001/v1')}"
                " 高层 + 脚本低层/兜底（需先起 vLLM 与 codex）"
            ),
            needs_cameras=cameras,
            requires_network=True,
        )
    raise KeyError(f"未知策略 {name!r}；可用：{', '.join(POLICY_NAMES)}")


__all__ = [
    "POLICY_NAMES",
    "Candidate",
    "Command",
    "Decision",
    "FallbackLowLevel",
    "Goal",
    "HighLevelPolicy",
    "LowLevelPolicy",
    "Observation",
    "PolicyPair",
    "ScriptedHighLevel",
    "ScriptedLowLevel",
    "StoneState",
    "build_policy",
]
