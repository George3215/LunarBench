"""TASK2 的 **Codex + QwenVL** 高层策略：用 agent 循环替代单轮 VLM JSON。

和 `vlm.py` 的关系
------------------
`vlm.py` 是"单轮补全"：一张请求直接要 `{"choice": <int>}`。本模块换成
**Codex CLI 当 agent**（`codex exec`），背后模型通过 Codex 的自定义 provider 指向本机
vLLM 上的 Qwen3-VL-8B（见 `assets/model/qwen3VL.sh`）。这样高层决策可以先用 Codex
的文件/图像/计算工具做分析，再给最终 JSON；代价是每步一次子进程，比单轮慢。

**wire_api 必须是 `responses`**：`codex-cli 0.135` 起已移除 `wire_api = "chat"`，
只接受 Responses API。vLLM 0.29 自带 `/v1/responses`（与 `/v1/chat/completions` 同端口
同时提供），所以 provider 直接指 `http://127.0.0.1:3001/v1` 即可，不需要额外网关。

为什么不直接用 GPT-as-Policy 的 app-server
------------------------------------------
`task2_stack/policy/GPT-as-Policy` 是为 RoboDojo/RoboLab 写的长连接 app-server 控制器，
它的 `StdioAppServer` 假定一整段 episode 只开一个 Codex 线程。task2 的执行器是
**同步调用** `decide()` / `step()`（见 `base.py`），所以这里用 `codex exec` 的
**一次性调用**更贴合：每次 `decide()` 起一个短命 Codex 进程，拿到 JSON 就结束。

配置（`task2.yaml` 的 `policy.agent` 段）
----------------------------------------
- `codex_bin`：Codex 可执行文件，默认 `codex`。
- `model` / `provider` / `base_url` / `api_key`：传给 Codex 的自定义 provider；默认
  指向本机 vLLM 的 `Qwen3-VL-8B`。
- `timeout_s` / `retries` / `strict`：子进程超时、修复重试、是否让异常外抛。
- `cameras` / `max_images` / `image_max_side` / `jpeg_quality`：送图的相机与压缩。
- `use_privileged_state`：是否把石块真值位姿写进 prompt（默认 false）。
- `workdir`：Codex 的工作目录（默认每回合一个临时目录，只读沙箱）。
- `extra_args`：追加到 `codex exec` 的参数（例如 `["--json"]`）。

失败处理与 `vlm.py` 一致且**可观测**：网络/子进程失败、JSON 解析失败、choice 越界都会
计入 `stats`，先做一次修复重试，仍失败则交给 `fallback`（通常是脚本高层），最后退化为
"取 score 最大的候选"。只有 `strict: true` 时才抛 `CodexAgentError`。
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base import Candidate, Decision, Observation
from .vlm import (
    _as_int,
    _record_event,
    _select_cameras,
    extract_json_object,
    new_stats,
)

__all__ = [
    "CodexAgentError",
    "CodexAgentHighLevel",
    "build_codex_command",
]

#: `self._recent` 环形缓冲长度，和 vlm.py 保持一致（有界）。
_EVENT_BUFFER = 64

#: 默认送图的相机顺序（只发送确实存在且带 RGB 的）。
_DEFAULT_CAMERA_ORDER: tuple[str, ...] = ("wrist", "top", "front")

_SYSTEM_INSTRUCTION = (
    "You are the high-level planner of a robotic dry-stone stacking cell. "
    "You will see camera images of the workspace and a numbered list of candidate "
    "(stone, slot) pairs. Choose exactly ONE candidate index that should be executed next, "
    "or -1 to stop when the structure is finished or no candidate is safe. "
    "Do not run shell commands or edit files; answer directly. "
    'Reply with STRICT JSON only, no prose, no code fence: {"choice": <int>, "reason": "<short>"}'
)


class CodexAgentError(RuntimeError):
    """`strict: true` 下 Codex 高层策略的显式失败。"""


def _toml_value(value: Any) -> str:
    """Codex `-c key=value` 的 value 按 TOML 解析；字符串必须带引号。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, dict):
        inner = ", ".join(f"{json.dumps(str(k))} = {_toml_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    return json.dumps(str(value))


def build_codex_command(
    config: Mapping[str, Any],
    *,
    prompt: str,
    image_paths: Sequence[Path],
    output_path: Path,
    workdir: Path,
) -> list[str]:
    """拼 `codex exec` 命令：自定义 provider + 图像 + 最终消息落盘。"""
    provider = str(config.get("provider", "vllm"))
    model = str(config.get("model", "Qwen3-VL-8B"))
    base_url = str(config.get("base_url", "http://127.0.0.1:3001/v1"))
    api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
    wire_api = str(config.get("wire_api", "responses"))
    sandbox = str(config.get("sandbox", "read-only"))

    command: list[str] = [
        str(config.get("codex_bin", "codex")),
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--color", "never",
        "-s", sandbox,
        "-C", str(workdir),
        "-c", f"model={_toml_value(model)}",
        "-c", f"model_provider={_toml_value(provider)}",
        "-c", "model_providers.%s=%s" % (
            provider,
            _toml_value({
                "name": provider,
                "base_url": base_url,
                "env_key": api_key_env,
                "wire_api": wire_api,
            }),
        ),
        "-o", str(output_path),
    ]
    for path in image_paths:
        command += ["-i", str(path)]
    command += [str(arg) for arg in config.get("extra_args", [])]
    command.append(prompt)
    return command


class CodexAgentHighLevel:
    """高层策略：Codex(QwenVL) 选候选；失败回退脚本高层，最后回退 score 最大。"""

    def __init__(self, config: Mapping[str, Any] | None = None, fallback: Any = None):
        self.config = dict(config or {})
        self.fallback = fallback
        self.model = str(self.config.get("model", "Qwen3-VL-8B"))
        self.provider = str(self.config.get("provider", "vllm"))
        self.codex_bin = str(self.config.get("codex_bin", "codex"))
        self.base_url = str(self.config.get("base_url", "http://127.0.0.1:3001/v1"))
        self.api_key = str(self.config.get("api_key", "EMPTY"))
        self.timeout_s = float(self.config.get("timeout_s", 180.0))
        self.retries = max(0, int(self.config.get("retries", 1)))
        self.strict = bool(self.config.get("strict", False))
        self.use_privileged_state = bool(self.config.get("use_privileged_state", False))
        self.image_max_side = self.config.get("image_max_side", 768)
        self.jpeg_quality = int(self.config.get("jpeg_quality", 88))
        self.instruction = str(self.config.get("instruction", ""))
        self.workdir_root = self.config.get("workdir")
        self.keep_workdir = bool(self.config.get("keep_workdir", False))
        self.stats = new_stats()
        self.name = f"codex-agent:{self.model}"
        self._recent: deque[dict[str, Any]] = deque(maxlen=_EVENT_BUFFER)
        self._episode = -1
        self._decision = 0
        self._instruction_override: str | None = None
        self._base_dir: Path | None = Path(self.workdir_root).expanduser() if self.workdir_root else None
        if self._base_dir is not None:
            self._base_dir.mkdir(parents=True, exist_ok=True)

    # -- 生命周期 ---------------------------------------------------------------------
    def reset(self, context: Mapping[str, Any] | None = None) -> None:
        context = dict(context or {})
        self._episode += 1
        self._recent.clear()
        self._instruction_override = None
        self._decision = 0
        if "instruction" in context and context["instruction"] is not None:
            self._instruction_override = str(context["instruction"])
        if self.fallback is not None and hasattr(self.fallback, "reset"):
            self.fallback.reset(context)
        _record_event(self._recent, "reset", episode=self._episode)

    def reset_stats(self) -> None:
        self.stats.clear()
        self.stats.update(new_stats())

    def recent_events(self) -> list[dict[str, Any]]:
        return list(self._recent)

    # -- prompt / 图像 ----------------------------------------------------------------
    def _effective_instruction(self, observation: Observation) -> str:
        if self._instruction_override:
            return self._instruction_override
        if observation.instruction:
            return observation.instruction
        return self.instruction

    def build_prompt(self, observation: Observation, candidates: Sequence[Candidate], camera_names: Sequence[str]) -> str:
        lines: list[str] = [_SYSTEM_INSTRUCTION, ""]
        lines.append("TASK INSTRUCTION:")
        lines.append(f"  {self._effective_instruction(observation) or '(none given)'}")
        lines.append("")
        lines.append("ROBOT STATE:")
        lines.append(f"  phase={observation.phase} step_index={observation.step_index}")
        lines.append(f"  tcp_xyz={[round(float(v), 4) for v in np.asarray(observation.tcp_pos).reshape(-1)]}")
        lines.append(f"  gripper_width_m={observation.gripper_width:.4f} held_stone={observation.held_stone or 'none'}")
        lines.append(f"  cameras_sent={list(camera_names)} (attached in this order)")
        lines.append("")
        lines.append(f"CANDIDATES ({len(candidates)}):")
        if not candidates:
            lines.append("  (empty)")
        for index, candidate in enumerate(candidates):
            line = f"[{index}] {candidate.describe()}"
            if candidate.note:
                line += f" note={candidate.note}"
            lines.append(line)
            if self.use_privileged_state:
                stone = candidate.stone
                lines.append(
                    "    stone_xyz="
                    + str([round(float(v), 4) for v in np.asarray(stone.pos).reshape(-1)])
                )
        lines.append("")
        lines.append('Choose the single best candidate index, or -1 to stop. Strict JSON: {"choice": <int>, "reason": "<short>"}')
        return "\n".join(lines)

    def _write_images(self, observation: Observation, directory: Path) -> tuple[list[Path], list[str], int]:
        """把选中的相机帧写成 JPEG 文件，返回 (路径, 相机名, 被跳过的帧数)。"""
        names, skipped = _select_cameras(observation, self.config)
        paths: list[Path] = []
        kept: list[str] = []
        for name in names:
            frame = observation.cameras[name]
            try:
                raw = frame.jpeg_bytes(quality=self.jpeg_quality, max_side=self.image_max_side)
            except Exception as error:  # noqa: BLE001 - 单帧坏了不该毁掉整次决策
                skipped += 1
                _record_event(self._recent, "image_error", camera=name, error=f"{type(error).__name__}: {error}")
                continue
            path = directory / f"camera_{len(paths)}_{name}.jpg"
            path.write_bytes(raw)
            paths.append(path)
            kept.append(name)
            _record_event(self._recent, "image", camera=name, jpeg_bytes=len(raw))
        return paths, kept, skipped

    # -- 子进程 -----------------------------------------------------------------------
    def _run_codex(self, prompt: str, image_paths: Sequence[Path], workdir: Path) -> tuple[str, float]:
        """跑一次 `codex exec`，返回 (agent 最终消息, 耗时秒)。失败抛 RuntimeError。"""
        if shutil.which(self.codex_bin) is None and not Path(self.codex_bin).exists():
            raise RuntimeError(f"找不到 Codex 可执行文件：{self.codex_bin}")
        output_path = workdir / "last_message.txt"
        command = build_codex_command(
            self.config, prompt=prompt, image_paths=image_paths,
            output_path=output_path, workdir=workdir,
        )
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = self.api_key
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"
        env.pop("OPENAI_BASE_URL", None)
        # Codex 的 in-process app-server 需要一个**可写的 CODEX_HOME**：默认的
        # ~/.codex 在只读家目录/沙箱里会直接 "Read-only file system (os error 30)"。
        # 每次决策用 workdir 下的私有 home，配合 --ignore-user-config --ephemeral
        # 做到既隔离又不留会话文件。
        codex_home = workdir / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command, cwd=str(workdir), env=env, capture_output=True, text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"codex exec 超时（>{self.timeout_s:.0f}s）") from error
        latency = time.perf_counter() - started
        (workdir / "codex_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
        (workdir / "codex_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-400:]
            raise RuntimeError(f"codex exec 退出码 {result.returncode}：{tail}")
        text = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
        if not text.strip():
            text = result.stdout or ""
        return text, latency

    # -- 决策 ------------------------------------------------------------------------
    def _fallback_decision(
        self, observation: Observation, candidates: Sequence[Candidate], cause: str, **extra: Any
    ) -> Decision:
        self.stats["fallbacks"] += 1
        _record_event(self._recent, "fallback", cause=cause, **extra)
        if self.fallback is not None:
            try:
                decision = self.fallback.decide(observation, candidates)
            except Exception as error:  # noqa: BLE001 - 外部策略什么都可能抛
                _record_event(self._recent, "fallback_error", error=f"{type(error).__name__}: {error}")
            else:
                return replace(
                    decision,
                    source=f"fallback:{getattr(self.fallback, 'name', 'external')}",
                    reason=f"codex_failed({cause}) -> fallback: {decision.reason}",
                )
        if not candidates:
            self.stats["stops"] += 1
            return Decision(action="stop", source="fallback:no_candidates",
                            reason=f"codex_failed({cause}) and no candidates -> stop")
        best_index = max(range(len(candidates)), key=lambda i: (float(candidates[i].score), -i))
        best = candidates[best_index]
        return Decision(
            action="place", stone=best.stone.name, slot=best.slot.key,
            target_pos=np.array(best.slot.target_pos, dtype=float, copy=True),
            target_quat=np.array(best.slot.target_quat, dtype=float, copy=True),
            source="fallback:argmax_score",
            reason=f"codex_failed({cause}) -> highest score candidate #{best_index}",
        )

    def _interpret(self, text: str, candidates: Sequence[Candidate], source: str, latency_s: float, images: int) -> tuple[Decision | None, str]:
        try:
            payload = extract_json_object(text)
        except ValueError as error:
            self.stats["parse_failures"] += 1
            self.stats["last_error"] = str(error)
            return None, f"json_parse_error: {error}"
        if "choice" not in payload:
            self.stats["parse_failures"] += 1
            self.stats["last_error"] = f"missing choice field: {sorted(payload)[:8]}"
            return None, f"missing_choice_field keys={sorted(payload)[:8]}"
        raw_choice = payload["choice"]
        choice = _as_int(raw_choice)
        stop_tokens = isinstance(raw_choice, str) and raw_choice.strip().lower() in {"stop", "none", "null", "done"}
        if choice is None and raw_choice is not None and not stop_tokens:
            self.stats["parse_failures"] += 1
            return None, f"choice_not_integer value={raw_choice!r}"
        reason = str(payload.get("reason", "")).strip().replace("\n", " ")[:240]
        base = f"model={self.model} images={images} latency={latency_s:.3f}s"
        if choice == -1 or (choice is None and stop_tokens and raw_choice is not None):
            self.stats["stops"] += 1
            return Decision(action="stop", source=source, reason=f"model_stop: {reason or '(no reason)'} | {base}"), ""
        if choice is None:
            return None, "choice_null"
        if not 0 <= choice < len(candidates):
            self.stats["out_of_range"] += 1
            self.stats["last_error"] = f"choice {choice} out of range [0,{len(candidates) - 1}]"
            return None, f"choice_out_of_range choice={choice} n_candidates={len(candidates)}"
        candidate = candidates[choice]
        return Decision(
            action="place", stone=candidate.stone.name, slot=candidate.slot.key,
            target_pos=np.array(candidate.slot.target_pos, dtype=float, copy=True),
            target_quat=np.array(candidate.slot.target_quat, dtype=float, copy=True),
            source=source, reason=f"codex_choice={choice} {reason or '(no reason)'} | {base}",
        ), ""

    def decide(self, observation: Observation, candidates: Sequence[Candidate]) -> Decision:
        self.stats["decisions"] += 1
        try:
            return self._decide_inner(observation, candidates)
        except Exception as error:  # noqa: BLE001 - 策略层不允许把异常漏给执行器
            if self.strict:
                raise CodexAgentError(f"CodexAgentHighLevel strict 失败：{type(error).__name__}: {error}") from error
            self.stats["failures"] += 1
            self.stats["last_error"] = f"unexpected {type(error).__name__}: {error}"
            return self._fallback_decision(observation, candidates, f"exception:{type(error).__name__}: {error}")

    def _decide_inner(self, observation: Observation, candidates: Sequence[Candidate]) -> Decision:
        candidates = list(candidates)
        view = observation if self.use_privileged_state else observation.image_only()
        if not candidates:
            self.stats["stops"] += 1
            return Decision(action="stop", source="codex-agent", reason="no candidates -> stop without querying the model")

        self._decision += 1
        if self._base_dir is not None:
            workdir = self._base_dir / f"ep{self._episode:03d}_decision{self._decision:04d}"
            if workdir.exists():
                shutil.rmtree(workdir, ignore_errors=True)
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            workdir = Path(tempfile.mkdtemp(prefix="task2_codex_"))

        source = "codex"
        last_cause = "unknown"
        try:
            image_paths, kept_names, skipped = self._write_images(view, workdir)
            self.stats["image_skipped"] += skipped
            self.stats["images_sent"] += len(image_paths)
            prompt = self.build_prompt(view, candidates, kept_names)
            for attempt in range(self.retries + 1):
                try:
                    text, latency = self._run_codex(prompt, image_paths, workdir)
                except Exception as error:  # noqa: BLE001 - 子进程/网络失败都归到这里
                    last_cause = f"codex_error: {error}"
                    self.stats["failures"] += 1
                    self.stats["last_error"] = str(error)
                    _record_event(self._recent, "codex_failure", attempt=attempt, error=str(error))
                    if attempt < self.retries:
                        self.stats["retries"] += 1
                        continue
                    break
                decision, failure = self._interpret(text, candidates, source, latency, len(image_paths))
                if decision is not None:
                    _record_event(self._recent, "decision", attempt=attempt, source=source, action=decision.action)
                    return decision
                last_cause = failure
                _record_event(self._recent, "parse_failure", attempt=attempt, error=failure)
                if attempt < self.retries:
                    self.stats["retries"] += 1
                    prompt = (prompt + "\n\nYour previous reply was rejected (" + failure + "). "
                              'Reply with STRICT JSON only: {"choice": <int>, "reason": "<short>"}.')
                    source = "codex_repair"

            if self.strict:
                raise CodexAgentError(f"CodexAgentHighLevel 无法得到可用决策：{last_cause}")
            return self._fallback_decision(observation, candidates, last_cause)
        finally:
            if not self.keep_workdir and self._base_dir is None:
                shutil.rmtree(workdir, ignore_errors=True)

    # -- 诊断 ------------------------------------------------------------------------
    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "episode": self._episode,
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "codex_bin": self.codex_bin,
            "use_privileged_state": self.use_privileged_state,
            "fallback": getattr(self.fallback, "name", None),
            "recent_events": self.recent_events(),
        }
