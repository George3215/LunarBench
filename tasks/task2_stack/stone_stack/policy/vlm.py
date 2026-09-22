"""TASK2 的 VLM 策略：高层"选哪块石头放哪个位"，低层"末端目标位姿怎么修"。

本模块只依赖 `stone_stack.policy.base` 里的冻结接口（`Observation` / `Candidate` /
`Decision` / `Command` / `Goal`）与 `requests` + `numpy`，**不导入 mujoco**：策略是
引擎无关的，IK、伺服、碰撞、成功判定都在执行器里。VLM 服务通过标准 OpenAI 兼容
HTTP 接口访问（`POST {base_url}/chat/completions`），图像走 base64 data URL。

设计取舍
--------
1. **为什么低层输出"绝对位姿"，而 VLM 只输出"增量"。**
   执行器需要的是世界系下的绝对 TCP 目标位姿（见 base.py 的"动作空间"一节），
   所以 `Command.tcp_pos/tcp_rot` 必须是绝对值。但让 VLM 直接看图吐绝对米制坐标
   是不可靠的：单目图像没有可靠尺度，模型的绝对数值基本是记忆里的先验，误差可达
   几十厘米。执行器/规划器给出的**标称位姿**（nominal pose）来自几何与 IK，是米制
   可信的；VLM 只负责在这个标称位姿上做**小幅感知修正**（"再往左一点、再压低一点"）。
   于是我们把任务拆成：几何给标称，VLM 给有界增量，执行器做绝对伺服。这样既保留了
   绝对位姿接口，又让 VLM 只需要做它擅长的相对判断。

2. **为什么要夹紧（clamp）。**
   VLM 的输出是自由文本，可能返回 1.5 m 的位移或 95° 的偏航（本仓库的测试里就专门
   构造了这种越界样本）。一旦直接下发，机械臂会撞桌面、撞墙或把石头甩出去。因此
   `delta_xyz` 逐轴夹到 ±`max_delta_m`、`delta_yaw_deg` 夹到 ±`max_yaw_deg`，并且
   **每次夹紧都计数**（`stats["clipped"]` / `clipped_xyz` / `clipped_yaw`）：报告里
   夹紧率长期偏高说明标称位姿本身就偏，或者 prompt 需要改，而不是"模型很激进"这种
   无法行动的结论。

3. **为什么必须兜底。**
   网络会超时、服务会 500、模型会返回 markdown 包裹的 JSON 甚至纯废话。任务不能因此
   中断，也不能静默吞掉：`decide` / `step` 内部先做一次"修复 prompt"重试，仍失败则走
   调用方给的 `fallback` 策略；高层没有 fallback 时退化为**确定性**的"取 score 最大的
   候选"，低层没有 fallback 时退化为**保持当前位姿**的 `Command`（`done=False`，让
   执行器自己超时）。所有失败路径都写进 `Decision.source` / `Command.source` 与
   `reason` / `note`，并累计到 `stats`，报告可回溯。只有显式配置 `strict: true` 时
   才允许异常穿透——这是给"我要立刻看到崩溃"的调试场景留的开关。

4. **特权信息。**
   `Observation.stones` 是仿真真值。默认 `use_privileged_state=False`，策略走
   `Observation.image_only()`，prompt 里只有图像 + `Candidate.describe()` 文本。
   注意 `Candidate.describe()`（冻结接口）本身含尺寸/质量与**空位目标坐标**，这是
   候选列表的固有内容、不是特权状态；真正的特权量是**石块自身的位姿真值**
   （`stone.pos/quat`），它只在 `use_privileged_state=True` 时以 `stone_xyz=` /
   `stone_quat=` 字段进入 prompt。

5. **可观测的统计。**
   两个类共享一组有界的 `self.stats` 计数器（requests / failures / parse_failures /
   images_sent / last_latency_s / clipped ...），另有固定长度的 `self._recent` 事件
   环形缓冲，不会随回合数无限增长。`reset()` 只清"每回合状态"（指令、标称位姿记忆、
   事件环形缓冲），累计计数器保留，便于整段评测后统计；需要清零时调用
   `reset_stats()`。
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
from collections import deque
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import requests

from .base import Candidate, Command, Decision, Goal, Observation

__all__ = [
    "VLMError",
    "VLMHttpClient",
    "VLMHighLevel",
    "VLMLowLevel",
    "check_server",
    "extract_json_object",
    "new_stats",
]

#: 打开后把每次 VLM 请求/原始回复/解析出的动作打到 stdout（终端观察用）。
#: 用法：`TASK2_VLM_TRACE=1`；`TASK2_VLM_TRACE_FULL=1` 连完整 prompt 一起打。
_TRACE = os.environ.get("TASK2_VLM_TRACE", "").strip().lower() in ("1", "true", "yes", "on")
_TRACE_FULL = os.environ.get("TASK2_VLM_TRACE_FULL", "").strip().lower() in ("1", "true", "yes", "on")


def _trace(label: str, message: str) -> None:
    if _TRACE:
        logging.getLogger(__name__).info("%s %s", label, message)

#: 默认的低层可接管目标类型。`hold`/`home` 一般交给脚本执行器，不劳烦 VLM。
DEFAULT_VLM_GOALS: tuple[str, ...] = ("place",)

#: 默认相机发送顺序（只发送确实存在且带 RGB 的）。
DEFAULT_CAMERA_ORDER: tuple[str, ...] = ("wrist", "top", "front")

#: JSON 抽取时允许扫描的最大文本长度，防止模型返回超长废话时解析器空转。
_MAX_SCAN_CHARS = 200_000

#: `self._recent` 环形缓冲长度：有界，不会随回合数增长。
_EVENT_BUFFER = 64


class VLMError(RuntimeError):
    """VLM 策略在 `strict: true` 下的显式失败（网络、解析、越界都归到这里）。"""


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------


def new_stats() -> dict[str, Any]:
    """新建一组有界计数器。两个策略类、HTTP 客户端共用同一套键名。"""
    return {
        "requests": 0,  # 真正发出的 HTTP 请求数（含重试）
        "failures": 0,  # 网络层失败（超时、连接错误、非 2xx、响应体不合法）
        "parse_failures": 0,  # 模型有回复但 JSON 抽取/字段校验失败
        "images_sent": 0,  # 累计发送的图像张数
        "last_latency_s": 0.0,  # 最近一次 HTTP 往返耗时
        "latency_total_s": 0.0,
        "latency_max_s": 0.0,
        "clipped": 0,  # 发生夹紧的步数（低层）
        "clipped_xyz": 0,  # 被夹紧的位移分量个数
        "clipped_yaw": 0,  # 被夹紧的偏航次数
        "clipped_grip": 0,  # 被夹紧的夹爪开口次数
        "retries": 0,  # 修复 prompt 重试次数
        "fallbacks": 0,  # 回退次数（含交给外部 fallback 与内部确定性回退）
        "decisions": 0,  # 高层 decide / 低层 step 的调用次数
        "stops": 0,  # 模型或回退给出的 stop 次数
        "out_of_range": 0,  # 候选下标越界次数
        "image_skipped": 0,  # 因缺 RGB / 编码失败被跳过的帧数
        "timeouts": 0,  # 其中属于请求超时的次数
        "http_errors": 0,  # 其中属于非 2xx 的次数
        "last_error": "",  # 最近一次失败的可读描述（字符串，有界）
    }


def _record_event(recent: deque[dict[str, Any]], kind: str, **fields: Any) -> None:
    """往固定长度的环形缓冲里追加一条事件；缓冲满后自动丢弃最旧的（内存有界）。"""
    recent.append({"kind": kind, "t": round(time.time(), 3), **fields})


def _bump_latency(stats: dict[str, Any], latency_s: float) -> None:
    stats["last_latency_s"] = round(float(latency_s), 4)
    stats["latency_total_s"] = round(float(stats["latency_total_s"]) + float(latency_s), 4)
    stats["latency_max_s"] = round(max(float(stats["latency_max_s"]), float(latency_s)), 4)


# --------------------------------------------------------------------------------------
# JSON 抽取
# --------------------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _iter_balanced_objects(text: str):
    """扫描出所有花括号平衡的 `{...}` 子串（跳过字符串与转义），按出现顺序产出。"""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : index + 1]
                    start = -1


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型回复里抽出第一个合法 JSON 对象。

    依次尝试：整段直接解析 -> markdown 代码围栏内容 -> 正文里第一个花括号平衡的子串。
    这样 `{"choice": 1}`、```` ```json\\n{...}\\n``` ````、以及"先寒暄再给 JSON 再总结"
    三种形态都能过。全部失败时抛 `ValueError`（调用方据此计 `parse_failures`）。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型回复为空")

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    for match in _FENCE_RE.finditer(text[:_MAX_SCAN_CHARS]):
        inner = match.group(1).strip()
        if inner:
            candidates.append(inner)
    candidates.extend(_iter_balanced_objects(text[:_MAX_SCAN_CHARS]))

    for chunk in candidates:
        try:
            parsed = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"回复中找不到合法 JSON 对象：{text[:180]!r}")


def _extract_message_text(payload: Mapping[str, Any]) -> str:
    """从 OpenAI 风格响应里取文本；兼容 content 为字符串或 parts 列表两种形态。"""
    try:
        choices = payload["choices"]
        message = choices[0]["message"] if isinstance(choices[0], Mapping) else choices[0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"响应缺少 choices[0].message：{str(payload)[:200]!r}") from error
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 有些服务把 content 拆成 [{"type": "text", "text": ...}]
        parts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
        text = "".join(part for part in parts if isinstance(part, str))
        if text:
            return text
    reasoning = message.get("reasoning_content") if isinstance(message, Mapping) else None
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    raise ValueError(f"响应没有可用文本内容：{str(payload)[:200]!r}")


# --------------------------------------------------------------------------------------
# HTTP 客户端（OpenAI 兼容 /chat/completions）
# --------------------------------------------------------------------------------------


class VLMHttpClient:
    """极简 OpenAI 兼容客户端：只用 `requests`，不依赖任何 `openai` SDK 版本。

    - `chat(messages)`：发一次 `/chat/completions`，返回文本；失败抛 `VLMError`。
    - `ping()`：`GET /models`，用于连通性检查（真实服务可选开启）。
    - 每次调用都带显式超时 `(connect_timeout_s, timeout_s)`，因此一定是可中断的。
    """

    def __init__(self, config: Mapping[str, Any], stats: dict[str, Any] | None = None):
        self.base_url = str(config.get("base_url", "http://127.0.0.1:3001/v1")).rstrip("/")
        self.model = str(config.get("model", "Qwen3-VL-8B"))
        self.api_key = str(config.get("api_key", "EMPTY"))
        self.timeout_s = float(config.get("timeout_s", 20.0))
        self.connect_timeout_s = float(config.get("connect_timeout_s", min(5.0, self.timeout_s)))
        self.max_tokens = int(config.get("max_tokens", 256))
        self.temperature = float(config.get("temperature", 0.0))
        self.stats = stats if stats is not None else new_stats()
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    # -- 内部 -------------------------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        timeout = (max(0.05, self.connect_timeout_s), max(0.05, self.timeout_s))
        started = time.perf_counter()
        try:
            response = self.session.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout as error:
            latency = time.perf_counter() - started
            self.stats["requests"] += 1
            self.stats["failures"] += 1
            self.stats["timeouts"] += 1
            self.stats["last_error"] = f"timeout after {latency:.2f}s: {error}"
            _bump_latency(self.stats, latency)
            raise VLMError(f"请求超时（{latency:.2f}s > {self.timeout_s}s）：{url}") from error
        except requests.exceptions.RequestException as error:
            latency = time.perf_counter() - started
            self.stats["requests"] += 1
            self.stats["failures"] += 1
            self.stats["last_error"] = f"{type(error).__name__}: {error}"
            _bump_latency(self.stats, latency)
            raise VLMError(f"HTTP 请求失败：{type(error).__name__}: {error}") from error
        latency = time.perf_counter() - started
        self.stats["requests"] += 1
        _bump_latency(self.stats, latency)
        if response.status_code >= 400:
            self.stats["failures"] += 1
            self.stats["http_errors"] += 1
            self.stats["last_error"] = f"HTTP {response.status_code}: {response.text[:160]}"
            raise VLMError(f"HTTP {response.status_code}：{response.text[:200]}")
        return response

    # -- 公开 -------------------------------------------------------------------------
    def chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """一次 chat completion，返回助手文本。任何失败都抛 `VLMError`。"""
        body = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        response = self._request("POST", f"{self.base_url}/chat/completions", json=body)
        try:
            payload = response.json()
        except ValueError as error:
            self.stats["failures"] += 1
            self.stats["last_error"] = f"响应不是 JSON：{response.text[:160]}"
            raise VLMError(f"响应不是合法 JSON：{response.text[:200]}") from error
        try:
            return _extract_message_text(payload)
        except ValueError as error:
            self.stats["failures"] += 1
            self.stats["last_error"] = str(error)
            raise VLMError(str(error)) from error

    def ping(self) -> dict[str, Any]:
        """`GET /models` 连通性探测；不抛异常，返回结构化结果（给真实服务开关用）。"""
        started = time.perf_counter()
        try:
            response = self._request("GET", f"{self.base_url}/models")
            payload = response.json()
        except (VLMError, ValueError) as error:
            return {
                "ok": False,
                "error": str(error),
                "latency_s": round(time.perf_counter() - started, 4),
                "base_url": self.base_url,
            }
        models: list[str] = []
        if isinstance(payload, Mapping):
            for entry in payload.get("data", []) or []:
                if isinstance(entry, Mapping) and "id" in entry:
                    models.append(str(entry["id"]))
        return {
            "ok": True,
            "models": models,
            "latency_s": round(time.perf_counter() - started, 4),
            "base_url": self.base_url,
        }


def check_server(
    base_url: str = "http://127.0.0.1:3001/v1",
    model: str = "Qwen3-VL-8B",
    api_key: str = "EMPTY",
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """一次性连通性检查，给"真实服务可选运行"的入口用。"""
    client = VLMHttpClient(
        {
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "timeout_s": timeout_s,
            "connect_timeout_s": min(2.0, timeout_s),
        }
    )
    result = client.ping()
    result["model"] = model
    return result


# --------------------------------------------------------------------------------------
# 图像编码 / 相机选择
# --------------------------------------------------------------------------------------


def _select_cameras(observation: Observation, config: Mapping[str, Any]) -> tuple[list[str], int]:
    """按配置决定发送哪些相机（顺序敏感），返回 (名单, 被跳过的帧数)。"""
    explicit = config.get("cameras")
    if explicit:
        order = [str(name) for name in explicit]
    else:
        declared = [str(name) for name in config.get("camera_order", DEFAULT_CAMERA_ORDER)]
        order = declared + [name for name in observation.camera_names() if name not in declared]

    max_images = int(config.get("max_images", 3))
    if max_images <= 0:
        return [], 0

    selected: list[str] = []
    skipped = 0
    for name in order:
        frame = observation.cameras.get(name)
        if frame is None:
            continue
        if frame.rgb is None:
            skipped += 1
            continue
        selected.append(name)
        if len(selected) >= max_images:
            break
    return selected, skipped


def _encode_data_url(frame: Any, max_side: int | None, quality: int) -> tuple[str, int]:
    """JPEG 编码 + base64 data URL。返回 (data_url, jpeg 字节数)。"""
    raw = frame.jpeg_bytes(quality=quality, max_side=max_side)
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"), len(raw)


def _image_message(names: Sequence[str], data_urls: Sequence[str]) -> list[dict[str, Any]]:
    """标准 OpenAI vision 消息格式：text part + 若干 image_url part。"""
    parts: list[dict[str, Any]] = []
    for name, url in zip(names, data_urls):
        parts.append({"type": "text", "text": f"[camera:{name}]"})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _format_vec(values: Any, precision: int = 4) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    return "[" + " ".join(f"{value:+.{precision}f}" for value in array) + "]"


def _rot_to_rpy_deg(rot: Any) -> tuple[float, float, float]:
    """ZYX 欧拉角（yaw-pitch-roll），只为 prompt 可读性，不参与任何计算。"""
    matrix = np.asarray(rot, dtype=float).reshape(3, 3)
    sy = math.hypot(matrix[0, 0], matrix[1, 0])
    if sy > 1e-9:
        return (
            math.degrees(math.atan2(matrix[2, 1], matrix[2, 2])),
            math.degrees(math.atan2(-matrix[2, 0], sy)),
            math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])),
        )
    return (
        math.degrees(math.atan2(-matrix[1, 2], matrix[1, 1])),
        math.degrees(math.atan2(-matrix[2, 0], sy)),
        0.0,
    )


def _yaw_matrix(degrees: float) -> np.ndarray:
    """绕世界系 +Z 的旋转矩阵。"""
    radians = math.radians(float(degrees))
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _as_int(value: Any) -> int | None:
    """把 choice 字段宽松地转成 int：接受 2 / 2.0 / "2"；bool 与 "stop" 另行处理。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if float(value).is_integer() else None
    if isinstance(value, str):
        token = value.strip()
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
        if re.fullmatch(r"[+-]?\d+\.0+", token):
            return int(float(token))
    return None


# --------------------------------------------------------------------------------------
# 高层策略
# --------------------------------------------------------------------------------------

_HIGH_SYSTEM_PROMPT = (
    "You are the high-level planner of a robotic dry-stone stacking cell. "
    "You see camera images of the workspace and a numbered list of candidate "
    "(stone, slot) pairs. Choose exactly ONE candidate index that should be executed next, "
    "or -1 to stop when the structure is finished or no candidate is safe. "
    'Reply with STRICT JSON only, no prose, no code fence: {"choice": <int>, "reason": "<short>"}'
)


class VLMHighLevel:
    """VLM 高层：在候选列表里挑一个 (石头, 空位)，或停机。

    失败路径（都可观测）：
    - 网络失败 / 非 2xx / 超时 -> 一次修复重试 -> 仍失败则回退；
    - 回复无法解析出 JSON -> 计 `parse_failures`，一次修复重试 -> 仍失败则回退；
    - `choice` 越界 -> 计 `out_of_range`，同上；
    - 回退优先交给构造时传入的 `fallback` 策略，否则取 `score` 最大的候选（确定性）。

    只有 `strict: true` 时才抛 `VLMError`。
    """

    def __init__(self, config: Mapping[str, Any] | None = None, fallback: Any = None):
        self.config = dict(config or {})
        self.fallback = fallback
        self.model = str(self.config.get("model", "Qwen3-VL-8B"))
        self.use_privileged_state = bool(self.config.get("use_privileged_state", False))
        self.strict = bool(self.config.get("strict", False))
        self.retries = max(0, int(self.config.get("retries", 1)))
        self.image_max_side = self.config.get("image_max_side", 768)
        self.jpeg_quality = int(self.config.get("jpeg_quality", 88))
        self.instruction = str(self.config.get("instruction", ""))
        self.stats = new_stats()
        self.client = VLMHttpClient(self.config, self.stats)
        self.name = f"vlm_high:{self.model}"
        self._recent: deque[dict[str, Any]] = deque(maxlen=_EVENT_BUFFER)
        self._episode = -1
        self._instruction_override: str | None = None

    # -- 生命周期 ---------------------------------------------------------------------
    def reset(self, context: Mapping[str, Any] | None = None) -> None:
        """开始新一回合：清掉每回合状态；`context["instruction"]` 覆盖默认指令。

        覆盖优先级：`context["instruction"]` > `observation.instruction` > `config["instruction"]`。
        累计 `stats` 不清零（整段评测后统一看），需要清零请调用 `reset_stats()`。
        """
        context = dict(context or {})
        self._episode += 1
        self._recent.clear()
        self._instruction_override = None
        if "instruction" in context and context["instruction"] is not None:
            self._instruction_override = str(context["instruction"])
        if "candidates" in context:  # 预留：允许执行器把候选集放进 context
            _record_event(self._recent, "context_candidates", count=len(context["candidates"]))
        if self.fallback is not None and hasattr(self.fallback, "reset"):
            self.fallback.reset(context)
        _record_event(self._recent, "reset", episode=self._episode)

    def reset_stats(self) -> None:
        """把累计计数器清零（典型用法：每个评测配置开始前调用一次）。"""
        self.stats.clear()
        self.stats.update(new_stats())

    def recent_events(self) -> list[dict[str, Any]]:
        return list(self._recent)

    # -- prompt ----------------------------------------------------------------------
    def _effective_instruction(self, observation: Observation) -> str:
        if self._instruction_override:
            return self._instruction_override
        if observation.instruction:
            return observation.instruction
        return self.instruction

    def _candidate_line(self, index: int, candidate: Candidate) -> str:
        line = f"[{index}] {candidate.describe()}"
        if candidate.note:
            line += f" note={candidate.note}"
        return line

    def _privileged_line(self, candidate: Candidate) -> str:
        """特权块：石块真值位姿。非特权模式下这一行绝不出现。"""
        stone = candidate.stone
        return (
            f"    stone_xyz={_format_vec(stone.pos)} stone_quat={_format_vec(stone.quat)} "
            f"stone_size_m={_format_vec(stone.size)} graspable={bool(stone.graspable)}"
        )

    def build_prompt(
        self,
        observation: Observation,
        candidates: Sequence[Candidate],
        camera_names: Sequence[str],
        privileged: Observation | None = None,
    ) -> str:
        """拼高层文本 prompt。`privileged` 只在开启特权模式时非空。"""
        lines: list[str] = []
        lines.append("TASK INSTRUCTION:")
        lines.append(f"  {self._effective_instruction(observation) or '(none given)'}")
        lines.append("")
        lines.append("ROBOT STATE:")
        lines.append(f"  phase={observation.phase} step_index={observation.step_index}")
        lines.append(f"  tcp_xyz={_format_vec(observation.tcp_pos)} gripper_width_m={observation.gripper_width:.4f}")
        held = observation.held_stone or "none"
        lines.append(f"  held_stone={held}")
        lines.append(f"  cameras_sent={list(camera_names)}")
        lines.append("")
        lines.append(f"CANDIDATES ({len(candidates)}):")
        if not candidates:
            lines.append("  (empty)")
        for index, candidate in enumerate(candidates):
            lines.append(self._candidate_line(index, candidate))
            if self.use_privileged_state and privileged is not None:
                lines.append(self._privileged_line(candidate))
        lines.append("")
        lines.append(
            "Choose the single best candidate index for the next placement, "
            "or -1 to stop. Answer with strict JSON: "
            '{"choice": <int>, "reason": "<short>"}'
        )
        return "\n".join(lines)

    # -- 决策 ------------------------------------------------------------------------
    def _fallback_decision(
        self, observation: Observation, candidates: Sequence[Candidate], cause: str, **extra: Any
    ) -> Decision:
        """确定性回退：优先外部 fallback 策略，否则取 score 最大的候选。"""
        self.stats["fallbacks"] += 1
        _record_event(self._recent, "fallback", cause=cause, **extra)
        if self.fallback is not None:
            try:
                decision = self.fallback.decide(observation, candidates)
            except Exception as error:  # noqa: BLE001 - 外部策略什么都可能抛
                _record_event(
                    self._recent, "fallback_error", error=f"{type(error).__name__}: {error}"
                )
            else:
                reason = f"vlm_failed({cause}) -> fallback={getattr(self.fallback, 'name', type(self.fallback).__name__)}: {decision.reason}"
                return replace(decision, source=f"fallback:{getattr(self.fallback, 'name', 'external')}", reason=reason)
        if not candidates:
            self.stats["stops"] += 1
            return Decision(
                action="stop",
                source="fallback:no_candidates",
                reason=f"vlm_failed({cause}) and no candidates to fall back on -> stop",
            )
        best_index = max(range(len(candidates)), key=lambda i: (float(candidates[i].score), -i))
        best = candidates[best_index]
        return Decision(
            action="place",
            stone=best.stone.name,
            slot=best.slot.key,
            target_pos=np.array(best.slot.target_pos, dtype=float, copy=True),
            target_quat=np.array(best.slot.target_quat, dtype=float, copy=True),
            source="fallback:argmax_score",
            reason=(
                f"vlm_failed({cause}) -> deterministic fallback: highest score "
                f"candidate #{best_index} score={float(best.score):.4f} ({best.describe()})"
            ),
        )

    def _interpret(
        self, text: str, candidates: Sequence[Candidate], source: str, latency_s: float, images: int
    ) -> tuple[Decision | None, str]:
        """把模型文本变成 Decision。返回 (decision, 失败原因)；失败时 decision 为 None。"""
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
            self.stats["last_error"] = f"choice not an integer: {raw_choice!r}"
            return None, f"choice_not_integer value={raw_choice!r}"

        reason = str(payload.get("reason", "")).strip().replace("\n", " ")[:240]
        base = f"model={self.model} images={images} latency={latency_s:.3f}s"

        if choice == -1 or (choice is None and stop_tokens and raw_choice is not None):
            self.stats["stops"] += 1
            return (
                Decision(action="stop", source=source, reason=f"model_stop: {reason or '(no reason)'} | {base}"),
                "",
            )
        if choice is None:
            self.stats["parse_failures"] += 1
            self.stats["last_error"] = f"choice is null: {raw_choice!r}"
            return None, "choice_null"

        if not 0 <= choice < len(candidates):
            self.stats["out_of_range"] += 1
            self.stats["last_error"] = f"choice {choice} out of range [0,{len(candidates) - 1}]"
            return None, f"choice_out_of_range choice={choice} n_candidates={len(candidates)}"

        candidate = candidates[choice]
        return (
            Decision(
                action="place",
                stone=candidate.stone.name,
                slot=candidate.slot.key,
                target_pos=np.array(candidate.slot.target_pos, dtype=float, copy=True),
                target_quat=np.array(candidate.slot.target_quat, dtype=float, copy=True),
                source=source,
                reason=f"model_choice={choice} {reason or '(no reason)'} | {base}",
            ),
            "",
        )

    def decide(self, observation: Observation, candidates: Sequence[Candidate]) -> Decision:
        """选一个候选或停机。默认只喂图像 + 候选文本；异常只在 `strict` 下外抛。"""
        self.stats["decisions"] += 1
        try:
            return self._decide_inner(observation, candidates)
        except Exception as error:  # noqa: BLE001 - 策略层不允许把异常漏给执行器
            if self.strict:
                raise VLMError(f"VLMHighLevel strict 失败：{type(error).__name__}: {error}") from error
            self.stats["failures"] += 1
            self.stats["last_error"] = f"unexpected {type(error).__name__}: {error}"
            return self._fallback_decision(
                observation, candidates, f"exception:{type(error).__name__}: {error}"
            )

    def _decide_inner(self, observation: Observation, candidates: Sequence[Candidate]) -> Decision:
        candidates = list(candidates)
        # 非特权模式：走 image_only() 剥掉石块真值与关节角，只留图像与 TCP 位姿。
        view = observation if self.use_privileged_state else observation.image_only()
        privileged = observation if self.use_privileged_state else None

        if not candidates:
            self.stats["stops"] += 1
            return Decision(
                action="stop",
                source="vlm",
                reason="no candidates supplied -> stop without querying the model",
            )

        camera_names, skipped = _select_cameras(view, self.config)
        self.stats["image_skipped"] += skipped
        data_urls: list[str] = []
        kept_names: list[str] = []
        for name in camera_names:
            try:
                url, nbytes = _encode_data_url(view.cameras[name], self.image_max_side, self.jpeg_quality)
            except Exception as error:  # noqa: BLE001 - 单帧坏了不该毁掉整次决策
                self.stats["image_skipped"] += 1
                _record_event(
                    self._recent, "image_error", camera=name, error=f"{type(error).__name__}: {error}"
                )
                continue
            data_urls.append(url)
            kept_names.append(name)
            _record_event(self._recent, "image", camera=name, jpeg_bytes=nbytes)

        prompt = self.build_prompt(view, candidates, kept_names, privileged=privileged)
        user_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        user_parts.extend(_image_message(kept_names, data_urls))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _HIGH_SYSTEM_PROMPT},
            {"role": "user", "content": user_parts},
        ]
        self.stats["images_sent"] += len(data_urls)

        _trace(
            "[VLM>]",
            f"HIGH candidates={len(candidates)} images={len(data_urls)} cameras={kept_names} "
            f"prompt_chars={len(prompt)}"
            + ("" if not _TRACE_FULL else "\n" + prompt),
        )

        last_cause = "unknown"
        source = "vlm"
        for attempt in range(self.retries + 1):
            try:
                text = self.client.chat(messages)
            except VLMError as error:
                last_cause = f"http_error: {error}"
                _record_event(self._recent, "http_failure", attempt=attempt, error=str(error))
                if attempt < self.retries:
                    self.stats["retries"] += 1
                    # 网络层失败时对话里没有 assistant 轮；补一个占位 assistant 轮，
                    # 免得某些严格交替 role 的 chat 模板拒绝"连续两条 user"。
                    messages = self._repair_messages(messages, "(no reply: request failed)", last_cause)
                    source = "vlm_repair"
                    continue
                break

            _trace("[VLM<]", f"HIGH raw reply ({self.stats['last_latency_s']:.2f}s): {text.strip()[:2000]}")
            decision, failure = self._interpret(text, candidates, source, self.stats["last_latency_s"], len(data_urls))
            if decision is not None:
                _record_event(
                    self._recent, "decision", attempt=attempt, source=source, action=decision.action
                )
                _trace(
                    "[VLM=]",
                    f"HIGH action -> action={decision.action} stone={decision.stone} slot={decision.slot} "
                    f"target={None if decision.target_pos is None else np.round(decision.target_pos, 3).tolist()} "
                    f"source={decision.source}",
                )
                return decision
            last_cause = failure
            _record_event(self._recent, "parse_failure", attempt=attempt, error=failure)
            if attempt < self.retries:
                self.stats["retries"] += 1
                messages = self._repair_messages(messages, text, failure)
                source = "vlm_repair"

        if self.strict:
            raise VLMError(f"VLMHighLevel 无法得到可用决策：{last_cause}")
        return self._fallback_decision(observation, candidates, last_cause)

    @staticmethod
    def _repair_messages(
        messages: Sequence[Mapping[str, Any]], bad_reply: str, cause: str
    ) -> list[dict[str, Any]]:
        """构造修复轮：把坏回复挂回对话，再要求"只输出严格 JSON"。

        修复轮**不追加新的图像 part**（第一轮 user 消息里的图像仍在对话历史里），
        所以上行只多了一段很短的文本；测试可以逐条消息核对这一点。
        """
        repaired = [dict(message) for message in messages]
        repaired.append({"role": "assistant", "content": bad_reply[:1500]})
        repaired.append(
            {
                "role": "user",
                "content": (
                    f"Your previous reply was rejected ({cause}). "
                    'Reply with STRICT JSON only, exactly: {"choice": <int>, "reason": "<short>"} '
                    "where <int> is a valid candidate index or -1 to stop. No markdown, no prose."
                ),
            }
        )
        return repaired

    # -- 诊断 ------------------------------------------------------------------------
    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "episode": self._episode,
            "model": self.model,
            "base_url": self.client.base_url,
            "use_privileged_state": self.use_privileged_state,
            "fallback": getattr(self.fallback, "name", None),
            "recent_events": self.recent_events(),
        }


# --------------------------------------------------------------------------------------
# 低层策略
# --------------------------------------------------------------------------------------

_LOW_SYSTEM_PROMPT = (
    "You are the low-level pose-correction module of a robotic arm controller. "
    "The motion planner has already produced a metrically reliable nominal TCP target pose "
    "in the WORLD frame. Your only job is to output a SMALL correction to that nominal pose, "
    "based on the camera images and the reported state. Never output absolute poses. "
    "Reply with STRICT JSON only, no prose, no code fence: "
    '{"delta_xyz": [dx, dy, dz], "delta_yaw_deg": <float>, "grip_width": <float|null>, "advance": <bool>}'
)


class VLMLowLevel:
    """VLM 低层：对标称目标位姿输出有界增量（世界系，米/度）。

    - `vlm_goals`（默认 `["place"]`）之外的目标类型直接交给 `fallback`；
    - 增量逐轴夹到 ±`max_delta_m`，偏航夹到 ±`max_yaw_deg`，夹紧计数进 `stats`；
    - `advance=true` -> `Command.done=True`；
    - 任何失败（无标称位姿、网络、解析）同样走 `fallback`；
    - 没有 `fallback` 时退化为"保持当前 TCP 位姿"的 `Command`（`done=False`）。
    """

    def __init__(self, config: Mapping[str, Any] | None = None, fallback: Any = None):
        self.config = dict(config or {})
        self.fallback = fallback
        self.model = str(self.config.get("model", "Qwen3-VL-8B"))
        self.vlm_goals = tuple(str(kind) for kind in self.config.get("vlm_goals", DEFAULT_VLM_GOALS))
        self.use_privileged_state = bool(self.config.get("use_privileged_state", False))
        self.strict = bool(self.config.get("strict", False))
        self.retries = max(0, int(self.config.get("retries", 1)))
        self.max_delta_m = float(self.config.get("max_delta_m", 0.03))
        self.max_yaw_deg = float(self.config.get("max_yaw_deg", 8.0))
        self.max_grip_width_m = float(self.config.get("max_grip_width_m", 0.14))
        self.image_max_side = self.config.get("image_max_side", 768)
        self.jpeg_quality = int(self.config.get("jpeg_quality", 88))
        self.instruction = str(self.config.get("instruction", ""))
        self.stats = new_stats()
        self.client = VLMHttpClient(self.config, self.stats)
        self.name = f"vlm_low:{self.model}"
        self._recent: deque[dict[str, Any]] = deque(maxlen=_EVENT_BUFFER)
        self._episode = -1
        self._instruction_override: str | None = None
        #: 每回合记住每种目标最近一次用过的标称位姿（`retreat` 缺省位姿时复用）。
        self._nominal_memory: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- 生命周期 ---------------------------------------------------------------------
    def reset(self, context: Mapping[str, Any] | None = None) -> None:
        """清掉标称位姿记忆与事件缓冲；`context["instruction"]` 覆盖默认指令。"""
        context = dict(context or {})
        self._episode += 1
        self._recent.clear()
        self._nominal_memory.clear()
        self._instruction_override = None
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

    # -- 标称位姿 ---------------------------------------------------------------------
    def _effective_instruction(self, observation: Observation) -> str:
        if self._instruction_override:
            return self._instruction_override
        if observation.instruction:
            return observation.instruction
        return self.instruction

    def nominal_pose(
        self, observation: Observation, goal: Goal
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """解析执行器给的标称位姿。

        优先级（都能拿到旋转时用对应旋转，否则用当前 TCP 旋转）：
        `pick` -> goal.pick_pos -> goal.stone.pos；
        `place` -> goal.place_pos；
        `retreat` -> goal.place_pos -> goal.pick_pos -> 本回合记住的 place/pick 标称；
        `home` / `hold` -> 当前 TCP 位姿。
        全都没有时返回 None，调用方直接走 fallback（不猜位姿）。
        """
        tcp_rot = np.array(observation.tcp_rot, dtype=float, copy=True)
        kind = goal.kind
        if kind == "pick":
            if goal.pick_pos is not None:
                rot = goal.pick_rot if goal.pick_rot is not None else tcp_rot
                return np.array(goal.pick_pos, dtype=float, copy=True), np.array(rot, dtype=float, copy=True), "goal.pick_pos"
            if goal.stone is not None:
                return np.array(goal.stone.pos, dtype=float, copy=True), tcp_rot, "goal.stone.pos"
        elif kind == "place":
            if goal.place_pos is not None:
                rot = goal.place_rot if goal.place_rot is not None else tcp_rot
                return np.array(goal.place_pos, dtype=float, copy=True), np.array(rot, dtype=float, copy=True), "goal.place_pos"
        elif kind == "retreat":
            for label, pos, rot in (
                ("goal.place_pos", goal.place_pos, goal.place_rot),
                ("goal.pick_pos", goal.pick_pos, goal.pick_rot),
            ):
                if pos is not None:
                    rotation = rot if rot is not None else tcp_rot
                    return np.array(pos, dtype=float, copy=True), np.array(rotation, dtype=float, copy=True), label
            for key in ("place", "pick"):
                if key in self._nominal_memory:
                    pos, rot = self._nominal_memory[key]
                    return pos.copy(), rot.copy(), f"memory.{key}"
        elif kind in ("home", "hold"):
            return np.array(observation.tcp_pos, dtype=float, copy=True), tcp_rot, "observation.tcp_pos"
        return None

    # -- prompt ----------------------------------------------------------------------
    def build_prompt(
        self,
        observation: Observation,
        goal: Goal,
        nominal_pos: np.ndarray,
        nominal_rot: np.ndarray,
        nominal_source: str,
        camera_names: Sequence[str],
    ) -> str:
        """拼低层文本 prompt：当前 TCP 位姿、标称目标位姿、相位、夹爪、指令、增量边界。"""
        tcp_rpy = _rot_to_rpy_deg(observation.tcp_rot)
        nominal_rpy = _rot_to_rpy_deg(nominal_rot)
        lines: list[str] = []
        lines.append("TASK INSTRUCTION:")
        lines.append(f"  {self._effective_instruction(observation) or '(none given)'}")
        lines.append("")
        lines.append("GOAL:")
        lines.append(f"  kind={goal.kind} stone={goal.stone.name if goal.stone else None} phase={observation.phase}")
        lines.append(f"  nominal_source={nominal_source}")
        lines.append(f"  timeout_s={goal.timeout_s:.2f} approach_height_m={goal.approach_height:.3f}")
        lines.append("")
        lines.append("CURRENT TCP POSE (world frame):")
        lines.append(f"  tcp_xyz_m={_format_vec(observation.tcp_pos)}")
        lines.append(f"  tcp_rpy_deg=[{tcp_rpy[0]:+.2f} {tcp_rpy[1]:+.2f} {tcp_rpy[2]:+.2f}]")
        lines.append(f"  tcp_rot_rows={_format_vec(np.asarray(observation.tcp_rot, dtype=float).reshape(-1), 3)}")
        lines.append(f"  gripper_width_m={observation.gripper_width:.4f} held_stone={observation.held_stone or 'none'}")
        lines.append("")
        lines.append("NOMINAL GOAL POSE FROM THE PLANNER (world frame, already reachable):")
        lines.append(f"  goal_xyz_m={_format_vec(nominal_pos)}")
        lines.append(f"  goal_rpy_deg=[{nominal_rpy[0]:+.2f} {nominal_rpy[1]:+.2f} {nominal_rpy[2]:+.2f}]")
        lines.append(f"  goal_rot_rows={_format_vec(np.asarray(nominal_rot, dtype=float).reshape(-1), 3)}")
        lines.append(f"  nominal_grip_width_m={goal.grip_width}")
        lines.append("")
        lines.append(f"  cameras_sent={list(camera_names)}")
        lines.append("")
        lines.append("OUTPUT (strict JSON, no markdown):")
        lines.append(
            '  {"delta_xyz": [dx, dy, dz], "delta_yaw_deg": <float>, "grip_width": <float|null>, "advance": <bool>}'
        )
        lines.append("RULES:")
        lines.append(
            f"  - delta_xyz is added to goal_xyz_m, in METERS, world frame, "
            f"each axis clamped to +/-{self.max_delta_m:.4f} m."
        )
        lines.append(
            f"  - delta_yaw_deg rotates about world +Z, applied as Rz(delta) @ goal_rot, "
            f"clamped to +/-{self.max_yaw_deg:.2f} deg."
        )
        lines.append(
            f"  - grip_width is the absolute gripper opening in METERS in [0, {self.max_grip_width_m:.3f}], "
            "or null to keep the nominal opening."
        )
        lines.append("  - advance=true means the corrected pose is good enough / the goal is satisfied.")
        lines.append("  - If the nominal pose already looks correct, return all-zero deltas and advance=true.")
        return "\n".join(lines)

    # -- 命令 ------------------------------------------------------------------------
    def _hold_command(self, observation: Observation, note: str, source: str) -> Command:
        """最小内部兜底：保持当前 TCP 位姿，`done=False`（让执行器自己超时或换策略）。"""
        _record_event(self._recent, "hold", note=note)
        return Command(
            tcp_pos=np.array(observation.tcp_pos, dtype=float, copy=True),
            tcp_rot=np.array(observation.tcp_rot, dtype=float, copy=True),
            grip_width=None,
            done=False,
            note=note,
            source=source,
        )

    def _delegate(self, observation: Observation, goal: Goal, cause: str, count_fallback: bool = True) -> Command:
        """交给外部 fallback；没有 fallback 时保持当前位姿。每次回退都计数。"""
        if count_fallback:
            self.stats["fallbacks"] += 1
        _record_event(self._recent, "delegate", goal=goal.kind, cause=cause)
        if self.fallback is not None:
            command = self.fallback.step(observation, goal)
            source = command.source or f"fallback:{getattr(self.fallback, 'name', type(self.fallback).__name__)}"
            note = command.note or f"vlm_low_delegated({cause})"
            return replace(command, source=source, note=note)
        return self._hold_command(observation, f"vlm_low_no_fallback: {cause}", "hold:no_fallback")

    def step(self, observation: Observation, goal: Goal) -> Command:
        """给执行器一条 TCP 目标命令。异常只在 `strict` 下外抛。"""
        self.stats["decisions"] += 1
        try:
            return self._step_inner(observation, goal)
        except Exception as error:  # noqa: BLE001 - 策略层不允许把异常漏给执行器
            if self.strict:
                raise VLMError(f"VLMLowLevel strict 失败：{type(error).__name__}: {error}") from error
            self.stats["failures"] += 1
            self.stats["last_error"] = f"unexpected {type(error).__name__}: {error}"
            return self._delegate(observation, goal, f"exception:{type(error).__name__}: {error}")

    def _step_inner(self, observation: Observation, goal: Goal) -> Command:
        if goal.kind not in self.vlm_goals:
            return self._delegate(observation, goal, f"goal_not_in_vlm_goals({goal.kind})")

        resolved = self.nominal_pose(observation, goal)
        if resolved is None:
            return self._delegate(observation, goal, f"no_nominal_pose_for({goal.kind})")
        nominal_pos, nominal_rot, nominal_source = resolved
        self._nominal_memory[goal.kind] = (nominal_pos.copy(), nominal_rot.copy())

        view = observation if self.use_privileged_state else observation.image_only()
        camera_names, skipped = _select_cameras(view, self.config)
        self.stats["image_skipped"] += skipped
        kept_names: list[str] = []
        data_urls: list[str] = []
        for name in camera_names:
            try:
                url, nbytes = _encode_data_url(view.cameras[name], self.image_max_side, self.jpeg_quality)
            except Exception as error:  # noqa: BLE001
                self.stats["image_skipped"] += 1
                _record_event(
                    self._recent, "image_error", camera=name, error=f"{type(error).__name__}: {error}"
                )
                continue
            kept_names.append(name)
            data_urls.append(url)
            _record_event(self._recent, "image", camera=name, jpeg_bytes=nbytes)

        prompt = self.build_prompt(view, goal, nominal_pos, nominal_rot, nominal_source, kept_names)
        user_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        user_parts.extend(_image_message(kept_names, data_urls))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _LOW_SYSTEM_PROMPT},
            {"role": "user", "content": user_parts},
        ]
        self.stats["images_sent"] += len(data_urls)

        _trace(
            "[VLM>]",
            f"LOW goal={goal.kind} stone={goal.stone.name if goal.stone else None} phase={observation.phase} "
            f"images={len(data_urls)} nominal={np.round(nominal_pos, 4).tolist()} "
            f"nominal_grip={goal.grip_width}"
            + ("" if not _TRACE_FULL else "\n" + prompt),
        )

        last_cause = "unknown"
        source = "vlm"
        for attempt in range(self.retries + 1):
            try:
                text = self.client.chat(messages)
            except VLMError as error:
                last_cause = f"http_error: {error}"
                _record_event(self._recent, "http_failure", attempt=attempt, error=str(error))
                if attempt < self.retries:
                    self.stats["retries"] += 1
                    # 同上：网络失败时补一个占位 assistant 轮，保持 role 交替。
                    messages = VLMHighLevel._repair_messages(messages, "(no reply: request failed)", last_cause)
                    source = "vlm_repair"
                    continue
                break

            _trace("[VLM<]", f"LOW raw reply ({self.stats['last_latency_s']:.2f}s): {text.strip()[:2000]}")
            command, failure = self._interpret(
                text, observation, goal, nominal_pos, nominal_rot, source, len(data_urls)
            )
            if command is not None:
                _record_event(
                    self._recent, "command", attempt=attempt, source=source, done=command.done
                )
                _trace(
                    "[VLM=]",
                    f"LOW action -> tcp_pos={np.round(command.tcp_pos, 4).tolist()} "
                    f"grip={command.grip_width} done={command.done} translation_only={command.pure_translation} "
                    f"source={command.source} note={command.note}",
                )
                return command
            last_cause = failure
            _record_event(self._recent, "parse_failure", attempt=attempt, error=failure)
            if attempt < self.retries:
                self.stats["retries"] += 1
                messages = VLMHighLevel._repair_messages(messages, text, failure)
                source = "vlm_repair"

        if self.strict:
            raise VLMError(f"VLMLowLevel 无法得到可用命令：{last_cause}")
        return self._delegate(observation, goal, last_cause)

    def _interpret(
        self,
        text: str,
        observation: Observation,
        goal: Goal,
        nominal_pos: np.ndarray,
        nominal_rot: np.ndarray,
        source: str,
        images: int,
    ) -> tuple[Command | None, str]:
        """解析增量 JSON -> 夹紧 -> Command。返回 (command, 失败原因)。"""
        try:
            payload = extract_json_object(text)
        except ValueError as error:
            self.stats["parse_failures"] += 1
            self.stats["last_error"] = str(error)
            return None, f"json_parse_error: {error}"

        if "delta_xyz" not in payload:
            self.stats["parse_failures"] += 1
            self.stats["last_error"] = f"missing delta_xyz: keys={sorted(payload)[:8]}"
            return None, f"missing_delta_xyz keys={sorted(payload)[:8]}"
        try:
            delta = np.asarray(payload["delta_xyz"], dtype=float).reshape(-1)
        except (TypeError, ValueError) as error:
            self.stats["parse_failures"] += 1
            self.stats["last_error"] = f"delta_xyz not numeric: {error}"
            return None, f"delta_xyz_not_numeric value={payload['delta_xyz']!r}"
        if delta.shape != (3,) or not np.isfinite(delta).all():
            self.stats["parse_failures"] += 1
            self.stats["last_error"] = f"delta_xyz shape/finite: {delta}"
            return None, f"delta_xyz_bad_shape value={payload['delta_xyz']!r}"

        try:
            yaw = float(payload.get("delta_yaw_deg", 0.0))
        except (TypeError, ValueError):
            self.stats["parse_failures"] += 1
            return None, f"delta_yaw_deg_not_numeric value={payload.get('delta_yaw_deg')!r}"
        if not math.isfinite(yaw):
            self.stats["parse_failures"] += 1
            return None, "delta_yaw_deg_not_finite"

        # ---- 夹紧：逐轴（位移）/ 标量（偏航），并计数
        clipped_delta = np.clip(delta, -self.max_delta_m, self.max_delta_m)
        clipped_yaw = float(np.clip(yaw, -self.max_yaw_deg, self.max_yaw_deg))
        axis_hits = int(np.count_nonzero(np.abs(clipped_delta - delta) > 1e-12))
        yaw_hit = int(abs(clipped_yaw - yaw) > 1e-12)
        if axis_hits or yaw_hit:
            self.stats["clipped"] += 1
            self.stats["clipped_xyz"] += axis_hits
            self.stats["clipped_yaw"] += yaw_hit
            _record_event(
                self._recent,
                "clamped",
                requested_xyz=[round(float(v), 4) for v in delta],
                applied_xyz=[round(float(v), 4) for v in clipped_delta],
                requested_yaw_deg=round(yaw, 4),
                applied_yaw_deg=round(clipped_yaw, 4),
            )

        target_pos = nominal_pos + clipped_delta
        target_rot = _yaw_matrix(clipped_yaw) @ nominal_rot

        # ---- 夹爪：null 表示沿用标称开口
        raw_grip = payload.get("grip_width", None)
        grip_note = "keep"
        grip_width: float | None = goal.grip_width
        if raw_grip is not None:
            try:
                requested_grip = float(raw_grip)
            except (TypeError, ValueError):
                self.stats["parse_failures"] += 1
                self.stats["last_error"] = f"grip_width not numeric: {raw_grip!r}"
                return None, f"grip_width_not_numeric value={raw_grip!r}"
            if not math.isfinite(requested_grip):
                self.stats["parse_failures"] += 1
                return None, "grip_width_not_finite"
            grip_width = float(np.clip(requested_grip, 0.0, self.max_grip_width_m))
            grip_note = "set"
            if abs(grip_width - requested_grip) > 1e-12:
                self.stats["clipped_grip"] += 1
                grip_note = f"clamped_from_{requested_grip:.4f}"

        advance = bool(payload.get("advance", False))
        note = (
            f"delta_xyz={_format_vec(clipped_delta)} delta_yaw_deg={clipped_yaw:+.2f} "
            f"grip={grip_note} advance={advance} nominal={_format_vec(nominal_pos)}"
            + (
                f" requested_xyz={_format_vec(delta)} requested_yaw_deg={yaw:+.2f}"
                if (axis_hits or yaw_hit)
                else ""
            )
        )
        return (
            Command(
                tcp_pos=target_pos,
                tcp_rot=target_rot,
                grip_width=grip_width,
                done=advance,
                note=note,
                source=source,
            ),
            "",
        )

    # -- 诊断 ------------------------------------------------------------------------
    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            "episode": self._episode,
            "model": self.model,
            "base_url": self.client.base_url,
            "vlm_goals": list(self.vlm_goals),
            "max_delta_m": self.max_delta_m,
            "max_yaw_deg": self.max_yaw_deg,
            "fallback": getattr(self.fallback, "name", None),
            "recent_events": self.recent_events(),
        }
