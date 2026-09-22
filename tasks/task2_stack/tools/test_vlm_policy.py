"""VLM 策略的端到端测试：桩服务 + 合成观测 + 逐项测量。

本机没有可用 GPU（`torch.cuda.is_available()` 为 False），起不了真正的
Qwen3-VL 服务，所以真实链路用 `tools/vlm_stub_server.py`（纯标准库的 OpenAI 兼容
桩服务）来跑通：策略只认 `POST {base_url}/chat/completions` 这个 HTTP 契约，
桩服务按脚本返回合法 JSON / 围栏 JSON / 垃圾文本 / HTTP 500 / 慢响应，
并记录每次请求的 prompt 文本、图像张数、base64 解码后的字节数、JPEG 魔数与 sha256。

测试做到的事：
- **不依赖 GPU、不导入 mujoco**：观测是用 numpy/PIL 画出来的合成场景（渐变月壤地面、
  灰色多边形石块、夹爪剪影），配上一组合乎尺寸的 `StoneState` / `PlanSlot` / `Candidate`。
- **测的都是可测量的量**：图像张数与字节数、图像 sha256 与哪一路相机对应、解析失败
  计数器、夹紧后的位姿与夹紧计数、超时耗时……每条断言都带实测值并写进报告 JSON。
- **真实服务是可选项**：`--real-url` 给一个真的 OpenAI 兼容地址；连不上就打一行
  `SKIP` 并以 0 退出（缺 GPU 不该让测试变红）。

用法：
    PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack \\
        /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python \\
        tasks/task2_stack/tools/test_vlm_policy.py
    # 可选：--real-url http://127.0.0.1:3001/v1 --real-model Qwen3-VL-8B
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests
from PIL import Image, ImageDraw

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

_MODULES_BEFORE = set(sys.modules)

from stone_stack.policy.base import (  # noqa: E402 - sys.path 先就位再导入
    CameraFrame,
    Candidate,
    Command,
    Decision,
    FallbackLowLevel,
    Goal,
    Observation,
    PlanSlot,
    StoneState,
)
from stone_stack.policy.vlm import (  # noqa: E402
    VLMError,
    VLMHighLevel,
    VLMLowLevel,
    check_server,
    extract_json_object,
    new_stats,
)

STUB_PATH = Path(__file__).resolve().parent / "vlm_stub_server.py"
REPORT_PATH = TASK_ROOT / "reports" / "vlm_policy_test.json"
DEFAULT_BASE_URL = "http://127.0.0.1:3001/v1"
DEFAULT_MODEL = "Qwen3-VL-8B"

#: 图像编码参数（策略与测试必须一致，否则 sha256 对不上）
IMAGE_MAX_SIDE = 768
JPEG_QUALITY = 88
#: 判定"图像不是空白噪声"的最小字节数
MIN_JPEG_BYTES = 2048

INSTRUCTION = "Build a level three-course dry-stone wall on the marked footprint."

#: 只有终端才上色；重定向到文件/日志时保持纯文本，方便 grep 与贴报告。
_USE_COLOR = sys.stdout.isatty()


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """把 numpy / Path / 元组等转成能进 JSON 的类型。"""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return round(value, 6)
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def frame_sha256(frame: CameraFrame, max_side: int = IMAGE_MAX_SIDE, quality: int = JPEG_QUALITY) -> str:
    """本地算某路相机的 JPEG sha256，用来核对桩服务收到的到底是哪张图。"""
    return hashlib.sha256(frame.jpeg_bytes(quality=quality, max_side=max_side)).hexdigest()


def yaw_matrix(degrees: float) -> np.ndarray:
    radians = float(np.deg2rad(degrees))
    cos, sin = float(np.cos(radians)), float(np.sin(radians))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rot_x(degrees: float) -> np.ndarray:
    radians = float(np.deg2rad(degrees))
    cos, sin = float(np.cos(radians)), float(np.sin(radians))
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]], dtype=float)


def vec_text(values: Any, precision: int = 4) -> str:
    """与 vlm.py 的 `_format_vec` 完全一致的格式化，用来在 prompt 里找石块真值。"""
    array = np.asarray(values, dtype=float).reshape(-1)
    return "[" + " ".join(f"{value:+.{precision}f}" for value in array) + "]"


class Recorder:
    """收集断言结果：不提前中断，跑完全部场景后统一报告并决定退出码。"""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.section = "general"

    def begin(self, section: str) -> None:
        self.section = section

    def check(self, name: str, ok: bool, detail: str = "", **measured: Any) -> bool:
        entry = {
            "section": self.section,
            "name": name,
            "status": "pass" if ok else "fail",
            "detail": detail,
            "measured": jsonable(measured),
        }
        self.checks.append(entry)
        marker = ("\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m") if _USE_COLOR else ("PASS" if ok else "FAIL")
        summary = ", ".join(f"{key}={jsonable(value)}" for key, value in measured.items())
        print(f"  [{marker}] {name}" + (f" :: {summary}" if summary else ""))
        if not ok and detail:
            print(f"         -> {detail}")
        return ok

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [entry for entry in self.checks if entry["status"] == "fail"]

    def measured_groups(self) -> dict[str, Any]:
        groups: dict[str, Any] = {}
        for entry in self.checks:
            groups.setdefault(entry["section"], {})[entry["name"]] = {
                "status": entry["status"],
                "measured": entry["measured"],
            }
        return groups


# --------------------------------------------------------------------------------------
# 合成场景（无 GPU、无渲染器、无 mujoco）
# --------------------------------------------------------------------------------------


def _polygon(cx: float, cy: float, radius: float, rng: np.random.Generator, vertices: int = 7) -> list[tuple[float, float]]:
    """一个不规则凸多边形，模拟石块轮廓。"""
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, vertices))
    jitter = rng.uniform(0.68, 1.18, vertices)
    return [
        (float(cx + radius * jitter[i] * np.cos(angles[i])), float(cy + radius * 0.62 * jitter[i] * np.sin(angles[i])))
        for i in range(vertices)
    ]


def render_camera_frame(name: str, width: int, height: int, seed: int, view: str) -> CameraFrame:
    """画一张"看起来像渲染结果"的 RGB-D 帧：渐变月壤地面 + 灰色多边形石块 + 夹爪剪影。

    真实感不重要，重要的是**非退化**：有梯度、有噪点、有明确的前景物体，JPEG 编出来
    远大于 2 KB，且能验证"哪路相机发了什么"（每路 seed/view 不同 -> 图不同 -> sha256 不同）。
    """
    rng = np.random.default_rng(seed)
    horizon = int(height * (0.10 if view == "top" else 0.40))
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    # 天空（真空黑）与月壤地面（按行渐变）
    rgb[:horizon] = np.array([10.0, 11.0, 16.0], dtype=np.float32)
    rows = np.arange(horizon, height, dtype=np.float32)
    ramp = (rows - horizon) / max(1.0, float(height - horizon))
    level = (52.0 + 104.0 * ramp)[:, None]
    rgb[horizon:, :, 0] = level * 1.00
    rgb[horizon:, :, 1] = level * 0.97
    rgb[horizon:, :, 2] = level * 0.90

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)

    # 地面上的陨石坑 / 阴影斑
    for _ in range(9):
        cx = float(rng.uniform(0, width))
        cy = float(rng.uniform(horizon + 4, height))
        rx = float(rng.uniform(12, 58))
        ry = rx * float(rng.uniform(0.18, 0.40))
        shade = int(rng.uniform(-26, -8))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(max(0, 90 + shade), max(0, 88 + shade), max(0, 82 + shade)))

    if view == "top":  # 俯视：画一圈"墙基 footprint"
        draw.rectangle(
            [width * 0.16, height * 0.18, width * 0.84, height * 0.40],
            outline=(196, 188, 168),
            width=3,
        )

    # 石块：灰色多边形 + 暗轮廓 + 亮高光边
    stone_depth_values: list[float] = []
    for index in range(6):
        cx = float(rng.uniform(width * 0.08, width * 0.92))
        cy = float(rng.uniform(horizon + height * 0.08, height * 0.94))
        radius = float(rng.uniform(width * 0.05, width * 0.13))
        points = _polygon(cx, cy, radius, rng)
        tone = int(rng.uniform(96, 168))
        draw.polygon(points, fill=(tone, int(tone * 0.97), int(tone * 0.90)), outline=(38, 36, 34))
        draw.line(points[:3], fill=(min(255, tone + 52),) * 3, width=2)
        stone_depth_values.append(float(rng.uniform(0.55, 1.85)))

    # 夹爪剪影：机身 + 两片夹指 + 金属高光
    grip_scale = 1.0 if view == "wrist" else 0.55
    gx = width * (0.50 if view == "wrist" else 0.30)
    gy = height * (0.42 if view == "wrist" else 0.68)
    body_w, body_h = width * 0.30 * grip_scale, height * 0.16 * grip_scale
    draw.rounded_rectangle(
        [gx - body_w / 2, gy - body_h * 2.1, gx + body_w / 2, gy - body_h * 0.5],
        radius=max(2, int(body_h * 0.25)),
        fill=(46, 48, 54),
        outline=(22, 22, 26),
        width=2,
    )
    for side in (-1.0, 1.0):
        pad_x = gx + side * body_w * 0.34
        draw.rounded_rectangle(
            [pad_x - body_w * 0.10, gy - body_h * 0.6, pad_x + body_w * 0.10, gy + body_h * 3.2],
            radius=max(2, int(body_h * 0.18)),
            fill=(74, 78, 86),
            outline=(26, 26, 30),
            width=2,
        )
        draw.line(
            [pad_x - body_w * 0.10, gy + body_h * 0.2, pad_x - body_w * 0.10, gy + body_h * 3.0],
            fill=(178, 182, 190),
            width=2,
        )

    grain = rng.normal(0.0, 5.0, size=(height, width, 3))
    rgb = np.clip(np.asarray(image, dtype=np.float32) + grain, 0, 255).astype(np.uint8)

    # 深度：0 表示没命中（天空），地面按行递增，石块与夹爪更近
    depth = np.zeros((height, width), dtype=np.float32)
    depth[horizon:, :] = (0.9 + 2.6 * ((np.arange(horizon, height, dtype=np.float32) - horizon) / max(1.0, height - horizon)))[:, None]
    depth_image = Image.new("F", (width, height), 0.0)
    depth_draw = ImageDraw.Draw(depth_image)
    for index in range(4):
        cx = float(rng.uniform(width * 0.10, width * 0.90))
        cy = float(rng.uniform(horizon + height * 0.10, height * 0.92))
        points = _polygon(cx, cy, float(rng.uniform(width * 0.05, width * 0.11)), rng)
        depth_draw.polygon(points, fill=stone_depth_values[index % len(stone_depth_values)])
    depth_draw.rectangle(
        [gx - body_w * 0.6, gy - body_h * 2.1, gx + body_w * 0.6, gy + body_h * 3.2],
        fill=0.28 if view == "wrist" else 0.62,
    )
    stone_mask = np.asarray(depth_image, dtype=np.float32)
    depth = np.where(stone_mask > 0.0, stone_mask, depth).astype(np.float32)

    pos = np.array([0.42, -0.18, 0.86], dtype=float)
    rot = rot_x(-90.0) if view != "top" else np.eye(3)
    return CameraFrame(
        name=name,
        rgb=rgb,
        depth=depth,
        fovy_deg=45.0 if view == "wrist" else 58.0,
        width=width,
        height=height,
        pos=pos,
        rot=rot,
    )


class Scene:
    """一整套合成观测 + 候选，供所有场景复用。"""

    def __init__(self) -> None:
        self.cameras = {
            "wrist": render_camera_frame("wrist", 640, 480, seed=11, view="wrist"),
            "top": render_camera_frame("top", 640, 480, seed=22, view="top"),
            "front": render_camera_frame("front", 800, 600, seed=33, view="front"),
        }
        self.stones = (
            StoneState(
                name="S0",
                pos=np.array([0.1370, -0.2120, 0.0350]),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                size=np.array([0.210, 0.140, 0.070]),
                mass=1.85,
            ),
            StoneState(
                name="S1",
                pos=np.array([-0.2840, 0.1660, 0.0320]),
                quat=np.array([0.9988, 0.0, 0.0, 0.0499]),
                size=np.array([0.180, 0.120, 0.064]),
                mass=1.42,
            ),
            StoneState(
                name="S2",
                pos=np.array([0.3520, 0.2810, 0.0410]),
                quat=np.array([0.9990, 0.0, 0.0, -0.0447]),
                size=np.array([0.240, 0.170, 0.082]),
                mass=2.35,
            ),
            StoneState(
                name="S3",
                pos=np.array([-0.1180, -0.3350, 0.0280]),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),
                size=np.array([0.150, 0.110, 0.056]),
                mass=0.94,
                graspable=False,
            ),
        )
        slots = (
            PlanSlot(course=0, slot_index=0, target_pos=np.array([-0.310, 0.050, 0.030]), target_quat=np.eye(3)),
            PlanSlot(course=0, slot_index=1, target_pos=np.array([-0.095, 0.050, 0.030]), target_quat=np.eye(3)),
            PlanSlot(course=1, slot_index=0, target_pos=np.array([-0.205, 0.050, 0.098]), target_quat=np.eye(3)),
            PlanSlot(course=2, slot_index=0, target_pos=np.array([-0.150, 0.050, 0.166]), target_quat=np.eye(3)),
        )
        scores = (0.42, 0.77, 0.61, 0.35)  # 最大值在 index 1 -> 确定性回退选它
        self.candidates = [
            Candidate(stone=self.stones[index], slot=slots[index], score=scores[index], note=f"stability={scores[index]:.2f}")
            for index in range(4)
        ]
        self.argmax_index = int(max(range(len(self.candidates)), key=lambda i: self.candidates[i].score))
        self.tcp_pos = np.array([0.2050, -0.0900, 0.3150], dtype=float)
        self.tcp_rot = yaw_matrix(180.0) @ rot_x(180.0)
        self.gripper_width = 0.0620
        self.nominal_pos = np.array([-0.2050, 0.0500, 0.1180], dtype=float)
        self.nominal_rot = yaw_matrix(15.0) @ rot_x(180.0)
        self.observation = self.make_observation()

    def make_observation(self, **overrides: Any) -> Observation:
        payload: dict[str, Any] = {
            "time": 12.5,
            "phase": "place",
            "joints": {"shoulder_pan_joint": 0.12, "elbow_joint": -1.44, "robotiq_85_left_knuckle_joint": 0.31},
            "tcp_pos": self.tcp_pos.copy(),
            "tcp_rot": self.tcp_rot.copy(),
            "gripper_width": self.gripper_width,
            "cameras": self.cameras,
            "stones": self.stones,
            "held_stone": "S2",
            "instruction": INSTRUCTION,
            "step_index": 42,
            "info": {"executor": "truth_stack", "privileged_note": "do not leak this"},
        }
        payload.update(overrides)
        return Observation(**payload)

    def place_goal(self, **overrides: Any) -> Goal:
        payload: dict[str, Any] = {
            "kind": "place",
            "stone": self.stones[2],
            "pick_pos": np.array([0.3520, 0.2810, 0.1500]),
            "pick_rot": yaw_matrix(90.0) @ rot_x(180.0),
            "place_pos": self.nominal_pos.copy(),
            "place_rot": self.nominal_rot.copy(),
            "grip_width": 0.0200,
            "approach_height": 0.18,
            "travel_height": 0.34,
            "timeout_s": 6.0,
        }
        payload.update(overrides)
        return Goal(**payload)

    def scene_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for name, frame in self.cameras.items():
            rgb = np.asarray(frame.rgb)
            jpeg = frame.jpeg_bytes(quality=JPEG_QUALITY, max_side=IMAGE_MAX_SIDE)
            with Image.open(io.BytesIO(jpeg)) as encoded:
                encoded_size = list(encoded.size)
            stats[name] = {
                "rgb_shape": list(rgb.shape),
                "rgb_mean": round(float(rgb.mean()), 3),
                "rgb_std": round(float(rgb.std()), 3),
                "unique_colors": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]),
                "jpeg_bytes": len(jpeg),
                "encoded_size": encoded_size,
                "sha256": hashlib.sha256(jpeg).hexdigest()[:16],
                **frame.depth_stats(),
            }
        return stats


# --------------------------------------------------------------------------------------
# 桩服务进程
# --------------------------------------------------------------------------------------


class StubHarness:
    """把 `vlm_stub_server.py` 拉起来（子进程 + 临时端口），并提供脚本/统计接口。"""

    def __init__(self, python: str, workdir: Path, verbose: bool = False) -> None:
        self.python = python
        self.workdir = workdir
        self.verbose = verbose
        self.process: subprocess.Popen[str] | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.ready: dict[str, Any] = {}
        self.stderr_path = workdir / "stub_stderr.log"
        self.initial_script_path = workdir / "stub_initial_script.json"
        self.stats_path = workdir / "stub_stats.json"
        self.session = requests.Session()
        self.exit_code: int | None = None
        self.root_url = ""

    # -- 生命周期 ---------------------------------------------------------------------
    def start(self, initial_script: Mapping[str, Any], timeout_s: float = 25.0) -> dict[str, Any]:
        self.initial_script_path.write_text(json.dumps(initial_script, ensure_ascii=False, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(TASK_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            self.python,
            str(STUB_PATH),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--script",
            str(self.initial_script_path),
            "--stats-out",
            str(self.stats_path),
            "--max-records",
            "200",
        ]
        if self.verbose:
            command.append("--verbose")
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=open(self.stderr_path, "w", encoding="utf-8"),
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                break
            if line.startswith("STUB_READY "):
                self.ready = json.loads(line[len("STUB_READY ") :])
                self.ready["command"] = command
                self.root_url = f"http://{self.ready['host']}:{self.ready['port']}"
                return self.ready
        raise RuntimeError(f"桩服务未在 {timeout_s}s 内就绪；stderr={self.stderr_text()[-800:]}")

    def _pump_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))

    def stderr_text(self) -> str:
        try:
            return self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @property
    def base_url(self) -> str:
        return str(self.ready["base_url"])

    # -- HTTP 接口 --------------------------------------------------------------------
    def _get(self, path: str, **kwargs: Any) -> Any:
        response = self.session.get(self.root_url + path, timeout=10, **kwargs)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: Mapping[str, Any] | None = None, timeout: float = 10.0) -> Any:
        response = self.session.post(self.root_url + path, json=dict(payload or {}), timeout=timeout)
        response.raise_for_status()
        return response.json()

    def models(self) -> list[str]:
        return [entry["id"] for entry in self._get("/v1/models")["data"]]

    def set_script(self, script: Mapping[str, Any]) -> dict[str, Any]:
        return self._post("/__script", script)

    def reset_records(self) -> dict[str, Any]:
        return self._post("/__reset")

    def requests(self) -> list[dict[str, Any]]:
        return list(self._get("/__stats")["requests"])

    def stats(self) -> dict[str, Any]:
        return self._get("/__stats")

    def dump(self, path: Path) -> dict[str, Any]:
        return self._post("/__dump", {"path": str(path)})

    def completion_content(self, text: str = "ping") -> str:
        """直接打一次 /v1/chat/completions，取回助手文本（用于验证脚本队列语义）。"""
        payload = {"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 16}
        response = self.session.post(self.base_url + "/chat/completions", json=payload, timeout=10)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def stop(self, timeout_s: float = 15.0) -> int | None:
        """优雅停机：先 `/__shutdown`，不行再 terminate/kill；返回退出码。"""
        if self.process is None:
            return None
        if self.process.poll() is None:
            try:
                self._post("/__shutdown", {}, timeout=5.0)
            except Exception:  # noqa: BLE001 - 停机路径要尽量温和，失败就升级手段
                self.process.terminate()
        try:
            self.exit_code = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.exit_code = self.process.wait(timeout=timeout_s)
        return self.exit_code


# --------------------------------------------------------------------------------------
# 供测试用的脚本化策略（spy / scripted）
# --------------------------------------------------------------------------------------


class SpyLowLevel:
    """记录调用次数的低层 fallback。"""

    def __init__(self, name: str = "spy_low") -> None:
        self.name = name
        self.calls: list[str] = []

    def reset(self, context: Mapping[str, Any] | None = None) -> None:
        self.calls.clear()
        self.last_context = dict(context or {})

    def step(self, observation: Observation, goal: Goal) -> Command:
        self.calls.append(goal.kind)
        return Command(
            tcp_pos=np.asarray(observation.tcp_pos, dtype=float).copy(),
            tcp_rot=np.asarray(observation.tcp_rot, dtype=float).copy(),
            grip_width=0.05,
            done=True,
            note=f"spy handled {goal.kind}",
            source="spy_fallback",
        )


class ScriptedHighLevel:
    """总是选固定下标的确定性高层 fallback。"""

    def __init__(self, index: int, name: str = "scripted_high") -> None:
        self.name = name
        self.index = index
        self.calls = 0

    def reset(self, context: Mapping[str, Any] | None = None) -> None:
        self.reset_context = dict(context or {})

    def decide(self, observation: Observation, candidates: Sequence[Candidate]) -> Decision:
        self.calls += 1
        candidate = candidates[self.index]
        return Decision(
            action="place",
            stone=candidate.stone.name,
            slot=candidate.slot.key,
            target_pos=np.array(candidate.slot.target_pos, dtype=float, copy=True),
            target_quat=np.array(candidate.slot.target_quat, dtype=float, copy=True),
            source="scripted",
            reason=f"scripted fallback picked {self.index}",
        )


# --------------------------------------------------------------------------------------
# 场景
# --------------------------------------------------------------------------------------


def base_high_config(stub: StubHarness, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base_url": stub.base_url,
        "model": DEFAULT_MODEL,
        "api_key": "EMPTY",
        "timeout_s": 5.0,
        "max_tokens": 128,
        "temperature": 0.0,
        "max_images": 3,
        "image_max_side": IMAGE_MAX_SIDE,
        "jpeg_quality": JPEG_QUALITY,
        "retries": 1,
        "strict": False,
    }
    config.update(overrides)
    return config


def base_low_config(stub: StubHarness, **overrides: Any) -> dict[str, Any]:
    config = base_high_config(stub)
    config.update({"max_delta_m": 0.03, "max_yaw_deg": 8.0, "vlm_goals": ["place"]})
    config.update(overrides)
    return config


def scenario_stub_script_modes(rec: Recorder, stub: StubHarness) -> None:
    """桩服务的队列语义：队列用完后按 fallback / repeat_last 退化（只打桩服务，不过策略）。"""
    rec.begin("stub_script")
    stub.set_script({"responses": [{"content": "QUEUED-A"}], "fallback": {"content": "FALLBACK-F"}})
    stub.reset_records()
    with_fallback = [stub.completion_content(), stub.completion_content(), stub.completion_content()]
    rec.check(
        "script queue drains then falls back to the 'fallback' spec",
        with_fallback == ["QUEUED-A", "FALLBACK-F", "FALLBACK-F"],
        f"contents={with_fallback}",
        contents=with_fallback,
        queue_remaining=stub.stats()["queue_remaining"],
    )

    stub.set_script({"responses": [{"content": "R1"}, {"content": "R2"}], "repeat_last": True})
    stub.reset_records()
    with_repeat = [stub.completion_content(), stub.completion_content(), stub.completion_content()]
    rec.check(
        "repeat_last=true repeats the final spec after the queue drains",
        with_repeat == ["R1", "R2", "R2"],
        f"contents={with_repeat}",
        contents=with_repeat,
    )

    stub.set_script({"responses": [{"status": 500, "body": {"error": {"message": "custom body"}}}]})
    stub.reset_records()
    body_message = ""
    status = 0
    response = stub.session.post(
        stub.base_url + "/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}, timeout=10
    )
    status = response.status_code
    body_message = response.json()["error"]["message"]
    rec.check(
        "scripted HTTP error honours status and custom body",
        status == 500 and body_message == "custom body",
        f"status={status} body={body_message}",
        status=status,
        body_message=body_message,
    )
    rec.check(
        "every stub request is recorded with timestamp and byte accounting",
        all(
            {"timestamp", "prompt_text", "image_count", "image_decoded_bytes", "request_bytes", "response_status"}
            <= set(record)
            for record in stub.requests()
        ),
        measured={
            "n_records": len(stub.requests()),
            "last_record_keys": sorted(stub.requests()[-1]) if stub.requests() else [],
        },
    )


def scenario_engine_agnostic(rec: Recorder) -> None:
    """策略模块不得把 mujoco 拖进来（测试机没有 GL/物理也要能跑）。"""
    rec.begin("engine_agnostic")
    added = sorted(name for name in set(sys.modules) - _MODULES_BEFORE if name.split(".")[0] == "mujoco")
    rec.check("policy modules did not import mujoco", not added, f"unexpected modules: {added}", added_modules=added)
    rec.check(
        "vlm module file path",
        "stone_stack/policy/vlm.py" in str(sys.modules["stone_stack.policy.vlm"].__file__),
        measured={"module_file": str(sys.modules["stone_stack.policy.vlm"].__file__)},
    )


def scenario_stub_and_transport(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """桩服务基本端点 + 高层一次成功决策：图像张数/字节数/魔数/sha256 全部核对。"""
    rec.begin("stub_server")
    models = stub.models()
    rec.check("GET /v1/models returns model list", DEFAULT_MODEL in models, f"models={models}", models=models)
    rec.check(
        "stub listens on ephemeral port (--port 0)",
        int(stub.ready["port"]) > 0,
        measured={"host": stub.ready["host"], "port": stub.ready["port"], "base_url": stub.base_url},
    )
    dumped = stub.dump(stub.workdir / "stub_dump_probe.json")
    rec.check(
        "POST /__dump writes stats json",
        bool(dumped.get("ok")) and (stub.workdir / "stub_dump_probe.json").exists(),
        measured={"path": dumped.get("path"), "exists": (stub.workdir / "stub_dump_probe.json").exists()},
    )

    rec.begin("high_level_transport")
    stub.set_script({"responses": [{"content": json.dumps({"choice": 2, "reason": "course 1 seat is free"})}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub))
    decision = policy.decide(scene.observation, scene.candidates)
    records = stub.requests()

    expected_sha = [frame_sha256(scene.cameras[name]) for name in ("wrist", "top", "front")]
    rec.check("exactly one HTTP request for one decide()", len(records) == 1, f"n={len(records)}", n_requests=len(records))
    if not records:
        return
    record = records[0]
    rec.check(
        "request carried 3 camera images",
        record["image_count"] == 3,
        f"image_count={record['image_count']}",
        image_count=record["image_count"],
    )
    rec.check(
        "images arrive in wrist,top,front order (sha256 match)",
        record["image_sha256"] == expected_sha,
        f"got={[h[:12] for h in record['image_sha256']]} want={[h[:12] for h in expected_sha]}",
        got_sha256=[h[:16] for h in record["image_sha256"]],
        want_sha256=[h[:16] for h in expected_sha],
    )
    rec.check(
        "every image is a plausible JPEG (>2KB, FFD8FF magic, data URL prefix)",
        all(size > MIN_JPEG_BYTES for size in record["image_decoded_bytes"])
        and all(record["image_magic_ok"])
        and all(prefix == "data:image/jpeg;base64" for prefix in record["image_prefixes"]),
        f"sizes={record['image_decoded_bytes']} magic={record['image_magic_ok']}",
        decoded_bytes=record["image_decoded_bytes"],
        min_bytes=min(record["image_decoded_bytes"]),
        magic_ok=record["image_magic_ok"],
        data_url_prefixes=record["image_prefixes"],
    )
    rec.check(
        "image bytes match local jpeg encode exactly",
        record["image_decoded_bytes"] == [len(scene.cameras[n].jpeg_bytes(quality=JPEG_QUALITY, max_side=IMAGE_MAX_SIDE)) for n in ("wrist", "top", "front")],
        measured={
            "stub_bytes": record["image_decoded_bytes"],
            "local_bytes": [len(scene.cameras[n].jpeg_bytes(quality=JPEG_QUALITY, max_side=IMAGE_MAX_SIDE)) for n in ("wrist", "top", "front")],
        },
    )
    rec.check(
        "prompt contains instruction and numbered candidate list",
        INSTRUCTION in record["prompt_text"]
        and "[0]" in record["prompt_text"]
        and "[3]" in record["prompt_text"],
        measured={"prompt_chars": record["prompt_chars"], "has_instruction": INSTRUCTION in record["prompt_text"]},
    )
    rec.check(
        "decision follows the stub's choice",
        decision.stone == scene.candidates[2].stone.name and decision.slot == scene.candidates[2].slot.key,
        f"stone={decision.stone} slot={decision.slot}",
        stone=decision.stone,
        slot=list(decision.slot or ()),
        action=decision.action,
    )
    rec.check("decision.source == 'vlm'", decision.source == "vlm", decision.source, source=decision.source)
    rec.check(
        "decision carries the chosen slot geometry",
        np.allclose(decision.target_pos, scene.candidates[2].slot.target_pos)
        and np.allclose(decision.target_quat, scene.candidates[2].slot.target_quat),
        measured={"target_pos": decision.target_pos, "reason": decision.reason},
    )
    rec.check(
        "stats after success",
        policy.stats["requests"] == 1
        and policy.stats["failures"] == 0
        and policy.stats["parse_failures"] == 0
        and policy.stats["images_sent"] == 3
        and policy.stats["last_latency_s"] > 0.0,
        measured={
            "requests": policy.stats["requests"],
            "failures": policy.stats["failures"],
            "parse_failures": policy.stats["parse_failures"],
            "images_sent": policy.stats["images_sent"],
            "last_latency_s": policy.stats["last_latency_s"],
        },
    )
    rec.check(
        "stats dict keys bounded/complete",
        set(new_stats()) <= set(policy.stats),
        measured={"stats_keys": sorted(policy.stats)},
    )
    rec.check(
        "event ring buffer is bounded",
        len(policy.recent_events()) <= 64,
        measured={"recent_events": len(policy.recent_events()), "buffer_max": 64},
    )


def scenario_camera_selection(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """相机子集 / 顺序 / 上限配置真的生效（用 sha256 证明发的是哪张图）。"""
    rec.begin("camera_selection")
    stub.set_script({"responses": [{"content": json.dumps({"choice": 0, "reason": "subset"})}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub, cameras=["top", "front"], max_images=2))
    policy.decide(scene.observation, scene.candidates)
    record = stub.requests()[0]
    want = [frame_sha256(scene.cameras["top"]), frame_sha256(scene.cameras["front"])]
    rec.check(
        "cameras=['top','front'] sends exactly those two, in order",
        record["image_count"] == 2 and record["image_sha256"] == want,
        f"n={record['image_count']} sha={[h[:10] for h in record['image_sha256']]}",
        image_count=record["image_count"],
        got_sha256=[h[:16] for h in record["image_sha256"]],
        want_sha256=[h[:16] for h in want],
    )
    rec.check(
        "images_sent counts only what was actually sent",
        policy.stats["images_sent"] == 2,
        measured={"images_sent": policy.stats["images_sent"]},
    )

    stub.set_script({"responses": [{"content": json.dumps({"choice": 1, "reason": "one image"})}]})
    stub.reset_records()
    policy_one = VLMHighLevel(base_high_config(stub, max_images=1))
    policy_one.decide(scene.observation, scene.candidates)
    record_one = stub.requests()[0]
    rec.check(
        "max_images=1 keeps only the first camera (wrist)",
        record_one["image_count"] == 1 and record_one["image_sha256"] == [frame_sha256(scene.cameras["wrist"])],
        measured={"image_count": record_one["image_count"], "sha256": record_one["image_sha256"][0][:16]},
    )
    stub.set_script({"responses": [{"content": json.dumps({"choice": 1, "reason": "no images"})}]})
    stub.reset_records()
    policy_zero = VLMHighLevel(base_high_config(stub, max_images=0))
    decision_zero = policy_zero.decide(scene.observation, scene.candidates)
    record_zero = stub.requests()[0]
    rec.check(
        "max_images=0 sends no image at all (text-only request)",
        record_zero["image_count"] == 0 and policy_zero.stats["images_sent"] == 0 and decision_zero.source == "vlm",
        measured={
            "image_count": record_zero["image_count"],
            "images_sent": policy_zero.stats["images_sent"],
            "source": decision_zero.source,
        },
    )


def scenario_json_robustness(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """围栏 JSON / 前后夹带散文的 JSON 都要能解析。"""
    rec.begin("json_robustness")
    fenced = 'Sure, here is my answer:\n```json\n{"choice": 1, "reason": "fenced reply"}\n```\nHope that helps.'
    stub.set_script({"responses": [{"content": fenced}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub))
    decision = policy.decide(scene.observation, scene.candidates)
    rec.check(
        "markdown-fenced JSON parses",
        decision.stone == scene.candidates[1].stone.name and decision.source == "vlm",
        f"stone={decision.stone} source={decision.source}",
        stone=decision.stone,
        source=decision.source,
        parse_failures=policy.stats["parse_failures"],
    )
    rec.check(
        "fenced reply needed no retry",
        policy.stats["parse_failures"] == 0 and policy.stats["retries"] == 0 and len(stub.requests()) == 1,
        measured={"parse_failures": policy.stats["parse_failures"], "retries": policy.stats["retries"], "requests": len(stub.requests())},
    )

    inline = 'Rocks look good. {"choice": 3, "reason": "inline json in prose"} -- done.'
    stub.set_script({"responses": [{"content": inline}]})
    stub.reset_records()
    policy_inline = VLMHighLevel(base_high_config(stub))
    decision_inline = policy_inline.decide(scene.observation, scene.candidates)
    rec.check(
        "JSON embedded in prose parses (brace scan)",
        decision_inline.stone == scene.candidates[3].stone.name and policy_inline.stats["parse_failures"] == 0,
        measured={"stone": decision_inline.stone, "parse_failures": policy_inline.stats["parse_failures"]},
    )

    nested = 'prefix {"choice": 2, "reason": "brace in {string} ok", "meta": {"k": [1,2,{"z":3}]}} suffix'
    stub.set_script({"responses": [{"content": nested}]})
    stub.reset_records()
    policy_nested = VLMHighLevel(base_high_config(stub))
    decision_nested = policy_nested.decide(scene.observation, scene.candidates)
    rec.check(
        "nested braces/strings inside the JSON do not break extraction",
        decision_nested.stone == scene.candidates[2].stone.name,
        measured={"stone": decision_nested.stone, "parse_failures": policy_nested.stats["parse_failures"]},
    )
    rec.check(
        "extract_json_object handles a raw fenced block",
        extract_json_object('```\n{"choice": 0}\n```')["choice"] == 0,
        measured={"parsed": extract_json_object('```\n{"choice": 0}\n```')},
    )
    raised = False
    try:
        extract_json_object("no json at all")
    except ValueError:
        raised = True
    rec.check("extract_json_object raises ValueError on garbage", raised, measured={"raised": raised})


def scenario_garbage_repair_fallback(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """垃圾 -> 修复 prompt 重试 -> 仍垃圾 -> 确定性回退（取 score 最大者）。"""
    rec.begin("garbage_repair_fallback")
    stub.set_script({"responses": [{"content": "I only see rocks, no JSON."}, {"content": "still not json <<<>>>"}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub))
    decision = policy.decide(scene.observation, scene.candidates)
    records = stub.requests()
    best = scene.candidates[scene.argmax_index]

    rec.check(
        "two requests: original + repair prompt",
        len(records) == 2,
        f"n={len(records)}",
        n_requests=len(records),
        image_counts=[r["image_count"] for r in records],
    )
    rec.check(
        "repair turn adds no new image parts (history still carries the 3 frames)",
        len(records) == 2
        and records[0]["images_per_message"] == [0, 3]
        and records[1]["roles"] == ["system", "user", "assistant", "user"]
        and records[1]["images_per_message"] == [0, 3, 0, 0],
        measured={
            "first_images_per_message": records[0]["images_per_message"] if records else None,
            "repair_roles": records[1]["roles"] if len(records) > 1 else None,
            "repair_images_per_message": records[1]["images_per_message"] if len(records) > 1 else None,
            "repair_total_images": records[1]["image_count"] if len(records) > 1 else None,
        },
    )
    rec.check(
        "repair prompt restates the strict JSON contract",
        len(records) > 1 and "STRICT JSON" in records[1]["prompt_text"] and "rejected" in records[1]["prompt_text"],
        measured={"repair_prompt_tail": records[1]["prompt_text"][-220:] if len(records) > 1 else ""},
    )
    rec.check(
        "parse_failures counts both unparseable replies",
        policy.stats["parse_failures"] == 2,
        f"parse_failures={policy.stats['parse_failures']}",
        parse_failures=policy.stats["parse_failures"],
    )
    rec.check(
        "retry counter records exactly one repair attempt",
        policy.stats["retries"] == 1,
        measured={"retries": policy.stats["retries"]},
    )
    rec.check(
        "deterministic fallback picks the highest-score candidate",
        decision.source == "fallback:argmax_score" and decision.stone == best.stone.name,
        f"source={decision.source} stone={decision.stone}",
        source=decision.source,
        stone=decision.stone,
        argmax_index=scene.argmax_index,
        argmax_score=float(best.score),
    )
    rec.check(
        "fallback decision carries the argmax slot geometry",
        np.allclose(decision.target_pos, best.slot.target_pos) and np.allclose(decision.target_quat, best.slot.target_quat),
        measured={"target_pos": decision.target_pos, "slot": list(decision.slot or ())},
    )
    rec.check(
        "fallback reason explains the failure and the rule",
        "highest score" in decision.reason and "json_parse_error" in decision.reason,
        measured={"reason": decision.reason[:220]},
    )
    rec.check(
        "stats mark the fallback and keep last_error",
        policy.stats["fallbacks"] == 1 and policy.stats["failures"] == 0 and bool(policy.stats["last_error"]),
        measured={"fallbacks": policy.stats["fallbacks"], "failures": policy.stats["failures"], "last_error": policy.stats["last_error"][:160]},
    )


def scenario_repair_success(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """第一次垃圾、修复轮给出合法 JSON -> 用修复轮结果，source='vlm_repair'。"""
    rec.begin("repair_success")
    stub.set_script(
        {
            "responses": [
                {"content": "not json at all"},
                {"content": '```json\n{"choice": 3, "reason": "fixed on retry"}\n```'},
            ]
        }
    )
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub))
    decision = policy.decide(scene.observation, scene.candidates)
    rec.check(
        "repair reply is used and marked as such",
        decision.source == "vlm_repair" and decision.stone == scene.candidates[3].stone.name,
        f"source={decision.source} stone={decision.stone}",
        source=decision.source,
        stone=decision.stone,
    )
    rec.check(
        "exactly one parse failure, one retry, two requests",
        policy.stats["parse_failures"] == 1 and policy.stats["retries"] == 1 and len(stub.requests()) == 2,
        measured={"parse_failures": policy.stats["parse_failures"], "retries": policy.stats["retries"], "requests": len(stub.requests())},
    )


def scenario_http_error(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """HTTP 500：重试一次后回退，绝不抛异常。"""
    rec.begin("http_500")
    stub.set_script({"responses": [{"status": 500}, {"status": 500}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub))
    raised = ""
    decision: Decision | None = None
    try:
        decision = policy.decide(scene.observation, scene.candidates)
    except Exception as error:  # noqa: BLE001 - 这里就是要证明它不抛
        raised = f"{type(error).__name__}: {error}"
    records = stub.requests()
    rec.check("HTTP 500 does not raise (strict=False)", raised == "", f"raised={raised}", raised=raised or None)
    rec.check(
        "HTTP 500 falls back to argmax candidate",
        decision is not None and decision.source == "fallback:argmax_score" and decision.stone == scene.candidates[scene.argmax_index].stone.name,
        f"decision={decision}",
        source=decision.source if decision else None,
        stone=decision.stone if decision else None,
    )
    rec.check(
        "error counters reflect two failed attempts",
        policy.stats["failures"] == 2 and policy.stats["http_errors"] == 2 and policy.stats["retries"] == 1,
        measured={
            "failures": policy.stats["failures"],
            "http_errors": policy.stats["http_errors"],
            "retries": policy.stats["retries"],
            "requests": policy.stats["requests"],
        },
    )
    rec.check(
        "stub recorded both 500 responses",
        len(records) == 2 and all(r["response_status"] == 500 for r in records),
        measured={"statuses": [r["response_status"] for r in records], "error_detail": policy.stats["last_error"][:120]},
    )


def scenario_timeout(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """慢响应触发客户端超时：耗时被 timeout_s 截断，随后回退，不抛出。"""
    rec.begin("timeout")
    stub.set_script({"responses": [{"delay_s": 2.5, "content": '{"choice": 0}'}, {"delay_s": 2.5, "content": '{"choice": 0}'}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub, timeout_s=0.6, retries=1))
    started = time.perf_counter()
    raised = ""
    decision: Decision | None = None
    try:
        decision = policy.decide(scene.observation, scene.candidates)
    except Exception as error:  # noqa: BLE001
        raised = f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started
    records = stub.requests()

    rec.check("slow server does not raise", raised == "", f"raised={raised}", raised=raised or None)
    rec.check(
        "timeout_s actually interrupts the call (2 attempts x 0.6s, far below 2.5s delay)",
        1.0 <= elapsed < 2.4,
        f"elapsed={elapsed:.3f}s",
        elapsed_s=round(elapsed, 3),
        configured_timeout_s=0.6,
        stub_delay_s=2.5,
    )
    rec.check(
        "timeout counters and fallback",
        policy.stats["timeouts"] == 2 and policy.stats["failures"] == 2 and policy.stats["fallbacks"] == 1,
        measured={
            "timeouts": policy.stats["timeouts"],
            "failures": policy.stats["failures"],
            "fallbacks": policy.stats["fallbacks"],
            "latency_max_s": policy.stats["latency_max_s"],
            "decision_source": decision.source if decision else None,
        },
    )
    rec.check(
        "stub still recorded the (abandoned) slow requests",
        len(records) == 2 and all(r["response_kind"] == "slow" for r in records),
        measured={"n": len(records), "kinds": [r["response_kind"] for r in records], "delays": [r["response_delay_s"] for r in records]},
    )
    rec.check(
        "measured latency is bounded by the timeout, not the server delay",
        policy.stats["latency_max_s"] < 1.5,
        measured={"latency_max_s": policy.stats["latency_max_s"], "stub_delay_s": 2.5},
    )
    rec.check(
        "decision still produced after timeout",
        decision is not None and decision.action == "place" and decision.source.startswith("fallback"),
        measured={"action": decision.action if decision else None, "source": decision.source if decision else None},
    )


def scenario_stop_and_bounds(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """choice=-1 停机路径；choice 越界的边界检查。"""
    rec.begin("stop_and_bounds")
    stub.set_script({"responses": [{"content": json.dumps({"choice": -1, "reason": "wall complete, nothing left"})}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub))
    decision = policy.decide(scene.observation, scene.candidates)
    rec.check(
        "choice=-1 maps to a stop Decision",
        decision.is_stop and decision.action == "stop" and decision.stone is None,
        f"action={decision.action}",
        action=decision.action,
        stone=decision.stone,
        slot=decision.slot,
    )
    rec.check(
        "stop is counted and reason is preserved",
        policy.stats["stops"] == 1 and "wall complete" in decision.reason,
        measured={"stops": policy.stats["stops"], "reason": decision.reason[:180]},
    )

    stub.set_script({"responses": [{"content": '{"choice": 99, "reason": "hallucinated index"}'}, {"content": '{"choice": 42}'}]})
    stub.reset_records()
    policy_oob = VLMHighLevel(base_high_config(stub))
    decision_oob = policy_oob.decide(scene.observation, scene.candidates)
    rec.check(
        "out-of-range indices are rejected twice then fall back",
        policy_oob.stats["out_of_range"] == 2
        and policy_oob.stats["parse_failures"] == 0
        and decision_oob.source == "fallback:argmax_score",
        measured={
            "out_of_range": policy_oob.stats["out_of_range"],
            "parse_failures": policy_oob.stats["parse_failures"],
            "source": decision_oob.source,
            "last_error": policy_oob.stats["last_error"][:120],
        },
    )
    rec.check(
        "out-of-range reason is recorded in the decision",
        "choice_out_of_range" in decision_oob.reason,
        measured={"reason": decision_oob.reason[:200]},
    )

    stub.set_script({"responses": [{"content": '{"reason": "forgot the choice field"}'}, {"content": '{"reason": "still forgot"}'}]})
    stub.reset_records()
    policy_missing = VLMHighLevel(base_high_config(stub))
    decision_missing = policy_missing.decide(scene.observation, scene.candidates)
    rec.check(
        "missing 'choice' field counts as a parse failure and falls back",
        policy_missing.stats["parse_failures"] == 2 and decision_missing.source == "fallback:argmax_score",
        measured={"parse_failures": policy_missing.stats["parse_failures"], "source": decision_missing.source},
    )

    rec.begin("empty_candidates")
    stub.set_script({"responses": [{"content": '{"choice": 0}'}]})
    stub.reset_records()
    policy_empty = VLMHighLevel(base_high_config(stub))
    decision_empty = policy_empty.decide(scene.observation, [])
    rec.check(
        "empty candidate list stops without any HTTP request",
        decision_empty.is_stop and len(stub.requests()) == 0 and policy_empty.stats["requests"] == 0,
        measured={"action": decision_empty.action, "requests": policy_empty.stats["requests"], "reason": decision_empty.reason},
    )


def scenario_privileged_state(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """use_privileged_state=False 时，石块位姿真值绝不能出现在 prompt 里。"""
    rec.begin("privileged_state")
    stub.set_script({"responses": [{"content": '{"choice": 1, "reason": "images only"}'}]})
    stub.reset_records()
    policy_off = VLMHighLevel(base_high_config(stub, use_privileged_state=False))
    policy_off.decide(scene.observation, scene.candidates)
    prompt_off = stub.requests()[0]["prompt_text"]

    stone_markers = [vec_text(stone.pos) for stone in scene.stones]
    quat_markers = [vec_text(stone.quat) for stone in scene.stones]
    rec.check(
        "privileged=False: no stone pose markers in the prompt",
        "stone_xyz=" not in prompt_off and "stone_quat=" not in prompt_off,
        measured={"has_stone_xyz": "stone_xyz=" in prompt_off, "has_stone_quat": "stone_quat=" in prompt_off},
    )
    rec.check(
        "privileged=False: no stone pose numbers leak into the prompt",
        not any(marker in prompt_off for marker in stone_markers) and not any(marker in prompt_off for marker in quat_markers),
        measured={"leaked_pos_markers": [m for m in stone_markers if m in prompt_off], "leaked_quat_markers": [m for m in quat_markers if m in prompt_off]},
    )
    rec.check(
        "privileged=False: candidate text list is still present",
        all(f"[{index}]" in prompt_off for index in range(len(scene.candidates)))
        and scene.candidates[0].stone.name in prompt_off,
        measured={"prompt_chars": len(prompt_off), "candidate_lines": sum(1 for i in range(4) if f"[{i}]" in prompt_off)},
    )
    rec.check(
        "privileged=False: joints/truth info dict stay out of the prompt",
        "shoulder_pan_joint" not in prompt_off and "privileged_note" not in prompt_off,
        measured={"has_joint_name": "shoulder_pan_joint" in prompt_off, "has_info_key": "privileged_note" in prompt_off},
    )

    stub.set_script({"responses": [{"content": '{"choice": 1, "reason": "with truth"}'}]})
    stub.reset_records()
    policy_on = VLMHighLevel(base_high_config(stub, use_privileged_state=True))
    policy_on.decide(scene.observation, scene.candidates)
    prompt_on = stub.requests()[0]["prompt_text"]
    rec.check(
        "privileged=True: stone poses are added for each candidate",
        "stone_xyz=" in prompt_on and all(marker in prompt_on for marker in stone_markers),
        measured={
            "has_stone_xyz": "stone_xyz=" in prompt_on,
            "n_stone_markers_found": sum(1 for marker in stone_markers if marker in prompt_on),
            "n_candidates": len(scene.candidates),
        },
    )
    rec.check(
        "privileged=True prompt is strictly longer than the image-only prompt",
        len(prompt_on) > len(prompt_off),
        measured={"prompt_chars_privileged": len(prompt_on), "prompt_chars_image_only": len(prompt_off)},
    )


def scenario_custom_fallback_and_strict(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """调用方给的 fallback 策略要真的被用上；strict=True 时才允许抛。"""
    rec.begin("custom_fallback")
    stub.set_script({"responses": [{"content": "garbage one"}, {"content": "garbage two"}]})
    stub.reset_records()
    fallback = ScriptedHighLevel(index=3, name="scripted_high")
    policy = VLMHighLevel(base_high_config(stub), fallback=fallback)
    context = {"instruction": "override me", "episode": 7}
    policy.reset(context)
    decision = policy.decide(scene.observation, scene.candidates)
    rec.check(
        "caller-supplied fallback policy decides when the VLM fails",
        fallback.calls == 1 and decision.stone == scene.candidates[3].stone.name,
        f"calls={fallback.calls} stone={decision.stone}",
        fallback_calls=fallback.calls,
        stone=decision.stone,
    )
    rec.check(
        "decision.source records which fallback answered",
        decision.source == "fallback:scripted_high",
        measured={"source": decision.source, "reason": decision.reason[:200]},
    )
    rec.check(
        "reset() propagates to the fallback policy",
        getattr(fallback, "reset_context", None) == context,
        measured={"fallback_reset_context_keys": sorted(getattr(fallback, "reset_context", {}) or {})},
    )

    rec.begin("strict_mode")
    stub.set_script({"responses": [{"content": "garbage"}, {"content": "garbage"}]})
    stub.reset_records()
    strict_policy = VLMHighLevel(base_high_config(stub, strict=True))
    raised = ""
    try:
        strict_policy.decide(scene.observation, scene.candidates)
    except VLMError as error:
        raised = str(error)
    rec.check(
        "strict=True raises VLMError instead of falling back",
        raised.startswith("VLMHighLevel strict"),
        f"raised={raised[:160]}",
        raised=raised[:200],
    )

    non_strict = VLMHighLevel(base_high_config(stub))
    raised_soft = ""
    try:
        soft_decision = non_strict.decide(scene.observation, scene.candidates)
    except Exception as error:  # noqa: BLE001 - 这里就是要证明它不抛
        raised_soft = f"{type(error).__name__}: {error}"
        soft_decision = None
    rec.check(
        "strict=False never raises out of decide()",
        raised_soft == "" and soft_decision is not None,
        f"raised={raised_soft}",
        raised=raised_soft or None,
        source=soft_decision.source if soft_decision else None,
    )


def scenario_low_level_clamp(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """低层增量夹紧：越界增量必须落到夹紧后的位姿，并计数。"""
    rec.begin("low_level_clamp")
    stub.set_script(
        {
            "responses": [
                {
                    "content": json.dumps(
                        {
                            "delta_xyz": [1.5, -0.9, 0.7],
                            "delta_yaw_deg": 95.0,
                            "grip_width": 0.05,
                            "advance": False,
                            "reason": "grossly out of range on purpose",
                        }
                    )
                }
            ]
        }
    )
    stub.reset_records()
    policy = VLMLowLevel(base_low_config(stub, max_delta_m=0.03, max_yaw_deg=8.0))
    goal = scene.place_goal()
    command = policy.step(scene.observation, goal)

    want_pos = np.asarray(goal.place_pos, dtype=float) + np.array([0.03, -0.03, 0.03])
    want_rot = yaw_matrix(8.0) @ np.asarray(goal.place_rot, dtype=float)
    rec.check(
        "clamped tcp_pos == nominal + per-axis clamped delta",
        bool(np.allclose(command.tcp_pos, want_pos, atol=1e-9)),
        f"got={command.tcp_pos} want={want_pos}",
        got_tcp_pos=command.tcp_pos,
        want_tcp_pos=want_pos,
        nominal_pos=goal.place_pos,
        requested_delta=[1.5, -0.9, 0.7],
        max_delta_m=0.03,
    )
    rec.check(
        "clamped yaw == +max_yaw_deg about world +Z",
        bool(np.allclose(command.tcp_rot, want_rot, atol=1e-9)),
        measured={
            "requested_yaw_deg": 95.0,
            "applied_yaw_deg": 8.0,
            "rot_det": float(np.linalg.det(command.tcp_rot)),
            "max_abs_rot_err": float(np.max(np.abs(command.tcp_rot - want_rot))),
        },
    )
    rec.check(
        "advance=False keeps done=False",
        command.done is False,
        measured={"done": command.done},
    )
    rec.check(
        "in-range grip_width is passed through unclamped",
        command.grip_width is not None and abs(command.grip_width - 0.05) < 1e-12,
        measured={"grip_width": command.grip_width, "clipped_grip": policy.stats["clipped_grip"]},
    )
    rec.check(
        "clip counters: 1 clipped step, 3 clipped axes, 1 clipped yaw",
        policy.stats["clipped"] == 1 and policy.stats["clipped_xyz"] == 3 and policy.stats["clipped_yaw"] == 1,
        measured={
            "clipped": policy.stats["clipped"],
            "clipped_xyz": policy.stats["clipped_xyz"],
            "clipped_yaw": policy.stats["clipped_yaw"],
            "clipped_grip": policy.stats["clipped_grip"],
        },
    )
    rec.check(
        "command note records both requested and applied deltas",
        "requested_xyz" in command.note and "delta_xyz" in command.note,
        measured={"note": command.note[:220], "source": command.source},
    )
    prompt = stub.requests()[0]["prompt_text"]
    rec.check(
        "low-level prompt carries tcp pose, nominal pose, phase, gripper, instruction, images",
        vec_text(scene.observation.tcp_pos) in prompt
        and vec_text(goal.place_pos) in prompt
        and "phase=place" in prompt
        and f"{scene.observation.gripper_width:.4f}" in prompt
        and INSTRUCTION in prompt
        and stub.requests()[0]["image_count"] == 3,
        measured={
            "image_count": stub.requests()[0]["image_count"],
            "has_tcp": vec_text(scene.observation.tcp_pos) in prompt,
            "has_nominal": vec_text(goal.place_pos) in prompt,
            "prompt_chars": len(prompt),
        },
    )

    # 未越界：不计数、位姿精确
    stub.set_script(
        {
            "responses": [
                {
                    "content": json.dumps(
                        {"delta_xyz": [0.01, -0.005, 0.0], "delta_yaw_deg": 2.5, "grip_width": 0.031, "advance": True}
                    )
                }
            ]
        }
    )
    stub.reset_records()
    policy_ok = VLMLowLevel(base_low_config(stub, max_delta_m=0.03, max_yaw_deg=8.0))
    command_ok = policy_ok.step(scene.observation, goal)
    want_pos_ok = np.asarray(goal.place_pos, dtype=float) + np.array([0.01, -0.005, 0.0])
    want_rot_ok = yaw_matrix(2.5) @ np.asarray(goal.place_rot, dtype=float)
    rec.check(
        "in-range delta applied exactly, nothing clipped",
        bool(np.allclose(command_ok.tcp_pos, want_pos_ok, atol=1e-12))
        and bool(np.allclose(command_ok.tcp_rot, want_rot_ok, atol=1e-12))
        and policy_ok.stats["clipped"] == 0,
        measured={
            "tcp_pos": command_ok.tcp_pos,
            "max_pos_err": float(np.max(np.abs(command_ok.tcp_pos - want_pos_ok))),
            "clipped": policy_ok.stats["clipped"],
        },
    )
    rec.check(
        "advance=True maps to done=True and grip_width is applied",
        command_ok.done is True and abs(float(command_ok.grip_width) - 0.031) < 1e-12,
        measured={"done": command_ok.done, "grip_width": command_ok.grip_width},
    )

    # grip_width=null -> 沿用标称开口；grip 越界 -> 夹到上限并计 clipped_grip
    stub.set_script(
        {
            "responses": [
                {"content": json.dumps({"delta_xyz": [0, 0, 0], "delta_yaw_deg": 0, "grip_width": None, "advance": True})},
                {"content": json.dumps({"delta_xyz": [0, 0, 0], "delta_yaw_deg": 0, "grip_width": 9.9, "advance": True})},
            ]
        }
    )
    stub.reset_records()
    policy_grip = VLMLowLevel(base_low_config(stub))
    command_null = policy_grip.step(scene.observation, goal)
    command_big = policy_grip.step(scene.observation, goal)
    rec.check(
        "grip_width=null keeps the nominal opening",
        command_null.grip_width is not None and abs(float(command_null.grip_width) - float(goal.grip_width)) < 1e-12,
        measured={"grip_width": command_null.grip_width, "goal_grip_width": goal.grip_width},
    )
    rec.check(
        "out-of-range grip_width is clamped to max_grip_width_m and counted",
        abs(float(command_big.grip_width) - 0.14) < 1e-12 and policy_grip.stats["clipped_grip"] == 1,
        measured={"grip_width": command_big.grip_width, "clipped_grip": policy_grip.stats["clipped_grip"]},
    )


def scenario_low_level_delegation(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """非 VLM 目标、缺标称位姿、网络失败都要落到 fallback（或无 fallback 时的保持位姿）。"""
    rec.begin("low_level_delegation")
    stub.set_script({"responses": [{"content": "garbage"}]})
    stub.reset_records()
    spy = SpyLowLevel()
    policy = VLMLowLevel(base_low_config(stub, vlm_goals=["place"]), fallback=spy)
    pick_goal = scene.place_goal(kind="pick")
    command = policy.step(scene.observation, pick_goal)
    rec.check(
        "goal kind outside vlm_goals is delegated to the fallback",
        spy.calls == ["pick"] and command.source == "spy_fallback",
        f"calls={spy.calls} source={command.source}",
        fallback_calls=list(spy.calls),
        source=command.source,
    )
    rec.check(
        "delegated goal issues no HTTP request",
        len(stub.requests()) == 0 and policy.stats["requests"] == 0,
        measured={"stub_requests": len(stub.requests()), "policy_requests": policy.stats["requests"], "fallbacks": policy.stats["fallbacks"]},
    )

    stub.reset_records()
    policy_hold = VLMLowLevel(base_low_config(stub, vlm_goals=["place"]))
    command_hold = policy_hold.step(scene.observation, pick_goal)
    rec.check(
        "no fallback -> minimal hold command with done=False",
        command_hold.done is False
        and command_hold.source == "hold:no_fallback"
        and command_hold.grip_width is None
        and bool(np.allclose(command_hold.tcp_pos, scene.observation.tcp_pos))
        and bool(np.allclose(command_hold.tcp_rot, scene.observation.tcp_rot)),
        measured={
            "done": command_hold.done,
            "source": command_hold.source,
            "tcp_pos": command_hold.tcp_pos,
            "current_tcp_pos": scene.observation.tcp_pos,
        },
    )
    rec.check(
        "hold path issues no HTTP request either",
        len(stub.requests()) == 0 and policy_hold.stats["fallbacks"] == 1,
        measured={"stub_requests": len(stub.requests()), "fallbacks": policy_hold.stats["fallbacks"]},
    )

    # place 目标但没有任何标称位姿 -> 不猜，直接回退
    policy_nominal = VLMLowLevel(base_low_config(stub, vlm_goals=["place"]))
    goal_without_pose = scene.place_goal(place_pos=None, place_rot=None)
    command_no_nominal = policy_nominal.step(scene.observation, goal_without_pose)
    rec.check(
        "place goal without a nominal pose delegates instead of guessing",
        command_no_nominal.done is False and "no_nominal_pose" in command_no_nominal.note,
        measured={"note": command_no_nominal.note, "requests": policy_nominal.stats["requests"]},
    )

    # 网络失败（500）走低层回退
    stub.set_script({"responses": [{"status": 503}, {"status": 503}]})
    stub.reset_records()
    spy_fail = SpyLowLevel(name="spy_low_fail")
    policy_fail = VLMLowLevel(base_low_config(stub), fallback=spy_fail)
    command_fail = policy_fail.step(scene.observation, scene.place_goal())
    rec.check(
        "low-level HTTP failure falls back without raising",
        command_fail.source == "spy_fallback"
        and spy_fail.calls == ["place"]
        and policy_fail.stats["http_errors"] == 2
        and policy_fail.stats["retries"] == 1,
        measured={
            "source": command_fail.source,
            "fallback_calls": list(spy_fail.calls),
            "http_errors": policy_fail.stats["http_errors"],
            "retries": policy_fail.stats["retries"],
        },
    )

    # 低层 strict=True 时才抛
    stub.set_script({"responses": [{"content": "garbage"}, {"content": "garbage"}]})
    stub.reset_records()
    policy_strict = VLMLowLevel(base_low_config(stub, strict=True))
    raised = ""
    try:
        policy_strict.step(scene.observation, scene.place_goal())
    except VLMError as error:
        raised = str(error)
    rec.check(
        "low-level strict=True raises VLMError",
        raised.startswith("VLMLowLevel strict"),
        measured={"raised": raised[:180]},
    )


def scenario_nominal_memory_and_reset(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """retreat 缺位姿时复用本回合记住的 place 标称；reset() 清掉这份记忆与指令覆盖。"""
    rec.begin("nominal_memory")
    stub.set_script({"responses": [{"content": json.dumps({"delta_xyz": [0, 0, 0], "delta_yaw_deg": 0, "grip_width": None, "advance": True})}] * 4})
    stub.reset_records()
    policy = VLMLowLevel(base_low_config(stub, vlm_goals=["place", "retreat"]))
    policy.step(scene.observation, scene.place_goal())
    # retreat 目标不带任何位姿（执行器没给），只能靠本回合记住的 place 标称位姿
    retreat_goal = scene.place_goal(kind="retreat", place_pos=None, place_rot=None, pick_pos=None, pick_rot=None)
    command = policy.step(scene.observation, retreat_goal)
    prompt = stub.requests()[-1]["prompt_text"]
    rec.check(
        "retreat reuses the remembered place nominal pose",
        "nominal_source=memory.place" in prompt and bool(np.allclose(command.tcp_pos, scene.nominal_pos)),
        measured={
            "has_memory_source": "nominal_source=memory.place" in prompt,
            "tcp_pos": command.tcp_pos,
            "nominal_pos": scene.nominal_pos,
            "requests": len(stub.requests()),
        },
    )

    policy.reset({"instruction": "RETREAT-ONLY-OVERRIDE"})
    stub.reset_records()
    command_after_reset = policy.step(scene.observation, retreat_goal)
    rec.check(
        "reset() clears the nominal memory (retreat delegates again)",
        command_after_reset.done is False and "no_nominal_pose" in command_after_reset.note,
        measured={"note": command_after_reset.note, "source": command_after_reset.source},
    )


def scenario_instruction_override(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """reset(context) 的 instruction 覆盖观测里的指令（两层都要生效）。"""
    rec.begin("instruction_override")
    override = "OVERRIDE-INSTRUCTION-42: place only on course 0"
    stub.set_script({"responses": [{"content": json.dumps({"choice": 0, "reason": "override"})}]})
    stub.reset_records()
    policy = VLMHighLevel(base_high_config(stub))
    policy.reset({"instruction": override})
    policy.decide(scene.observation, scene.candidates)
    prompt = stub.requests()[0]["prompt_text"]
    rec.check(
        "high level: context instruction wins over observation.instruction",
        override in prompt and INSTRUCTION not in prompt,
        measured={"has_override": override in prompt, "has_observation_instruction": INSTRUCTION in prompt},
    )

    stub.set_script({"responses": [{"content": json.dumps({"delta_xyz": [0, 0, 0], "delta_yaw_deg": 0, "grip_width": None, "advance": True})}]})
    stub.reset_records()
    low = VLMLowLevel(base_low_config(stub))
    low.reset({"instruction": override})
    low.step(scene.observation, scene.place_goal())
    prompt_low = stub.requests()[0]["prompt_text"]
    rec.check(
        "low level: context instruction wins over observation.instruction",
        override in prompt_low and INSTRUCTION not in prompt_low,
        measured={"has_override": override in prompt_low, "has_observation_instruction": INSTRUCTION in prompt_low},
    )

    stub.set_script({"responses": [{"content": json.dumps({"choice": 0, "reason": "no override"})}]})
    stub.reset_records()
    policy_plain = VLMHighLevel(base_high_config(stub))
    policy_plain.reset({})
    policy_plain.decide(scene.observation, scene.candidates)
    prompt_plain = stub.requests()[0]["prompt_text"]
    rec.check(
        "without context override the observation instruction is used",
        INSTRUCTION in prompt_plain,
        measured={"has_observation_instruction": INSTRUCTION in prompt_plain},
    )


def scenario_fallback_low_level_wrapper(rec: Recorder, stub: StubHarness, scene: Scene) -> None:
    """base.FallbackLowLevel 必须能直接包住 VLMLowLevel（primary），且命令合法。"""
    rec.begin("fallback_low_level_wrapper")
    stub.set_script({"responses": [{"content": "garbage"}, {"content": "garbage"}]})
    stub.reset_records()
    primary = VLMLowLevel(base_low_config(stub))
    backup = SpyLowLevel(name="backup_low")
    wrapped = FallbackLowLevel(primary=primary, backup=backup)
    wrapped.reset({})
    command = wrapped.step(scene.observation, scene.place_goal())
    rec.check(
        "VLMLowLevel works as FallbackLowLevel.primary (valid Command, no wrapper failure)",
        wrapped.failures == 0 and command.source in {"hold:no_fallback", "vlm"},
        f"failures={wrapped.failures} source={command.source}",
        wrapper_failures=wrapped.failures,
        source=command.source,
        done=command.done,
        backup_calls=list(backup.calls),
    )
    rec.check(
        "internal self-heal keeps the wrapper's backup unused",
        backup.calls == [] and primary.stats["fallbacks"] == 1,
        measured={"backup_calls": list(backup.calls), "primary_fallbacks": primary.stats["fallbacks"]},
    )


def scenario_real_server(rec: Recorder, scene: Scene, args: argparse.Namespace) -> dict[str, Any]:
    """可选：对一个真实 OpenAI 兼容服务发一次请求；连不上就 SKIP。"""
    rec.begin("real_server")
    if not args.real_url:
        print(f"  [SKIP] real server not requested (pass --real-url to enable); would probe {DEFAULT_BASE_URL}")
        rec.check(
            "real server run is opt-in",
            True,
            measured={"requested": False, "default_would_be": DEFAULT_BASE_URL, "note": "no GPU on this machine"},
        )
        return {"status": "not_requested", "base_url": DEFAULT_BASE_URL}

    probe = check_server(args.real_url, model=args.real_model, timeout_s=args.real_timeout)
    if not probe.get("ok"):
        print(
            f"  [SKIP] real VLM server unreachable at {args.real_url} "
            f"({probe.get('error')}) -- skipping real-model check, this machine has no usable GPU"
        )
        rec.check(
            "real server unreachable -> skip cleanly",
            True,
            measured={"probe": probe, "verdict": "SKIP"},
        )
        return {"status": "skipped_unreachable", "probe": probe, "base_url": args.real_url}

    print(f"  [INFO] real server reachable: {probe}")
    config = {
        "base_url": args.real_url,
        "model": args.real_model,
        "timeout_s": args.real_timeout,
        "max_images": 1,
        "image_max_side": 512,
        "jpeg_quality": 80,
        "max_tokens": 128,
        "retries": 0,
    }
    policy = VLMHighLevel(config)
    outcome: dict[str, Any] = {"status": "ran", "probe": probe}
    raised = ""
    try:
        decision = policy.decide(scene.observation, scene.candidates)
    except Exception as error:  # noqa: BLE001 - 真实服务什么都可能返回
        raised = f"{type(error).__name__}: {error}"
        decision = None
    outcome.update(
        {
            "raised": raised,
            "decision_source": decision.source if decision else None,
            "decision_stone": decision.stone if decision else None,
            "decision_reason": decision.reason[:300] if decision else None,
            "stats": policy.stats_snapshot(),
        }
    )
    rec.check(
        "real server: one request completed without raising",
        raised == "" and decision is not None,
        f"raised={raised}",
        **{key: outcome[key] for key in ("decision_source", "decision_stone", "decision_reason")},
    )
    return outcome


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------


def print_summary(rec: Recorder) -> None:
    width = max((len(entry["name"]) for entry in rec.checks), default=10)
    print("")
    print("=" * 108)
    print(f"{'RESULT':<7} {'SECTION':<26} {'CHECK':<{width}}  MEASURED")
    print("-" * 108)
    for entry in rec.checks:
        measured = ", ".join(f"{key}={jsonable(value)}" for key, value in entry["measured"].items())
        if len(measured) > 96:
            measured = measured[:93] + "..."
        print(f"{entry['status'].upper():<7} {entry['section']:<26} {entry['name']:<{width}}  {measured}")
    print("=" * 108)
    sections: dict[str, list[int]] = {}
    for entry in rec.checks:
        bucket = sections.setdefault(entry["section"], [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if entry["status"] == "pass" else 0
    for name, (total, passed) in sections.items():
        print(f"  {name:<28} {passed}/{total} passed")
    print(f"  {'TOTAL':<28} {len(rec.checks) - len(rec.failures)}/{len(rec.checks)} passed")


def build_report(
    rec: Recorder,
    stub: StubHarness,
    scene: Scene,
    args: argparse.Namespace,
    started_at: float,
    real_result: Mapping[str, Any],
    scene_stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "report_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": round(time.time() - started_at, 3),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "cwd": os.getcwd(),
        "task_root": str(TASK_ROOT),
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "requests": requests.__version__,
            "platform": sys.platform,
            "cuda_available": _cuda_available(),
        },
        "stub_server": {
            **{key: stub.ready.get(key) for key in ("host", "port", "base_url", "command")},
            "exit_code": stub.exit_code,
            "stderr_tail": stub.stderr_text()[-400:],
        },
        "synthetic_scene": scene_stats,
        "candidates": [
            {
                "index": index,
                "describe": candidate.describe(),
                "score": float(candidate.score),
                "stone_pos": candidate.stone.pos,
                "slot_target": candidate.slot.target_pos,
            }
            for index, candidate in enumerate(scene.candidates)
        ],
        "summary": {
            "total": len(rec.checks),
            "passed": len(rec.checks) - len(rec.failures),
            "failed": len(rec.failures),
            "ok": not rec.failures,
            "failed_checks": [entry["name"] for entry in rec.failures],
        },
        "checks": rec.checks,
        "measurements": rec.measured_groups(),
        "real_server": jsonable(dict(real_result)),
        "options": {key: jsonable(value) for key, value in vars(args).items()},
    }


def _cuda_available() -> bool | str:
    try:
        import torch  # noqa: PLC0415 - 只在这个诊断字段里用，缺 torch 也不影响测试
    except Exception as error:  # noqa: BLE001
        return f"torch unavailable: {type(error).__name__}"
    return bool(torch.cuda.is_available())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLM 策略端到端测试（桩服务；无需 GPU）")
    parser.add_argument("--report", default=str(REPORT_PATH), help=f"报告 JSON 路径（默认 {REPORT_PATH}）")
    parser.add_argument("--real-url", default=None, help="真实 OpenAI 兼容服务地址，例如 http://127.0.0.1:3001/v1")
    parser.add_argument("--real-model", default=DEFAULT_MODEL, help="真实服务使用的模型名")
    parser.add_argument("--real-timeout", type=float, default=8.0, help="真实服务探测超时（秒）")
    parser.add_argument("--verbose", action="store_true", help="打印桩服务访问日志")
    parser.add_argument("--keep-tmp", action="store_true", help="保留临时目录（调试用）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.time()
    rec = Recorder()
    workdir = Path(tempfile.mkdtemp(prefix="vlm_policy_test_"))
    stub = StubHarness(sys.executable, workdir, verbose=args.verbose)
    scene = Scene()
    scene_stats = scene.scene_stats()
    real_result: dict[str, Any] = {"status": "not_run"}

    print("VLM policy end-to-end test")
    print(f"  python      : {sys.executable}")
    print(f"  cwd         : {os.getcwd()}")
    print(f"  workdir     : {workdir}")
    print(f"  scene       : " + ", ".join(f"{name}={info['rgb_shape']} jpeg={info['jpeg_bytes']}B" for name, info in scene_stats.items()))
    print("")

    try:
        stub.start({"responses": [{"content": json.dumps({"choice": 2, "reason": "initial script"})}]})
        print(f"  stub server : {stub.base_url} (pid={stub.ready.get('pid')}, ephemeral port)")
        print("")
        scenario_engine_agnostic(rec)
        scenario_stub_and_transport(rec, stub, scene)
        scenario_stub_script_modes(rec, stub)
        scenario_camera_selection(rec, stub, scene)
        scenario_json_robustness(rec, stub, scene)
        scenario_garbage_repair_fallback(rec, stub, scene)
        scenario_repair_success(rec, stub, scene)
        scenario_http_error(rec, stub, scene)
        scenario_timeout(rec, stub, scene)
        scenario_stop_and_bounds(rec, stub, scene)
        scenario_privileged_state(rec, stub, scene)
        scenario_custom_fallback_and_strict(rec, stub, scene)
        scenario_low_level_clamp(rec, stub, scene)
        scenario_low_level_delegation(rec, stub, scene)
        scenario_nominal_memory_and_reset(rec, stub, scene)
        scenario_instruction_override(rec, stub, scene)
        scenario_fallback_low_level_wrapper(rec, stub, scene)
        real_result = scenario_real_server(rec, scene, args)
    except Exception as error:  # noqa: BLE001 - 任何未预期异常都要进报告而不是只有栈
        rec.check(
            "harness completed without crashing",
            False,
            f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc()[-1200:],
        )
    finally:
        exit_code = stub.stop()
        rec.check(
            "stub server terminates cleanly",
            exit_code == 0,
            f"exit_code={exit_code}",
            exit_code=exit_code,
            stderr_tail=stub.stderr_text()[-200:],
        )

    report = build_report(rec, stub, scene, args, started_at, real_result, scene_stats)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(rec)
    print("")
    # 报告本身也要自证：重新读回来核对条数与通过数（这一步失败同样让退出码非 0）
    verify_error = ""
    try:
        reloaded = json.loads(report_path.read_text(encoding="utf-8"))
        if reloaded["summary"]["total"] != len(rec.checks) or reloaded["summary"]["failed"] != len(rec.failures):
            verify_error = f"summary mismatch: {reloaded['summary']}"
    except Exception as error:  # noqa: BLE001 - 报告读不回来就该失败
        verify_error = f"{type(error).__name__}: {error}"
    if verify_error:
        print(f"REPORT VERIFY FAILED: {verify_error}")
    else:
        print(
            f"report: {report_path} "
            f"(verified: {reloaded['summary']['passed']}/{reloaded['summary']['total']} checks, "
            f"{report_path.stat().st_size} bytes)"
        )
    print(f"exit  : {'0 (all checks passed)' if not rec.failures and not verify_error else '1 (failures present)'}")

    if not args.keep_tmp:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if not rec.failures and not verify_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
