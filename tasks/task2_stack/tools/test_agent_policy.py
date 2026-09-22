#!/usr/bin/env python3
"""`qwen-agent`（Codex + QwenVL 高层）的离线测试：不需要 GPU、vLLM 或真 Codex。

用一个 **假 codex 脚本**替换 `codex exec`：它接受相同的命令行，把预设的
`last_message` 写进 `-o` 指定的文件。于是可以离线验证：

1. `build_codex_command` 拼出的配置里确实带自定义 provider / base_url / wire_api；
2. 合法 JSON -> 正确的 `Decision`；
3. `choice=-1` / `"stop"` -> 停机；
4. 越界或垃圾输出 -> 修复重试 -> 回退脚本高层 -> 最终回退 argmax(score)；
5. 子进程退出码非 0 -> 回退；
6. 图像确实以文件形式传给 `codex exec -i`（张数、JPEG 魔数）。

    python tools/test_agent_policy.py
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

import numpy as np  # noqa: E402

from stone_stack.policy.agent import CodexAgentHighLevel, build_codex_command  # noqa: E402
from stone_stack.policy.base import CameraFrame, Candidate, Observation, PlanSlot, StoneState  # noqa: E402

CHECKS = {"passed": 0, "failed": 0, "details": []}


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        CHECKS["passed"] += 1
        CHECKS["details"].append({"name": name, "ok": True})
    else:
        CHECKS["failed"] += 1
        CHECKS["details"].append({"name": name, "ok": False, "detail": detail})
        print(f"  FAIL {name}: {detail}")


def make_fake_codex(directory: Path, *, reply: str = '{"choice": 1, "reason": "fake"}', exit_code: int = 0) -> Path:
    """生成一个假的 codex 可执行文件：记录 argv，并把 reply 写进 -o。"""
    script = directory / "fake_codex.sh"
    argv_log = directory / "argv.txt"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > "{argv_log}"\n'
        'out=""\n'
        'while [[ $# -gt 0 ]]; do\n'
        '  case "$1" in -o) out="$2"; shift 2;; *) shift;; esac\n'
        "done\n"
        f"cat > \"$out\" <<'REPLY'\n{reply}\nREPLY\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def sample() -> tuple[Observation, list[Candidate]]:
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[10:20, 10:20] = 200
    cam = CameraFrame(name="wrist", rgb=rgb, depth=None, fovy_deg=75.0, width=64, height=48)
    stone = StoneState(name="rock_0", pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]),
                       size=np.array([0.05, 0.04, 0.03]), mass=0.3, grasp_quality=0.5)
    slots = [PlanSlot(0, 0, np.array([0.1, 0.2, 0.3]), np.array([1.0, 0, 0, 0])),
             PlanSlot(0, 1, np.array([0.2, 0.2, 0.3]), np.array([1.0, 0, 0, 0]))]
    candidates = [Candidate(stone=stone, slot=slots[0], score=0.1),
                  Candidate(stone=stone, slot=slots[1], score=0.9)]
    obs = Observation(time=0.0, phase="idle", joints={}, tcp_pos=np.zeros(3),
                      tcp_rot=np.eye(3), gripper_width=0.0, cameras={"wrist": cam},
                      instruction="stack the stones")
    return obs, candidates


class _ScriptedFallback:
    name = "scripted"

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, context=None) -> None:  # noqa: ANN001
        pass

    def decide(self, observation, candidates):  # noqa: ANN001
        from stone_stack.policy.base import Decision

        self.calls += 1
        candidate = candidates[0]
        return Decision(action="place", stone=candidate.stone.name, slot=candidate.slot.key,
                        target_pos=candidate.slot.target_pos, target_quat=candidate.slot.target_quat,
                        source="scripted", reason="scripted fallback")


def run() -> int:
    obs, candidates = sample()
    with tempfile.TemporaryDirectory(prefix="agent_policy_test_") as tmp:
        tmpdir = Path(tmp)

        print("[1] build_codex_command")
        cmd = build_codex_command(
            {"base_url": "http://127.0.0.1:3001/v1", "model": "Qwen3-VL-8B", "provider": "vllm"},
            prompt="hello", image_paths=[tmpdir / "a.jpg"], output_path=tmpdir / "o.txt", workdir=tmpdir,
        )
        joined = " ".join(cmd)
        check("含自定义 provider 定义", "model_providers.vllm=" in joined, joined)
        check("provider 用 responses wire_api", '"responses"' in joined or "'responses'" in joined, joined)
        check("base_url 正确", "http://127.0.0.1:3001/v1" in joined, joined)
        check("带图像 -i", "-i" in cmd, joined)

        print("[2] 合法 JSON -> Decision")
        good = CodexAgentHighLevel({"codex_bin": str(make_fake_codex(tmpdir)), "keep_workdir": True}, fallback=None)
        good.reset({})
        decision = good.decide(obs, candidates)
        check("choice=1 命中候选 1", decision.slot == (0, 1), str(decision.slot))
        check("source 是 codex", decision.source == "codex", decision.source)
        check("送了 1 张图", good.stats["images_sent"] == 1, str(good.stats["images_sent"]))

        print("[3] 停机")
        stop_dir = tmpdir / "stop"
        stop_dir.mkdir()
        stopper = CodexAgentHighLevel({"codex_bin": str(make_fake_codex(stop_dir, reply='{"choice": -1, "reason": "done"}'))}, fallback=None)
        stopper.reset({})
        check("choice=-1 -> stop", stopper.decide(obs, candidates).is_stop)
        stop_dir2 = tmpdir / "stop2"
        stop_dir2.mkdir()
        stopper2 = CodexAgentHighLevel({"codex_bin": str(make_fake_codex(stop_dir2, reply='{"choice": "stop"}'))}, fallback=None)
        stopper2.reset({})
        check('choice="stop" -> stop', stopper2.decide(obs, candidates).is_stop)

        print("[4] 越界 -> 修复重试 -> 回退")
        bad_dir = tmpdir / "bad"
        bad_dir.mkdir()
        fallback = _ScriptedFallback()
        bad = CodexAgentHighLevel({"codex_bin": str(make_fake_codex(bad_dir, reply='{"choice": 99}')), "retries": 1}, fallback=fallback)
        bad.reset({})
        tried = bad.decide(obs, candidates)
        check("越界走 fallback", tried.source.startswith("fallback:"), tried.source)
        check("retries 计数为 1", bad.stats["retries"] == 1, str(bad.stats["retries"]))
        check("回退到脚本高层", fallback.calls >= 1, str(fallback.calls))

        print("[5] 子进程失败 -> 回退 argmax")
        fail = CodexAgentHighLevel({"codex_bin": "/bin/false", "retries": 0}, fallback=None)
        fail.reset({})
        failed = fail.decide(obs, candidates)
        check("argmax(score) 命中候选 1", failed.slot == (0, 1), str(failed.slot))
        check("source=argmax_score", failed.source == "fallback:argmax_score", failed.source)

        print("[6] 图像确实落盘为 JPEG")
        img_dir = tmpdir / "img"
        img_dir.mkdir()
        writer = CodexAgentHighLevel({"codex_bin": str(make_fake_codex(img_dir)), "keep_workdir": True,
                                      "workdir": str(img_dir)}, fallback=None)
        writer.reset({})
        writer.decide(obs, candidates)
        jpgs = sorted(img_dir.rglob("camera_*.jpg"))
        check("生成 1 个 JPEG", len(jpgs) == 1, str(jpgs))
        if jpgs:
            check("JPEG 魔数 FFD8FF", jpgs[0].read_bytes()[:3] == b"\xff\xd8\xff", "bad magic")

    print(f"\nTOTAL {CHECKS['passed']}/{CHECKS['passed'] + CHECKS['failed']} passed")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
