#!/usr/bin/env python3
"""抓取自检：把每块候选石头真的抓起来抬一次，量它到底夹不夹得住。

为什么需要它：夹爪"能夹多宽"不能只看几何。指垫是软的（solref/solimp + condim=4），
石头形状不规则，夹爪闭合时可能咬住棱角、也可能把石头推走。这个脚本对每块候选石头
跑一遍**完整的抓取路径**（接近 → 下降 → 闭合 → 抬起），用真值量抬升量，输出
"哪块石头夹得住、对应的宽度是多少"，用来定 `max_opening_m` 和候选池过滤阈值。

    python tools/grasp_smoke.py --arm ur5e --out reports/grasp_smoke_ur5e.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

import numpy as np  # noqa: E402

from stone_stack.execution import ExecutionConfig, Executor  # noqa: E402
from stone_stack.policy.base import Goal, PolicyPair, StoneState  # noqa: E402
from stone_stack.policy.scripted import ScriptedHighLevel, ScriptedLowLevel  # noqa: E402
from stone_stack.robots import SceneOptions, build_scene, build_workcell, get_profile, scale_stones  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TASK2 抓取自检")
    parser.add_argument("--arm", default="ur5e")
    parser.add_argument("--stones", type=int, default=0, help="试抓多少块（0=全部候选）")
    parser.add_argument("--stone-scale", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--servo-seconds", type=float, default=0.6)
    return parser.parse_args()


def make_stones(profile, count: int):
    from stone_stack.moonsim_rocks import make_moonsim_wall_rocks

    return scale_stones(make_moonsim_wall_rocks(seed=17, count=count), profile.stone_scale)


def main() -> int:
    args = parse_args()
    profile = get_profile(args.arm)
    scale = args.stone_scale if args.stone_scale is not None else profile.stone_scale
    count = args.stones or 14
    stones = make_stones(profile, count)
    workcell = build_workcell(profile, stones, (3, 2, 1))
    built = build_scene(profile, stones, workcell, SceneOptions(camera_width=320, camera_height=240))
    pair = PolicyPair(name="probe", high=ScriptedHighLevel(), low=ScriptedLowLevel())
    config = ExecutionConfig(
        servo_seconds=args.servo_seconds,
        capture_cameras=False,
        reset_seconds=0.2,
        grasp_retries=0,
        contact_aware_place=False,
    )
    executor = Executor(built, profile, pair, config)
    executor.controller.open_gripper()
    executor.settle(0.3)

    tilt_hint = workcell.wall_center - workcell.base_pos
    rotation = executor.controller.grasp_rotation(np.array([0.0, 1.0, 0.0]), tilt_hint=tilt_hint)

    records = []
    started = time.time()
    for index, stone in enumerate(stones):
        executor.reset_pick_scene(stone.name)
        executor.settle(config.reset_seconds)
        pick_pos, pocket_quat = executor.stone_pose(stone.name)
        size = executor._stone_size(stone.name)
        goal = Goal(
            kind="pick",
            stone=StoneState(name=stone.name, pos=pick_pos, quat=pocket_quat, size=size, mass=stone.mass, graspable=True),
            pick_pos=executor.controller.tcp_for_pad_center(pick_pos, rotation),
            pick_rot=rotation,
            grip_width=max(0.002, float(size[1]) * 0.85),
            timeout_s=25.0,
        )
        metrics: list[dict] = []
        executor._run_goal(goal, phase="pick", instruction="", metrics=metrics)
        lifted, _ = executor.stone_pose(stone.name)
        lift_gain = float(lifted[2] - pick_pos[2])
        contact = executor.contact_count(stone.name)
        # 抬起来时指垫的实际开口：量它有没有真的贴住石头
        opening = executor.controller.measured_opening()
        record = {
            "name": stone.name,
            "size_mm": [round(float(value) * 1000, 1) for value in size],
            "mass_kg": round(float(stone.mass), 3),
            "lift_gain_m": round(lift_gain, 4),
            "grasped": bool(lift_gain > 0.03),
            "pad_opening_mm": round(opening * 1000, 1),
            "stone_contacts": int(contact),
            "policy_steps": len(metrics),
            "final_tcp_error_mm": round(float(metrics[-1]["pos_error_m"]) * 1000, 2) if metrics else None,
        }
        records.append(record)
        print(
            f"{stone.name:12s} W={record['size_mm'][1]:6.1f}mm T={record['size_mm'][2]:5.1f}mm "
            f"m={record['mass_kg']:.2f}kg lift={record['lift_gain_m']*1000:7.1f}mm "
            f"grasped={record['grasped']} pad_open={record['pad_opening_mm']:6.1f}mm"
        )

    grasped = [record for record in records if record["grasped"]]
    widest_ok = max((record["size_mm"][1] for record in grasped), default=0.0)
    narrowest_fail = min((record["size_mm"][1] for record in records if not record["grasped"]), default=None)
    summary = {
        "arm": profile.name,
        "stone_scale": scale,
        "profile_max_opening_mm": round(profile.gripper.max_opening_m * 1000, 2),
        "graspable_margin_used": 0.95,
        "tried": len(records),
        "grasped": len(grasped),
        "widest_grasped_mm": widest_ok,
        "narrowest_failed_mm": narrowest_fail,
        "suggested_max_grasp_width_mm": widest_ok,
        "duration_s": round(time.time() - started, 1),
        "records": records,
    }
    print(
        f"\n{profile.name}: {len(grasped)}/{len(records)} 抓起，最宽成功 {widest_ok:.1f} mm"
        f"（配置上限 {profile.gripper.max_opening_m * 1000:.1f} mm × 0.95）"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"报告 {args.out}")
    return 0 if grasped else 1


if __name__ == "__main__":
    sys.exit(main())
