#!/usr/bin/env python3
"""TASK2 入口：--config 选配置，--mode 选操作，--view 打开窗口。

机械臂、策略、石头、trial 预算和输出路径都在 YAML 中设置。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

# MuJoCo 的 GL 后端必须在 import mujoco 之前定下来。本机 osmesa 的 PyOpenGL 路径
# 直接导入失败、glfw 需要真实 GLX，egl 是唯一在本机实测可用的离屏后端。
os.environ.setdefault("MUJOCO_GL", "egl")

TASK_ROOT = Path(__file__).resolve().parent
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

import numpy as np  # noqa: E402
import mujoco.viewer  # noqa: E402

from stone_stack.execution import ExecutionConfig, Executor, write_report  # noqa: E402
from stone_stack.policy import ScriptedHighLevel, ScriptedLowLevel, build_policy  # noqa: E402
from stone_stack.robots import (  # noqa: E402
    SceneOptions,
    build_scene,
    build_workcell,
    get_profile,
    list_profiles,
)
from stone_stack.tools.stones import load_stones

from stone_stack.task_config import default_report_path, load_config, parse_courses  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML 配置，默认 task2.yaml")
    parser.add_argument("--mode", choices=("run", "camera", "list", "train", "eval"), default="run", help="运行 / 拍照 / 列表 / RL训练 / RL评估")
    parser.add_argument("--view", action="store_true", help="打开 MuJoCo 窗口")
    return parser.parse_args(argv)


def build_everything(config: dict):
    """由 YAML 装配场景与策略。"""
    np.random.seed(int(config["seed"]))
    if config["scene"]["layout"] == "upstream":
        from stone_stack.upstream_scene import build_upstream
        built, workcell, profile = build_upstream(config)
    else:
        profile = get_profile(config["robot"]["arm"])
        scale = config["robot"]["stone_scale"] or profile.stone_scale
        stones = load_stones(config, scale)
        courses = parse_courses(config["robot"]["courses"])

        options = SceneOptions(
            stone_visual_style=config["scene"]["stone_visual_style"],
            stone_visual_roughness=float(config["scene"]["stone_visual_roughness"]),
            stone_visual_subdivisions=int(config["scene"]["stone_visual_subdivisions"]),
            robot_visual=config["scene"]["robot_visual"],
            camera_width=int(config["scene"]["camera_width"]),
            camera_height=int(config["scene"]["camera_height"]),
            timestep=float(config["scene"]["timestep"]),
            top_camera_height_m=float(config["scene"]["top_camera_height_m"]),
            front_camera_offset=tuple(config["scene"]["front_camera_offset"]),
            overview_offset=tuple(config["scene"]["overview_offset"]),
        )
        workcell = build_workcell(profile, stones, courses, wall_distance_scale=float(config["robot"]["wall_distance_scale"]))
        built = build_scene(profile, stones, workcell, options)

    scripted = config["scripted"]
    high = ScriptedHighLevel()
    low = ScriptedLowLevel(
        approach_height_m=float(scripted["approach_height_m"]),
        travel_height_m=float(scripted["travel_height_m"]),
        place_clearance_m=float(scripted["place_clearance_m"]),
        retreat_height_m=float(scripted["retreat_height_m"]),
        close_fraction=float(scripted["close_fraction"]),
    )
    pair = build_policy(config["policy"]["name"], high, low, config)
    return built, workcell, profile, pair, config


def save_frames(built, directory: Path) -> list[str]:
    from stone_stack.cameras import open_camera_rig
    from PIL import Image

    names = tuple(built.model.camera(i).name for i in range(built.model.ncam))
    rig = open_camera_rig(built, names)
    frames = rig.capture()
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in frames.items():
        path = directory / f"{built.workcell.profile.name}_{name}.png"
        Image.fromarray(frame.rgb).save(path)
        written.append(str(path))
        if frame.depth is not None:
            depth = frame.depth.copy()
            valid = depth[depth > 0]
            if valid.size:
                scaled = np.clip((depth - valid.min()) / max(valid.max() - valid.min(), 1e-6), 0, 1)
            else:
                scaled = np.zeros_like(depth)
            Image.fromarray((scaled * 255).astype(np.uint8)).save(directory / f"{built.workcell.profile.name}_{name}_depth.png")
    rig.close()
    return written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "list":
        for profile in list_profiles():
            logger.info(f"{profile.name:6s} {profile.description}")
            logger.info(f"       关节 {len(profile.joints)} 个，夹爪开口 {profile.gripper.max_opening_m * 1000:.1f} mm，"
                  f"石块缩放 {profile.stone_scale}，基座 {profile.base_pos}")

        logger.info("scripted  纯几何脚本策略（高层：铺层选石；低层：路径点伺服）")
        logger.info("rl        第二条主线：DrQ-v2 图像强化学习 / SAC 状态对照")
        logger.info("qwen-direct QwenVL 结构化工具调用：看图 → TCP/夹爪 → 新观测（直接控制主线）")
        logger.info("vlm       Qwen3-VL 高层 + 低层，失败自动回退脚本（需要 vLLM 服务）")
        logger.info("vlm-high  只把高层换成 VLM，低层仍走脚本")
        logger.info("vlm-low   只把低层换成 VLM，高层仍走脚本")
        logger.info("qwen-agent Codex agent(背后 Qwen3-VL) 高层选石 + 脚本低层，失败回退脚本（需 vLLM + codex）")
        return 0

    config = load_config(args.config)
    if config["output"]["log"]:
        log_path = Path(config["output"]["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
    if config["policy"]["name"] == "rl" and args.mode != "camera":
        from stone_stack.rl.train import run_rl
        return run_rl(config, args.mode, args.view)
    if args.mode in ("train", "eval"):
        raise ValueError("train/eval require policy.name: rl")
    built, workcell, profile, pair, config = build_everything(config)
    output_config = config["output"]
    logger.info("机械臂 %s：%s", profile.name, profile.description)
    logger.info("工作区 %s", workcell.describe())

    if output_config["save_xml"]:
        xml_path = Path(output_config["save_xml"])
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(built.xml, encoding="utf-8")
        logger.info("已保存 MJCF：%s", xml_path)

    if args.mode == "camera":
        written = save_frames(built, Path(output_config["save_frames"] or TASK_ROOT / "outputs/camera"))
        for path in written:
            logger.info("已保存相机图：%s", path)
        return 0

    # 真正跑任务
    from stone_stack.cameras import open_camera_rig

    rig = None
    if pair.needs_cameras or config["execution"]["capture_cameras"]:
        rig = open_camera_rig(built, tuple(pair.needs_cameras) or ("wrist", "top", "front"))
    # 执行参数从 YAML 拷贝，再填入日志、相机和路径。
    execution_options = dict(config["execution"])
    execution_options.pop("max_placements")
    execution_options["camera_names"] = tuple(pair.needs_cameras) or ("wrist", "top", "front")
    execution_options["log"] = logger.info
    if output_config["save_frames"]:
        execution_options["save_frames"] = Path(output_config["save_frames"])
    execution = ExecutionConfig(**execution_options)
    viewer = None
    try:
        if args.view:
            viewer = mujoco.viewer.launch_passive(built.model, built.data)
        executor = Executor(built, profile, pair, execution, cameras=rig, viewer=viewer)
        if pair.name == "qwen-direct":
            from datetime import datetime
            from stone_stack.trials import run_trials
            output = TASK_ROOT / "outputs" / ("direct_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            report = run_trials(executor, pair, config["instruction"], output)
        else:
            placements = int(config["execution"]["max_placements"]) or None
            report = executor.run(instruction=config["instruction"], max_placements=placements)
    finally:
        if viewer is not None:
            viewer.close()
            # close 只发出退出请求；等待 MuJoCo UI 线程释放 GL，再关闭相机和解释器。
            for thread in threading.enumerate():
                if thread.name.endswith("(_launch_internal)"):
                    thread.join()
        if rig is not None:
            rig.close()

    report_path = config["output"]["report"] or default_report_path(profile.name, pair.name)
    report_path = Path(report_path)
    write_report(report, report_path)
    if pair.name == "qwen-direct":
        logger.info(f"直接控制结束：{report['status']}，动作 {report['executed_actions']}，报告 {report_path}")
        # 退出 0 仅表示运行正常结束；success=null 不代表堆叠成功。
        return 0
    logger.info(
        f"完成：placed {report['placed_count']}/{report['executed_placements']}，"
        f"stacked {report['stacked_count']}，报告 {report_path}"
    )
    return 0 if report["placed_count"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
