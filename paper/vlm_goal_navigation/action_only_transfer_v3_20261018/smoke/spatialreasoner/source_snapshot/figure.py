"""Render the TASK3 Experiment 0 figure for the staircase scene.

Top row: what the robot's own camera sees when the target is visible, when it
has just been lost, and after the re-observing action.
Bottom row: the three pieces of decision evidence - the oracle visibility curve
along the approach, the camera-measured target pixels per candidate, and the
geometric displacement of each candidate.

The figure is generated from the live MuJoCo scene, not from stored arrays.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from .experiment import action_state, robot_anchor, target_anchor, visibility
    from .go2_piper_camera import close_scene, create_scene, render_observation, set_robot_pose
    from .scenes import CAMERA_OFFSET_XYZ, ROBOT_BASE_HEIGHT_M, scene_for
    from .terrain import generate
except ImportError:
    from experiment import action_state, robot_anchor, target_anchor, visibility
    from go2_piper_camera import close_scene, create_scene, render_observation, set_robot_pose
    from scenes import CAMERA_OFFSET_XYZ, ROBOT_BASE_HEIGHT_M, scene_for
    from terrain import generate

CANDIDATES = ["turn_left", "turn_right", "forward", "climb", "back"]
REOBSERVE = "back"


def main() -> None:
    p = argparse.ArgumentParser(description="Render the TASK3 staircase figure")
    p.add_argument("--terrain", default="staircase10")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("docs/images/task3_reobservation.png"))
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terrain = generate(args.terrain, args.seed)
    scene = scene_for(terrain.name)
    target = target_anchor(terrain, scene)
    lost = {"robot": robot_anchor(terrain, scene.lost_xy), "yaw": scene.yaw_rad, "pitch": 0.0}
    approach = {"robot": robot_anchor(terrain, scene.approach_xy), "yaw": scene.yaw_rad, "pitch": 0.0}
    after = {name: action_state(lost, name, terrain, scene) for name in CANDIDATES}

    views, pixels = {}, {}
    cam = create_scene(terrain, target)
    try:
        for name, state in [("approach", approach), ("lost", lost)] + \
                [(f"after_{n}", after[n]) for n in CANDIDATES]:
            set_robot_pose(cam, state["robot"], state["yaw"])
            cam.renderer.update_scene(cam.data, camera=cam.camera_name)
            views[name] = np.asarray(cam.renderer.render()).copy()
            pixels[name] = render_observation(cam)["target_pixel_count"]
    finally:
        close_scene(cam)

    xs = np.arange(-5.6, 2.61, 0.05)
    visible = np.array([visibility(terrain, robot_anchor(terrain, (float(x), 0.0)),
                                   scene.yaw_rad, 0.0, target)["visible_oracle"] for x in xs])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))
    for ax, (key, title) in zip(axes[0], [
            ("approach", f"approach  ({scene.approach_xy[0]:+.1f} m)  target visible"),
            ("lost", f"target lost  ({scene.lost_xy[0]:+.1f} m)"),
            (f"after_{REOBSERVE}", f"after '{REOBSERVE}'  re-observed")]):
        ax.imshow(views[key])
        ax.set_title(f"{title}\nred target pixels: {pixels[key]}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1][0]
    ax.step(xs, visible.astype(float), where="post", color="#1f77b4", lw=2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["lost", "visible"])
    ax.set_ylim(-0.15, 1.2); ax.set_xlim(xs[0], xs[-1])
    for label, x, color in [("approach", scene.approach_xy[0], "#2ca02c"),
                            ("lost", scene.lost_xy[0], "#d62728"),
                            (REOBSERVE, after[REOBSERVE]["robot"][0], "#ff7f0e"),
                            ("climb", after["climb"]["robot"][0], "#9467bd")]:
        ax.axvline(x, color=color, ls="--", lw=1)
        ax.text(x, 1.06, label, rotation=90, fontsize=7, ha="right", va="top", color=color)
    ax.set_xlabel("robot base x on the approach (m)")
    ax.set_title("privileged geometric oracle", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.bar(CANDIDATES, [pixels[f"after_{n}"] for n in CANDIDATES], color="#d62728")
    ax.set_ylabel("red target pixels")
    ax.set_title("Go2+Piper camera evidence after each candidate", fontsize=10)
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][2]
    gains = [float(after[n]["robot"][0] - lost["robot"][0]) for n in CANDIDATES]
    colors = ["#2ca02c" if v > 0 else "#7f7f7f" for v in gains]
    colors[CANDIDATES.index(REOBSERVE)] = "#ff7f0e"
    colors[CANDIDATES.index("climb")] = "#9467bd"
    ax.bar(CANDIDATES, gains, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("forward displacement (m)")
    ax.set_title("geometric action magnitude", fontsize=10)
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("TASK3 Experiment 0 - target re-observability on a 10-level staircase "
                 "(MuJoCo Go2+Piper, oracle geometry)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")
    print("camera target pixels:", {k: v for k, v in pixels.items()})
    print("visibility at anchors:",
          {"approach": bool(visibility(terrain, approach["robot"], scene.yaw_rad, 0.0, target)["visible_oracle"]),
           "lost": bool(visibility(terrain, lost["robot"], scene.yaw_rad, 0.0, target)["visible_oracle"])})


if __name__ == "__main__":
    main()
