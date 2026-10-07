"""Render scene screenshots for the TASK3 terrains, including the robot's own view.

For each of the newer terrains this renders three panels:

``overview``  a free camera framing the whole terrain, so the shape is readable;
``eye``       the image from the camera actually injected into Go2+Piper, i.e.
              what the robot sees, with the target marker visible or not;
``behind``    a free camera just behind and above the robot, for context.

The basin row is the point of the figure: the *same* scene is rendered with the
robot at the far anchor and at the rim, and the eye view flips from no target to
target because the near rim stops occluding at a grazing angle.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from .go2_piper_camera import CAMERA_NAME
    from .scenes import ROBOT_BASE_HEIGHT_M, scene_for
    from .terrain_viewer import build, place, yaw_of
except ImportError:
    from go2_piper_camera import CAMERA_NAME
    from scenes import ROBOT_BASE_HEIGHT_M, scene_for
    from terrain_viewer import build, place, yaw_of

ROOT = Path(__file__).resolve().parents[2]

# (title, terrain, anchor, camera)
#   anchor: "approach" | "lost" | (x, y)
#   camera: "overview" | "eye" | "behind"
PANELS = [
    ("basin: robot far away\ncamera view: target hidden", "basin", "lost", "eye"),
    ("basin: at the rim\ncamera view: target visible in the pit", "basin", "approach", "eye"),
    ("basin: same far pose\nfrom behind the robot", "basin", "lost", "behind"),
    ("basin overview\n(red marker on the pit floor)", "basin", "approach", "overview"),
    ("perlin: in a trough\ncamera view: target hidden", "perlin", "lost", "eye"),
    ("perlin: on the crest\ncamera view: target visible", "perlin", "approach", "eye"),
    ("perlin overview\nPerlin/fBm relief, walkable steps", "perlin", "lost", "overview"),
    ("perlin_rough overview\ntighter, deeper relief", "perlin_rough", "approach", "overview"),
]


def render(name: str, anchor, camera: str, controller_name: str, ground: str,
           width: int, height: int, policy: Path | None = None):
    import mujoco

    model, controller, scene, terrain, target = build(name, controller_name, ground, policy)
    data = mujoco.MjData(model)
    xy = {"approach": scene.approach_xy, "lost": scene.lost_xy}.get(anchor, anchor)
    place(model, data, terrain, xy, controller, scene.yaw_rad)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    if camera == "eye":
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = model.camera(CAMERA_NAME).id
    elif camera == "behind":
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        base = data.qpos[:3]
        cam.lookat[:] = [base[0] + 1.0, base[1], base[2] + 0.1]
        cam.distance = 4.5
        cam.azimuth = 160.0
        cam.elevation = -18.0
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [(float(terrain.x[0]) + float(terrain.x[-1])) / 2, 0.0,
                         float(terrain.heights.max()) * 0.4]
        cam.distance = 10.5
        cam.azimuth = 125.0
        cam.elevation = -20.0

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=cam)
        return np.asarray(renderer.render()).copy()
    finally:
        close = getattr(renderer, "close", None)
        if close:
            close()


def main() -> None:
    p = argparse.ArgumentParser(description="Render TASK3 scene screenshots")
    p.add_argument("--controller", choices=["go2rl", "leggedmanip"], default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--output", type=Path, default=Path("docs/images/task3_scenes.png"))
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = args.cols
    rows = (len(PANELS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows), squeeze=False)
    for i, (title, terrain, anchor, camera) in enumerate(PANELS):
        ax = axes[i // cols][i % cols]
        try:
            tile = render(terrain, anchor, camera, args.controller, args.ground, 480, 360)
            ax.imshow(tile)
        except Exception as exc:  # noqa: BLE001 - one bad panel must not kill the sheet
            ax.text(0.5, 0.5, f"render failed\n{exc!r}", ha="center", va="center",
                    fontsize=7, wrap=True)
        ax.set_title(title, fontsize=9, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(PANELS), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("TASK3 scenes: the overview, the view from behind the robot, and the camera "
                 "actually injected into Go2+Piper", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.30, wspace=0.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
