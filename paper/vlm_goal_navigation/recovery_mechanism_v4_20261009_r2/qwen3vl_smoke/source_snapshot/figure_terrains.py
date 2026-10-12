"""Render a contact sheet of every TASK3 terrain with the robot standing on it.

Uses the same scene builder and the same robots as the experiments, so what the
sheet shows is exactly what the interactive viewer opens.  This is the offline
way to eyeball the terrains; ``terrain_viewer.py`` is the interactive one.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .terrain_viewer import ALL_TERRAINS, build, describe
except ImportError:
    from terrain_viewer import ALL_TERRAINS, build, describe

ROOT = Path(__file__).resolve().parents[2]


def render_tile(name: str, controller_name: str, ground: str, width: int, height: int,
                policy: Path | None = None) -> np.ndarray:
    import mujoco

    model, controller, scene, terrain, target = build(name, controller_name, ground, policy)
    data = mujoco.MjData(model)
    x, y = float(scene.approach_xy[0]), float(scene.approach_xy[1])
    data.qpos[:3] = [x, y, terrain.height(x, y) + 0.30]
    data.qpos[3:7] = [math.cos(scene.yaw_rad / 2), 0, 0, math.sin(scene.yaw_rad / 2)]
    data.qpos[7:] = controller.default_angles
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mid_x = (float(terrain.x[0]) + float(terrain.x[-1])) / 2
    cam.lookat[:] = [mid_x, 0.0, float(terrain.heights.max()) * 0.4]
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
    p = argparse.ArgumentParser(description="Render a contact sheet of the TASK3 terrains")
    p.add_argument("--controller", choices=["go2rl", "leggedmanip"], default="go2rl")
    p.add_argument("--ground", choices=["auto", "slabs", "boxes", "hfield"], default="auto")
    p.add_argument("--check-report", type=Path,
                   default=Path("tasks/task3_reobservability/generated/terrain_check.json"))
    p.add_argument("--output", type=Path,
                   default=Path("docs/images/task3_terrains.png"))
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdicts = {}
    if args.check_report.exists():
        report = json.loads(args.check_report.read_text(encoding="utf-8"))
        verdicts = {r["terrain"]: r for r in report["results"]}

    names = ALL_TERRAINS
    cols = args.cols
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows), squeeze=False)

    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        try:
            tile = render_tile(name, args.controller, args.ground, 480, 340)
            ax.imshow(tile)
        except Exception as exc:  # noqa: BLE001 - one bad terrain must not kill the sheet
            ax.text(0.5, 0.5, f"render failed\n{exc!r}", ha="center", va="center",
                    fontsize=7, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        info = describe(name)
        v = verdicts.get(name, {})
        if v.get("error"):
            status = "ERROR"
        elif not v:
            status = "not checked"
        elif not v["verdict"].get("locomotion_tested", True):
            status = "geometry OK, locomotion N/A (hfield)" if v["all_pass"] else "geometry MISMATCH"
        else:
            status = ("PASS" if v["all_pass"] else "FAIL") + f"  ({v['ground_kind']})"
        ax.set_title(f"{name}\nz {info['z_range_m'][0]:+.2f}..{info['z_range_m'][1]:+.2f} m   {status}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(names), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("TASK3 terrains, generated programmatically as exact box slabs "
                 "(plus the legacy height-field profiles), with the Go2+Piper standing on each",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
