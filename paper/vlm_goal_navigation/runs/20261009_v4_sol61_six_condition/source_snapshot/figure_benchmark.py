"""Render the TASK3 Experiment 2 figure from a benchmark report.

One row per scene:

left   the terrain cross-section, the measured oracle visibility runs V(x) along
       the nominal approach, and the frozen lost state;
right  the physically measured visibility curve V(a, t) of every candidate
       action, as step traces, so "different action -> different future
       observability" is visible directly rather than asserted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .scenes import BENCHMARK_SCENES, scene_for
    from .benchmark import build_scene_terrain
except ImportError:
    from scenes import BENCHMARK_SCENES, scene_for
    from benchmark import build_scene_terrain

ACTION_COLORS = {
    "forward": "#d62728", "back": "#ff7f0e", "left": "#1f77b4", "right": "#17becf",
    "turn_left": "#8c8c8c", "turn_right": "#bbbbbb", "toward": "#2ca02c",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Render the TASK3 Experiment 2 figure")
    p.add_argument("--benchmark", type=Path,
                   default=Path("tasks/task3_reobservability/generated/benchmark.json"))
    p.add_argument("--output", type=Path,
                   default=Path("docs/images/task3_benchmark.png"))
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.benchmark.read_text(encoding="utf-8"))
    names = [n for n in BENCHMARK_SCENES if n in report["scenes"]]

    fig, axes = plt.subplots(len(names), 2, figsize=(14.0, 2.7 * len(names)),
                             squeeze=False)
    for row, name in enumerate(names):
        entry = report["scenes"][name]
        scene = scene_for(name)
        terrain = build_scene_terrain(scene)

        # ---- left: terrain profile and V(x) ----
        ax = axes[row][0]
        xs = np.linspace(float(terrain.x[0]), float(terrain.x[-1]), 600)
        zs = np.array([terrain.height(float(x), 0.0) for x in xs])
        ax.fill_between(xs, zs, zs.min() - 0.35, color="#d9d9d9", zorder=1)
        ax.plot(xs, zs, color="#555555", lw=1.6, zorder=2)
        band_y = zs.max() + 0.22
        for run in entry.get("profile", {}).get("runs", []):
            ax.plot([run["x_from"], run["x_to"]], [band_y, band_y], lw=6,
                    solid_capstyle="butt",
                    color="#2ca02c" if run["visible"] else "#d62728", zorder=3)
        ax.plot([float(scene.approach_xy[0])], [band_y], marker="v", ms=9,
                color="#2ca02c", zorder=4)
        ax.plot([float(scene.lost_xy[0])], [band_y], marker="v", ms=9,
                color="#d62728", zorder=4)
        target = np.array(entry["target_xyz"])
        ax.plot([target[0]], [target[2]], marker="*", ms=15, color="#d62728", zorder=4)
        ax.set_ylim(zs.min() - 0.3, band_y + 0.55)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylabel("z (m)")
        ax.set_title(f"{name}: {scene.mechanism}\n"
                     f"V(x): " + " ".join(
                         ("V" if r["visible"] else "X") for r in
                         entry.get("profile", {}).get("runs", [])),
                     fontsize=9)
        ax.grid(alpha=0.25)

        # ---- right: V(a, t) per candidate ----
        ax = axes[row][1]
        for action in entry["actions"]:
            curve = action["curve"]
            if not curve:
                continue
            ts = [c["t_s"] for c in curve]
            vs = [1.0 if c["visible"] else 0.0 for c in curve]
            label = (f"{action['action']}  rsr={int(action['rsr'])}  "
                     f"t={action['t_reobs_s']}"
                     + ("  (fell)" if action["fallen"] else ""))
            ax.step(ts, vs, where="post", lw=1.8,
                    color=ACTION_COLORS.get(action["action"], "k"), label=label)
        ax.set_ylim(-0.15, 1.35)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["lost", "visible"])
        ax.set_xlabel("time in the rollout (s)")
        ax.set_title(f"{name}: physical visibility V(a, t)   best={entry['best_action']}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="center right", ncol=1, framealpha=0.9)
        ax.grid(alpha=0.25)

    fig.suptitle("TASK3 Experiment 2 - action-dependent target re-observability "
                 f"(controller: {report.get('controller')})", fontsize=12)
    fig.text(0.5, 0.005,
             "Left panels: V(x) is the geometric oracle profile and includes positions the robot "
             "cannot physically reach (e.g. standing on top of the barrier) - the physical "
             "rollouts on the right, not this curve, are the ground truth.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.015, 1, 0.965))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
