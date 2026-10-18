"""Render the physical Experiment 0 figure from a locomotion episode report.

Left: top-down path of every candidate action from the frozen lost state, with
the step edges and the remembered target marked.
Right: the same rollouts as base height against forward distance, with the
staircase profile drawn behind them, so "walked up the stairs" and "re-saw the
target" can be read separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Render the TASK3 physical-action figure")
    p.add_argument("--episode", type=Path,
                   default=Path("tasks/task3_reobservability/generated/"
                                "episode_staircase10_go2rl.json"))
    p.add_argument("--output", type=Path,
                   default=Path("docs/images/task3_physical_actions.png"))
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.episode.read_text(encoding="utf-8"))
    spec = report["terrain"]
    tread = float(spec.get("step_tread_m", 0.30))
    rise = float(spec.get("step_rise_m", 0.15))
    steps = int(spec.get("step_count", 10))
    target = report["target_xyz"]
    lost = report["lost_state"]["base_xyz"]

    colors = {"turn_left": "#9467bd", "turn_right": "#8c564b", "forward": "#7f7f7f",
              "climb": "#d62728", "climb_toward": "#2ca02c", "back": "#ff7f0e"}

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- top-down plane -----------------------------------------------------
    for i in range(steps + 1):
        ax_top.axvline(i * tread, color="#bbbbbb", lw=0.8, zorder=0)
    ax_top.axvspan(0.0, steps * tread, color="#dddddd", alpha=0.5, zorder=0,
                   label="step block")
    ax_top.plot([target[0]], [target[1]], marker="*", ms=18, color="#d62728", zorder=5,
                label="remembered target")
    ax_top.plot([lost[0]], [lost[1]], marker="o", ms=9, color="black", zorder=5,
                label="lost state")
    for c in report["candidates"]:
        trace = c["rollout"]["trace"]
        xy = np.array([[t["xyz"][0], t["xyz"][1]] for t in trace])
        if len(xy) == 0:
            continue
        style = "-" if c["reobserved"] else "--"
        ax_top.plot(xy[:, 0], xy[:, 1], style, color=colors.get(c["action"], "k"), lw=2,
                    label=f"{c['action']} ({'re-observed' if c['reobserved'] else 'no'})")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_title("Physical candidate rollouts from the lost state (top-down)")
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=8, loc="upper left")

    # --- base height against forward distance -------------------------------
    xs = np.linspace(-3.0, steps * tread + 2.0, 400)
    profile = np.clip(np.floor(np.maximum(xs, 0.0) / tread) + 1, 0, steps) * rise
    profile = np.where(xs < 0.0, 0.0, profile)
    profile = np.where(xs >= steps * tread, steps * rise, profile)
    ax_side.plot(xs, profile, color="#555555", lw=2, label="step surface")
    for c in report["candidates"]:
        trace = c["rollout"]["trace"]
        arr = np.array([[t["xyz"][0], t["xyz"][2]] for t in trace])
        if len(arr) == 0:
            continue
        style = "-" if c["reobserved"] else "--"
        ax_side.plot(arr[:, 0], arr[:, 1], style, color=colors.get(c["action"], "k"), lw=2,
                     label=c["action"])
    ax_side.axhline(target[2], color="#d62728", ls=":", lw=1.5, label="target height")
    ax_side.set_xlabel("x (m)")
    ax_side.set_ylabel("base height (m)")
    ax_side.set_title("Base height against forward distance")
    ax_side.grid(alpha=0.3)
    ax_side.legend(fontsize=8, loc="upper left")

    reached = report.get("physically_reobserved_by")
    fig.suptitle(
        f"TASK3 Experiment 0 - physical candidate actions "
        f"(controller: {report.get('controller', '?')}, policy: "
        f"{Path(report.get('policy', {}).get('policy', '?')).name})"
        f"   |   first re-observing action: {reached}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")
    for c in report["candidates"]:
        r = c["rollout"]
        print(f"  {c['action']:13s} reobserved={c['reobserved']!s:5s} "
              f"fwd={r['forward_m']:+.2f} dz={r['height_gain_m']:+.2f} "
              f"level={c['after']['stair_level']} fallen={r['fallen']}")


if __name__ == "__main__":
    main()
