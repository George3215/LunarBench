"""Render the TASK3 Experiment 1 (history necessity) figure.

Top-left   top-down view of one paired world: same terrain, same lost state and
           heading in both worlds; the target sits behind the window on opposite
           sides.
Top-right  for every pair, the lateral positions that reveal target A and
           target B, with the shared lost state marked.  The supports are
           disjoint and on opposite sides in every accepted pair.
Bottom-left the decision rules measured over every world: reactive forward,
           history-only, initial-heading lock, and the oracle counterfactual.
Bottom-right the physically measured V(a, t) curves of the first pair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .scenes import HISTORY_PAIRS, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from .benchmark import build_scene_terrain
except ImportError:
    from scenes import HISTORY_PAIRS, ROBOT_BASE_HEIGHT_M, Scene, scene_for
    from benchmark import build_scene_terrain

ACTION_COLORS = {
    "forward": "#d62728", "back": "#ff7f0e", "left": "#1f77b4", "right": "#17becf",
    "left_hold": "#0b5394", "right_hold": "#0f7b7b",
    "turn_left": "#8c8c8c", "turn_right": "#bbbbbb", "toward": "#2ca02c",
    "heading_lock": "#9467bd",
}
SHOWN = ["forward", "back", "left_hold", "right_hold", "toward", "heading_lock"]
BASELINE_LABELS = {
    "A_reactive_forward": "A reactive\nforward",
    "B_history_only_toward": "B history-only\ntoward memory",
    "D_initial_heading_lock": "D heading lock\n(IMU only)",
    "C_oracle_counterfactual": "C oracle\ncounterfactual",
}


def scenes_for_pair_id(pair_id: str) -> list[Scene]:
    """Rebuild a pair's scenes from the descriptor, or from the static registry."""
    if pair_id in HISTORY_PAIRS:
        return HISTORY_PAIRS[pair_id]
    try:
        from history_pairs import load_descriptor, scenes_for
    except ImportError:
        from .history_pairs import load_descriptor, scenes_for
    for params in load_descriptor():
        if params.pair_id == pair_id:
            return scenes_for(params)
    raise KeyError(f"unknown pair id {pair_id!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render the TASK3 history-necessity figure")
    p.add_argument("--report", type=Path,
                   default=Path("tasks/task3_reobservability/generated/history_necessity.json"))
    p.add_argument("--output", type=Path,
                   default=Path("docs/images/task3_history_necessity.png"))
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text(encoding="utf-8"))
    worlds = report["worlds"]
    pairs = report["pairs"]
    agg = report.get("aggregate", {})
    focus = pairs[0]
    focus_names = focus["worlds"]
    focus_scenes = scenes_for_pair_id(focus["pair_id"])
    terrain = build_scene_terrain(focus_scenes[0])
    spec = terrain.spec

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))

    # ---- (a) top-down of the focus pair ----------------------------------
    ax = axes[0][0]
    wx, ww = spec["wall_x_m"], spec["wall_width_m"]
    why = spec["window_half_y_m"]
    y_lim = spec["size_m"][1] / 2
    for y0, y1 in ((-y_lim, -why), (why, y_lim)):
        ax.fill_between([wx, wx + ww], [y0, y0], [y1, y1], color="#555555")
    ax.fill_between([wx, wx + ww], [-why, -why], [why, why], color="#cccccc",
                    label=f"window (sill {spec['sill_z_m']:.2f} m)")
    for i, name in enumerate(focus_names):
        w = worlds[name]
        target = w["target_xyz"]
        ax.plot([target[0]], [target[1]], marker="*" if i == 0 else "P", ms=16,
                color="#d62728" if i == 0 else "#1f77b4",
                label=f"target {'A (+y)' if i == 0 else 'B (-y)'} - hidden from the lost state")
        hist = w["history_vantage_xy"]
        ax.plot([hist[0]], [hist[1]], marker="v", ms=9,
                color="#d62728" if i == 0 else "#1f77b4", alpha=0.65)
        ax.annotate("last seen here", xy=hist, xytext=(hist[0] - 0.3, hist[1] - 0.85),
                    fontsize=7, color="#666666")
        for act, style in (("left_hold", ":"), ("right_hold", "-.")):
            row = next((r for r in w["actions"] if r["action"] == act), None)
            if row is None:
                continue
            xy = np.array([[c["xyz"][0], c["xyz"][1]] for c in row["curve"]])
            ax.plot(xy[:, 0], xy[:, 1], style, lw=1.4, color=ACTION_COLORS[act],
                    alpha=0.8, label=f"{act} in {'A' if i == 0 else 'B'}"
                                     f" ({'hit' if row['rsr'] else 'miss'})")
    lost = worlds[focus_names[0]]["lost_state_xyz"]
    ax.plot([lost[0]], [lost[1]], marker="o", ms=9, color="black",
            label="shared lost state")
    ax.arrow(lost[0], lost[1], 0.9, 0, head_width=0.16, color="black",
             length_includes_head=True)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"pair {focus['pair_id']}: same lost state and heading, mirrored history",
                 fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)
    ax.grid(alpha=0.25)

    # ---- (b) reveal intervals over every pair ----------------------------
    ax = axes[0][1]
    for i, pair in enumerate(pairs):
        for j, name in enumerate(pair["worlds"]):
            ys = worlds[name]["lateral_profile"]["revealing_y"]
            colour = "#d62728" if j == 0 else "#1f77b4"
            if ys:
                ax.plot([min(ys), max(ys)], [i, i], lw=8, solid_capstyle="butt",
                        color=colour)
            ax.plot([worlds[name]["lost_state_xyz"][1]], [i], marker="o", ms=5,
                    color="black", zorder=5)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p["pair_id"] for p in pairs], fontsize=8)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("robot y at the lost state's x (m)")
    ax.set_title("Which lateral position reveals the target, per pair\n"
                 "(red = target A, blue = target B, black dot = shared lost state)",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    ax.invert_yaxis()

    # ---- (c) decision rules over every world -----------------------------
    ax = axes[1][0]
    labels = list(BASELINE_LABELS)
    rates = [agg.get("baseline_rsr_over_worlds", {}).get(k, 0.0) * 100 for k in labels]
    colours = ["#7f7f7f", "#2ca02c", "#9467bd", "#d62728"]
    bars = ax.bar([BASELINE_LABELS[k] for k in labels], rates, color=colours)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2, f"{rate:.0f}%",
                ha="center", fontsize=9)
    n_worlds = agg.get("n_worlds", len(worlds))
    ax.set_ylim(0, 118)
    ax.set_ylabel("RSR over worlds (%)")
    ax.set_title(f"Decision rules over {n_worlds} worlds "
                 f"({agg.get('n_pairs', len(pairs))} pairs)\n"
                 f"H2 holds on {agg.get('pairs_with_no_history_independent_action', 0)}"
                 f"/{agg.get('n_pairs', len(pairs))} pairs", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # ---- (d) V(a, t) for the focus pair ----------------------------------
    ax = axes[1][1]
    styles = ["-", "--"]
    for i, name in enumerate(focus_names):
        for action in worlds[name]["actions"]:
            if action["action"] not in SHOWN:
                continue
            ts = [c["t_s"] for c in action["curve"]]
            vs = [1.0 if c["visible"] else 0.0 for c in action["curve"]]
            ax.step(ts, vs, where="post", lw=1.6, ls=styles[i],
                    color=ACTION_COLORS[action["action"]],
                    label=f"{action['action']} | {'A' if i == 0 else 'B'}"
                          f" rsr={int(action['rsr'])}")
    ax.set_ylim(-0.15, 1.35)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["lost", "visible"])
    ax.set_xlabel("time in the rollout (s)")
    ax.set_title(f"physical V(a, t) for pair {focus['pair_id']} "
                 f"(solid = world A, dashed = world B)", fontsize=10)
    ax.legend(fontsize=6.5, loc="center right", ncol=1, framealpha=0.9)
    ax.grid(alpha=0.25)

    eq_max = agg.get("camera_mean_abs_pixel_diff_max")
    ident_min = agg.get("camera_pixels_identical_fraction_min")
    fig.suptitle(
        "TASK3 Experiment 1 - history necessity: the current observation does not "
        "determine the optimal action\n"
        f"{agg.get('n_pairs', len(pairs))} pairs / {agg.get('n_worlds', len(worlds))} worlds"
        f"   -   observations equivalent in {agg.get('pairs_with_equivalent_observations', '?')}"
        f"/{agg.get('n_pairs', len(pairs))} pairs (max mean pixel diff {eq_max}, "
        f"min identical-pixel fraction {ident_min})\n"
        f"optimal actions opposite in {agg.get('pairs_with_opposite_optimal_actions', '?')}"
        f"/{agg.get('n_pairs', len(pairs))} pairs   -   "
        f"no history-independent action in "
        f"{agg.get('pairs_with_no_history_independent_action', '?')}"
        f"/{agg.get('n_pairs', len(pairs))} pairs", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
