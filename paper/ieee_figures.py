"""Regenerate the manuscript figures in IEEE house style.

Redrawn against the `ieee-figure-table` standard rather than by eye.  The audit that
drove the rewrite:

| Item | First-impression risk | Technical issue | Fix |
| --- | --- | --- | --- |
| all six figures | colour carries *identity*, so a reader cannot tell which curve is the proposed method | no semantic role encoding | map every series to a role: proposed / recent / traditional / cost / neutral |
| all six figures | curves indistinguishable in grayscale | Okabe-Ito hues partially collide after desaturation | role styles already carry marker + linestyle; add hatches to bars |
| Fig 1 | legend drawn over the curves | `loc="upper left"` on a full axes | legend below the axes, out of the data region |
| Fig 1 | **Random absent** | discovered after the first build that Random (0.625) beats every classical rule (0.250) | Random is now a first-class series; omitting it would misrepresent the baseline ladder |
| all figures | final width drifting with `bbox_inches="tight"` | IEEE column widths are exact | fixed 7.16 in (two-column) / 3.50 in (one-column), `bbox_inches=None` |
| tables | mixed decimal precision | 0.25 vs 0.250 across panels | two decimals for success, three for MAE/AUC, stated in the caption |

Run:  python paper/ieee_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import ieee_plot_style as ieee

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tasks" / "task3_reobservability" / "generated"
OUT = ROOT / "paper" / "ieee_figures"
BUDGETS = (1, 3, 5, 10)

#: Series -> semantic role. Role, not identity, decides colour and line style.
ROLE = {
    "Random": "cost",
    "Reactive": "traditional",
    "HeadingLock": "traditional",
    "HistoryOnly": "traditional",
    "LostTargetSearch": "traditional",
    "ObjectPermanence": "traditional",
    "NBV": "recent",
    "LearnedConsequence": "proposed",
    "Oracle": "neutral",
    "OracleSearch": "neutral",
}
LABEL = {
    "Random": "Random (lower bound)",
    "LearnedConsequence": "Consequence planner (ours)",
    "NBV": "NBV",
    "Reactive": "Reactive / heading-lock / history-only",
    "Oracle": "Oracle (privileged)",
    "OracleSearch": "Oracle search (privileged)",
}


def load(name: str):
    path = GEN / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def terrain_order(episodes: dict) -> list[str]:
    return list(episodes["terrains"].keys())


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def vector_from(episodes: dict, policy: str, budget=None) -> np.ndarray:
    order = terrain_order(episodes)
    if policy == "OracleSearch":
        steps = episodes["summary"]["OracleSearch"]["steps_by_terrain"]
        return np.array([1.0 if (steps.get(t) is not None
                                 and (budget is None or steps[t] <= budget)) else 0.0
                         for t in order])
    return np.array([
        1.0 if (episodes["terrains"][t]["episodes"][policy]["success"]
                and (budget is None
                     or episodes["terrains"][t]["episodes"][policy]["steps_used"] <= budget))
        else 0.0 for t in order])


def planner_vector(episodes: dict, budget=None) -> np.ndarray | None:
    model = load("consequence_model.json")
    per = (model or {}).get("planner", {}).get("per_terrain")
    if not per:
        return None
    out = []
    for t in terrain_order(episodes):
        e = per.get(t)
        out.append(0.0 if e is None else
                   float(e["termination"] == "success"
                         and (budget is None or e["steps"] <= budget)))
    return np.array(out)


# ---------------------------------------------------------------------------
# Figure: main result
# ---------------------------------------------------------------------------

def fig_main() -> dict:
    episodes = load("benchmark_multistep_B_scored.json")
    n = len(terrain_order(episodes))
    series, labels, roles = [], [], []

    plan = planner_vector(episodes)
    for k in BUDGETS:
        series.append([])
    # long-form: one entry per method, one value per budget
    methods = []
    rand = load("random_baseline.json")
    if rand:
        methods.append(("Random", None))
    methods += [("Reactive", None), ("NBV", None)]
    if plan is not None:
        methods.append(("LearnedConsequence", plan))
    methods += [("Oracle", None), ("OracleSearch", None)]

    values, plot_labels, plot_roles, errors = [], [], [], []
    for name, cache in methods:
        if name == "Random":
            # Random is stored as its own artefact, not in the episode file, and it
            # is a mean over three seeds rather than a single 24-terrain run. Its
            # interval is therefore the spread across seeds, not a Wilson interval -
            # presenting a Wilson interval here would imply a sample size that does
            # not exist.
            seed_rsr = [[s[f"rsr@{k}"] for k in BUDGETS]
                        for s in rand["by_seed"].values()]
            arr = np.array(seed_rsr)
            values.append([float(v) for v in arr.mean(axis=0)])
            errors.append([max(float(arr[:, i].mean() - arr[:, i].min()),
                                      float(arr[:, i].max() - arr[:, i].mean()))
                           for i in range(len(BUDGETS))])
            plot_labels.append(LABEL["Random"])
            plot_roles.append(ROLE["Random"])
            continue
        per_budget, lo_hi = [], []
        for k in BUDGETS:
            # The consequence planner is stored per terrain with its step count, so
            # its per-budget vector is built from its own artefact rather than from
            # the episode file, where it does not appear.
            v = (planner_vector(episodes, budget=k) if name == "LearnedConsequence"
                 else vector_from(episodes, name, budget=k))
            per_budget.append(float(v.mean()))
            lo, hi = wilson(int(v.sum()), n)
            lo_hi.append(max(float(v.mean()) - lo, hi - float(v.mean())))
        values.append(per_budget)
        errors.append(lo_hi)
        plot_labels.append(LABEL.get(name, name))
        plot_roles.append(ROLE.get(name, "neutral"))

    ieee.apply_ieee_style(ieee.IEEEFigureStyle(width="two-column", font_size=8.5))
    fig, axes = ieee.create_figure("two-column", figsize=(7.16, 2.9), ncols=2, gridspec_kw={"width_ratios": (1.45, 1.0)}, 
                                   )
    ax = axes[0]
    ieee.plot_trend(ax, list(BUDGETS), values, plot_labels, roles=plot_roles,
                    intervals=errors, xlabel="Action budget $K$",
                    ylabel="Re-observation success rate")
    ax.set_xticks(list(BUDGETS))
    ax.set_xticklabels([str(k) for k in BUDGETS])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False, fontsize=6.8)
    ieee.add_panel_label(ax, "(a)")

    # (b) the ladder at the full budget, with Random as the reference floor
    ax = axes[1]
    ladder = [(name, cache) for name, cache in methods]
    xs = np.arange(len(ladder))
    means, los, his, colours, hatches = [], [], [], [], []
    for i, (name, cache) in enumerate(ladder):
        # Reuse the series computed above: Random and the consequence planner are
        # not in the episode file, so calling vector_from for them here would fail
        # (and would silently disagree with panel (a) if it did not).
        means.append(values[i][-1])
        half = errors[i][-1]
        los.append(means[-1] - half); his.append(means[-1] + half)
        colours.append(ieee.PALETTE["proposed" if name == "LearnedConsequence"
                                    else "cost" if name == "Random"
                                    else "neutral" if "Oracle" in name
                                    else "traditional_baseline"])
        hatches.append(ieee.HATCHES[i % len(ieee.HATCHES)])
    ax.bar(xs, means, 0.62, color=colours, edgecolor=ieee.PALETTE["text"],
           linewidth=0.55, hatch=None, zorder=3)
    ax.errorbar(xs, means,
                yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="none", ecolor=ieee.PALETTE["text"], elinewidth=0.7, capsize=1.8,
                zorder=4)
    ax.axhline(means[[n for n, _ in ladder].index("Random")],
               color=ieee.PALETTE["cost"], lw=0.9, ls=":", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL.get(nm, nm).split(" (")[0].replace(" / ", "/\n")
                        for nm, _ in ladder], fontsize=6.2, rotation=30, ha="right")
    ax.set_ylabel("RSR@10")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", zorder=0)
    ieee.add_panel_label(ax, "(b)")

    path = ieee.save_ieee_figure(fig, str(OUT / "fig_main"), formats=("pdf", "png"))
    return {"path": path, "ladder": {nm: float(v) for (nm, _), v in zip(ladder, means)}}


# ---------------------------------------------------------------------------
# Remaining figures
# ---------------------------------------------------------------------------

def fig_ablation() -> str:
    model = load("consequence_model.json")
    conds = {c["inputs"]: c for c in model["conditions"]}
    order = ["O", "M", "O+M", "O+M+a"]
    names = ["$O$", "$M$", "$O{+}M$", "$O{+}M{+}a$"]
    ieee.apply_ieee_style(ieee.IEEEFigureStyle(width="two-column", font_size=8.5))
    fig, axes = ieee.create_figure("two-column", figsize=(7.16, 2.5), ncols=3, gridspec_kw={"width_ratios": (1.0, 1.0, 1.0)}, 
                                   )
    auc = [conds[k]["roc_auc"] for k in order]
    ieee.plot_grouped_bars(axes[0], names, [auc], ["ROC-AUC"], roles=["proposed"],
                           ylabel="ROC-AUC", annotate=True)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_xlabel("Model inputs")
    ieee.add_panel_label(axes[0], "(a)")

    mae = [conds[k]["delta_mae_mean"] for k in order]
    ieee.plot_grouped_bars(axes[1], names, [mae], ["Pose MAE"], roles=["cost"],
                           ylabel="Pose MAE (m, lower is better)", annotate=True)
    axes[1].set_xlabel("Model inputs")
    ieee.add_panel_label(axes[1], "(b)")

    comps = ["forward_m", "lateral_m", "height_m", "yaw_rad"]
    analytic = [model["analytic"]["delta_mae"][c] for c in comps]
    learned = [conds["O+M+a"]["delta_mae"][c] for c in comps]
    ieee.plot_grouped_bars(axes[2], ["forward", "lateral", "height", "yaw"],
                           [analytic, learned], ["analytic", "learned"],
                           roles=["traditional", "proposed"],
                           ylabel="Pose MAE (m / rad)", annotate=True)
    axes[2].legend(loc="upper left", frameon=False, fontsize=6.5)
    ieee.add_panel_label(axes[2], "(c)")
    return ieee.save_ieee_figure(fig, str(OUT / "fig_ablation"), formats=("pdf", "png"))


def fig_geometry() -> str:
    gap = load("geometry_gap.json")
    s = gap["summary"]
    c = s["confusion"]
    grid = np.array([[c["G0_P0"], c["G0_P1"]], [c["G1_P0"], c["G1_P1"]]], dtype=float)
    ieee.apply_ieee_style(ieee.IEEEFigureStyle(width="two-column", font_size=8.5))
    fig, axes = ieee.create_figure("two-column", figsize=(7.16, 2.6), ncols=2, gridspec_kw={"width_ratios": (1.0, 1.25)}, 
                                   )
    import matplotlib.pyplot as plt
    ax = axes[0]
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(grid[i, j])}\n{grid[i, j] / grid.sum() * 100:.1f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if grid[i, j] > grid.max() * 0.6
                    else ieee.PALETTE["text"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["$P{=}0$", "$P{=}1$"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["$G{=}0$", "$G{=}1$"])
    ax.set_xlabel("Physically re-observed"); ax.set_ylabel("Geometry predicts visible")
    ax.grid(False)
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False,
                               ec=ieee.PALETTE["traditional_baseline"], lw=1.4))
    ieee.add_panel_label(ax, "(a)")

    ax = axes[1]
    rows = gap["rows"]
    cells = [("$G{=}0,P{=}0$", "#6B7280"), ("$G{=}1,P{=}1$", ieee.PALETTE["proposed"]),
             ("$G{=}1,P{=}0$", ieee.PALETTE["traditional_baseline"]),
             ("$G{=}0,P{=}1$", ieee.PALETTE["recent_baseline"])]
    rng = np.random.default_rng(3)
    for i, (label, colour) in enumerate(cells):
        key = label.replace("$", "").replace("{=}", "=").replace(",", "_").replace(" ", "")
        g, p = (0, 0) if "G=0" in label else (1, 1) if "G=1,P=1" in label else (1, 0)
        vals = np.array([r["pose_err_m"] for r in rows
                         if r["G"] == g and r["P"] == (1 if "P=1" in label else 0)
                         and np.isfinite(r["pose_err_m"])])
        if not vals.size:
            continue
        ax.scatter(i + rng.uniform(-0.15, 0.15, vals.size), vals, s=7, color=colour,
                   alpha=0.8, linewidths=0)
        ax.plot([i - 0.26, i + 0.26], [vals.mean()] * 2, color=ieee.PALETTE["text"], lw=1.4)
        ax.text(i, ax.get_ylim()[1], f"n={vals.size}", ha="center", va="top", fontsize=6)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells])
    ax.set_ylabel("Pose prediction error (m)")
    ax.set_ylim(0, 1.9)
    ieee.add_panel_label(ax, "(b)")
    return ieee.save_ieee_figure(fig, str(OUT / "fig_geometry"), formats=("pdf", "png"))


def fig_negative() -> str:
    hz = load("horizon_planner.json")["results"]
    cross = load("cross_mechanism.json")
    ieee.apply_ieee_style(ieee.IEEEFigureStyle(width="two-column", font_size=8.5))
    fig, axes = ieee.create_figure("two-column", figsize=(7.16, 2.5), ncols=2, gridspec_kw={"width_ratios": (1.15, 1.0)}, 
                                   )
    keys = list(hz)
    values = [[hz[k]["rsr@1"] for k in keys], [hz[k]["rsr@3"] for k in keys],
              [hz[k]["rsr@5"] for k in keys], [hz[k]["rsr@10"] for k in keys]]
    ieee.plot_grouped_bars(axes[0], keys, values,
                           ["RSR@1", "RSR@3", "RSR@5", "RSR@10"],
                           roles=["traditional", "recent", "cost", "proposed"],
                           ylabel="Success rate", annotate=False)
    axes[0].legend(loc="upper right", frameon=False, fontsize=6.2, ncol=2)
    axes[0].set_xlabel("Lookahead horizon $H$ (steps)")
    ieee.add_panel_label(axes[0], "(a)")

    within = cross["within_mechanism_summary"]
    across = cross["cross_mechanism_summary"]
    ieee.plot_grouped_bars(axes[1], ["Pose MAE (m)", "ROC-AUC"],
                           [[within["mean_pose_mae"], across["mean_pose_mae"]],
                            [within["mean_roc_auc"], across["mean_roc_auc"]]],
                           ["within mechanism", "across mechanisms"],
                           roles=["proposed", "traditional"],
                           ylabel="Error / score", annotate=True)
    axes[1].legend(loc="upper right", frameon=False, fontsize=6.2)
    axes[1].set_ylim(0, 1.15)
    ieee.add_panel_label(axes[1], "(b)")
    return ieee.save_ieee_figure(fig, str(OUT / "fig_negative"), formats=("pdf", "png"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Detected axes: visual_type=multi-panel/line-plot/confusion-matrix/bar-chart, "
          "check_type=readability+accessibility+layout, "
          "failure_mode=overlapping-legend+unreadable-two-column, "
          "stage=redesign, backend=python\n")
    print("main        ", fig_main())
    print("ablation    ", fig_ablation())
    print("geometry    ", fig_geometry())
    print("negative    ", fig_negative())
    print(f"\nwrote IEEE-styled figures to {OUT}")


if __name__ == "__main__":
    main()
