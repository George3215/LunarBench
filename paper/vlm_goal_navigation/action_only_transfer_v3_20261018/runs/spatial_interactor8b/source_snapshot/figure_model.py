"""Render the TASK3 Experiment 3 (learned visibility predictor) figure.

Left    decision-rule RSR, separately for the paired-history worlds (where the
        current observation is provably insufficient) and for the multi-terrain
        scenes (where the model has never seen that terrain family).
Middle  which action each planner picked, world by world, against the true best
        action: the observation-only model collapses to a near-constant choice,
        the model with target memory alternates with the world.
Right   out-of-fold AUC of the predictor per feature set, including the control
        whose memory features are permuted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ACTION_ORDER = ["forward", "back", "left", "right", "left_hold", "right_hold",
                "turn_left", "turn_right", "toward", "heading_lock"]
RULES = ["A_reactive_forward", "B_history_only_toward", "D_initial_heading_lock",
         "learned", "C_oracle_counterfactual"]
RULE_LABELS = {"A_reactive_forward": "A\nforward", "B_history_only_toward": "B\nmemory-aim",
               "D_initial_heading_lock": "D\nheading lock", "learned": "learned",
               "C_oracle_counterfactual": "C\noracle"}
FEATURE_SETS = ["obs_only", "obs_memory_shuffled_control", "obs_memory"]
SET_LABELS = {"obs_only": "obs only\n(no memory)", "obs_memory_shuffled_control":
              "memory shuffled\n(control)", "obs_memory": "obs + memory"}


def main() -> None:
    p = argparse.ArgumentParser(description="Render the TASK3 learned-predictor figure")
    p.add_argument("--model", type=Path,
                   default=Path("tasks/task3_reobservability/generated/visibility_model.json"))
    p.add_argument("--history", type=Path,
                   default=Path("tasks/task3_reobservability/generated/history_necessity.json"))
    p.add_argument("--output", type=Path,
                   default=Path("docs/images/task3_visibility_model.png"))
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = json.loads(args.model.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # ---- (a) decision-rule RSR per scope ---------------------------------
    ax = axes[0]
    scopes = ["history_pairs", "multi_terrain"]
    scope_labels = {"history_pairs": "paired-history worlds (16)",
                    "multi_terrain": "unseen terrain families (5)"}
    width = 0.36
    for si, scope in enumerate(scopes):
        vals = [model["feature_sets"]["obs_memory"]["decisions"][scope][r] or 0.0
                for r in RULES]
        xs = np.arange(len(RULES)) + (si - 0.5) * width
        bars = ax.bar(xs, [v * 100 for v in vals], width,
                      color=["#7f7f7f", "#2ca02c", "#9467bd", "#d62728", "#1f77b4"][0:1] * 0
                      + (["#d62728"] * len(RULES) if scope == "history_pairs"
                         else ["#7f7f7f"] * len(RULES)),
                      alpha=0.9 if scope == "history_pairs" else 0.45,
                      label=scope_labels[scope])
        for x, v in zip(xs, vals):
            ax.text(x, v * 100 + 2, f"{v * 100:.0f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(RULES)))
    ax.set_xticklabels([RULE_LABELS[r] for r in RULES], fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("RSR (%)")
    ax.set_title("Decision rules (solid = paired history,\nfaded = unseen terrain)",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # ---- (b) chosen action per world -------------------------------------
    ax = axes[1]
    hist = json.loads(args.history.read_text(encoding="utf-8"))
    hist_worlds = [w["pair_id"] + side for w in hist["pairs"]
                   for side in ("A", "B")] if False else None
    # Rebuild the world order the model report used, from the paired report.
    decisions_by_world: dict[str, dict] = {}
    report_worlds = hist["worlds"]
    order = [w["pair_id"] for w in hist["pairs"]]
    pairs_of = {p["pair_id"]: p["worlds"] for p in hist["pairs"]}
    ordered = []
    for pid in order:
        for side in pairs_of[pid]:
            ordered.append((pid, side, report_worlds[side]))
    for pid, side, entry in ordered:
        decisions_by_world[side] = entry
    n_hist = len(ordered)

    model_chosen = model.get("chosen_actions", {})
    # The model report lists worlds in the order of the pooled sample table.
    learned_actions = model_chosen.get("obs_memory", [])
    only_actions = model_chosen.get("obs_only", [])
    n_worlds = len(learned_actions)
    xs = np.arange(n_worlds)
    for x in xs:
        truth = None
        world = None
        if x < n_hist:
            pid, side, entry = ordered[x]
            world = side
            truth = entry["optimal_action"]
        ax.axvspan(x - 0.5, x + 0.5,
                   color="#ffe9e9" if x < n_hist else "#eef2f7", zorder=0)
        if truth:
            ax.plot([x], [ACTION_ORDER.index(truth)], marker="o", ms=8,
                    mfc="none", mec="black", mew=1.4, zorder=4)
    for series, marker, colour, label in ((only_actions, "x", "#7f7f7f", "learned, obs only"),
                                          (learned_actions, "o", "#d62728", "learned, obs+memory")):
        if not series:
            continue
        ax.plot(xs, [ACTION_ORDER.index(a) for a in series], marker=marker, ls="none",
                ms=8, color=colour, label=label, zorder=3)
    ax.set_yticks(range(len(ACTION_ORDER)))
    ax.set_yticklabels(ACTION_ORDER, fontsize=7)
    ax.set_xlabel("world (first 16 = paired history, last 5 = multi-terrain)")
    ax.set_xlim(-0.6, n_worlds - 0.4)
    ax.set_title("Chosen action per world\n(open circle = fastest true action; ties are counted as successes)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="x")

    # ---- (c) predictor quality -------------------------------------------
    ax = axes[2]
    aucs = [model["feature_sets"][s]["oof_auc"] or 0.0 for s in FEATURE_SETS]
    bars = ax.bar([SET_LABELS[s] for s in FEATURE_SETS], [a * 100 for a in aucs],
                  color=["#7f7f7f", "#bbbbbb", "#d62728"])
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, a * 100 + 1.5, f"{a:.3f}",
                ha="center", fontsize=9)
    ax.axhline(50, color="black", ls="--", lw=1)
    ax.text(-0.45, 51, "chance", fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("out-of-fold AUC (%)")
    ax.set_title(f"Predicting re-observability\n{model['n_samples']} samples, "
                 f"{model['n_groups']} groups, leave-one-group-out", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    hp = model["feature_sets"]["obs_memory"]["decisions"]["history_pairs"]
    fig.suptitle(
        "TASK3 Experiment 3 - the counterfactual upper bound replaced by a learned predictor\n"
        f"on the {hp['n_worlds']} paired-history worlds: A {hp['A_reactive_forward']:.2f}, "
        f"B {hp['B_history_only_toward']:.2f}, D {hp['D_initial_heading_lock']:.2f}, "
        f"learned-with-memory {hp['learned']:.2f}, oracle {hp['C_oracle_counterfactual']:.2f}",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
