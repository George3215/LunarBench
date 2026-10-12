"""Render the TASK3 Experiment 4 (history horizon) figure.

Left    decision RSR against horizon, per feature block, with the no-history
        target-position ceiling drawn as a line.
Middle  the well-powered curve: out-of-fold AUC against horizon.  This one is
        measured over every hidden sample, not just the 24 decision states.
Right   the mechanism: how far back the last sighting is, and what fraction of
        decision states has a window that actually reaches it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = ["H0", "H1", "H2", "H4", "H8", "H16", "H32", "H64", "Hfull"]
BLOCK_STYLE = {
    "all": ("everything the horizon provides", "#d62728", "-", "o"),
    "memory": ("target memory only", "#1f77b4", "--", "s"),
    "motion": ("recent-motion summary only", "#2ca02c", ":", "^"),
}
H_VALUES = [0, 1, 2, 4, 8, 16, 32, 64, 10 ** 6]


def main() -> None:
    p = argparse.ArgumentParser(description="Render the TASK3 history-horizon figure")
    p.add_argument("--report", type=Path,
                   default=Path("tasks/task3_reobservability/generated/history_horizon.json"))
    p.add_argument("--output", type=Path, default=Path("docs/images/task3_history_horizon.png"))
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads(args.report.read_text(encoding="utf-8"))
    blocks = rep["blocks"]
    x = np.arange(len(LABELS))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # ---- (a) decision RSR -------------------------------------------------
    ax = axes[0]
    for name, (label, colour, ls, marker) in BLOCK_STYLE.items():
        rsr = [r["rsr"] * 100 for r in blocks[name]]
        ax.plot(x, rsr, ls, color=colour, marker=marker, lw=1.8, ms=6, label=label)
    ceiling = blocks["oracle_target"][0]["rsr"] * 100
    ax.axhline(ceiling, color="black", ls="--", lw=1.3,
               label=f"true target position, no history ({ceiling:.0f}%)")
    ax.axhline(50, color="grey", ls=":", lw=1)
    ax.text(0.05, 51.5, "chance", fontsize=8, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylim(30, 105)
    ax.set_ylabel("decision RSR (%)")
    ax.set_xlabel("history horizon $H_k$ (10 Hz samples)")
    ax.set_title(f"Decision quality vs horizon\n{rep['n_worlds'] if 'n_worlds' in rep else blocks['all'][0]['n_worlds']} "
                 f"decisions over {rep['n_groups']} groups -> resolution "
                 f"1/{blocks['all'][0]['n_worlds']} per point", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    # ---- (b) predictor AUC ------------------------------------------------
    ax = axes[1]
    for name, (label, colour, ls, marker) in BLOCK_STYLE.items():
        auc = [r["oof_auc"] * 100 for r in blocks[name]]
        ax.plot(x, auc, ls, color=colour, marker=marker, lw=1.8, ms=6, label=label)
    base = blocks["all"][0]["oof_auc"] * 100
    ax.axhline(base, color="grey", ls=":", lw=1)
    ax.annotate("", xy=(8, blocks["all"][-1]["oof_auc"] * 100), xytext=(8, base),
                arrowprops=dict(arrowstyle="<->", color="grey", lw=1))
    ax.text(7.6, (base + blocks["all"][-1]["oof_auc"] * 100) / 2,
            f"+{blocks['all'][-1]['oof_auc'] * 100 - base:.1f} AUC\nover H0",
            fontsize=8, ha="right", color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel("out-of-fold AUC (%)")
    ax.set_xlabel("history horizon $H_k$")
    ax.set_title(f"Well-powered curve: every hidden sample\n"
                 f"{rep['n_samples']} samples, leave-one-group-out", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    # ---- (c) mechanism ----------------------------------------------------
    ax = axes[2]
    gaps = rep["sighting_gap_steps"]
    finite = [v for v in gaps.values() if v is not None]
    never = sum(1 for v in gaps.values() if v is None)
    ax.hist(finite, bins=np.arange(0, 100, 5), color="#9467bd", alpha=0.75,
            label=f"steps since the last sighting\n(median {int(np.median(finite))}, "
                  f"max {max(finite)}; {never} never seen)")
    ax.axvline(64, color="#d62728", ls="--", lw=1.6, label="H64 = 6.4 s")
    ax.set_xlabel("gap between the last sighting and the decision state (10 Hz steps)")
    ax.set_ylabel("decision states")
    ax2 = ax.twinx()
    avail = [r["states_with_memory"] for r in blocks["all"]]
    ax2.plot(x, avail, color="#2ca02c", marker="o", lw=1.8, ms=6,
             label="states whose window reaches a sighting")
    ax2.set_ylabel("states with memory available", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, len(gaps) + 1)
    ax.set_title("Mechanism: the window has to reach the last sighting", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle("TASK3 Experiment 4 - history horizon: a 0.4 s window already recovers the "
                 "decision information, and the full trajectory adds nothing", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
