"""Publication figures and statistics for the TASK3 experiments.

The diagnostic figures elsewhere in the repo explain *how* an experiment works.
This module produces the *statistical* figures a paper needs, from the recorded
JSON artefacts only, with three rules applied throughout:

1.  **Every proportion carries an interval.**  Binomial proportions with small
    $n$ get a Wilson score interval, not a normal approximation, which is badly
    wrong near 0 and 1 - exactly where several of these results sit.
2.  **Every comparison is paired.**  All policies run on the same terrains, so a
    difference in RSR is tested by resampling *terrains* with replacement and
    looking at the distribution of the paired difference.  An unpaired test would
    ignore that the terrain is a shared, high-variance factor.
3.  **Vector output.**  PDF for the paper, PNG for inspection, and the numbers
    also written as CSV and LaTeX so a table cannot drift from a figure.

    python tasks/task3_reobservability/paper_figures.py

Writes into ``paper/figures/`` and ``paper/tables/``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "tasks" / "task3_reobservability" / "generated"
FIGDIR = ROOT / "paper" / "figures"
TABDIR = ROOT / "paper" / "tables"

# Colour-blind-safe qualitative palette (Okabe-Ito), one colour per policy.
POLICY_COLOR = {
    "Reactive": "#999999",
    "HeadingLock": "#E69F00",
    "HistoryOnly": "#56B4E9",
    "LostTargetSearch": "#009E73",
    "ObjectPermanence": "#0072B2",
    "NBV": "#D55E00",
    "Oracle": "#CC79A7",
    "OracleSearch": "#000000",
}
POLICY_MARKER = {
    "Reactive": "o", "HeadingLock": "s", "HistoryOnly": "^", "LostTargetSearch": "v",
    "ObjectPermanence": "D", "NBV": "P", "Oracle": "X", "OracleSearch": "*",
}
DEFAULT_COLOR = "#444444"
BUDGETS = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because several of these results sit
    at or near 0 % / 100 %, where the normal interval produces bounds outside
    [0, 1] and understates the uncertainty.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 20000,
                     seed: int = 0) -> dict:
    """Paired bootstrap of ``mean(a) - mean(b)`` over shared units.

    Both vectors are indexed by the same terrain, so resampling indices resamples
    terrains, which is the unit of variation.  Returns the point difference, a
    95 % percentile interval, and the fraction of resamples in which the
    difference is not positive (a one-sided bootstrap p-value analogue).

    Two degenerate cases are labelled rather than dressed up as statistics:

    * identical inputs (every terrain agrees) - there is no variance to
      bootstrap, so the interval is [0, 0] and reporting a p-value would be
      meaningless.  This happens for real here: five of the baselines make
      outcome-identical choices on every terrain.
    * an all-zero difference distribution - the p-value is reported as a bound
      ``1 / n_resamples`` rather than as an exact 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs equal-length vectors")
    diff = float(a.mean() - b.mean())
    n = a.size
    if n == 0:
        return {"diff": None, "lo": None, "hi": None, "p_not_positive": None,
                "n": 0, "identical": False}
    if np.array_equal(a, b):
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_not_positive": None,
                "n": int(n), "identical": True}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    p = float((deltas <= 0).mean())
    return {
        "diff": round(diff, 4),
        "lo": round(float(np.percentile(deltas, 2.5)), 4),
        "hi": round(float(np.percentile(deltas, 97.5)), 4),
        "p_not_positive": round(p, 6) if p > 0 else f"<{1 / n_resamples:.0e}",
        "n": int(n),
        "identical": False,
    }


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

def setup_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print(f"  wrote paper/figures/{name}.pdf / .png")


def load(name: str) -> dict:
    path = GENERATED / name
    if not path.exists():
        raise SystemExit(f"missing artefact {path}; run the experiment first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------

def policy_vectors(episodes: dict, policy: str, budget: int | None = None) -> np.ndarray:
    """Per-terrain success vector for one policy, in the fixed terrain order.

    ``OracleSearch`` is a special case: it comes from the solvability audit rather
    than from the episode run, so it has no per-terrain episode entry.  Its
    per-terrain step counts are carried in its summary row and are expanded here,
    which keeps every consumer working from the same terrain order.
    """
    order = terrain_order(episodes)
    if policy == "OracleSearch":
        steps = episodes["summary"]["OracleSearch"]["steps_by_terrain"]
        return np.array([
            1.0 if (steps.get(t) is not None and (budget is None or steps[t] <= budget))
            else 0.0 for t in order])
    out = []
    for terrain in order:
        entry = episodes["terrains"][terrain]["episodes"][policy]
        if budget is None:
            out.append(float(bool(entry["success"])))
        else:
            out.append(float(bool(entry["success"]) and entry["steps_used"] <= budget))
    return np.array(out)


def planner_vectors(episodes: dict, budget: int | None = None) -> np.ndarray | None:
    """Per-terrain success for the E4 learned consequence planner, if it has run.

    Read from ``consequence_model.json`` rather than merged into the episode file,
    so the benchmark artefact stays exactly what the benchmark produced.
    """
    path = GENERATED / "consequence_model.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_terrain = payload.get("planner", {}).get("per_terrain")
    if not per_terrain:
        return None
    out = []
    for terrain in terrain_order(episodes):
        entry = per_terrain.get(terrain)
        if entry is None:
            out.append(0.0)
            continue
        ok = entry["termination"] == "success"
        out.append(float(ok and (budget is None or entry["steps"] <= budget)))
    return np.array(out)


def terrain_order(episodes: dict) -> list[str]:
    return list(episodes["terrains"].keys())


def group_of(terrain: str) -> str:
    return "paired-history" if terrain.startswith(("hist", "mem")) else "multi-terrain"


# ---------------------------------------------------------------------------
# Figure 1: multi-step benchmark
# ---------------------------------------------------------------------------

def fig_benchmark(episodes: dict) -> dict:
    import matplotlib.pyplot as plt

    policies = list(episodes["summary"])
    order = terrain_order(episodes)
    n = len(order)
    learned10 = None
    learned10 = planner_vectors(episodes, budget=10)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) RSR@K curves with Wilson intervals at each K
    ax = axes[0]
    stats: dict = {}
    for policy in policies:
        means, los, his = [], [], []
        for k in BUDGETS:
            v = policy_vectors(episodes, policy, budget=k)
            lo, hi = wilson(int(v.sum()), n)
            means.append(v.mean())
            los.append(lo)
            his.append(hi)
        colour = POLICY_COLOR.get(policy, DEFAULT_COLOR)
        ls = "--" if policy in ("Oracle", "OracleSearch") else "-"
        ax.errorbar(BUDGETS, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                    color=colour, marker=POLICY_MARKER.get(policy, "o"), ls=ls,
                    lw=1.3, ms=4, capsize=2, elinewidth=0.8, label=policy)
        stats[policy] = {"rsr@10": means[-1], "ci_lo": los[-1], "ci_hi": his[-1]}
    learned = planner_vectors(episodes)
    if learned is not None:
        means, los, his = [], [], []
        for k in BUDGETS:
            v = planner_vectors(episodes, budget=k)
            lo, hi = wilson(int(v.sum()), n)
            means.append(v.mean()); los.append(lo); his.append(hi)
        ax.errorbar(BUDGETS, means,
                    yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                    color="#000000", marker="*", ls="-", lw=1.8, ms=8, capsize=2,
                    elinewidth=0.8, label="LearnedConsequence (E4)")
        stats["LearnedConsequence"] = {"rsr@10": means[-1], "ci_lo": los[-1],
                                       "ci_hi": his[-1]}
    ax.set_xscale("log")
    ax.set_xticks(list(BUDGETS))
    ax.set_xticklabels([str(k) for k in BUDGETS])
    ax.set_xlabel("action budget $K$ (steps)")
    ax.set_ylabel("RSR@$K$")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title(f"(a) Re-observation success vs budget\n"
                 f"{n} terrains; Wilson 95 % intervals", loc="left")
    # The five rule-based baselines coincide exactly at every budget - that IS the
    # result - so one legend entry is added for the group rather than five
    # overlapping ones.
    handles, labels_ = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels_)
            if l not in ("Reactive", "HeadingLock", "HistoryOnly", "LostTargetSearch")]
    ax.legend([h for h, _ in keep], [l for _, l in keep],
              loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
              columnspacing=1.0, handlelength=1.6)
    ax.annotate("five rule-based baselines coincide\n"
                "(Reactive, HeadingLock, HistoryOnly,\nLostTargetSearch, ObjectPermanence)",
                xy=(3, 0.25), xytext=(1.1, 0.47), fontsize=6.2, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))

    # (b) per-group split at the full budget, as a dot plot so that the many
    #     zero values stay readable
    ax = axes[1]
    groups = ["multi-terrain", "paired-history"]
    group_colour = {"multi-terrain": "#0072B2", "paired-history": "#D55E00"}
    names = list(policies)
    vectors = {p: policy_vectors(episodes, p) for p in policies}
    if learned10 is not None:
        names.append("LearnedConsequence (E4)")
        vectors["LearnedConsequence (E4)"] = learned10
    y = np.arange(len(names))[::-1]
    for g in groups:
        mask = np.array([group_of(t) == g for t in order])
        xs, los, his = [], [], []
        for policy in names:
            sub = vectors[policy][mask]
            lo, hi = wilson(int(sub.sum()), sub.size)
            xs.append(sub.mean()); los.append(lo); his.append(hi)
        ax.hlines(y, los, his, color=group_colour[g], lw=1.2, alpha=0.85)
        ax.plot(xs, y, "o", ms=4.5, color=group_colour[g],
                label=f"{g} (n={int(mask.sum())})")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("RSR@$K{=}10$")
    ax.set_title("(b) Where the discrimination is\nWilson 95 % intervals", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              columnspacing=1.0, handlelength=1.2)

    fig.tight_layout()
    save(fig, "fig1_benchmark")
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# Figure 2: paired-world evidence (H1 and H2)
# ---------------------------------------------------------------------------

def fig_pairs(history: dict, memory: dict) -> dict:
    import matplotlib.pyplot as plt

    def frac(count, total):
        lo, hi = wilson(count, total)
        return count / total, lo, hi

    h_pairs = history["pairs"]
    h_n = len(h_pairs)
    h_eq = sum(1 for p in h_pairs if p["verdict"]["observations_equivalent"])
    h_opp = sum(1 for p in h_pairs if p["verdict"]["optimal_actions_differ"])
    h_no = sum(1 for p in h_pairs if not p["verdict"]["history_independent_action_exists"])

    m_agg = memory["aggregate"]
    m_n = m_agg["n_pairs"]
    m_rows = memory["pairs"]
    m_eq = m_agg["pairs_with_identical_observations"]
    m_mem = m_agg["pairs_with_identical_memory"]
    m_opp = m_agg["pairs_with_different_optimal_actions"]
    m_dis = m_agg["pairs_with_no_common_successful_action"]

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    rows = [
        ("H1  $O_A{=}O_B$", *frac(h_eq, h_n)),
        ("H1  $a_A^*{\\neq}a_B^*$", *frac(h_opp, h_n)),
        ("H1  no common action", *frac(h_no, h_n)),
        ("H2  $O_A{=}O_B$", *frac(m_eq, m_n)),
        ("H2  $M_A{=}M_B$", *frac(m_mem, m_n)),
        ("H2  $a_A^*{\\neq}a_B^*$", *frac(m_opp, m_n)),
        ("H2  no common action", *frac(m_dis, m_n)),
    ]
    y = np.arange(len(rows))[::-1]
    colours = ["#0072B2"] * 3 + ["#D55E00"] * 4
    for yi, (label, mean, lo, hi), colour in zip(y, rows, colours):
        ax.plot([lo, hi], [yi, yi], color=colour, lw=1.4, solid_capstyle="butt")
        ax.plot([mean], [yi], marker="o", ms=5, color=colour)
        ax.text(1.04, yi, f"{int(round(mean * (h_n if label.startswith('H1') else m_n)))}"
                          f"/{h_n if label.startswith('H1') else m_n}",
                va="center", fontsize=7, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axhline(3.5, color="#bbbbbb", lw=0.8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fraction of pairs")
    ax.set_title("Paired worlds: what the construction establishes\n"
                 f"Wilson 95% intervals; H1 n={h_n}, H2 n={m_n}",
                 loc="left")
    fig.tight_layout()
    save(fig, "fig2_paired_worlds")
    plt.close(fig)
    return {"H1": {"n": h_n, "obs_eq": h_eq, "opposite": h_opp, "no_common": h_no},
            "H2": {"n": m_n, "obs_eq": m_eq, "mem_eq": m_mem, "opposite": m_opp,
                   "disjoint": m_dis}}


# ---------------------------------------------------------------------------
# Figure 3: geometry vs physical re-observability (H3)
# ---------------------------------------------------------------------------

def fig_geometry(gap: dict) -> dict:
    import matplotlib.pyplot as plt

    rows = gap["rows"]
    s = gap["summary"]
    c = s["confusion"]
    grid = np.array([[c["G0_P0"], c["G0_P1"]], [c["G1_P0"], c["G1_P1"]]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                            gridspec_kw={"width_ratios": [1.0, 1.25]})

    # (a) confusion matrix
    ax = axes[0]
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max())
    for i in range(2):
        for j in range(2):
            val = int(grid[i, j])
            frac = val / grid.sum()
            ax.text(j, i, f"{val}\n({frac * 100:.1f} %)", ha="center", va="center",
                    fontsize=8, color="white" if grid[i, j] > grid.max() * 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["$P{=}0$", "$P{=}1$"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["$G{=}0$", "$G{=}1$"])
    ax.set_xlabel("physical re-observation")
    ax.set_ylabel("geometric prediction")
    ax.grid(False)
    # outline the failure cell
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec="#D55E00", lw=1.6))
    ax.set_title("(a) $G$ vs $P$ over %d (state, action) pairs\n"
                 "orange: geometry endorses, physics fails" % s["n"], loc="left")

    # (b) pose prediction error by agreement, with paired bootstrap
    ax = axes[1]
    cells = [("G0,P0", [r for r in rows if r["G"] == 0 and r["P"] == 0], "#bbbbbb"),
             ("G1,P1", [r for r in rows if r["G"] == 1 and r["P"] == 1], "#0072B2"),
             ("G1,P0", [r for r in rows if r["G"] == 1 and r["P"] == 0], "#D55E00"),
             ("G0,P1", [r for r in rows if r["G"] == 0 and r["P"] == 1], "#009E73")]
    rng = np.random.default_rng(3)
    for i, (label, subset, colour) in enumerate(cells):
        vals = np.array([r["pose_err_m"] for r in subset
                         if np.isfinite(r["pose_err_m"])])
        if vals.size == 0:
            continue
        x = i + rng.uniform(-0.16, 0.16, size=vals.size)
        ax.scatter(x, vals, s=9, color=colour, alpha=0.75, linewidths=0)
        mean = float(vals.mean())
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.plot([i - 0.26, i + 0.26], [mean, mean], color="black", lw=1.4)
        ax.plot([i, i], [lo, hi], color="black", lw=0.9)
        ax.text(i, ax.get_ylim()[1] * 0.97, f"n={vals.size}", ha="center",
                va="top", fontsize=6.5, color="#555555")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells])
    ax.set_ylabel("pose prediction error (m)")

    # The contrast that carries the claim: does the pose error really separate the
    # failed geometric predictions from the successful ones?  Bootstrapped over
    # pairs, which is the unit of variation.
    gap_vals = np.array([r["pose_err_m"] for r in rows
                         if r["G"] == 1 and r["P"] == 0 and np.isfinite(r["pose_err_m"])])
    hit_vals = np.array([r["pose_err_m"] for r in rows
                         if r["G"] == 1 and r["P"] == 1 and np.isfinite(r["pose_err_m"])])
    test = None
    if gap_vals.size and hit_vals.size:
        # unpaired here: the two cells contain different (state, action) pairs, so
        # the bootstrap resamples within each cell rather than pairing them
        rng2 = np.random.default_rng(11)
        deltas = (rng2.choice(gap_vals, (20000, gap_vals.size)).mean(axis=1)
                  - rng2.choice(hit_vals, (20000, hit_vals.size)).mean(axis=1))
        test = {"diff": round(float(gap_vals.mean() - hit_vals.mean()), 3),
                "lo": round(float(np.percentile(deltas, 2.5)), 3),
                "hi": round(float(np.percentile(deltas, 97.5)), 3),
                "p_not_positive": round(float((deltas <= 0).mean()), 4)}
        ax.annotate(f"$\\Delta$ = {test['diff']:+.2f} m "
                    f"[{test['lo']:+.2f}, {test['hi']:+.2f}]\n"
                    f"bootstrap $p$ = {max(test['p_not_positive'], 1 / 20000):.4f}",
                    xy=(2, float(np.mean([r['pose_err_m'] for r in rows
                                          if r['G'] == 1 and r['P'] == 0
                                          and np.isfinite(r['pose_err_m'])]))),
                    xytext=(1.3, 1.62), fontsize=6.5, color="#333333",
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7))
    ax.set_title("(b) The gap is pose prediction, not the oracle\n"
                 "oracle at executed pose matches $P$ in %.1f %% of pairs"
                 % (s["G_actual_equals_P_rate"] * 100), loc="left")

    fig.tight_layout()
    save(fig, "fig3_geometry_gap")
    plt.close(fig)
    return {**s, "pose_error_test_G1P0_vs_G1P1": test}


# ---------------------------------------------------------------------------
# Figure 4: history horizon
# ---------------------------------------------------------------------------

def fig_horizon(horizon: dict) -> dict:
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in horizon["blocks"]["all"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    n = horizon["blocks"]["all"][0]["n_worlds"]
    for block, colour, ls in (("all", "#000000", "-"),
                              ("memory", "#0072B2", "--"),
                              ("motion", "#009E73", ":")):
        series = horizon["blocks"][block]
        means = [r["rsr"] for r in series]
        los, his = [], []
        for r in series:
            lo, hi = wilson(int(round(r["rsr"] * r["n_worlds"])), r["n_worlds"])
            los.append(lo); his.append(hi)
        ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                    color=colour, ls=ls, marker="o", ms=3.5, lw=1.3, capsize=2,
                    elinewidth=0.7, label=block)
    ceiling = horizon["blocks"]["oracle_target"][0]["rsr"]
    ax.axhline(ceiling, color="#D55E00", ls="-.", lw=1.2,
               label=f"true target, no history ({ceiling:.2f})")
    ax.axhline(0.5, color="#999999", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("history horizon $H_k$ at 10 Hz")
    ax.set_ylabel("decision RSR")
    ax.set_title(f"(a) Decision quality saturates immediately\n"
                 f"{n} decision states; Wilson 95 % intervals", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              columnspacing=1.0, handlelength=1.6)

    ax = axes[1]
    for block, colour, ls in (("all", "#000000", "-"),
                              ("memory", "#0072B2", "--"),
                              ("motion", "#009E73", ":")):
        series = horizon["blocks"][block]
        ax.plot(x, [r["oof_auc"] for r in series], color=colour, ls=ls,
                marker="o", ms=3.5, lw=1.3, label=block)
    ax.axhline(0.5, color="#999999", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("history horizon $H_k$")
    ax.set_ylabel("out-of-fold AUC")
    ax.set_title(f"(b) Predictor information saturates at H64\n"
                 f"{horizon['n_samples']} samples, leave-one-group-out", loc="left")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6)

    fig.tight_layout()
    save(fig, "fig4_history_horizon")
    plt.close(fig)
    return {"n_states": n}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def write_tables(episodes: dict, stats: dict) -> None:
    TABDIR.mkdir(parents=True, exist_ok=True)
    order = terrain_order(episodes)
    n = len(order)
    policies = list(episodes["summary"])

    lines = ["policy,n,RSR@1,RSR@3,RSR@5,RSR@10,RSR@10_lo,RSR@10_hi"]
    for policy in policies:
        v10 = policy_vectors(episodes, policy, budget=10)
        lo, hi = wilson(int(v10.sum()), n)
        row = [policy, n]
        for k in BUDGETS:
            row.append(f"{policy_vectors(episodes, policy, budget=k).mean():.3f}")
        row += [f"{lo:.3f}", f"{hi:.3f}"]
        lines.append(",".join(str(x) for x in row))
    (TABDIR / "table_benchmark.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote paper/tables/table_benchmark.csv")

    # paired comparisons against the strongest non-privileged baseline
    rows = []
    identical_to = {}
    reference = "HistoryOnly"
    for policy in policies:
        a = policy_vectors(episodes, policy)
        if np.array_equal(a, policy_vectors(episodes, reference)) and policy != reference:
            identical_to[policy] = reference
        for other in ("NBV", "HistoryOnly"):
            if policy == other or other not in policies:
                continue
            test = paired_bootstrap(a, policy_vectors(episodes, other))
            rows.append({"policy": policy, "vs": other, **test})
    import csv
    with (TABDIR / "table_paired_tests.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["policy", "vs", "diff", "lo", "hi",
                                                "p_not_positive", "identical", "n"])
        writer.writeheader()
        writer.writerows(rows)
    print("  wrote paper/tables/table_paired_tests.csv")
    if identical_to:
        print(f"  outcome-identical on every terrain: {identical_to}")

    tex = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Policy & RSR@1 & RSR@5 & RSR@10 & 95\\% CI (RSR@10) \\\\",
        "\\midrule",
    ]
    for policy in policies:
        v10 = policy_vectors(episodes, policy, budget=10)
        lo, hi = wilson(int(v10.sum()), n)
        m1 = policy_vectors(episodes, policy, budget=1).mean()
        m5 = policy_vectors(episodes, policy, budget=5).mean()
        tex.append(f"{policy.replace('_', chr(92) + '_')} & {m1:.2f} & {m5:.2f} & "
                   f"{v10.mean():.2f} & [{lo:.2f}, {hi:.2f}] \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TABDIR / "table_benchmark.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("  wrote paper/tables/table_benchmark.tex")


def fig_safety(calib: dict) -> dict:
    """E4-5b: prediction quality versus decision usefulness.

    (a) is the diagnosis - calibration error averaged over all predictions versus
    calibration error restricted to the band where a threshold actually acts.
    (b) is the consequence - the success/coverage trade-off curve.  A variant is
    only useful if some point on (b) rises above the no-safety line.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8),
                             gridspec_kw={"width_ratios": [1.0, 1.2]})

    ax = axes[0]
    diag = calib["calibration_diagnostics"]
    names = ["raw", "temperature", "isotonic"]
    x = np.arange(len(names))
    overall = [diag[n]["ece"] for n in names]
    boundary = [diag[n]["decision_boundary_ece"] for n in names]
    ax.bar(x - 0.2, overall, 0.38, color="#0072B2", label="overall ECE")
    ax.bar(x + 0.2, boundary, 0.38, color="#D55E00", label="ECE at decision band")
    for xi, v in zip(x - 0.2, overall):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
    for xi, v in zip(x + 0.2, boundary):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("expected calibration error")
    ax.set_ylim(0, max(boundary) * 1.35)
    ax.set_title("(a) Well calibrated on average,\nbadly calibrated where the threshold acts",
                 loc="left")
    ax.legend(loc="upper left")

    ax = axes[1]
    colours = {"raw": "#999999", "temperature": "#0072B2", "isotonic": "#009E73"}
    for name in names:
        curve = calib["sweeps"][name]["curve"]
        ax.plot([c["coverage"] for c in curve], [c["success_rate"] for c in curve],
                marker="o", ms=3.5, lw=1.2, color=colours[name], label=name)
    no_safety = calib["no_safety"]["success_rate"]
    ax.axhline(no_safety, color="#000000", ls="--", lw=1.1,
               label=f"no safety ({no_safety:.3f})")
    ax.axhline(calib["oracle_v"]["success_rate"], color="#CC79A7", ls=":", lw=1.1,
               label=f"Oracle-V ({calib['oracle_v']['success_rate']:.3f})")
    ax.axhline(calib["oracle_s"]["success_rate"], color="#D55E00", ls="-.", lw=1.1,
               label=f"Oracle-S ({calib['oracle_s']['success_rate']:.3f})")
    ax.set_xlabel("coverage (fraction of states where safety changed the action)")
    ax.set_ylabel("one-step safe re-observation")
    ax.set_title("(b) No operating point beats not using safety\n"
                 "so the limit is the rule, not the calibration", loc="left")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.4)

    fig.tight_layout()
    save(fig, "fig5_safety_calibration")
    plt.close(fig)
    return {"no_safety": no_safety, "oracle_s": calib["oracle_s"]["success_rate"],
            "best": {n: calib["sweeps"][n]["best"] for n in names}}


def fig_horizon_ablation(payload: dict) -> dict:
    """E4d: does lookahead depth help, cost more, or accumulate model error?

    Three panels because the three explanations are distinguishable only together:
    (a) success against budget for each horizon, (b) the action cost and time each
    horizon spends, (c) the measured one-step prediction error.  If success rises
    with H while cost and error stay flat, lookahead is doing real work; if success
    is flat while cost rises, it is just spending actions; if error rises with H,
    the model is the limit.
    """
    import matplotlib.pyplot as plt

    results = payload["results"]
    names = list(results)
    budgets = (1, 3, 5, 10)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    # (a) RSR@K
    ax = axes[0]
    colours = ["#999999", "#0072B2", "#009E73", "#D55E00"]
    for name, colour in zip(names, colours):
        s = results[name]
        ys = [s[f"rsr@{k}"] for k in budgets]
        ax.plot(budgets, ys, marker="o", ms=4, lw=1.4, color=colour, label=name)
    ax.set_xscale("log")
    ax.set_xticks(list(budgets))
    ax.set_xticklabels([str(k) for k in budgets])
    ax.set_xlabel("action budget $K$")
    ax.set_ylabel("RSR@$K$")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) Success by lookahead depth", loc="left")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8)

    # (b) what each horizon spends
    ax = axes[1]
    x = np.arange(len(names))
    cost = [results[n]["mean_motion_cost_m"] for n in names]
    ax.bar(x, cost, 0.6, color=colours[:len(names)])
    for xi, v in zip(x, cost):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean action cost (m)")
    ax.set_ylim(0, max(cost) * 1.25)
    ax.set_title("(b) What the lookahead spends", loc="left")

    # (c) measured model error
    ax = axes[2]
    err = [results[n]["model_error"]["mean_position_error_m"] for n in names]
    ax.plot(x, err, marker="s", ms=4, lw=1.4, color="#CC79A7")
    for xi, v in zip(x, err):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("one-step pose error (m)")
    ax.set_ylim(0, max(err) * 1.3)
    ax.set_title("(c) Model error actually measured", loc="left")

    fig.tight_layout()
    save(fig, "fig6_horizon_ablation")
    plt.close(fig)
    return {"horizons": names,
            "rsr10": {n: results[n]["rsr@10"] for n in names},
            "pose_error": dict(zip(names, err))}


def horizon_failure_analysis(payload: dict, oracle_steps: dict | None = None) -> dict:
    """Which terrains change verdict with the horizon, and in which direction.

    The two cases worth separating are the ones the horizon ablation exists to
    find: terrains that need more than one step (so a longer lookahead rescues
    them) and terrains that *lose* success as the lookahead grows (so the planner
    is being misled by its own chained predictions).
    """
    results = payload["results"]
    names = list(results)
    per = {n: results[n]["per_terrain"] for n in names}
    terrains = list(per[names[0]])
    rows = []
    for terrain in terrains:
        entry = {}
        for n in names:
            info = per[n].get(terrain, {})
            entry[n] = (info.get("termination") == "success", info.get("steps"))
        verdicts = [entry[n][0] for n in names]
        rows.append({"terrain": terrain, **{f"{n}_success": entry[n][0] for n in names},
                     **{f"{n}_steps": entry[n][1] for n in names},
                     "gains_with_horizon": (not verdicts[0]) and any(verdicts[1:]),
                     "loses_with_horizon": verdicts[0] and not all(verdicts) or
                     (any(verdicts[:-1]) and not verdicts[-1])})

    rescued = [r["terrain"] for r in rows if r["gains_with_horizon"]]
    degraded = [r["terrain"] for r in rows
                if r[f"{names[-1]}_success"] is False
                and any(r[f"{n}_success"] for n in names[:-1])]
    # terrains the privileged reference planner needs more than one step for
    multi_step = []
    if oracle_steps:
        multi_step = [t for t, k in oracle_steps.items() if k and k > 1]
    return {
        "horizons": names,
        "rescued_by_lookahead": rescued,
        "degraded_at_max_horizon": degraded,
        "oracle_requires_multiple_steps": sorted(multi_step),
        "per_terrain": rows,
    }


def horizon_paired_tests(payload: dict) -> dict:
    """Is any lookahead depth significantly better than H=1?

    With 24 terrains a one-terrain difference is 0.042, which looks like a result
    and is not one.  Everything is therefore tested as a paired bootstrap over
    terrains against H=1 - the same discipline the baseline table uses - and cost is
    reported alongside, because "no gain at 40 % more motion" is a different claim
    from "no gain".
    """
    results = payload["results"]
    names = list(results)
    terrains = list(results[names[0]]["per_terrain"])
    vectors, step_vectors = {}, {}
    for name in names:
        per = results[name]["per_terrain"]
        vectors[name] = np.array([1.0 if per[t]["termination"] == "success" else 0.0
                                  for t in terrains])
        step_vectors[name] = np.array([float(per[t]["steps"] or 0) for t in terrains])
    base = names[0]
    tests = {}
    for name in names[1:]:
        success_test = paired_bootstrap(vectors[name], vectors[base])
        step_test = paired_bootstrap(step_vectors[name], step_vectors[base])
        tests[f"{name}_vs_{base}"] = {
            "success_diff": success_test["diff"],
            "success_ci": [success_test["lo"], success_test["hi"]],
            "success_p_not_better": success_test["p_not_positive"],
            "extra_steps": step_test["diff"],
            "extra_steps_ci": [step_test["lo"], step_test["hi"]],
            "cost_ratio": round(results[name]["mean_motion_cost_m"]
                                / results[base]["mean_motion_cost_m"], 3),
        }
    return {"baseline": base, "n_terrains": len(terrains), "tests": tests}


def main() -> None:
    p = argparse.ArgumentParser(description="Publication figures for TASK3")
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of: benchmark pairs geometry horizon safety lookahead")
    args = p.parse_args()

    setup_style()
    wanted = set(args.only) if args.only else {"benchmark", "pairs", "geometry", "horizon", "safety", "lookahead"}

    if "benchmark" in wanted:
        episodes = load("benchmark_multistep_B_scored.json")
        print("fig1 benchmark ...")
        stats = fig_benchmark(episodes)
        write_tables(episodes, stats)
        best = max((k for k in episodes["summary"]
                    if k not in ("Oracle", "OracleSearch")),
                   key=lambda k: episodes["summary"][k]["rsr@10"])
        print(f"  best non-privileged policy at RSR@10: {best}")

    if "pairs" in wanted:
        print("fig2 paired worlds ...")
        summary = fig_pairs(load("history_necessity.json"), load("memory_sufficiency.json"))
        print(f"  {summary}")

    if "geometry" in wanted:
        print("fig3 geometry gap ...")
        s = fig_geometry(load("geometry_gap.json"))
        print(f"  gap share of geometric positives: {s['gap_share_of_geometric_positives']}")

    if "horizon" in wanted:
        print("fig4 history horizon ...")
        summary = fig_horizon(load("history_horizon.json"))
        print(f"  {summary}")

    if "safety" in wanted:
        print("fig5 safety calibration ...")
        summary = fig_safety(load("safety_calibration.json"))
        print(f"  no safety {summary['no_safety']} vs Oracle-S {summary['oracle_s']}; "
              f"best per variant: {summary['best']}")

    if "lookahead" in wanted:
        print("fig6 horizon ablation ...")
        payload = load("horizon_planner.json")
        summary = fig_horizon_ablation(payload)
        print(f"  RSR@10 by horizon: {summary['rsr10']}")
        oracle_steps = None
        try:
            oracle_steps = load("benchmark_multistep_B_scored.json")["summary"]["OracleSearch"]["steps_by_terrain"]
        except SystemExit:
            pass
        analysis = horizon_failure_analysis(payload, oracle_steps)
        analysis["paired_tests"] = horizon_paired_tests(payload)
        print(f"  rescued by lookahead : {analysis['rescued_by_lookahead']}")
        print(f"  degraded at max H    : {analysis['degraded_at_max_horizon']}")
        print(f"  oracle needs >1 step : {analysis['oracle_requires_multiple_steps']}")
        (TABDIR / "table_horizon_ablation.json").write_text(
            json.dumps(analysis, indent=2), encoding="utf-8")
        print("  wrote paper/tables/table_horizon_ablation.json")


if __name__ == "__main__":
    main()
