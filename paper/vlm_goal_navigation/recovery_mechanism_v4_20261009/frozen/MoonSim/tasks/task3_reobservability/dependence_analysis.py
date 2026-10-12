"""E4e-0: what is the actual relationship between visibility and safety?

The joint consequence head failed (E4e: -0.013 on one-step success), and that result
on its own is uninterpretable.  It is consistent with at least four different
worlds:

  A. ``V`` and ``S`` are essentially independent, so joint modelling is unnecessary
     and ``p_V * p_S`` would suffice;
  B. they are strongly coupled, but 2240 samples cannot fit a joint head;
  C. they are coupled and ``p_V * p_S`` works, so the problem is how the joint model
     was fitted rather than whether joint modelling is right;
  D. even the ground-truth ``Oracle-VS`` is no better than ``Oracle-V``, so safety is
     not the current bottleneck at all.

This module distinguishes them with **no model training whatsoever** - only counts
over the 2240 recorded (state, action) outcomes.

**The statistical trap this experiment exists to avoid.**  A significant overall
odds ratio does *not* establish ``V not independent of S | (O, M, a)``.  The action
itself plausibly moves both quantities in opposite directions - forward may raise
visibility while lowering safety, a lateral hold may do the reverse - so a strong
marginal association can appear even under conditional independence.  The analysis
is therefore reported at three levels and the conclusion is restricted to what each
level can support:

  Level 1  empirical coupling              P(V,S) vs P(V)P(S), odds ratio
  Level 2  coupling *within* each action   the same statistic stratified by action
  Level 3  coupling *within* each terrain  the same statistic stratified by mechanism

Only a within-stratum association speaks to conditional dependence, and even then a
single outcome per (state, action) cell means the 2240 samples cannot prove
conditional independence - they can only fail to find dependence.

The action-set analysis is the part that speaks directly to planning: for each
decision state, how often is there a *conflict*, i.e. no action that is both visible
and safe.

    python tasks/task3_reobservability/dependence_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .consequence_model import SAFETY_MARGIN_KEEP_UPRIGHT, load
except ImportError:
    from consequence_model import SAFETY_MARGIN_KEEP_UPRIGHT, load

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "dependence_analysis.json"


def visible(row: dict) -> bool:
    return bool(row["reobserved"])


def safe(row: dict) -> bool:
    """Physical safety, defined *independently* of visibility.

    This separation is the whole point: ``safe`` must not be ``reobserved and
    safe``, or the 2x2 table would be degenerate by construction.
    """
    return not (row.get("out_of_bounds")
                or float(row.get("upright_min", 1.0)) < SAFETY_MARGIN_KEEP_UPRIGHT)


def contingency(rows: list[dict]) -> dict:
    """The 2x2 table and the statistics that can be read from it."""
    n11 = sum(1 for r in rows if visible(r) and safe(r))
    n10 = sum(1 for r in rows if visible(r) and not safe(r))
    n01 = sum(1 for r in rows if not visible(r) and safe(r))
    n00 = sum(1 for r in rows if not visible(r) and not safe(r))
    n = len(rows)
    if n == 0:
        return {"n": 0}
    p_v = (n11 + n10) / n
    p_s = (n11 + n01) / n
    p_vs = n11 / n
    out = {
        "n": n,
        "table": {"V1_S1": n11, "V1_S0": n10, "V0_S1": n01, "V0_S0": n00},
        "P(V)": round(p_v, 4), "P(S)": round(p_s, 4), "P(V,S)": round(p_vs, 4),
        "P(V)P(S)": round(p_v * p_s, 4),
        "delta_independence": round(p_vs - p_v * p_s, 4),
        "P(S|V)": round(n11 / (n11 + n10), 4) if (n11 + n10) else None,
        "P(S|not V)": round(n01 / (n01 + n00), 4) if (n01 + n00) else None,
    }
    # Woolf log-odds-ratio interval, with a Haldane-Anscombe correction so that a
    # zero cell widens the interval instead of producing an infinite estimate.
    a, b, c, d = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    out["odds_ratio"] = round(math.exp(log_or), 3)
    out["odds_ratio_ci"] = [round(math.exp(log_or - 1.96 * se), 3),
                            round(math.exp(log_or + 1.96 * se), 3)]
    out["independent"] = bool(out["odds_ratio_ci"][0] <= 1.0 <= out["odds_ratio_ci"][1])
    out["zero_cell"] = bool(0 in (n11, n10, n01, n00))
    return out


def stratified(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {name: contingency(subset) for name, subset in sorted(groups.items())}


def action_sets(rows: list[dict]) -> dict:
    """Per decision state, how many actions are visible, safe, and both."""
    terrain = np.array([r["terrain"] for r in rows])
    state = np.array([r["state_id"] for r in rows])
    records, conflicts, only_visible, only_safe = [], 0, 0, 0
    for t in dict.fromkeys(terrain):
        for s in sorted(set(state[terrain == t])):
            idx = np.where((terrain == t) & (state == s))[0]
            if idx.size == 0:
                continue
            a_v = {int(i) for i in idx if visible(rows[i])}
            a_s = {int(i) for i in idx if safe(rows[i])}
            both = a_v & a_s
            if a_v and a_s and not both:
                conflicts += 1
            if a_v and not a_s:
                only_visible += 1
            if a_s and not a_v:
                only_safe += 1
            records.append({
                "terrain": t, "state": int(s),
                "n_visible": len(a_v), "n_safe": len(a_s), "n_both": len(both),
                "conflict": bool(a_v and a_s and not both),
            })
    n = len(records)
    tv = sum(r["n_visible"] for r in records)
    ts = sum(r["n_safe"] for r in records)
    tb = sum(r["n_both"] for r in records)
    return {
        "n_decision_states": n,
        "states_with_conflict": conflicts,
        "states_with_conflict_rate": round(conflicts / n, 4) if n else None,
        "states_with_visible_but_no_safe": only_visible,
        "states_with_safe_but_no_visible": only_safe,
        "safety_coverage_of_visible_actions": round(tb / tv, 4) if tv else None,
        "visibility_coverage_of_safe_actions": round(tb / ts, 4) if ts else None,
        "mean_visible_actions": round(tv / n, 2) if n else None,
        "mean_safe_actions": round(ts / n, 2) if n else None,
        "mean_both_actions": round(tb / n, 2) if n else None,
        "per_state": records,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="E4e-0 visibility-safety dependence analysis")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    payload = load(args.data)
    rows = payload["rows"]

    report = {
        "n_samples": len(rows),
        "level_1_overall": contingency(rows),
        "level_2_by_action": stratified(rows, lambda r: r["action"]),
        "level_3_by_terrain": stratified(rows, lambda r: r["terrain"]),
        "level_3_by_mechanism": stratified(rows, lambda r: r["mechanism"]),
        "action_sets": action_sets(rows),
    }

    print("=== E4e-0 visibility vs safety dependence (no model trained) ===")
    print(f"samples: {len(rows)}\n")
    o = report["level_1_overall"]
    print("Level 1 - overall")
    print(f"  table {o['table']}")
    print(f"  P(V)={o['P(V)']}  P(S)={o['P(S)']}  P(V,S)={o['P(V,S)']}  "
          f"P(V)P(S)={o['P(V)P(S)']}  delta={o['delta_independence']:+}")
    print(f"  P(S|V)={o['P(S|V)']}  P(S|notV)={o['P(S|not V)']}  "
          f"OR={o['odds_ratio']} CI={o['odds_ratio_ci']}  "
          f"independent={o['independent']}")
    if o["zero_cell"]:
        print("  NOTE: a zero cell is present; the interval uses Haldane-Anscombe "
              "correction and is widened accordingly.")

    print("\nLevel 2 - within each action (this is what speaks to conditional "
          "dependence)")
    print(f"  {'action':14s} {'n':>5s} {'P(V)':>7s} {'P(S)':>7s} {'delta':>8s} "
          f"{'OR':>7s}  {'OR 95% CI':>16s}  indep")
    for name, c in report["level_2_by_action"].items():
        print(f"  {name:14s} {c['n']:>5d} {c['P(V)']:>7.3f} {c['P(S)']:>7.3f} "
              f"{c['delta_independence']:>+8.4f} {c['odds_ratio']:>7.3f}  "
              f"{str(c['odds_ratio_ci']):>16s}  {c['independent']}")

    print("\nLevel 3 - within each terrain mechanism")
    print(f"  {'mechanism':14s} {'n':>5s} {'P(V)':>7s} {'P(S)':>7s} {'delta':>8s} "
          f"{'OR':>7s}  indep")
    for name, c in report["level_3_by_mechanism"].items():
        print(f"  {name:14s} {c['n']:>5d} {c['P(V)']:>7.3f} {c['P(S)']:>7.3f} "
              f"{c['delta_independence']:>+8.4f} {c['odds_ratio']:>7.3f}  "
              f"{c['independent']}")

    a = report["action_sets"]
    print("\nAction-set analysis (per decision state)")
    print(f"  decision states                : {a['n_decision_states']}")
    print(f"  mean visible / safe / both     : {a['mean_visible_actions']} / "
          f"{a['mean_safe_actions']} / {a['mean_both_actions']}")
    print(f"  safety coverage of visible     : {a['safety_coverage_of_visible_actions']}")
    print(f"  visibility coverage of safe    : {a['visibility_coverage_of_safe_actions']}")
    print(f"  states with a conflict         : {a['states_with_conflict']} "
          f"({a['states_with_conflict_rate']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
