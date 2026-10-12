"""E4-5b: is the safety information useless, or is the way we use it useless?

E4-5 established a clean puzzle:

* the safety term is worth having - with ground-truth labels, adding it raises
  one-step success from 0.3884 to **0.4107**;
* the safety head is genuinely predictive - ROC-AUC **0.900**;
* and yet plugging the head into the planner makes it **worse** at every penalty
  weight, hard and soft alike (0.3661 -> 0.3125 ... 0.3571).

Two explanations fit. Either the safety signal carries no usable information, or
the *decision rule* that consumes it is wrong - a probability is not a decision.
This module separates them by fixing the prediction and varying only how it is
used:

``Learned-V``        argmax of predicted visibility, no safety
``hard safety``      remove any action with ``p_unsafe >= tau``, then argmax visibility
``calibrated``       same rule, but on a calibrated probability (temperature / isotonic)
``selective``        intervene only where the safety head is confident, i.e. a high tau
``Oracle-V``         ground-truth visibility, no safety
``Oracle-S``         ground-truth visibility and ground-truth safety

and by reporting the **trade-off curve** rather than a single operating point:

    RSR@1(tau)   versus   coverage(tau)

where coverage is the fraction of decision states at which the safety term actually
changed the chosen action.  A variant is only useful if some point on that curve
beats no-safety *at a coverage a deployment could tolerate*.

**Calibration is fitted out-of-fold, by terrain.**  A calibrator fitted on the same
predictions it is scored on would manufacture the improvement it is supposed to
demonstrate, so each decision state's calibrator is fitted only on the other
terrains' out-of-fold predictions.

    python tasks/task3_reobservability/safety_calibration.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .consequence_model import (SAFETY_MARGIN_KEEP_UPRIGHT, full_consequence_heads,
                                    load, safety_labels)
except ImportError:
    from consequence_model import (SAFETY_MARGIN_KEEP_UPRIGHT, full_consequence_heads,
                                   load, safety_labels)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "safety_calibration.json"

TAUS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises NLL of the logits, by scalar search.

    A one-parameter fit, deliberately: with a few hundred positives, a richer
    calibrator would fit the calibration split rather than the distribution.
    """
    z = logit(p)
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.25, 4.0, 76):
        q = 1.0 / (1.0 + np.exp(-z / t))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        nll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logit(p) / t))


def fit_isotonic(p: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(p, y)
    return model


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def decision_boundary_ece(y: np.ndarray, p: np.ndarray, band: float = 0.15) -> float:
    """Calibration restricted to the region where the decision actually flips.

    Average ECE is the wrong diagnostic here.  A head can be well calibrated on
    average because most actions are obviously safe (p near 0), while being badly
    calibrated exactly where a threshold would act.  This measures only the band
    around 0.5.
    """
    mask = np.abs(p - 0.5) <= band
    if not mask.any():
        return float("nan")
    return expected_calibration_error(y[mask], p[mask], bins=5)


def states(rows: list[dict]) -> list[dict]:
    """Group rows into decision states, in a fixed order."""
    terrain = np.array([r["terrain"] for r in rows])
    state = np.array([r["state_id"] for r in rows])
    out = []
    for t in dict.fromkeys(terrain):
        for s in sorted(set(state[terrain == t])):
            idx = np.where((terrain == t) & (state == s))[0]
            if idx.size:
                out.append({"terrain": t, "state": int(s), "idx": idx})
    return out


def is_safe_win(row: dict) -> bool:
    return bool(row["reobserved"]
                and not row.get("out_of_bounds")
                and float(row.get("upright_min", 1.0)) >= SAFETY_MARGIN_KEEP_UPRIGHT)


def choose(p_vis: np.ndarray, p_unsafe: np.ndarray | None, tau: float | None,
           idx: np.ndarray) -> int:
    """Pick an action index under a visibility score with optional safety filter.

    When the filter removes every candidate the visibility ranking is used
    unchanged, rather than abstaining - abstention would silently turn a decision
    rule into a budget-burning policy and confound the comparison.
    """
    if p_unsafe is not None and tau is not None:
        allowed = idx[p_unsafe[idx] < tau]
        if allowed.size:
            return int(allowed[np.argmax(p_vis[allowed])])
    return int(idx[np.argmax(p_vis[idx])])


def evaluate(rows: list[dict], p_vis: np.ndarray, p_unsafe: np.ndarray | None,
             tau: float | None, baseline_actions: dict | None = None) -> dict:
    """One-step success and how often the safety term changed the action."""
    solved = total = changed = 0
    chosen = {}
    for st in states(rows):
        idx = st["idx"]
        pick = choose(p_vis, p_unsafe, tau, idx)
        picked = rows[pick]["action"]
        chosen[(st["terrain"], st["state"])] = picked
        total += 1
        solved += int(is_safe_win(rows[pick]))
        if baseline_actions is not None:
            changed += int(baseline_actions[(st["terrain"], st["state"])] != picked)
    return {"states": total, "solved": solved,
            "success_rate": round(solved / total, 4) if total else None,
            "interventions": changed,
            "coverage": round(changed / total, 4) if total and baseline_actions else 0.0,
            "chosen": chosen}


def run(rows: list[dict], inputs: str = "O+M+a") -> dict:
    heads = full_consequence_heads(rows, inputs,
                                   sorted({r["action"] for r in rows}))
    p_vis, p_unsafe = heads["p_visible"], heads["p_unsafe"]
    y_safe = heads["y_unsafe"]
    true_vis = heads["y_visible"]

    # --- out-of-fold calibration, by terrain -------------------------------
    terrain = np.array([r["terrain"] for r in rows])
    p_temp = np.zeros_like(p_unsafe)
    p_iso = np.zeros_like(p_unsafe)
    for t in np.unique(terrain):
        test, train = terrain == t, terrain != t
        if not train.any() or len(np.unique(y_safe[train])) < 2:
            p_temp[test], p_iso[test] = p_unsafe[test], p_unsafe[test]
            continue
        temperature = fit_temperature(p_unsafe[train], y_safe[train])
        p_temp[test] = apply_temperature(p_unsafe[test], temperature)
        p_iso[test] = fit_isotonic(p_unsafe[train], y_safe[train]).predict(p_unsafe[test])

    diagnostics = {}
    for name, p in (("raw", p_unsafe), ("temperature", p_temp), ("isotonic", p_iso)):
        diagnostics[name] = {
            "ece": round(expected_calibration_error(y_safe, p), 4),
            "decision_boundary_ece": round(decision_boundary_ece(y_safe, p), 4),
            "mean_p": round(float(p.mean()), 4),
        }

    # --- reference operating points ----------------------------------------
    no_safety = evaluate(rows, p_vis, None, None)
    base_actions = no_safety["chosen"]
    result = {
        "no_safety": {k: v for k, v in no_safety.items() if k != "chosen"},
        "oracle_v": {k: v for k, v in evaluate(rows, true_vis.astype(float), None,
                                               None).items() if k != "chosen"},
        "oracle_s": {k: v for k, v in evaluate(
            rows, true_vis.astype(float), heads["y_unsafe"], 0.5).items()
            if k != "chosen"},
        "calibration_diagnostics": diagnostics,
        "true_safety_positive_rate": round(float(y_safe.mean()), 4),
        "sweeps": {},
    }

    for name, p in (("raw", p_unsafe), ("temperature", p_temp), ("isotonic", p_iso)):
        curve = []
        for tau in TAUS:
            entry = evaluate(rows, p_vis, p, tau, baseline_actions=base_actions)
            curve.append({"tau": tau, "success_rate": entry["success_rate"],
                          "coverage": entry["coverage"],
                          "solved": entry["solved"], "states": entry["states"]})
        best = max(curve, key=lambda c: (c["success_rate"], -c["coverage"]))
        result["sweeps"][name] = {
            "curve": curve, "best": best,
            "beats_no_safety": bool(best["success_rate"]
                                    > result["no_safety"]["success_rate"]),
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="E4-5b decision-aware safety calibration")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--inputs", default="O+M+a")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    payload = load(args.data)
    report = run(payload["rows"], args.inputs)

    print("=== E4-5b is the safety information useless, or its use? ===\n")
    print("calibration diagnostics (out-of-fold, by terrain)")
    print(f"  {'variant':12s} {'ECE':>8s} {'ECE@boundary':>14s} {'mean p':>8s}")
    for name, d in report["calibration_diagnostics"].items():
        print(f"  {name:12s} {d['ece']:>8.4f} {d['decision_boundary_ece']:>14.4f} "
              f"{d['mean_p']:>8.4f}")

    print(f"\nreference operating points (one-step safe re-observation, "
          f"{report['no_safety']['states']} decision states)")
    for name, key in (("no safety", "no_safety"), ("Oracle-V", "oracle_v"),
                      ("Oracle-S", "oracle_s")):
        e = report[key]
        print(f"  {name:12s} {e['success_rate']:.4f}  ({e['solved']}/{e['states']})")

    print("\ntau sweeps (success at the fraction of states where safety intervened)")
    for name, sweep in report["sweeps"].items():
        print(f"\n  {name}")
        for point in sweep["curve"]:
            flag = "  <-- best" if point is sweep["best"] else ""
            print(f"    tau={point['tau']:.2f}  success={point['success_rate']:.4f}  "
                  f"coverage={point['coverage']:.3f}{flag}")
        print(f"    best: {sweep['best']['success_rate']:.4f} at coverage "
              f"{sweep['best']['coverage']:.3f}  "
              f"beats no-safety: {sweep['beats_no_safety']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
