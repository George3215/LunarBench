"""E4e: a joint consequence decision, instead of serially composing two heads.

E4c and E4d failed in the same shape.  In E4c the safety head carries real
information (Oracle-S > Oracle-V) and is accurate (AUC 0.900), yet adding it makes
the planner worse at every penalty weight, and halving its decision-band
calibration error changes nothing.  In E4d a longer lookahead carries real
information (some terrains genuinely need two steps) yet costs twice the motion and
breaks eight terrains that one step solves.

The common cause is that both were **serial compositions** of two quantities that
were predicted independently:

* ``argmax P(V)`` first, then a safety term that vetoes or penalises the ranking;
* a per-step visibility score, then a ``max`` over chained steps to stand in for
  "visible at some point".

A veto is the wrong shape.  The planner does not want to know whether an action is
unsafe; it wants to know whether the action achieves the goal, where the goal is
*re-observed and still on its feet*.  That is a single event, so it should be a
single predicted quantity.

**Four decisions, one prediction each:**

``V-only``       ``argmax P(V)`` - the E4-3 planner, the current baseline
``joint head``   ``argmax P(V and safe)`` - one head on the event the planner wants
``selective``    ``P(V)`` ranking with conformal filtering: reject an action only
                 when its safety score exceeds the out-of-fold quantile of the
                 *safe* class, which bounds how often a good action is discarded
``both``         conformal filtering, then the joint head
``Oracle-V`` / ``Oracle-S``   privileged references for the information available

The conformal rule is the targeted fix for the measured failure.  E4c showed that
the head's errors land on exactly the marginal decisions, and that a fixed
threshold therefore discards good and bad actions at similar rates.  Calibrating
the threshold against the *negative* class distribution, out of fold, bounds that
rate by construction rather than hoping a tuned lambda does.

    python tasks/task3_reobservability/joint_decision.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .consequence_model import (SAFETY_MARGIN_KEEP_UPRIGHT, build_training_matrix,
                                    load)
except ImportError:
    from consequence_model import (SAFETY_MARGIN_KEEP_UPRIGHT, build_training_matrix,
                                   load)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "joint_decision.json"

QUANTILES = (0.80, 0.90, 0.95, 0.975, 0.99)


def is_safe_win(row: dict) -> bool:
    return bool(row["reobserved"]
                and not row.get("out_of_bounds")
                and float(row.get("upright_min", 1.0)) >= SAFETY_MARGIN_KEEP_UPRIGHT)


def decision_states(rows: list[dict]) -> list[dict]:
    terrain = np.array([r["terrain"] for r in rows])
    state = np.array([r["state_id"] for r in rows])
    out = []
    for t in dict.fromkeys(terrain):
        for s in sorted(set(state[terrain == t])):
            idx = np.where((terrain == t) & (state == s))[0]
            if idx.size:
                out.append({"terrain": t, "state": int(s), "idx": idx})
    return out


def out_of_fold_scores(rows, inputs, action_ids, joint: bool):
    """Out-of-fold P(V) and P(V and safe), with terrain held out.

    For the joint head the target is the *conjunction*, not a product of two
    marginals: the planner's objective is a single event, and predicting its
    marginals separately and multiplying them would reintroduce exactly the
    independence assumption this experiment exists to test.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = build_training_matrix(rows, inputs, action_ids)
    terrain = np.array([r["terrain"] for r in rows])
    y_vis = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
    y_joint = np.array([1.0 if is_safe_win(r) else 0.0 for r in rows])
    y_unsafe = np.array([1.0 if (r.get("out_of_bounds")
                                 or float(r.get("upright_min", 1.0))
                                 < SAFETY_MARGIN_KEEP_UPRIGHT) else 0.0 for r in rows])

    p_vis = np.zeros(len(rows))
    p_joint = np.zeros(len(rows))
    p_unsafe = np.zeros(len(rows))
    for t in np.unique(terrain):
        test, train = terrain == t, terrain != t
        if not train.any():
            continue
        for y, out in ((y_vis, p_vis), (y_joint, p_joint), (y_unsafe, p_unsafe)):
            if len(np.unique(y[train])) < 2:
                out[test] = y[train].mean()
                continue
            model = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                                   learning_rate=0.08, random_state=7)
            model.fit(X[train], y[train])
            out[test] = model.predict_proba(X[test])[:, 1]
    return {"p_vis": p_vis, "p_joint": p_joint, "p_unsafe": p_unsafe,
            "y_vis": y_vis, "y_joint": y_joint, "y_unsafe": y_unsafe}


def conformal_thresholds(scores: dict, rows: list[dict]) -> dict:
    """Quantiles of the *safe* class's unsafe-score, out of fold.

    Rejecting an action whose safety score is below this quantile would discard
    that fraction of genuinely safe actions.  Doing it out of fold is what makes
    the bound meaningful: a threshold fitted on the same scores it filters would
    simply describe them.
    """
    y_unsafe = scores["y_unsafe"]
    p = scores["p_unsafe"]
    safe_scores = p[y_unsafe == 0]
    return {q: float(np.quantile(safe_scores, q)) for q in QUANTILES}


def choose(score: np.ndarray, p_unsafe: np.ndarray, tau: float | None,
           idx: np.ndarray) -> int:
    if tau is not None:
        allowed = idx[p_unsafe[idx] < tau]
        if allowed.size:
            return int(allowed[np.argmax(score[allowed])])
    return int(idx[np.argmax(score[idx])])


def evaluate(rows: list[dict], scores: dict, rule: str, tau: float | None) -> dict:
    p_vis, p_joint, p_unsafe = scores["p_vis"], scores["p_joint"], scores["p_unsafe"]
    solved = total = intervened = 0
    for st in decision_states(rows):
        idx = st["idx"]
        if rule == "v_only":
            pick = choose(p_vis, p_unsafe, None, idx)
        elif rule == "product":
            # The textbook joint score under an independence assumption. E4e-0
            # measured that V and S are *negatively* coupled (OR 0.35), so this is
            # the wrong model on purpose: it is the cheap baseline the dependence
            # analysis has to justify or rule out.
            pick = int(idx[np.argmax((p_vis * (1.0 - p_unsafe))[idx])])
        elif rule == "joint":
            pick = choose(p_joint, p_unsafe, None, idx)
        elif rule == "selective":
            pick = choose(p_vis, p_unsafe, tau, idx)
        elif rule == "both":
            pick = choose(p_joint, p_unsafe, tau, idx)
        else:
            raise ValueError(rule)
        baseline = choose(p_vis, p_unsafe, None, idx)
        intervened += int(pick != baseline)
        total += 1
        solved += int(is_safe_win(rows[pick]))
    return {"states": total, "solved": solved,
            "success_rate": round(solved / total, 4) if total else None,
            "coverage": round(intervened / total, 4) if total else 0.0}


def run(rows: list[dict], inputs: str = "O+M+a") -> dict:
    action_ids = sorted({r["action"] for r in rows})
    scores = out_of_fold_scores(rows, inputs, action_ids, joint=True)
    thresholds = conformal_thresholds(scores, rows)

    true_vis = scores["y_vis"]
    true_joint = scores["y_joint"]

    result = {
        "n_states": len(decision_states(rows)),
        "positive_rates": {
            "visibility": round(float(scores["y_vis"].mean()), 4),
            "visibility_and_safe": round(float(scores["y_joint"].mean()), 4),
            "unsafe": round(float(scores["y_unsafe"].mean()), 4),
        },
        "conformal_thresholds": {str(q): round(v, 4)
                                 for q, v in thresholds.items()},
        "reference": {
            "Oracle-V": evaluate(rows, {"p_vis": true_vis.astype(float),
                                        "p_joint": true_vis.astype(float),
                                        "p_unsafe": scores["p_unsafe"]}, "v_only", None),
            "Oracle-S": evaluate(rows, {"p_vis": true_joint.astype(float),
                                        "p_joint": true_joint.astype(float),
                                        "p_unsafe": scores["p_unsafe"]}, "v_only", None),
        },
        "baseline_v_only": evaluate(rows, scores, "v_only", None),
        "product_pV_pS": evaluate(rows, scores, "product", None),
        "joint_head": evaluate(rows, scores, "joint", None),
        "selective": {str(q): evaluate(rows, scores, "selective", t)
                      for q, t in thresholds.items()},
        "both": {str(q): evaluate(rows, scores, "both", t)
                 for q, t in thresholds.items()},
    }

    base = result["baseline_v_only"]["success_rate"]
    best_selective = max(result["selective"].items(),
                         key=lambda kv: kv[1]["success_rate"])
    result["verdict"] = {
        "baseline": base,
        "oracle_v": result["reference"]["Oracle-V"]["success_rate"],
        "oracle_s": result["reference"]["Oracle-S"]["success_rate"],
        "joint_head_gain": round(result["joint_head"]["success_rate"] - base, 4),
        "best_selective": {"quantile": best_selective[0], **best_selective[1]},
        "best_selective_gain": round(best_selective[1]["success_rate"] - base, 4),
        "any_rule_beats_baseline": bool(
            result["joint_head"]["success_rate"] > base
            or best_selective[1]["success_rate"] > base
            or max(v["success_rate"] for v in result["both"].values()) > base),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="E4e joint consequence decision")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--inputs", default="O+M+a")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    payload = load(args.data)
    report = run(payload["rows"], args.inputs)

    print("=== E4e joint consequence decision ===\n")
    print(f"decision states: {report['n_states']}   positive rates: "
          f"{report['positive_rates']}")
    print(f"conformal thresholds (quantile of the safe class): "
          f"{report['conformal_thresholds']}\n")

    print(f"  {'rule':24s} {'success':>8s} {'coverage':>9s}")
    for name in ("Oracle-V", "Oracle-S"):
        e = report["reference"][name]
        print(f"  {name:24s} {e['success_rate']:>8.4f} {e['coverage']:>9.3f}")
    e = report["baseline_v_only"]
    print(f"  {'V-only (E4-3 baseline)':24s} {e['success_rate']:>8.4f} "
          f"{e['coverage']:>9.3f}")
    e = report["joint_head"]
    print(f"  {'joint head P(V and safe)':24s} {e['success_rate']:>8.4f} "
          f"{e['coverage']:>9.3f}")
    print("  selective (conformal safety filter)")
    for q, e in report["selective"].items():
        print(f"      q={q:<6s} {e['success_rate']:>8.4f} {e['coverage']:>9.3f}")
    print("  both (conformal filter + joint head)")
    for q, e in report["both"].items():
        print(f"      q={q:<6s} {e['success_rate']:>8.4f} {e['coverage']:>9.3f}")

    v = report["verdict"]
    print(f"\nverdict: baseline {v['baseline']:.4f}  Oracle-V {v['oracle_v']:.4f}  "
          f"Oracle-S {v['oracle_s']:.4f}")
    print(f"  joint head gain {v['joint_head_gain']:+.4f}")
    print(f"  best selective gain {v['best_selective_gain']:+.4f}")
    print(f"  any rule beats baseline: {v['any_rule_beats_baseline']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
