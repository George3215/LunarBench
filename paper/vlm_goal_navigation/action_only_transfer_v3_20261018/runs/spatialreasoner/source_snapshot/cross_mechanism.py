"""E-OOD: does the action-conditioned consequence structure survive across mechanisms?

The question is no longer whether the phenomenon exists - H1, H2, H3, E4, E4c, E4d
and E-Odom established that - but whether the *structure* generalises:

    O, M, a -> S_future

The unit of generalisation here is the **mechanism**, not the terrain.  An earlier
single-step predictor was fitted and scored with one terrain held out, which for the
16 window-wall worlds still leaves fifteen members of the same mechanism in the
training set; that measures within-mechanism generalisation and cannot answer this
question.  This module holds out an entire mechanism family at once and reports how
much the consequence model and the visibility model degrade.

Reported per held-out mechanism:

* pose-consequence MAE, against the analytic nominal-displacement baseline, so the
  comparison is "did the learned model still beat the kinematic model?" rather than
  an absolute number with no reference;
* visibility ROC-AUC and PR-AUC;
* the same quantities when only a single terrain is held out, as the within-mechanism
  reference, so the drop attributable to the mechanism change is visible.

    python tasks/task3_reobservability/cross_mechanism.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .consequence_model import (analytic_delta_mae, build_training_matrix, load,
                                    pr_auc, roc_auc)
except ImportError:
    from consequence_model import (analytic_delta_mae, build_training_matrix, load,
                                   pr_auc, roc_auc)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "cross_mechanism.json"

#: Terrain -> mechanism family, from the benchmark taxonomy. The window-wall family
#: is the 16 paired-history worlds, which are one mechanism by construction.
MECHANISM_OF = {
    "barrier": "occlusion direction",
    "stairs_up": "elevation",
    "trench": "elevation",
    "pit": "spatial scale",
    "basin": "spatial scale",
    "waves": "visibility dynamics",
    "perlin": "terrain complexity",
    "perlin_rough": "terrain complexity",
}


def mechanism(terrain: str) -> str:
    if terrain.startswith(("hist", "mem")):
        return "occlusion direction"
    return MECHANISM_OF.get(terrain, "other")


def fit_and_score(rows: list[dict], action_ids: list[str], train_mask, test_mask,
                  inputs: str = "O+M+a") -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    X = build_training_matrix(rows, inputs, action_ids)
    Y = np.array([r["delta_pose"] for r in rows], dtype=float)
    y = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])

    pose_pred = np.zeros((int(test_mask.sum()), Y.shape[1]))
    for j in range(Y.shape[1]):
        model = HistGradientBoostingRegressor(max_depth=3, max_iter=250,
                                              learning_rate=0.08, random_state=7)
        model.fit(X[train_mask], Y[train_mask, j])
        pose_pred[:, j] = model.predict(X[test_mask])
    pose_mae = float(np.abs(pose_pred - Y[test_mask]).mean())

    if len(np.unique(y[train_mask])) < 2:
        return {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                "pose_mae_learned": round(pose_mae, 4), "roc_auc": None,
                "pr_auc": None}
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                         learning_rate=0.08, random_state=7)
    clf.fit(X[train_mask], y[train_mask])
    probs = clf.predict_proba(X[test_mask])[:, 1]
    return {
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "pose_mae_learned": round(pose_mae, 4),
        "roc_auc": roc_auc(y[test_mask], probs),
        "pr_auc": pr_auc(y[test_mask], probs),
    }


def analytic_on(rows: list[dict], test_mask) -> float:
    errors = []
    for row, keep in zip(rows, test_mask):
        if not keep:
            continue
        vx, vy, wz, duration = row["action_vec"]
        predicted = np.array([vx * duration, vy * duration, 0.0, wz * duration])
        errors.append(float(np.abs(predicted - np.array(row["delta_pose"])).mean()))
    return round(float(np.mean(errors)), 4) if errors else None


def main() -> None:
    p = argparse.ArgumentParser(description="E-OOD leave-one-mechanism-out")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--inputs", default="O+M+a")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    payload = load(args.data)
    rows = payload["rows"]
    action_ids = payload["action_ids"]
    terrain = np.array([r["terrain"] for r in rows])
    mech = np.array([mechanism(r["terrain"]) for r in rows])

    report = {"n_rows": len(rows), "by_mechanism": {}, "within_mechanism": {}}

    print("=== E-OOD leave-one-mechanism-out ===")
    print(f"rows={len(rows)}  mechanisms={sorted(set(mech))}\n")
    print(f"  {'held-out':22s} {'n_train':>8s} {'n_test':>7s} {'pose MAE':>9s} "
          f"{'analytic':>9s} {'AUC':>7s} {'PR-AUC':>7s}")

    for name in sorted(set(mech)):
        test = mech == name
        train = ~test
        if not train.any():
            continue
        result = fit_and_score(rows, action_ids, train, test, args.inputs)
        result["analytic_pose_mae"] = analytic_on(rows, test)
        result["learned_beats_analytic"] = bool(
            result["analytic_pose_mae"] is not None
            and result["pose_mae_learned"] < result["analytic_pose_mae"])
        report["by_mechanism"][name] = result
        print(f"  {name:22s} {result['n_train']:>8d} {result['n_test']:>7d} "
              f"{result['pose_mae_learned']:>9.4f} {result['analytic_pose_mae']:>9.4f} "
              f"{str(result['roc_auc'])[:6]:>7s} {str(result['pr_auc'])[:6]:>7s}")

    print("\n  within-mechanism reference (one terrain held out)")
    for name in sorted(set(terrain)):
        test = terrain == name
        train = ~test
        result = fit_and_score(rows, action_ids, train, test, args.inputs)
        report["within_mechanism"][name] = result
    maes = [v["pose_mae_learned"] for v in report["within_mechanism"].values()
            if v["pose_mae_learned"] is not None]
    aucs = [v["roc_auc"] for v in report["within_mechanism"].values()
            if v["roc_auc"] is not None]
    report["within_mechanism_summary"] = {
        "mean_pose_mae": round(float(np.mean(maes)), 4),
        "mean_roc_auc": round(float(np.mean(aucs)), 4),
    }
    print(f"    mean pose MAE {report['within_mechanism_summary']['mean_pose_mae']}  "
          f"mean AUC {report['within_mechanism_summary']['mean_roc_auc']}")

    out_maes = [v["pose_mae_learned"] for v in report["by_mechanism"].values()]
    out_aucs = [v["roc_auc"] for v in report["by_mechanism"].values() if v["roc_auc"]]
    report["cross_mechanism_summary"] = {
        "mean_pose_mae": round(float(np.mean(out_maes)), 4),
        "mean_roc_auc": round(float(np.mean(out_aucs)), 4) if out_aucs else None,
        "mechanisms_where_learned_beats_analytic": sum(
            1 for v in report["by_mechanism"].values() if v["learned_beats_analytic"]),
        "n_mechanisms": len(report["by_mechanism"]),
    }
    s = report["cross_mechanism_summary"]
    print(f"    mean pose MAE {s['mean_pose_mae']}  mean AUC {s['mean_roc_auc']}")
    print(f"    learned beats analytic on {s['mechanisms_where_learned_beats_analytic']}"
          f"/{s['n_mechanisms']} held-out mechanisms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
