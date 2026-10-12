"""EXP-0: information regime audit, and the memory-vs-vision substitution test.

The reframing raises one question that decides whether the existing results stand:

    does ``O+M+a`` contain privileged information?

If the target-memory block is a ground-truth world position, then E4-3 measures
*localisation plus planning*, not *visual reacquisition*, and it has to be redone.
So this script does two things and nothing else:

**Audit.** Print, feature by feature, exactly what each block contains and where the
value comes from, so the classification is checked rather than asserted.

**Substitution test.** The decisive ablation is not "vision vs no vision" but:

    state                pose, proprioception, history, action      - no memory, no camera
    state + memory       the current E4-3 input
    state + vision       the camera *instead of* the memory
    state + memory + vis both

If ``state + vision`` recovers what ``state + memory`` provides, then a deployable
visual agent could in principle substitute for the structured memory and the paper's
story survives. If it does not, the gap is an information gap that a visual
representation has to close, which is the honest form of the argument.

    python tasks/task3_reobservability/regime_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .consequence_model import build_training_matrix, load, pr_auc, roc_auc
except ImportError:
    from consequence_model import build_training_matrix, load, pr_auc, roc_auc

ROOT = Path(__file__).resolve().parents[2]
GEN = Path(__file__).resolve().parent / "generated"
OUT = GEN / "regime_audit.json"

#: Every feature block, classified by whether a deployed robot could obtain it.
REGIME = {
    "pose (x, y, z, sin yaw, cos yaw)":
        ("deployment", "odometry / self-motion"),
    "projected gravity (3)":
        ("deployment", "IMU"),
    "pose history (4 x H)":
        ("deployment", "past odometry"),
    "action vector (4)": ("deployment", "chosen by the agent"),
    "action one-hot (10)": ("deployment", "chosen by the agent"),
    "memory: body-frame target vector, range, bearing, age (7)":
        ("conditional", "detection at the last sighting + odometry; exact, drift-free, "
                         "and the target is assumed static"),
    "memory presence flag (1)": ("deployment", "whether a sighting exists"),
    "RGB / depth": ("deployment", "onboard camera"),
}


def audit() -> dict:
    print("=== EXP-0 information regime audit ===\n")
    print(f"  {'feature block':62s} {'regime':12s} source")
    for block, (regime, source) in REGIME.items():
        print(f"  {block:62s} {regime:12s} {source}")
    print()
    print("  No block is ground-truth target position: the memory stores a body-frame")
    print("  vector reconstructed from the last camera detection using the robot's own")
    print("  pose, which the robot holds as odometry. It is therefore not injected")
    print("  state. It is nevertheless *exact* - noise-free and drift-free - which a")
    print("  real detector would not be, and the target is static. That is the honest")
    print("  classification: deployment-valid in kind, optimistic in quality.")
    return {"regime": REGIME}


def substitution() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    payload = load(GEN / "consequence_dataset.json")
    rows = payload["rows"]
    action_ids = payload["action_ids"]
    y = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
    terrain = np.array([r["terrain"] for r in rows])

    from consequence_model import feature_matrix
    pose = feature_matrix(rows, "O", 8)[:, :8 + 32]
    memory = feature_matrix(rows, "M", 0)
    action_vec = np.array([r["action_vec"] for r in rows], dtype=float)
    onehot = np.array([[1.0 if a == r["action"] else 0.0 for a in action_ids]
                       for r in rows])
    state = np.hstack([pose, action_vec, onehot])
    state_memory = np.hstack([state, memory])

    blob = np.load(GEN / "vision_frames.npz", allow_pickle=True)
    index = {str(k): i for i, k in enumerate(blob["keys"])}
    idx = np.array([index[f"{r['terrain']}|{r['state_id']}"] for r in rows])
    import vision_ablation as va
    vis = va.encode(blob["rgb"])[idx]
    dep = va.depth_features(blob["depth"])[idx]

    from sklearn.decomposition import PCA
    conditions = {
        "state (no memory, no camera)": ("raw", state, None),
        "state + memory  (current E4-3 input)": ("raw", state_memory, None),
        "state + vision  (camera, no memory)": ("pca", state, np.hstack([vis, dep])),
        "state + memory + vision": ("pca", state_memory, np.hstack([vis, dep])),
    }
    results = {}
    for name, (mode, raw, block) in conditions.items():
        probs = np.zeros(len(rows))
        for t in np.unique(terrain):
            test, train = terrain == t, terrain != t
            if len(np.unique(y[train])) < 2:
                probs[test] = y[train].mean()
                continue
            if mode == "raw":
                Xtr, Xte = raw[train], raw[test]
            else:
                p = PCA(n_components=16, random_state=0)
                tr, te = p.fit_transform(block[train]), p.transform(block[test])
                Xtr, Xte = np.hstack([raw[train], tr]), np.hstack([raw[test], te])
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.08, random_state=7)
            m.fit(Xtr, y[train])
            probs[test] = m.predict_proba(Xte)[:, 1]
        results[name] = {"roc_auc": roc_auc(y, probs), "pr_auc": pr_auc(y, probs),
                         "n_features": int(Xtr.shape[1])}
        print(f"  {name:40s} features={Xtr.shape[1]:>4d}  AUC={results[name]['roc_auc']:.4f}"
              f"  PR-AUC={results[name]['pr_auc']:.4f}")

    no_mem = results["state (no memory, no camera)"]["roc_auc"]
    with_mem = results["state + memory  (current E4-3 input)"]["roc_auc"]
    with_vis = results["state + vision  (camera, no memory)"]["roc_auc"]
    both = results["state + memory + vision"]["roc_auc"]
    return {"conditions": results, "verdict": {
        "memory_contribution": round(with_mem - no_mem, 4),
        "vision_contribution_without_memory": round(with_vis - no_mem, 4),
        "vision_recovers_memory": bool(with_vis >= with_mem - 0.01),
        "vision_contribution_given_memory": round(both - with_mem, 4),
    }}


def main() -> None:
    report = audit()
    report["substitution"] = substitution()
    v = report["substitution"]["verdict"]
    print("\n=== memory vs vision substitution ===")
    for k, val in v.items():
        print(f"  {k}: {val}")
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
