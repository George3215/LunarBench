"""TASK3 Experiment 3: replace the oracle counterfactual with a learned predictor.

Experiments 1 and 2 measured action-conditioned future observability by running
every candidate in MuJoCo.  That is an upper bound, not a policy: it costs one
physical rollout per candidate per decision.  This module asks whether the same
decision can be made by *predicting* re-observability instead.

The predictor is trained on the physical rollouts already recorded by
``history.py`` and ``benchmark.py``: every sample is one (robot state, candidate
action) pair with the measured label "did the target become visible during the
rest of this rollout".  Two feature sets are compared, and the difference
between them is the whole point:

``obs_only``
    Robot pose and the candidate action.  No target memory.  In a paired world
    the two worlds share pose, heading and camera image *exactly* (measured
    pixel-identical in Experiment 1), so any function of those inputs is
    identical in both worlds and this model cannot possibly choose opposite
    actions for them.
``obs_memory``
    The same, plus the target position the robot remembers from before it lost
    sight of it.

Evaluation is leave-one-group-out over scenes and pairs, so a world is never
scored by a model that saw it.  The learned planner picks the action with the
highest predicted probability and is scored by the *measured* outcome of that
action, against the same baselines Experiment 1 used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from .benchmark import BENCHMARK_ACTIONS
except ImportError:
    from benchmark import BENCHMARK_ACTIONS

ROOT = Path(__file__).resolve().parents[2]
ACTION_NAMES = sorted(BENCHMARK_ACTIONS)
SUBSAMPLE_S = 0.5


def _action_vector(name: str) -> np.ndarray:
    p = BENCHMARK_ACTIONS[name]
    onehot = np.zeros(len(ACTION_NAMES), dtype=float)
    onehot[ACTION_NAMES.index(name)] = 1.0
    return np.concatenate([[p.vx, p.vy, p.wz, p.duration_s], onehot])


ACTION_VECTORS = {name: _action_vector(name) for name in ACTION_NAMES}


def _target_of(entry: dict) -> np.ndarray:
    return np.asarray(entry["target_xyz"], dtype=float)


def _pose_features(xyz, yaw) -> np.ndarray:
    return np.array([xyz[0], xyz[1], xyz[2], math.sin(yaw), math.cos(yaw)], dtype=float)


def _memory_features(xyz, yaw, target) -> np.ndarray:
    d = target - np.asarray(xyz, dtype=float)
    rng = float(np.linalg.norm(d))
    bearing = math.atan2(d[1], d[0]) - yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    return np.array([d[0], d[1], d[2], rng, math.sin(bearing), math.cos(bearing)], dtype=float)


def load_samples(reports: list[dict], subsample_s: float = SUBSAMPLE_S) -> list[dict]:
    """Flatten recorded rollouts into (state, action, label) samples."""
    samples: list[dict] = []
    for source in reports:
        # history.py stores worlds, benchmark.py stores scenes.
        entries = source.get("worlds") or source.get("scenes") or {}
        for key, entry in entries.items():
            target = _target_of(entry)
            group = entry.get("pair_id") or entry.get("scene") or key.rsplit("_", 1)[0]
            for action in entry["actions"]:
                curve = action["curve"]
                if not curve:
                    continue
                visible = [bool(c["visible"]) for c in curve]
                # The recorder samples on a fixed step (0.1 s in practice); keep
                # every stride-th sample rather than testing the timestamp
                # modulo, which never matches because t starts at one timestep.
                dt = curve[1]["t_s"] - curve[0]["t_s"] if len(curve) > 1 else subsample_s
                stride = max(1, int(round(subsample_s / dt))) if dt > 0 else 1
                for i, c in enumerate(curve):
                    if i % stride:
                        continue
                    future = any(v for v in visible[i:])
                    samples.append({
                        "source": source["task"], "group": group, "world": key,
                        "action": action["action"], "t_s": c["t_s"],
                        "is_decision_state": bool(i == 0),
                        "pose": _pose_features(c["xyz"], c["yaw_rad"]),
                        "memory": _memory_features(c["xyz"], c["yaw_rad"], target),
                        "action_vec": ACTION_VECTORS[action["action"]],
                        "label": int(future),
                    })
    return samples


def build_matrix(samples: list[dict], use_memory: bool,
                 shuffle_memory: bool = False, seed: int = 7) -> np.ndarray:
    """Feature matrix.  ``shuffle_memory`` is the control: memory values are
    permuted across samples, so their marginals survive but their association
    with the state does not."""
    parts = [np.stack([s["pose"] for s in samples])]
    if use_memory:
        mem = np.stack([s["memory"] for s in samples])
        if shuffle_memory:
            mem = mem[np.random.default_rng(seed).permutation(len(mem))]
        parts.append(mem)
    parts.append(np.stack([s["action_vec"] for s in samples]))
    return np.hstack(parts)


def out_of_fold_probabilities(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                              seed: int = 7) -> np.ndarray:
    """Leave-one-group-out predictions, so no world is scored by its own model."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    pred = np.zeros(len(y), dtype=float)
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        if not test.any() or len(np.unique(y[train])) < 2:
            pred[test] = float(y[train].mean()) if train.any() else 0.0
            continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                             learning_rate=0.08, random_state=seed)
        clf.fit(X[train], y[train])
        pred[test] = clf.predict_proba(X[test])[:, 1]
    return pred


def decision_table(samples: list[dict], probs: np.ndarray) -> list[dict]:
    """For each world, the action a planner would pick at the lost state."""
    rows: list[dict] = []
    by_world: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        if s["is_decision_state"]:
            by_world.setdefault(s["world"], []).append(i)
    for world, idx in by_world.items():
        best = max(idx, key=lambda i: probs[i])
        rows.append({
            "world": world, "group": samples[best]["group"],
            "source": samples[best]["source"],
            "learned_action": samples[best]["action"],
            "learned_prob": round(float(probs[best]), 4),
            "learned_would_succeed": bool(samples[best]["label"]),
            "candidates": [{"action": samples[i]["action"],
                            "prob": round(float(probs[i]), 4),
                            "label": int(samples[i]["label"])} for i in idx],
        })
    return rows


def summarise_decisions(decisions: list[dict]) -> dict:
    """RSR of each decision rule, over all worlds and per experiment."""
    scopes = {
        "all_worlds": decisions,
        "history_pairs": [d for d in decisions if "history" in d["source"]],
        "multi_terrain": [d for d in decisions if "history" not in d["source"]],
    }
    out: dict = {"n_worlds": len(decisions)}

    def rate(rows, picker):
        if not rows:
            return None
        hits = 0
        for d in rows:
            want = picker(d)
            row = next((c for c in d["candidates"] if c["action"] == want), None)
            if row and row["label"]:
                hits += 1
        return round(hits / len(rows), 4)

    oracle_pick = lambda d: min((c for c in d["candidates"] if c["label"]),
                                key=lambda c: c["action"])["action"]
    for name, rows in scopes.items():
        out[name] = {
            "n_worlds": len(rows),
            "A_reactive_forward": rate(rows, lambda d: "forward"),
            "B_history_only_toward": rate(rows, lambda d: "toward"),
            "D_initial_heading_lock": rate(rows, lambda d: "heading_lock"),
            "learned": rate(rows, lambda d: d["learned_action"]),
            "C_oracle_counterfactual": rate(rows, oracle_pick),
        }
    return out


def run(report_paths: list[Path], seed: int = 7) -> dict:
    from sklearn.metrics import roc_auc_score

    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    samples = load_samples(reports)
    groups = np.array([s["group"] for s in samples])
    y = np.array([s["label"] for s in samples])
    result: dict = {"n_samples": len(samples), "n_groups": int(len(np.unique(groups))),
                    "positive_rate": round(float(y.mean()), 4),
                    "sources": [str(p) for p in report_paths], "feature_sets": {}}
    variants = (("obs_only", False, False),
                ("obs_memory", True, False),
                ("obs_memory_shuffled_control", True, True))
    for label, use_memory, shuffle in variants:
        X = build_matrix(samples, use_memory, shuffle, seed)
        probs = out_of_fold_probabilities(X, y, groups, seed)
        auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None
        decisions = decision_table(samples, probs)
        chosen: dict[str, int] = {}
        for d in decisions:
            chosen[d["learned_action"]] = chosen.get(d["learned_action"], 0) + 1
        result["feature_sets"][label] = {
            "n_features": int(X.shape[1]),
            "oof_auc": None if auc is None else round(auc, 4),
            "oof_accuracy": round(float(((probs > 0.5).astype(int) == y).mean()), 4),
            "decisions": summarise_decisions(decisions),
            "chosen_action_histogram": chosen,
        }
        if not shuffle:
            result.setdefault("chosen_actions", {})[label] = [d["learned_action"]
                                                              for d in decisions]
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 Experiment 3: learned visibility predictor")
    p.add_argument("--reports", nargs="*", type=Path, default=[
        Path("tasks/task3_reobservability/generated/history_necessity.json"),
        Path("tasks/task3_reobservability/generated/benchmark.json")])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/visibility_model.json"))
    args = p.parse_args()

    out = run(args.reports, args.seed)
    print(f"samples={out['n_samples']} groups={out['n_groups']} "
          f"positive_rate={out['positive_rate']}")
    for name, entry in out["feature_sets"].items():
        print(f"\n--- {name} ({entry['n_features']} features) "
              f"oof_auc={entry['oof_auc']} acc={entry['oof_accuracy']}")
        for scope in ("all_worlds", "history_pairs", "multi_terrain"):
            d = entry["decisions"][scope]
            print(f"    {scope} (n={d['n_worlds']}): " + "  ".join(
                f"{k}={v}" for k, v in d.items() if k != "n_worlds"))
        print(f"    chosen histogram: {entry['chosen_action_histogram']}")
    for label, actions in out.get("chosen_actions", {}).items():
        print(f"\nchosen per world ({label}):")
        print("   ", actions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
