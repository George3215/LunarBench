"""E4: what information does the re-observation decision actually need?

H1, H2 and H3 each remove one explanation.  E4 asks what is left, in four parts
that share one dataset and one evaluation protocol:

**E4-1 Information ablation.**  ``O`` / ``M`` / ``O+M`` / ``O+M+a`` predicting
re-observation.  The comparison that matters is ``O+M`` vs ``O+M+a``: if adding
the action buys nothing, "action-conditioned" is an assumption rather than a
finding.  Reported with ROC-AUC, PR-AUC, Brier, expected calibration error and
paired-world accuracy - the last because a good average AUC can still hide the
ambiguity H1 and H2 construct.

**E4-2 Analytic vs learned pose consequence.**  H3 located the missing piece:
the oracle at the *executed* pose agrees with the physical outcome 97.9 % of the
time, so what fails is ``a -> S_future``.  So this part predicts the executed
**body-frame pose delta** and compares a learned model against the analytic
nominal displacement.  The visibility oracle is then applied to the predicted
pose, which keeps the already-validated part validated.

**E4-3 Prediction -> planning.**  Both models drive a planner whose score is
``P(re-observe | O, M, a)``, evaluated on the multi-step episode protocol against
the classical baselines.

**E4-4 History ablation.**  ``H0 / H1 / H2 / H4 / H8``.  H3 measured a 0.40 m
height error, which is gait, so a single frame may not know whether the robot is
climbing, descending or settled.

Splits are by **mechanism**, never random: the same terrain family must not appear
on both sides, or terrain-specific correlation leaks.

    python tasks/task3_reobservability/consequence_model.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
ACTION_HISTORY_DIM = 4
#: Components of the body-frame pose delta predicted in E4-2.
DELTA_NAMES = ["forward_m", "lateral_m", "height_m", "yaw_rad"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}; run consequence_dataset.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def feature_matrix(rows: list[dict], inputs: str, history: int = 8) -> np.ndarray:
    """Build the feature matrix for one ablation condition.

    ``O``    pose + proprioception + ``history`` pose deltas
    ``M``    target memory only (a zero vector plus a presence flag when absent)
    ``O+M``  both
    ``O+M+a`` both plus the action
    ``a``    action only

    The history block is trimmed *by dropping the oldest entries*, never by
    changing the feature definition, so H0 and H8 differ only in how far back the
    window reaches.
    """
    keep = history * ACTION_HISTORY_DIM
    out = []
    for row in rows:
        parts = []
        if "O" in inputs:
            obs = list(row["observation"])
            pose_proprio = obs[:len(obs) - 4 * 8]
            hist = obs[len(obs) - 4 * 8:]
            hist = hist[len(hist) - keep:] if keep else []
            while len(hist) < keep:
                hist = [0.0] + hist
            parts.extend(pose_proprio)
            parts.extend(hist)
        if "M" in inputs:
            memory = row["memory"]
            if memory is None:
                # absence is its own signal, not a zero vector
                parts.extend([0.0] * 7 + [0.0])
            else:
                parts.extend([float(v) for v in memory] + [1.0])
        if "a" in inputs:
            parts.extend([float(v) for v in row["action_vec"]])
            onehot = [0.0] * len(row.get("_actions", []))
            parts.extend(onehot)
        out.append(parts)
    return np.array(out, dtype=float)


def action_onehots(rows: list[dict], action_ids: list[str]) -> None:
    for row in rows:
        row["_actions"] = action_ids


def mechanism_split(rows: list[dict]) -> list[str]:
    return sorted({r["terrain"] for r in rows})


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Standard ECE over equal-width probability bins.

    Reported because Brier alone cannot distinguish "wrong" from
    "confidently wrong", and a planner takes argmax over these probabilities.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


# ---------------------------------------------------------------------------
# E4-1 / E4-2
# ---------------------------------------------------------------------------

def fit_binary(X_train, y_train, X_test, seed: int = 7) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), float(y_train.mean()))
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                           learning_rate=0.08, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def fit_regression(X_train, Y_train, X_test, seed: int = 7) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = np.zeros((len(X_test), Y_train.shape[1]))
    for j in range(Y_train.shape[1]):
        model = HistGradientBoostingRegressor(max_depth=3, max_iter=250,
                                              learning_rate=0.08, random_state=seed)
        model.fit(X_train, Y_train[:, j])
        out[:, j] = model.predict(X_test)
    return out


def leave_one_terrain_out(rows: list[dict], inputs: str, history: int,
                          action_ids: list[str], target: str) -> tuple[np.ndarray, dict]:
    """Out-of-fold predictions with the test terrain never seen in training."""
    X = feature_matrix(rows, inputs, history)
    if "a" in inputs:
        onehot = np.zeros((len(rows), len(action_ids)))
        for i, row in enumerate(rows):
            onehot[i, action_ids.index(row["action"])] = 1.0
        X = np.hstack([X, onehot])
    y = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
    Y = np.array([r["delta_pose"] for r in rows], dtype=float)
    terrain = np.array([r["terrain"] for r in rows])

    probs = np.zeros(len(rows))
    deltas = np.zeros_like(Y)
    for t in np.unique(terrain):
        test = terrain == t
        train = ~test
        if not train.any():
            continue
        if target == "visibility":
            probs[test] = fit_binary(X[train], y[train], X[test])
        else:
            deltas[test] = fit_regression(X[train], Y[train], X[test])
    if target == "visibility":
        return probs, {"y": y}
    return deltas, {"Y": Y}


def evaluate_condition(rows, inputs, history, action_ids, seed=7) -> dict:
    probs, ctx = leave_one_terrain_out(rows, inputs, history, action_ids, "visibility")
    y = ctx["y"]
    deltas, dctx = leave_one_terrain_out(rows, inputs, history, action_ids, "delta")
    Y = dctx["Y"]
    mae = np.abs(deltas - Y).mean(axis=0)
    return {
        "inputs": inputs,
        "history": history,
        "n": len(rows),
        "roc_auc": roc_auc(y, probs),
        "pr_auc": pr_auc(y, probs),
        "brier": brier(y, probs),
        "ece": expected_calibration_error(y, probs),
        "delta_mae": {name: round(float(v), 4) for name, v in zip(DELTA_NAMES, mae)},
        "delta_mae_mean": round(float(mae.mean()), 4),
        "probs": probs,
    }


def analytic_delta_mae(rows: list[dict], action_ids) -> dict:
    """The analytic baseline: the action moves the robot by its nominal command.

    This is exactly what a kinematic / next-best-view predictor assumes, and H3
    already showed it is wrong; E4-2 quantifies by how much, against the same
    target as the learned model.
    """
    nominal = {}
    for row in rows:
        vx, vy, wz, duration = row["action_vec"]
        yaw = row["start_pose"][3]
        dx_local, dy_local = vx * duration, vy * duration
        nominal.setdefault(row["action"], []).append(
            [dx_local, dy_local, 0.0, wz * duration])
    errors = []
    per_action: dict[str, list[float]] = {}
    for row in rows:
        vx, vy, wz, duration = row["action_vec"]
        predicted = np.array([vx * duration, vy * duration, 0.0, wz * duration])
        actual = np.array(row["delta_pose"])
        err = np.abs(predicted - actual)
        errors.append(err)
        per_action.setdefault(row["action"], []).append(float(err.mean()))
    errors = np.array(errors)
    return {
        "delta_mae": {name: round(float(v), 4) for name, v in zip(DELTA_NAMES,
                                                                 errors.mean(axis=0))},
        "delta_mae_mean": round(float(errors.mean()), 4),
        "per_action_mean_abs_error": {k: round(float(np.mean(v)), 4)
                                      for k, v in per_action.items()},
    }


def paired_world_accuracy(rows: list[dict], probs: np.ndarray) -> dict:
    """On the H1/H2 paired worlds, does the model pick a winning action?

    A high average AUC can coexist with a model that picks the same action in both
    worlds of a pair.  This measures the thing the pairs were built to test.
    """
    by_world: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row["state_id"] == 0:
            by_world.setdefault(row["terrain"], []).append(i)
    pairs: dict[str, dict] = {}
    for terrain, idx in by_world.items():
        if not terrain.startswith("hist"):
            continue
        best = max(idx, key=lambda i: probs[i])
        winners = {rows[i]["action"] for i in idx if rows[i]["reobserved"]}
        pairs[terrain] = {"chosen": rows[best]["action"], "winners": sorted(winners),
                          "success": rows[best]["action"] in winners}
    by_pair: dict[str, dict] = {}
    for terrain, entry in pairs.items():
        pair_id = terrain.rsplit("_", 1)[0]
        by_pair.setdefault(pair_id, {})[terrain.rsplit("_", 1)[1]] = entry
    differing = sum(1 for v in by_pair.values()
                    if len(v) == 2 and v["left"]["chosen"] != v["right"]["chosen"])
    solved = sum(1 for e in pairs.values() if e["success"])
    return {
        "n_worlds": len(pairs),
        "n_pairs": len(by_pair),
        "world_accuracy": round(solved / len(pairs), 4) if pairs else None,
        "pairs_where_the_model_switches_action": differing,
        "pairs": by_pair,
    }


def live_features(observation, action_vec: list[float], action_onehot: list[float],
                  inputs: str) -> list[float]:
    """Features for one live (observation, action) pair, matching the training layout.

    Kept structurally identical to :func:`feature_matrix`: pose, proprioception,
    memory-with-presence-flag, action vector and action one-hot.  The planner uses
    ``H0`` so the history block is empty; E4-4 measured that the visibility AUC is
    unchanged without history.
    """
    pose = observation.pose
    gravity = observation.proprioception[-3:]
    parts = list([pose.x, pose.y, pose.z, math.sin(pose.yaw), math.cos(pose.yaw),
                  *[float(g) for g in gravity]])
    if "M" in inputs:
        memory = observation.target_memory
        if memory is None:
            parts.extend([0.0] * 7 + [0.0])
        else:
            d = memory.xyz - pose.xyz
            c, s = math.cos(-pose.yaw), math.sin(-pose.yaw)
            body = [d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2]]
            bearing = math.atan2(d[1], d[0]) - pose.yaw
            parts.extend([float(body[0]), float(body[1]), float(body[2]),
                          float(np.linalg.norm(d)), math.sin(bearing),
                          math.cos(bearing), float(memory.age_s), 1.0])
    if "a" in inputs:
        parts.extend(action_vec)
        parts.extend(action_onehot)
    return parts


class ConsequencePlanner:
    """Score every candidate with a trained consequence model, take the argmax.

    The model is retrained for each evaluated terrain with that terrain held out,
    so the planner is never scored on data it saw.
    """

    def __init__(self, model, action_ids: list[str], action_vectors: dict,
                 inputs: str = "O+a", name: str = "LearnedConsequence"):
        self.name = name
        self.requires = frozenset({"proprioception", "target_memory"})
        self.model = model
        self.action_ids = action_ids
        self.action_vectors = action_vectors
        self.inputs = inputs

    def reset(self, observation) -> None:
        pass

    def scores(self, observation) -> dict:
        rows = []
        for action_id in self.action_ids:
            onehot = [1.0 if a == action_id else 0.0 for a in self.action_ids]
            rows.append(live_features(observation, self.action_vectors[action_id],
                                      onehot, self.inputs))
        X = np.array(rows, dtype=float)
        probs = self.model.predict_proba(X)[:, 1]
        return dict(zip(self.action_ids, probs))

    def act(self, observation) -> str:
        scores = self.scores(observation)
        return max(scores, key=lambda a: (scores[a], a))


def build_training_matrix(rows: list[dict], inputs: str, action_ids: list[str],
                          history: int = 0) -> np.ndarray:
    """Training features for a condition that includes the action.

    ``inputs`` is one of ``O+a``, ``M+a``, ``O+M+a``; the action block (vector plus
    one-hot) is appended in the same order :func:`live_features` uses, so the
    planner can rebuild it exactly.
    """
    base = feature_matrix(rows, inputs.replace("+a", ""), history)
    action_vec = np.array([r["action_vec"] for r in rows], dtype=float)
    onehot = np.array([[1.0 if a == r["action"] else 0.0 for a in action_ids]
                       for r in rows])
    return np.hstack([base, action_vec, onehot])


class ConsequencePlanner:
    """Score every candidate with a trained consequence model, take the argmax.

    The model is retrained for each evaluated terrain with that terrain held out,
    so the planner is never scored on data it saw.
    """

    def __init__(self, model, action_ids: list[str], action_vectors: dict,
                 inputs: str = "O+M+a", name: str = "LearnedConsequence"):
        self.name = name
        self.requires = frozenset({"proprioception", "target_memory"})
        self.model = model
        self.action_ids = action_ids
        self.action_vectors = action_vectors
        self.inputs = inputs

    def reset(self, observation) -> None:
        pass

    def scores(self, observation) -> dict:
        rows = []
        for action_id in self.action_ids:
            onehot = [1.0 if a == action_id else 0.0 for a in self.action_ids]
            # Pass ``self.inputs`` through unchanged: live_features appends the
            # action block itself, so stripping "+a" here would drop the very
            # features the condition is about and change the column count.
            rows.append(live_features(observation, self.action_vectors[action_id],
                                      onehot, self.inputs))
        probs = self.model.predict_proba(np.array(rows, dtype=float))[:, 1]
        return dict(zip(self.action_ids, probs))

    def act(self, observation) -> str:
        scores = self.scores(observation)
        return max(scores, key=lambda a: (scores[a], a))


def run_planner(data: Path, inputs: str = "O+M+a", max_steps: int = 10) -> dict:
    """E4-3: drive the multi-step protocol with the learned consequence planner."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    import api
    import episode as ep

    payload = load(data)
    rows = payload["rows"]
    action_ids = payload["action_ids"]
    X = build_training_matrix(rows, inputs, action_ids)
    y = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
    terrain_of = np.array([r["terrain"] for r in rows])
    action_vectors = {r["action"]: r["action_vec"] for r in rows}

    results = []
    for name, scene in api.scene_catalogue().items():
        train = terrain_of != name
        if not train.any() or len(np.unique(y[train])) < 2:
            continue
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.08, random_state=7)
        model.fit(X[train], y[train])
        planner = ConsequencePlanner(model, action_ids, action_vectors, inputs=inputs)
        environment = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table())
        try:
            episode = ep.run_episode(planner, environment, max_steps=max_steps,
                                     protocol="B")
            episode.failure_type = ep.classify_failure(episode, None, None, [])
            results.append(episode)
            print(f"  {name:18s} {episode.termination:18s} steps={episode.steps_used} "
                  f"{'ok' if episode.success else 'fail'}  {episode.actions}",
                  flush=True)
        finally:
            environment.close()
    summary = ep.summarise(results)
    summary["per_terrain"] = {e.scene: {"termination": e.termination,
                                        "steps": e.steps_used,
                                        "actions": e.actions}
                              for e in results}
    return summary


#: An action is "unsafe" when it leaves the terrain or drops below this uprightness
#: margin.  A binary *fall* label is deliberately not used: it has zero positives in
#: 720 single-action rollouts, so it carries no learnable signal, while the margin
#: and the boundary check are defined for every action.
SAFETY_MARGIN_KEEP_UPRIGHT = 0.90


def safety_labels(rows: list[dict]) -> np.ndarray:
    return np.array([1.0 if (r.get("out_of_bounds")
                             or r.get("upright_min", 1.0) < SAFETY_MARGIN_KEEP_UPRIGHT)
                     else 0.0 for r in rows])


def fit_binary_multi(X_train, Y_train, X_test, seed: int = 7) -> np.ndarray:
    """One probability column per binary head."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    out = np.zeros((len(X_test), Y_train.shape[1]))
    for j in range(Y_train.shape[1]):
        column = Y_train[:, j]
        if len(np.unique(column)) < 2:
            out[:, j] = float(column.mean())
            continue
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.08, random_state=seed)
        model.fit(X_train, column)
        out[:, j] = model.predict_proba(X_test)[:, 1]
    return out


def full_consequence_heads(rows: list[dict], inputs: str, action_ids: list[str],
                           history: int = 0) -> dict:
    """Out-of-fold predictions for every consequence head, leaving out each terrain."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    X = build_training_matrix(rows, inputs, action_ids, history)
    y_vis = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
    y_safe = safety_labels(rows)
    Y_pose = np.array([r["delta_pose"] for r in rows], dtype=float)
    Y_upright = np.array([r.get("upright_min", 1.0) for r in rows], dtype=float)
    terrain = np.array([r["terrain"] for r in rows])

    p_vis = np.zeros(len(rows)); p_safe = np.zeros(len(rows))
    pose = np.zeros_like(Y_pose); upright = np.zeros(len(rows))
    for t in np.unique(terrain):
        test, train = terrain == t, terrain != t
        if not train.any():
            continue
        if len(np.unique(y_vis[train])) >= 2:
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.08, random_state=7)
            m.fit(X[train], y_vis[train]); p_vis[test] = m.predict_proba(X[test])[:, 1]
        else:
            p_vis[test] = y_vis[train].mean()
        if len(np.unique(y_safe[train])) >= 2:
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.08, random_state=7)
            m.fit(X[train], y_safe[train]); p_safe[test] = m.predict_proba(X[test])[:, 1]
        else:
            p_safe[test] = y_safe[train].mean()
        for j in range(Y_pose.shape[1]):
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=250,
                                              learning_rate=0.08, random_state=7)
            m.fit(X[train], Y_pose[train, j]); pose[test, j] = m.predict(X[test])
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=250,
                                          learning_rate=0.08, random_state=7)
        m.fit(X[train], Y_upright[train]); upright[test] = m.predict(X[test])

    return {"p_visible": p_vis, "p_unsafe": p_safe, "pose": pose,
            "upright_min": upright, "y_visible": y_vis, "y_unsafe": y_safe,
            "Y_pose": Y_pose, "Y_upright": Y_upright}


def safety_head_report(heads: dict) -> dict:
    y, p = heads["y_unsafe"], heads["p_unsafe"]
    return {
        "positive_rate": round(float(y.mean()), 4),
        "n_positive": int(y.sum()),
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
        "brier": round(brier(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
    }


def oracle_planner_ablation(rows: list[dict], action_ids: list[str],
                            inputs: str = "O+M+a") -> dict:
    """Where does the learned planner's remaining loss come from?

    Four planners that differ only in which consequence they know:

    ``Oracle-V``   knows the true visibility outcome, ignores safety
    ``Oracle-S``   knows the true visibility outcome *and* the true safety outcome
    ``Learned-V``  predicted visibility only - the E4-3 planner
    ``Learned-V+S`` predicted visibility *and* predicted safety

    Comparing ``Learned-V`` against ``Oracle-V`` isolates prediction error from
    planning error; comparing ``Oracle-V`` against ``Oracle-S`` isolates how much
    of the remaining loss is the safety term at all.
    """
    heads = full_consequence_heads(rows, inputs, action_ids)
    terrain = np.array([r["terrain"] for r in rows])
    state = np.array([r["state_id"] for r in rows])

    def pick(score: np.ndarray, mask: np.ndarray) -> dict:
        """Best action per decision state under a score, and whether it worked."""
        chosen, solved, total = [], 0, 0
        for t in np.unique(terrain):
            for s in np.unique(state[mask & (terrain == t)]):
                sel = mask & (terrain == t) & (state == s)
                idx = np.where(sel)[0]
                if idx.size == 0:
                    continue
                best = idx[int(np.argmax(score[idx]))]
                chosen.append(rows[best]["action"])
                total += 1
                solved += int(bool(rows[best]["reobserved"])
                              and not bool(rows[best].get("out_of_bounds"))
                              and float(rows[best].get("upright_min", 1.0)) >=
                              SAFETY_MARGIN_KEEP_UPRIGHT)
        return {"states": total, "solved": solved,
                "success_rate": round(solved / total, 4) if total else None,
                "chosen_actions": chosen}

    mask = np.ones(len(rows), dtype=bool)
    true_vis = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
    true_unsafe = safety_labels(rows)
    result = {
        "Oracle-V": pick(true_vis.astype(float), mask),
        "Oracle-S": pick(true_vis - 1e3 * true_unsafe, mask),
        "Learned-V": pick(heads["p_visible"], mask),
        "Learned-V+S": pick(heads["p_visible"] - 1e3 * heads["p_unsafe"], mask),
        "safety_head": safety_head_report(heads),
        "pose_mae_learned": {name: round(float(np.abs(heads["pose"][:, j]
                                                     - heads["Y_pose"][:, j]).mean()), 4)
                             for j, name in enumerate(DELTA_NAMES)},
        "upright_mae": round(float(np.abs(heads["upright_min"]
                                          - heads["Y_upright"]).mean()), 4),
    }

    # A hard veto is a strong assumption about the safety head's precision.  The
    # oracle shows the safety *term* is worth having; if the learned variant loses
    # instead, the question is whether that is the term or the veto.  Sweeping a
    # soft penalty answers it: a soft penalty that beats both the no-safety and the
    # hard-veto variant means the information is good and only the decision rule
    # was wrong.
    result["soft_penalty_sweep"] = {
        f"lambda={lam:g}": pick(heads["p_visible"] - lam * heads["p_unsafe"], mask)
        for lam in (0.5, 1.0, 2.0, 5.0)
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="TASK3 E4: information and consequence ablations")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path,
                   default=Path("tasks/task3_reobservability/generated/consequence_model.json"))
    p.add_argument("--planner", action="store_true",
                   help="also run E4-3: the learned consequence planner on the episode protocol")
    p.add_argument("--planner-inputs", default="O+M+a",
                   choices=["O+a", "M+a", "O+M+a"])
    p.add_argument("--full", action="store_true",
                   help="also run E4-5: full consequence heads and the oracle ablation")
    args = p.parse_args()

    payload = load(args.data)
    rows = payload["rows"]
    action_ids = payload["action_ids"]
    print(f"rows={len(rows)}  terrains={len(set(r['terrain'] for r in rows))}  "
          f"positive_rate={np.mean([r['reobserved'] for r in rows]):.3f}\n")

    print("E4-1 information ablation (leave-one-terrain-out)")
    conditions = []
    for inputs in ("O", "M", "O+M", "O+M+a"):
        result = evaluate_condition(rows, inputs, payload["history_len"], action_ids)
        conditions.append(result)
        print(f"  {inputs:7s} AUC={result['roc_auc']:.4f}  PR-AUC={result['pr_auc']:.4f}  "
              f"Brier={result['brier']:.4f}  ECE={result['ece']:.4f}  "
              f"pose MAE={result['delta_mae_mean']:.4f} m/rad")

    print("\nE4-2 analytic vs learned pose consequence")
    analytic = analytic_delta_mae(rows, action_ids)
    learned = conditions[-1]
    print(f"  analytic  pose MAE mean = {analytic['delta_mae_mean']:.4f}")
    print(f"  learned   pose MAE mean = {learned['delta_mae_mean']:.4f}")
    for name in DELTA_NAMES:
        print(f"    {name:11s} analytic={analytic['delta_mae'][name]:.4f}  "
              f"learned={learned['delta_mae'][name]:.4f}")

    print("\nE4-1b paired-world behaviour (the thing H1/H2 construct)")
    paired = paired_world_accuracy(rows, conditions[-1]["probs"])
    print(f"  worlds={paired['n_worlds']}  accuracy={paired['world_accuracy']}  "
          f"pairs where the model switches action="
          f"{paired['pairs_where_the_model_switches_action']}/{paired['n_pairs']}")

    print("\nE4-4 history ablation")
    history_results = []
    for history in (0, 1, 2, 4, 8):
        result = evaluate_condition(rows, "O+M+a", history, action_ids)
        history_results.append(result)
        print(f"  H{history:<2d} AUC={result['roc_auc']:.4f}  Brier={result['brier']:.4f}  "
              f"pose MAE={result['delta_mae_mean']:.4f}")

    out = {
        "task": "TASK3_e4_consequence_model",
        "n_rows": len(rows),
        "conditions": [{k: v for k, v in c.items() if k != "probs"} for c in conditions],
        "analytic": analytic,
        "paired_worlds": paired,
        "history_ablation": [{k: v for k, v in c.items() if k != "probs"}
                             for c in history_results],
    }

    if args.planner:
        print(f"\nE4-3 learned consequence planner on the episode protocol "
              f"(inputs={args.planner_inputs})")
        out["planner"] = run_planner(args.data, args.planner_inputs)
        s = out["planner"]
        print(f"  RSR@1={s['rsr@1']:.3f}  RSR@3={s['rsr@3']:.3f}  "
              f"RSR@5={s['rsr@5']:.3f}  RSR@10={s['rsr@10']:.3f}  "
              f"cost={s['mean_motion_cost_m']:.2f} m")

    if args.full:
        print("\nE4-5 full consequence model: visibility + pose + safety")
        ablation = oracle_planner_ablation(rows, action_ids, args.planner_inputs)
        out["full_consequence"] = ablation
        for name, entry in ablation["soft_penalty_sweep"].items():
            print(f"  soft {name:10s} one-step success {entry['success_rate']} "
                  f"({entry['solved']}/{entry['states']})")
        head = ablation["safety_head"]
        print(f"  safety head: positive rate {head['positive_rate']} "
              f"(n={head['n_positive']})  AUC={head['roc_auc']}  "
              f"PR-AUC={head['pr_auc']}  Brier={head['brier']}  ECE={head['ece']}")
        print(f"  upright margin MAE = {ablation['upright_mae']}")
        for name in ("Oracle-V", "Oracle-S", "Learned-V", "Learned-V+S"):
            entry = ablation[name]
            print(f"  {name:12s} one-step success {entry['success_rate']} "
                  f"({entry['solved']}/{entry['states']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
