"""E4d: is a single-step consequence model enough, or is the action *sequence* required?

The scientific question is not "does searching help" but:

    is a one-step consequence model sufficient to support re-observation, or must
    the action sequence be modelled explicitly?

**The constraint that makes this a real test.**  For H > 1 the intermediate states
must come from the model:

    S_hat_{t+1} = f(O_t, M_t, a_t)
    S_hat_{t+2} = f(O_hat_{t+1}, M_t, a_{t+1})
    ...

Never from ground-truth stepping.  Stepping the real simulator and then reading off
the true visibility would measure an oracle simulator, not what horizon does to a
learned consequence model.  The only real execution in the whole experiment is:

    real state -> predict H-step sequence -> choose -> execute FIRST action
               -> real observation -> replan

which is receding-horizon control, and which also means model error at step k is fed
back into the input at step k+1 exactly as it would be on a robot.

**Two things are reported that a plain success rate would hide.**

* ``T_reobs`` and ``C_action`` alongside ``RSR@K``, so "longer horizon helps",
  "longer horizon just spends more actions" and "longer horizon accumulates model
  error" can be told apart.
* the **chained-prediction error**: at each decision the planner records where it
  believes the first action will take the robot, and this is compared with where the
  robot actually ends up.  That is model-error accumulation measured directly rather
  than inferred from a drop in success.

    python tasks/task3_reobservability/horizon_planner.py --horizons 1 2 3 5
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
    from .consequence_model import build_training_matrix, load
except ImportError:
    import api
    import episode as ep
    from consequence_model import build_training_matrix, load

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "horizon_planner.json"


@dataclass
class Belief:
    """The robot's own predicted state, used as model input during lookahead."""

    x: float
    y: float
    z: float
    yaw: float
    gravity: tuple[float, float, float]


def memory_block(target_world, belief: Belief, age_s: float) -> list[float]:
    """Target memory in the *body* frame of a predicted pose.

    Two things have to be maintained here, and both are easy to get silently wrong:

    * the block is **body-relative**, so it must be recomputed at every chained
      step.  Holding it fixed while the predicted pose moves would feed the model a
      target that follows the robot;
    * the **age** must advance with the lookahead.  The training rows carry the real
      memory age, so a rollout that pins it to zero is feeding the model a state
      from a different distribution than it was fitted on - and at H=1 it would
      also stop the planner from reproducing the single-step baseline it is
      supposed to be the H=1 case of.
    """
    if target_world is None:
        return [0.0] * 7 + [0.0]
    d = target_world - np.array([belief.x, belief.y, belief.z])
    c, s = math.cos(-belief.yaw), math.sin(-belief.yaw)
    body = [d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2]]
    bearing = math.atan2(d[1], d[0]) - belief.yaw
    return [float(body[0]), float(body[1]), float(body[2]), float(np.linalg.norm(d)),
            math.sin(bearing), math.cos(bearing), float(age_s), 1.0]


def feature_row(belief: Belief, target_world, action_vec, action_onehot,
                age_s: float = 0.0) -> list[float]:
    return [belief.x, belief.y, belief.z, math.sin(belief.yaw), math.cos(belief.yaw),
            *[float(g) for g in belief.gravity],
            *memory_block(target_world, belief, age_s),
            *[float(v) for v in action_vec], *action_onehot]


def advance(belief: Belief, delta, gravity) -> Belief:
    """Apply a predicted body-frame pose delta and gravity vector."""
    dfwd, dlat, dz, dyaw = (float(v) for v in delta)
    c, s = math.cos(belief.yaw), math.sin(belief.yaw)
    return Belief(x=belief.x + dfwd * c - dlat * s,
                  y=belief.y + dfwd * s + dlat * c,
                  z=belief.z + dz,
                  yaw=belief.yaw + dyaw,
                  gravity=tuple(float(g) for g in gravity))


class ConsequenceHeads:
    """Pose-delta, gravity and visibility heads, fitted with terrain held out."""

    def __init__(self, rows: list[dict], action_ids: list[str], held_out: str,
                 inputs: str = "O+M+a"):
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        X = build_training_matrix(rows, inputs, action_ids)
        terrain = np.array([r["terrain"] for r in rows])
        train = terrain != held_out
        # Per-component fits: this sklearn version's gradient booster does not take
        # a multi-output target, and fitting four scalars is clearer anyway.
        Y = np.array([r["delta_pose"] for r in rows], dtype=float)
        self.pose = []
        for j in range(Y.shape[1]):
            model = HistGradientBoostingRegressor(max_depth=3, max_iter=250,
                                                  learning_rate=0.08, random_state=7)
            model.fit(X[train], Y[train, j])
            self.pose.append(model)
        # Gravity has to be predicted too: it is an input feature, so a chained
        # rollout that held it constant would be feeding the model a state the
        # robot cannot be in.
        gravity = np.array([r["observation"][5:8] for r in rows])
        self.gravity = []
        for j in range(gravity.shape[1]):
            model = HistGradientBoostingRegressor(max_depth=3, max_iter=250,
                                                  learning_rate=0.08, random_state=7)
            model.fit(X[train], gravity[train, j])
            self.gravity.append(model)
        y = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
        self.visible = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                                      learning_rate=0.08, random_state=7)
        if len(np.unique(y[train])) >= 2:
            self.visible.fit(X[train], y[train])

    def predict(self, feats: np.ndarray, with_pose: bool = True):
        out = {}
        if with_pose:
            out["delta"] = np.stack([m.predict(feats) for m in self.pose], axis=1)
            out["gravity"] = np.stack([m.predict(feats) for m in self.gravity], axis=1)
        out["visible"] = self.visible.predict_proba(feats)[:, 1]
        return out


class HorizonPlanner:
    """Receding-horizon planner over a *learned* consequence model.

    Enumerates action sequences with beam search (a full $|A|^H$ enumeration is
    100 000 sequences at H=5), scores each by the best visibility probability
    reached anywhere along it - which matches the task, "visible at some point
    within the budget" - then executes only the first action and replans.
    """

    def __init__(self, heads: ConsequenceHeads, action_ids: list[str],
                 action_vectors: dict, horizon: int = 1, beam: int = 8,
                 name: str | None = None):
        self.heads = heads
        self.action_ids = action_ids
        self.action_vectors = action_vectors
        self.horizon = horizon
        self.beam = beam
        self.name = name or f"Horizon{horizon}"
        self.requires = frozenset({"proprioception", "target_memory"})
        self.prediction_error: list[dict] = []
        self._pending: Belief | None = None

    # -- lookahead ---------------------------------------------------------
    def _expand(self, belief: Belief, target_world, age_s: float):
        rows, meta = [], []
        for action_id in self.action_ids:
            onehot = [1.0 if a == action_id else 0.0 for a in self.action_ids]
            rows.append(feature_row(belief, target_world, self.action_vectors[action_id],
                                    onehot, age_s))
            meta.append(action_id)
        out = self.heads.predict(np.array(rows, dtype=float))
        return meta, out

    def plan(self, observation) -> list[str]:
        target = observation.target_memory
        target_world = None if target is None else np.asarray(target.xyz, dtype=float)
        age0 = 0.0 if target is None else float(target.age_s)
        step_s = float(self.action_vectors[self.action_ids[0]][3])
        pose = observation.pose
        belief = Belief(pose.x, pose.y, pose.z, pose.yaw,
                        tuple(float(g) for g in observation.proprioception[-3:]))
        # beam entries: (score, sequence, belief, age)
        beam = [(1.0, [], belief, age0)]
        # The belief reached after the FIRST action, kept for error tracking: this
        # is the only prediction the robot can actually check, because it only ever
        # executes one action before replanning.
        first_step_belief = None
        for _ in range(self.horizon):
            candidates = []
            for score, sequence, node, age in beam:
                actions, out = self._expand(node, target_world, age)
                for i, action_id in enumerate(actions):
                    nxt = advance(node, out["delta"][i], out["gravity"][i])
                    p = float(out["visible"][i])
                    candidates.append((max(score, p) if sequence else p,
                                       sequence + [action_id], nxt, age + step_s, p))
            candidates.sort(key=lambda c: -c[0])
            beam = [(c[0], c[1], c[2], c[3]) for c in candidates[:self.beam]]
        best_score, best_sequence, best_belief, _ = max(beam, key=lambda c: c[0])
        # recompute the one-step belief for the chosen first action so the logged
        # prediction is exactly the one the planner commits to
        actions, out = self._expand(belief, target_world, age0)
        index = actions.index(best_sequence[0])
        first_step_belief = advance(belief, out["delta"][index], out["gravity"][index])
        self._pending = first_step_belief
        return best_sequence

    # -- policy interface --------------------------------------------------
    def reset(self, observation) -> None:
        self.prediction_error = []
        self._pending = None

    def act(self, observation) -> str:
        # Check the previous commitment before planning the next one: the pose the
        # planner predicted for the end of its first action versus where the robot
        # actually is now.
        if self._pending is not None:
            pose = observation.pose
            error = float(np.linalg.norm(
                np.array([self._pending.x, self._pending.y, self._pending.z])
                - np.array([pose.x, pose.y, pose.z])))
            self.prediction_error.append({
                "step": len(self.prediction_error) + 1,
                "predicted": [self._pending.x, self._pending.y, self._pending.z,
                              self._pending.yaw],
                "actual": [pose.x, pose.y, pose.z, pose.yaw],
                "position_error_m": error,
            })
        sequence = self.plan(observation)
        return sequence[0]


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def run_horizon(data: Path, horizon: int, beam: int, max_steps: int,
                inputs: str = "O+M+a") -> dict:
    payload = load(data)
    rows = payload["rows"]
    action_ids = payload["action_ids"]
    action_vectors = {r["action"]: r["action_vec"] for r in rows}

    episodes, errors = [], []
    for name, scene in api.scene_catalogue().items():
        heads = ConsequenceHeads(rows, action_ids, held_out=name, inputs=inputs)
        planner = HorizonPlanner(heads, action_ids, action_vectors,
                                 horizon=horizon, beam=beam)
        environment = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table())
        try:
            episode = ep.run_episode(planner, environment, max_steps=max_steps,
                                     protocol="B")
            episode.failure_type = ep.classify_failure(episode, None, None, [])
            episodes.append(episode)
            for record in planner.prediction_error:
                errors.append({"terrain": name, **record})
            print(f"    {name:18s} {episode.termination:18s} steps={episode.steps_used} "
                  f"{'ok' if episode.success else 'fail'}  {episode.actions}",
                  flush=True)
        finally:
            environment.close()

    summary = ep.summarise(episodes)
    # Model-error accumulation, measured directly: the pose the planner predicted
    # for the end of its committed first action versus where the robot actually was
    # when it replanned.  This is the quantity that should explain any H=5 drop,
    # rather than the drop being asserted as "error accumulation".
    finite = [e for e in errors if np.isfinite(e["position_error_m"])]
    summary["model_error"] = {
        "n_records": len(finite),
        "mean_position_error_m": (round(float(np.mean(
            [e["position_error_m"] for e in finite])), 4) if finite else None),
        "p90_position_error_m": (round(float(np.percentile(
            [e["position_error_m"] for e in finite], 90)), 4) if finite else None),
        "max_position_error_m": (round(float(np.max(
            [e["position_error_m"] for e in finite])), 4) if finite else None),
    }
    summary["per_terrain"] = {e.scene: {"termination": e.termination,
                                        "steps": e.steps_used, "actions": e.actions}
                              for e in episodes}
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="E4d action-horizon ablation")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    results = {}
    for horizon in args.horizons:
        print(f"\n=== H={horizon} (beam {args.beam}) ===", flush=True)
        results[f"H{horizon}"] = run_horizon(args.data, horizon, args.beam,
                                             args.max_steps)

    print("\n=== E4d action-horizon ablation (real execution trace) ===")
    header = ("horizon", "RSR@1", "RSR@3", "RSR@5", "RSR@10", "steps", "T_reobs",
              "cost_m", "pose_err_m")
    print(f"  {header[0]:>7s} " + " ".join(f"{h:>8s}" for h in header[1:]))
    for name, s in results.items():
        print(f"  {name:>7s} {s['rsr@1']:>8.3f} {s['rsr@3']:>8.3f} {s['rsr@5']:>8.3f} "
              f"{s['rsr@10']:>8.3f} {s['mean_steps_when_solved']:>8} "
              f"{str(s['mean_t_reobs_s']):>8s} {s['mean_motion_cost_m']:>8.3f} "
              f"{s['model_error']['mean_position_error_m']:>8.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"task": "TASK3_e4d_horizon_ablation",
                                       "beam": args.beam, "results": results},
                                      indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
