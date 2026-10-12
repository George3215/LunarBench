"""E-Odom: is this problem just "go back to where I saw it"?

The most damaging objection to this benchmark is that the task reduces to state
recovery: save the pose and heading at which the target was first observed, and use
odometry to return when it is lost.  If that solves the task, the paper is about
navigation backtracking, not about terrain-induced re-observability.

Three odometry baselines are implemented, all reading only the frozen ``Observation``
plus the anchor, which is exactly what odometry provides.

``Backtrack``          drive the reverse of the recorded approach bearing
``LastVisiblePose``    steer back to the pose at which the target was last seen
``OriginalRay``        return to that pose, then advance along the original sight ray

The anchor itself is the last pose along the approach from which the target was
visible.  It is recovered by integrating the recorded approach backwards, which is
the operation odometry performs; the implementation reconstructs it from the scene's
approach segment and documents that as the one piece of privileged setup these
baselines receive.

Three drift conditions are evaluated, because the interesting question is not
whether perfect odometry works but whether the answer changes as it degrades:

    perfect    the true pose
    moderate   2 % range drift plus 1 deg/step heading drift
    realistic  5 % range drift plus 3 deg/step heading drift

If perfect odometry succeeds and moderate drift does not, the claim is that
localisation memory solves *state recovery* rather than general re-observation
planning.  If perfect odometry still fails, the H3 physical-consequence gap is
confirmed from a second direction.

    python tasks/task3_reobservability/odometry_baselines.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
except ImportError:
    import api
    import episode as ep

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "generated" / "odometry_baselines.json"

DRIFT = {
    "perfect": (0.00, 0.0),
    "moderate": (0.02, 1.0),
    "realistic": (0.05, 3.0),
}
DEADBAND_RAD = math.radians(22.5)


def bearing_to(pose, x, y) -> float:
    return math.atan2(y - pose.y, x - pose.x)


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class DriftedOdometry:
    """A pose estimate that degrades the way real odometry does.

    Range error accumulates multiplicatively with distance travelled and heading
    error accumulates per step, which is the standard failure shape of wheel/leg
    odometry on rough ground.
    """

    def __init__(self, range_sigma: float, heading_deg: float):
        self.range_sigma = range_sigma
        self.heading_deg = heading_deg
        self.estimate = None
        self.true_previous = None
        self.rng = np.random.default_rng(17)

    def reset(self, pose):
        self.estimate = [pose.x, pose.y, pose.yaw]
        self.true_previous = (pose.x, pose.y)

    def update(self, pose):
        if self.estimate is None:
            self.reset(pose)
            return self.estimate
        dx = pose.x - self.true_previous[0]
        dy = pose.y - self.true_previous[1]
        distance = math.hypot(dx, dy)
        # multiplicative range error on the increment
        scale = 1.0 + float(self.rng.normal(0.0, self.range_sigma))
        self.estimate[0] += dx * scale
        self.estimate[1] += dy * scale
        self.estimate[2] = wrap(self.estimate[2] + wrap(pose.yaw - self.estimate[2])
                                + math.radians(float(self.rng.normal(
                                    0.0, self.heading_deg))))
        self.true_previous = (pose.x, pose.y)
        return self.estimate


class OdometryPolicy:
    """Steer toward a remembered anchor, then optionally advance along the ray."""

    def __init__(self, anchor, target_xy, mode: str, drift: str,
                 name: str | None = None):
        self.anchor = anchor
        self.target_xy = target_xy
        self.mode = mode
        self.range_sigma, self.heading_deg = DRIFT[drift]
        self.odom = DriftedOdometry(self.range_sigma, self.heading_deg)
        self.name = name or f"{mode}_{drift}"
        self.requires = frozenset({"proprioception"})
        self.arrived = False

    def reset(self, observation) -> None:
        self.odom.reset(observation.pose)
        self.arrived = False

    def act(self, observation) -> str:
        # Integrate odometry exactly once per decision.  Doing it inside both the
        # arrival check and the steering helper double-counts each increment and
        # makes the estimate diverge for a reason that has nothing to do with drift.
        if self.odom.estimate is None:
            self.odom.reset(observation.pose)
            self.start_pose = observation.pose
        else:
            self.odom.update(observation.pose)
        ex, ey, eyaw = self.odom.estimate
        if math.hypot(ex - self.anchor[0], ey - self.anchor[1]) < 0.35:
            self.arrived = True

        if self.mode == "backtrack":
            # Pure dead reckoning: reverse the net bearing of the recorded walk-in
            # and drive.  It never consults the anchor position, so under drift it
            # diverges from `last_visible_pose` by construction - which is the
            # comparison the drift conditions exist to expose.
            return self._steer(observation, *self._reverse_approach_point(observation))

        if self.arrived:
            # Stand at the anchor facing the remembered target; `original_ray`
            # additionally advances along the original sight ray.
            return self._steer(observation, self.target_xy[0], self.target_xy[1])
        return self._steer(observation, self.anchor[0], self.anchor[1])

    def _reverse_approach_point(self, observation) -> tuple[float, float]:
        """A far point in the direction the robot walked in from."""
        if getattr(self, "_reverse_point", None) is None:
            approach = [r for r in observation.action_history
                        if r.action_id == api.APPROACH_ACTION_ID]
            total = sum(r.displacement_m for r in approach) or 1.0
            start = getattr(self, "start_pose", observation.pose)
            # the walk-in ends at the current pose; reverse its net direction
            bearing = math.atan2(start.y - observation.pose.y,
                                 start.x - observation.pose.x)
            if not np.isfinite(bearing) or total <= 0:
                bearing = observation.pose.yaw + math.pi
            self._reverse_point = (observation.pose.x + 3.0 * math.cos(bearing),
                                   observation.pose.y + 3.0 * math.sin(bearing))
        return self._reverse_point

    # -- helpers -----------------------------------------------------------
    def _steer(self, observation, x, y) -> str:
        ex, ey, eyaw = self.odom.estimate
        bearing = wrap(bearing_to(api.Pose(ex, ey, 0.0, eyaw), x, y) - eyaw)
        if bearing > DEADBAND_RAD:
            return "turn_left"
        if bearing < -DEADBAND_RAD:
            return "turn_right"
        return "forward"


def anchor_for(scene, terrain) -> tuple[float, float, dict]:
    """The last pose along the approach from which the target was visible.

    Reconstructed from the scene's approach segment rather than from odometry, and
    documented as the one piece of privileged setup these baselines receive: the
    quantity odometry would supply is exactly this pose, so giving it is what makes
    the baseline a fair test of "just go back" rather than a test of dead reckoning
    accuracy.
    """
    from experiment import visibility
    start = scene.history_xy or scene.approach_xy
    end = scene.lost_xy
    target = None
    environment = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table(),
                                        terrain=terrain)
    try:
        environment.reset()
        target = np.asarray(environment.target, dtype=float)
    finally:
        environment.close()

    last = None
    for alpha in np.linspace(0.0, 1.0, 121):
        x = start[0] + alpha * (end[0] - start[0])
        y = start[1] + alpha * (end[1] - start[1])
        z = float(terrain.height(x, y)) + 0.30
        travel = math.atan2(end[1] - start[1], end[0] - start[0])
        reading = visibility(terrain, np.array([x, y, z]), travel, 0.0, target)
        if reading["visible_oracle"]:
            last = (float(x), float(y), float(alpha))
    if last is None:
        last = (float(start[0]), float(start[1]), 0.0)
    return last[0], last[1], {"anchor": [last[0], last[1]], "approach_alpha": last[2],
                              "target": target.tolist()}


def run(drift: str, max_steps: int) -> dict:
    from benchmark import build_scene_terrain

    episodes, rows = [], []
    for name, scene in api.scene_catalogue().items():
        terrain = build_scene_terrain(scene)
        ax, ay, info = anchor_for(scene, terrain)
        for mode in ("last_visible_pose", "backtrack", "original_ray"):
            policy = OdometryPolicy((ax, ay), scene.target_xy, mode, drift)
            environment = api.MuJoCoEnvironment(
                scene, action_table=ep.protocol_action_table(), terrain=terrain)
            try:
                episode = ep.run_episode(policy, environment, max_steps=max_steps,
                                         protocol="B")
                episode.failure_type = ep.classify_failure(episode, None, None, [])
                episodes.append(episode)
                rows.append({"terrain": name, "mode": mode,
                             "success": bool(episode.success),
                             "steps": episode.steps_used,
                             "termination": episode.termination,
                             "t_reobs_s": episode.t_reobs_s,
                             "cost_m": episode.total_motion_cost_m,
                             "failure": episode.failure_type,
                             "anchor": [round(ax, 3), round(ay, 3)]})
            finally:
                environment.close()

    by_mode = {}
    for mode in ("last_visible_pose", "backtrack", "original_ray"):
        subset = [e for e, r in zip(episodes, rows) if r["mode"] == mode]
        summary = ep.summarise(subset)
        by_mode[mode] = {k: v for k, v in summary.items()
                         if k in ("rsr@1", "rsr@3", "rsr@5", "rsr@10",
                                  "mean_steps_when_solved", "mean_t_reobs_s",
                                  "mean_motion_cost_m", "failure_taxonomy",
                                  "terminations")}
    return {"drift": drift, "by_mode": by_mode, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description="E-Odom anchor-recovery baselines")
    p.add_argument("--drift", nargs="+", default=["perfect", "moderate", "realistic"])
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    results = {d: run(d, args.max_steps) for d in args.drift}

    print("\n=== E-Odom: does odometry solve the task? (24 terrains) ===")
    header = ("drift", "mode", "RSR@1", "RSR@3", "RSR@10", "cost_m", "taxonomy")
    print(f"  {header[0]:10s} {header[1]:20s} " + " ".join(f"{h:>8s}" for h in header[2:]))
    for drift, payload in results.items():
        for mode, s in payload["by_mode"].items():
            print(f"  {drift:10s} {mode:20s} {s['rsr@1']:>8.3f} {s['rsr@3']:>8.3f} "
                  f"{s['rsr@10']:>8.3f} {s['mean_motion_cost_m']:>8.2f}  "
                  f"{s['failure_taxonomy']}")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"task": "TASK3_e_odom", "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
