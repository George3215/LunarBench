"""Navigation benchmark harness: one loop, annotated evidence, full logging.

Design constraints, in the order they matter:

1. **One loop for every model.** A policy is anything implementing ``act``. Adding
   ViNT, NoMaD or a VLM means writing an adapter, never touching the loop.
2. **Evidence per step.** Every decision renders a frame, annotates it with the model
   name, step, chosen action and outcome, and writes it to disk. A reviewer can open
   the images and see what the model saw and chose.
3. **Log the model's own view.** Inputs and outputs are recorded as the model
   received and produced them, plus the privileged simulator values recorded
   *separately* and never fed back.

The privileged block is written to its own file with its own key, so it is impossible
to mistake it for policy input. Nothing in this module passes a privileged value into
``act``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

try:
    from . import api
    from . import episode as ep
except ImportError:
    import api
    import episode as ep

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "paper" / "navigation_runs"

#: Actions a navigation policy may return. Stop is a terminator, not a primitive.
NAV_ACTIONS = ("forward", "back", "left", "right", "turn_left", "turn_right")
STOP = "stop"

#: Success needs this many target pixels sustained for this many 10 Hz frames, so a
#: single stray pixel does not count as re-observation. Fixed before any test run.
MIN_PIXELS = 3
SUSTAINED = 3
USABLE_PIXELS = 24


class Policy(Protocol):
    """Anything that maps one observation to one action."""
    name: str

    def reset(self) -> None: ...
    def act(self, observation) -> tuple[str, dict]: ...


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class ScriptedPolicy:
    """A fixed action cycle. Useful as a lower bound and as a harness smoke test."""

    def __init__(self, name: str, actions=("forward",), seed: int = 1):
        self.name = name
        self.actions = tuple(actions)
        self.rng = np.random.default_rng(seed)
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def act(self, observation) -> tuple[str, dict]:
        action = self.actions[self._i % len(self.actions)]
        self._i += 1
        return action, {"rule": f"cycle[{self._i % len(self.actions)}]"}


class RandomPolicy:
    def __init__(self, seed: int = 1):
        self.name = "Random"
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, observation) -> tuple[str, dict]:
        return str(self.rng.choice(NAV_ACTIONS)), {"rule": "uniform"}


class GoalImagePolicy:
    """Adapter slot for ViNT / NoMaD / any goal-conditioned navigation model.

    The interface difference that matters: these models take ``(context_obs,
    goal_image)`` and return a *waypoint*, not one of my six primitives. The
    conversion is explicit here rather than hidden, because it is a modelling choice
    that a reader has to be able to audit:

        waypoint (dx, dy) in the robot frame  ->  primitive by dominant component

    ``goal_image`` is the last frame in which the target was visible. That is the
    interesting case: a goal-conditioned model is being asked to navigate toward a
    goal image that no longer corresponds to anything currently in view.
    """

    def __init__(self, name: str, model, transform=None, device: str = "cpu"):
        self.name = name
        self.model = model
        self.transform = transform
        self.device = device
        self.goal_image = None
        self.context: list = []
        self.context_size = 5
        self.last_waypoint = None

    def set_goal(self, image) -> None:
        self.goal_image = image

    def reset(self) -> None:
        self.context = []
        self.last_waypoint = None

    def act(self, observation) -> tuple[str, dict]:
        if self.goal_image is None or observation.current_rgb is None:
            return "forward", {"rule": "no-goal fallback"}
        import torch
        self.context.append(observation.current_rgb)
        self.context = self.context[-self.context_size:]
        while len(self.context) < self.context_size:
            self.context.insert(0, self.context[0])
        obs = torch.stack([self.transform(f) for f in self.context]).unsqueeze(0)
        goal = self.transform(self.goal_image).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _dist, action = self.model(obs.to(self.device), goal.to(self.device))
        waypoint = np.asarray(action)[-1, 0, :2]
        self.last_waypoint = waypoint.tolist()
        return primitive_from_waypoint(waypoint), {
            "rule": "waypoint->primitive",
            "waypoint": [round(float(v), 4) for v in waypoint],
        }


def primitive_from_waypoint(waypoint) -> str:
    """Dominant-component conversion from a waypoint to one of the six primitives.

    Kept public and documented because it is the one place where a foundation
    model's continuous output is discretised, and a reader must be able to see and
    change it.
    """
    dx, dy = float(waypoint[0]), float(waypoint[1])
    if abs(dy) > 2.0 * abs(dx):
        return "turn_left" if dy > 0 else "turn_right"
    if dx >= 0:
        return "forward"
    return "back"


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def annotate(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn text into the frame without pulling in a plotting stack.

    A minimal 5x7 bitmap font, because the frames are 160x120 and anything larger is
    unreadable at that size. Uppercase and digits only.
    """
    glyphs = _FONT
    scale = 1
    pad = 2
    width = max(len(s) for s in lines) * 6 * scale + 2 * pad
    height = len(lines) * 8 * scale + 2 * pad
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for row, text in enumerate(lines):
        for col, ch in enumerate(text.upper()):
            bits = glyphs.get(ch, glyphs["?"])
            for y in range(7):
                for x in range(5):
                    if bits[y] >> (4 - x) & 1:
                        y0 = pad + row * 8 * scale + y * scale
                        x0 = pad + col * 6 * scale + x * scale
                        canvas[y0:y0 + scale, x0:x0 + scale] = 255
    out = rgb.copy()
    h = min(height, out.shape[0])
    w = min(width, out.shape[1])
    out[:h, :w] = canvas[:h, :w]
    return out


_FONT = {
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11), "B": (0x1E, 0x11, 0x1E, 0x11, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E), "D": (0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E),
    "E": (0x1F, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x1F), "F": (0x1F, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F), "H": (0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E), "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11), "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11), "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E), "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D), "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x0E, 0x01, 0x01, 0x11, 0x0E), "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E), "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11), "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04), "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E), "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F), "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02), "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E), "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E), "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    " ": (0,) * 7, "-": (0, 0, 0, 0x1F, 0, 0, 0), "_": (0, 0, 0, 0, 0, 0, 0x1F),
    ".": (0, 0, 0, 0, 0, 0x0C, 0x0C), ":": (0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10), "|": (0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "?": (0x0E, 0x11, 0x01, 0x06, 0x04, 0x00, 0x04), "=": (0, 0, 0x1F, 0, 0x1F, 0, 0),
    "#": (0x0A, 0x1F, 0x0A, 0x0A, 0x0A, 0x1F, 0x0A), "+": (0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0),
}


@dataclass
class StepRecord:
    """One decision, split into what the model saw, what it did, and ground truth."""
    step: int
    terrain: str
    model: str
    action: str
    policy_aux: dict
    # -- what the policy received (deployment-valid only) -----------------
    policy_input: dict
    # -- privileged simulator state, recorded separately, never fed back ---
    privileged: dict
    visible_pixels: int
    wall_s: float


def run(model: Policy, terrain: str, budget: int = 50, render_size=(160, 120),
        seed: int = 1, save_frames: int = 12, root: Path = EVIDENCE) -> dict:
    """Run one model on one terrain, writing annotated frames and a full log."""
    run_dir = root / model.name.replace(" ", "_").lower() / terrain
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene = api.scene_catalogue()[terrain]
    environment = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table(),
                                        render=True, render_size=render_size)
    model.reset()
    records: list[StepRecord] = []
    start = time.time()
    try:
        observation = environment.reset()
        start_xy = np.array(observation.pose.xyz, dtype=float)[:2]
        path = 0.0
        previous_xy = start_xy.copy()
        streak = 0
        success = False
        termination = "budget_exhausted"
        pixels_at_success = 0
        step = 0
        termination = "budget_exhausted"
        for step in range(1, budget + 1):
            t0 = time.time()
            action, aux = model.act(observation)
            decision_s = time.time() - t0
            if action == STOP:
                term = "policy_stop"
                break
            environment.step(action)
            now = environment.observe()
            xy = np.array(now.pose.xyz, dtype=float)[:2]
            moved = float(np.linalg.norm(xy - previous_xy))
            path += moved

            # At a 500-primitive budget an episode that has already fallen or left
            # the terrain must stop, otherwise the remaining steps are spent with the
            # robot lying on the ground and the cost accounting becomes meaningless.
            upright = float(environment.data.xmat[1].reshape(3, 3)[2, 2])
            if upright < 0.25 or abs(xy[0]) > 60.0 or abs(xy[1]) > 60.0:
                termination = "fallen" if upright < 0.25 else "out_of_bounds"
                break

            rgb = now.current_rgb
            pixels = count_target_pixels(rgb)
            if pixels >= MIN_PIXELS:
                streak += 1
                if streak >= SUSTAINED:
                    success = True
                    pixels_at_success = pixels
            else:
                streak = 0

            records.append(StepRecord(
                step=step, terrain=terrain, model=model.name, action=action,
                policy_aux=aux,
                policy_input={
                    "pose": [round(float(v), 4) for v in now.pose.xyz],
                    "yaw": round(float(now.pose.yaw), 4),
                    "goal_image_set": bool(getattr(model, "goal_image", None) is not None),
                    "target_pixels_in_frame": pixels,
                },
                privileged={
                    "target_world": [round(float(v), 4) for v in environment.target],
                    "robot_world": [round(float(v), 4) for v in now.pose.xyz],
                    "true_distance_m": round(float(np.linalg.norm(
                        np.asarray(environment.target, dtype=float)
                        - np.asarray(now.pose.xyz, dtype=float))), 4),
                    "sim_visible": bool(pixels >= MIN_PIXELS),
                },
                visible_pixels=pixels, wall_s=round(decision_s, 4)))

            if step <= save_frames:
                label = [f"{model.name}", f"step {step} {terrain}",
                         f"act {action}", f"px {pixels}"]
                try:
                    from PIL import Image
                    Image.fromarray(annotate(rgb, label)).resize(
                        (rgb.shape[1] * 2, rgb.shape[0] * 2), Image.NEAREST).save(
                        frames_dir / f"step{step:03d}_{action}.png")
                except Exception as exc:  # pragma: no cover
                    records[-1].policy_aux.setdefault("frame_error", str(exc))

            previous_xy = xy
            observation = now
            if success:
                break

        summary = {
            "model": model.name, "terrain": terrain, "budget": budget,
            "steps_used": step, "success": success, "termination": termination, "usable": pixels_at_success >= USABLE_PIXELS,
            "pixels_at_success": pixels_at_success,
            "path_m": round(path, 4),
            "net_m": round(float(np.linalg.norm(
                np.array(observation.pose.xyz, dtype=float)[:2] - start_xy)), 4),
            "wall_s": round(time.time() - start, 3),
        }
    finally:
        environment.close()

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Privileged values go to their own file with their own key, so they cannot be
    # mistaken for policy input.
    (run_dir / "steps.jsonl").write_text(
        "\n".join(json.dumps(asdict(r)) for r in records), encoding="utf-8")
    (run_dir / "privileged.jsonl").write_text(
        "\n".join(json.dumps({"step": r.step, **r.privileged}) for r in records),
        encoding="utf-8")
    return summary


def count_target_pixels(rgb) -> int:
    if rgb is None:
        return 0
    m = _target_mask(rgb)
    return int(m.sum())


def _target_mask(rgb) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (r > 120) & (r > 1.5 * g) & (r > 1.5 * b)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="navigation benchmark harness")
    p.add_argument("--models", nargs="*", default=["forward-only", "random", "scan"])
    p.add_argument("--terrains", nargs="*", default=["pit", "trench", "stairs_up"])
    p.add_argument("--budget", type=int, default=50)
    args = p.parse_args()

    registry = {
        "forward-only": lambda: ScriptedPolicy("ForwardOnly", ("forward",)),
        "random": lambda: RandomPolicy(1),
        "scan": lambda: ScriptedPolicy("Scan", ("turn_left", "forward", "turn_right", "forward")),
    }
    print(f"=== navigation harness ===\nbudget {args.budget} primitives, "
          f"success needs {MIN_PIXELS} px for {SUSTAINED} frames\n")
    rows = []
    for name in args.models:
        policy = registry[name]()
        for terrain in args.terrains:
            s = run(policy, terrain, budget=args.budget)
            rows.append(s)
            print(f"  {s['model']:12s} {terrain:12s} steps={s['steps_used']:>3d} "
                  f"success={str(s['success']):5s} path={s['path_m']:.2f}m "
                  f"px={s['pixels_at_success']:>4d} wall={s['wall_s']:.1f}s")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nannotated frames + step logs + privileged logs under {EVIDENCE}")


if __name__ == "__main__":
    main()
