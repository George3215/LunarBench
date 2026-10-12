"""Qwen3-VL-8B trials on the terrain_design_v4 strict-admitted terrains.

Wires the recovery_mechanism_v4 strict-gate records (initial-visible -> real
terrain occlusion -> physically recovered) into the same image-history x
sensor-condition trial loop used for the earlier task3 VLM runs.

Conditions: MODES ('rgb','rgbd') x HORIZONS ('current','first_current',
'all_history'), i.e.
  only the current frame
  first frame + current frame
  current frame + all recorded history frames
Budget: at most --max-steps autonomous VLM actions per trial (default 3000).

The system prompt is fixed in context_navigation_policy.SYSTEM and describes
only the task and action interface; it never names or describes the terrain.
Every request is recorded (messages.json, rendered_prompt.txt, context.json).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

import navigation_trial as nt
import all_terrain_context_trial as act
import terrain_design_v4 as td4
from all_terrain_setup import place
from context_navigation_policy import (
    HORIZONS,
    INPUT_TOKEN_LIMIT,
    MODES,
    OUTPUT_TOKEN_LIMIT,
    SYSTEM,
    Policy,
)
from go2_piper_camera import CAMERA_NAME
from scenes import Scene
from visible_start_trial import actions
from vlm_benchmark import dump, sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADMISSION = ROOT / "paper/vlm_goal_navigation/recovery_mechanism_v4_20261009_r2"


def ray_check(camera):
    """Strict-gate occlusion test: target centre in FoV and first ray hit is terrain/entity.

    Mirrors strict_recovery_gate.ray so the VLM trial replays exactly the state
    the admission accepted (barrier cubes and window jambs are entities too).
    """
    env = camera.env
    data = camera.render_data
    cid = env.model.camera(CAMERA_NAME).id
    eye = data.cam_xpos[cid]
    target = data.geom_xpos[env.model.geom("task3_target_geom").id]
    local = data.cam_xmat[cid].reshape(3, 3).T @ (target - eye)
    h, w = camera.depth.shape
    vertical = math.tan(math.radians(float(env.model.cam_fovy[cid])) / 2)
    fov = bool(
        local[2] < 0
        and abs(local[1] / local[2]) < vertical
        and abs(local[0] / local[2]) < vertical * w / h
    )
    hit = np.array([-1], np.int32)
    direction = (target - eye) / np.linalg.norm(target - eye)
    distance = mujoco.mj_ray(env.model, data, eye, direction, None, 1, -1, hit)
    name = (
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0]))
        if hit[0] >= 0
        else None
    )
    names = {s.name for s in env.terrain.slabs} if hasattr(env.terrain, "slabs") else set()
    names.update(
        re.findall(
            r'<geom\s+name="([^"]+)"',
            env.terrain.spec.get("navigation_extra_geom_xml", "")
            + env.terrain.spec.get("navigation_dynamic_body_xml", ""),
        )
    )
    names.update(["task3_terrain", "task3_slab", "task3_apron", "task3_landing"])
    blocked = name in names or bool(name and name.startswith("task3_step_"))
    return {
        "target_centre_in_fov": fov,
        "blocking_geom": name,
        "hit_distance_m": float(distance),
        "entity_occlusion": bool(
            fov and blocked and distance >= 0 and distance < np.linalg.norm(target - eye)
        ),
    }


def v4_loss_kind(camera):
    check = ray_check(camera)
    if not check["target_centre_in_fov"]:
        return "fov_exit", check["blocking_geom"]
    if check["entity_occlusion"]:
        return "terrain_occlusion", check["blocking_geom"]
    return "other_occlusion", check["blocking_geom"]


class V4Approach(nt.Physics):
    """Approach that declares loss exactly like the strict admission gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = False
        self.loss_run = 0
        self.loss = None
        self.start_xy = self.env.data.qpos[:2].copy()
        self.extra_terminal = None

    def record(self, step, action):
        state = super().record(step, action)
        m = state["evaluation"]
        if self.enabled:
            valid = m["target_segment_pixels"] <= 2 and m["upright"] >= 0.25 and m["in_bounds"]
            moved = float(np.linalg.norm(self.env.data.qpos[:2] - self.start_xy))
            check = ray_check(self.camera)
            self.loss_run = self.loss_run + 1 if valid and moved >= 0.25 and check["entity_occlusion"] else 0
            if self.loss_run >= 3:
                self.loss = {
                    "step": step,
                    "measurement": m,
                    "ray": check,
                    "kind": "terrain_occlusion",
                    "blocking_geom": check["blocking_geom"],
                    "lost_frames": self.loss_run,
                    "displacement_m": moved,
                    "qpos": self.env.data.qpos.tolist(),
                }
                self.extra_terminal = "approach_loss"
        return state


# all_terrain_context_trial.Trial resolves Approach/loss_kind as module globals;
# point them at the strict-gate-compatible versions.
act.Approach = V4Approach
act.loss_kind = v4_loss_kind


def build_setups(admission: Path, terrains: list[str] | None):
    manifest = json.loads((admission / "formal_manifest.json").read_text())
    inventory = json.loads((admission / "candidate_inventory.json").read_text())
    configs = {item["terrain_id"]: item["config"] for item in inventory}
    setups = []
    terrain_map = {}
    for record in manifest:
        terrain_id = record["terrain_id"]
        if not record["accepted"]:
            continue
        if terrains and terrain_id not in terrains:
            continue
        config = configs[terrain_id]
        terrain = td4.build_terrain_v3(config)
        terrain.spec.update(record["terrain_spec"])
        scene = Scene(**record["scene"])
        terrain_map[scene.name] = terrain
        setups.append(
            {
                "scene_name": scene.name,
                "terrain_id": terrain_id,
                "terrain_class": record["terrain_class"],
                "adjusted_scene": record["scene"],
                "approach_action": record["approach_actions"][0],
                "approach_actions": len(record["approach_actions"]),
                "terrain_spec": record["terrain_spec"],
                "loss": record["loss"],
                "gates": record["gates"],
                "added_occluder": record["terrain_spec"].get("navigation_added_occluder"),
                "original_terrain_geometry_unchanged": True,
                "strict_gate_evidence": record["evidence_path"],
            }
        )
    if not setups:
        raise SystemExit("no accepted terrains selected")
    return setups, terrain_map


def verify(setups, terrain_map):
    """Re-run each physical initial/approach gate without loading the model."""
    torch.set_num_threads(1)
    actions()
    report = []
    for setup in setups:
        scene = Scene(**setup["adjusted_scene"])
        terrain = terrain_map[scene.name]
        env = nt.api.MuJoCoEnvironment(scene, terrain=terrain)
        cam = nt.EvaluatorCamera(env, render_size=(640, 480))
        place(env)
        folder = Path(os.environ.get("DSH_TMP", "/tmp")) / f"verify_{scene.name}"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with (folder / "state.jsonl").open("w") as sf, (folder / "evaluator.jsonl").open("w") as ef:
                physics = V4Approach(env, cam, sf, ef, folder)
                physics.record(0, "RESET")
                stable = True
                for _ in range(6):
                    if physics.execute("STOP", 0):
                        stable = False
                        break
                initial = cam.read(env.model, env.data)
                ok_initial = (
                    stable
                    and initial["target_visible_rgb_pixels"] >= 40
                    and not initial["near_and_visible"]
                )
                physics.start_xy = env.data.qpos[:2].copy()
                physics.enabled = True
                terminal = None
                for step in range(1, setup["approach_actions"] + 1):
                    terminal = physics.execute(setup["approach_action"], step)
                    if terminal:
                        break
                after = cam.read(env.model, env.data)
                kind, geom = v4_loss_kind(cam)
                ok = (
                    ok_initial
                    and terminal == "approach_loss"
                    and kind == "terrain_occlusion"
                    and after["target_segment_pixels"] <= 2
                    and bool(physics.loss)
                )
                report.append(
                    {
                        "terrain_id": setup["terrain_id"],
                        "scene": scene.name,
                        "ok": ok,
                        "initial_pixels": initial["target_visible_rgb_pixels"],
                        "terminal": terminal,
                        "loss_kind": kind,
                        "blocking_geom": geom,
                        "approach_action": setup["approach_action"],
                        "approach_steps": setup["approach_actions"],
                    }
                )
                print("VERIFY", setup["terrain_id"], ok, terminal, kind, geom, flush=True)
        finally:
            cam.close()
            env.close()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    p.add_argument("--terrains", nargs="*")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--modes", nargs="*", default=list(MODES))
    p.add_argument("--horizons", nargs="*", default=list(HORIZONS))
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()

    setups, terrain_map = build_setups(a.admission.resolve(), a.terrains)
    if a.verify_only:
        report = verify(setups, terrain_map)
        dump(a.out.resolve() / "verify_report.json", report)
        print("VERIFY_PASS", sum(r["ok"] for r in report), "/", len(report))
        return

    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    actions()
    act.MAX_STEPS = a.max_steps

    total = len(setups) * len(a.modes) * len(a.horizons)
    dump(
        root / "protocol.json",
        {
            "version": "terrain_design_v4-qwen3vl-context-depth/1",
            "terrain_design_module": "terrain_design_v4",
            "admission_source": str(a.admission.resolve()),
            "scene_count": len(setups),
            "scenes": [s["terrain_id"] for s in setups],
            "trial_count": total,
            "sensor_conditions": a.modes,
            "context_conditions": a.horizons,
            "max_steps": a.max_steps,
            "system_prompt": SYSTEM,
            "model": str(nt.MODEL),
            "model_max_context_tokens": 262144,
            "runtime_input_token_limit": INPUT_TOKEN_LIMIT,
            "max_new_tokens": OUTPUT_TOKEN_LIMIT,
            "frame_preprocessing_area": 65536,
            "raw_camera_resolution": [640, 480],
            "no_truncation_or_sampling": True,
            "history_definition": "all action-boundary camera observations from settled initial frame through current, including common approach; not 10Hz evaluator samples",
            "current": "current frame only, no previous outputs/actions/depth/images",
            "first_current": "first frame always retained + current frame, no previous outputs/actions",
            "all_history": "every recorded camera observation, no summaries/frame dropping or prior assistant outputs",
            "depth": "aligned ideal rendered image and RGB-derived red-region metric range for selected frames only; no GT",
            "terrain_content_disclosed_to_model": False,
            "prompt_discloses_terrain": False,
            "success": "visible target >=20 pixels and median 3D surface range <1m for 3 consecutive 10Hz samples; upright/in bounds",
            "resource_limits_are_not_navigation_failure_evidence": True,
            "real_hardware": False,
            "seed": 7,
            "decoding": "greedy",
        },
    )
    dump(root / "scene_setup.json", setups)
    (root / "SYSTEM_PROMPT.txt").write_text(SYSTEM + "\n")
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())

    terrains = terrain_map
    setups_by_name = {s["scene_name"]: s for s in setups}
    policy = Policy()
    print("MODEL_READY", flush=True)
    tasks = iter(
        [(s, m, h) for s in setups for m in a.modes for h in a.horizons]
    )
    active = []
    start = time.perf_counter()
    max_steps = a.max_steps
    try:
        while True:
            while len(active) < a.batch_size:
                task = next(tasks, None)
                if task is None:
                    break
                setup, mode, horizon = task
                active.append(
                    act.Trial(setup, terrains[setup["scene_name"]], root, mode, horizon)
                )
            if not active:
                break
            act.status(root, active, total, start, policy)
            if shutil.disk_usage(root).free < 5 * 1024**3:
                for trial in active:
                    trial.finish("disk_space_limit")
                raise RuntimeError("Explicit disk-space guard; remaining tasks not started")
            records = policy.decide_batch([t.prepare() for t in active])
            retry = []
            for trial, record in zip(active, records):
                trial.calls.append(record)
                if record["parsed"] is None and not record.get("termination"):
                    retry.append(trial)
            if retry:
                records = policy.decide_batch(
                    [(t.request, t.step_folder / "inference_2", 2) for t in retry]
                )
                for trial, record in zip(retry, records):
                    trial.calls.append(record)
            for trial in active:
                trial.complete()
            active = [t for t in active if not t.done]
            act.status(root, active, total, start, policy)
        act.status(root, active, total, start, policy)["status"] = "trials_complete_pending_analysis"
        dump(root / "RUN_STATUS.json", act.status(root, active, total, start, policy))
    except Exception as exc:
        dump(root / "ERROR.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        for trial in active:
            if not trial.done:
                trial.close()


if __name__ == "__main__":
    main()
