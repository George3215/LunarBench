"""Matched checkpoint transfer test on unchanged V4 admitted terrain.

New recovery protocol, not a replay of the deleted Oct-17 memory runner.
Only images, RGB-derived depth, ideal self-odometry and model plans reach VLM.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, GenerationConfig, LogitsProcessorList, Qwen3VLForConditionalGeneration

import all_terrain_context_trial as act
import context_navigation_policy as cp
import navigation_trial as nt
from v4_qwen3vl_trial import DEFAULT_ADMISSION, build_setups, ray_check
from visible_start_trial import actions
from vlm_benchmark import dump, sha
from action_grammar import ActionGrammar, ActionVocabulary
from model_output_adapter import decision_json, initial_plan

VERSION = "v4-spatial-transfer-recovery/2-action-grammar"
SYSTEM = cp.SYSTEM + '''
You also receive ideal simulated SELF odometry relative to the initial robot frame,
recent measured action effects, and up to four event-selected RGB-D observations.
These are robot measurements, not a map or hidden target coordinates. Body forward
is +x and left is +y; positive yaw turns left. Depth is ideal camera depth.
The initial target evidence remains available. Maintain an uncertain spatial belief
when the target disappears; choose a route to recover its view and approach it.
The initial plan is your own earlier visual plan, not an oracle route. Correct it
when fresh observations or actual motion contradict it. Do not assume commanded
displacement was achieved. No external planner executes or corrects your actions.
STOP is forbidden unless CURRENT RGB has at least 20 red pixels and CURRENT finite
red-region surface range is strictly between 0 and 1 metre. A historical sighting
never permits STOP. If STOP is unavailable, choose one of the six motion actions.
Optional JSON fields: "spatial_belief" (brief uncertain bearing/distance/occlusion
assessment), "plan_update" (brief revised route), "expected_reappearance_m"
(a nonnegative number or null; an uncertain prediction, never a known fact).
Additional permitted evidence_sources: "self_odometry", "recent_executed_motion",
"own_initial_or_revised_plan", matching the measurements and own plan provided.
'''
PLAN_SYSTEM = '''You are the visual route planner for a simulated quadruped.
From ONLY its initial forward RGB-D observation, propose a tentative route toward
the red sphere. Terrain behind visible surfaces is unknown. Estimate likely target
direction, obstacles, a broad motion strategy, and when to reconsider it. Mark
uncertainty explicitly. Do not invent hidden geometry. This plan is advisory and
will be revised using later images and measured self-motion. No map or target
world coordinates are given. Return one JSON object with keys goal_evidence,
tentative_route, occlusion_hypothesis, uncertainty, revision_trigger (short strings).
'''


def pose(data):
    return {"xyz": data.qpos[:3].copy(), "rotation": data.xmat[1].reshape(3, 3).copy()}


def odometry(value, origin):
    r0 = origin["rotation"]
    yaw0 = math.atan2(r0[1, 0], r0[0, 0])
    c, s = math.cos(yaw0), math.sin(yaw0)
    delta = value["xyz"] - origin["xyz"]
    r = value["rotation"]
    yaw = math.atan2(r[1, 0], r[0, 0])
    return {"x_m": round(c * delta[0] + s * delta[1], 4),
            "y_m": round(-s * delta[0] + c * delta[1], 4),
            "z_m": round(float(delta[2]), 4),
            "yaw_rad": round(math.atan2(math.sin(yaw-yaw0), math.cos(yaw-yaw0)), 4),
            "roll_rad": round(math.atan2(r[2, 1], r[2, 2]), 4),
            "pitch_rad": round(math.asin(float(np.clip(-r[2, 0], -1, 1))), 4)}


def stop_allowed(sensor):
    distance = sensor["red_region_median_surface_range_m"]
    return bool(sensor["red_pixel_count"] >= 20 and distance is not None
                and math.isfinite(distance) and 0 < distance < 1)


def event_frames(frames):
    """First, last visible, latest visible-to-hidden transition, current; <=4."""
    selected = {0, len(frames)-1}
    visible = lambda f: f["depth_sensor"]["red_pixel_count"] >= 20
    sightings = [i for i, f in enumerate(frames) if visible(f)]
    losses = [i for i in range(1, len(frames)) if visible(frames[i-1]) and not visible(frames[i])]
    if sightings:
        selected.add(sightings[-1])
    if losses:
        selected.add(losses[-1])
    return [frames[i] for i in sorted(selected)]


class Policy(cp.Policy):
    def __init__(self, model_path, processor_path, max_tokens):
        self.processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 65536}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa").to("cuda:0").eval()
        # The spatial release saves training-time use_cache=false and a different
        # EOS list. Match Qwen's inference settings instead of inheriting these.
        self.model.generation_config = GenerationConfig.from_pretrained(
            str(processor_path), local_files_only=True)
        self.model.generation_config.use_cache = True
        self.model.config.text_config.use_cache = True
        self.generation_calls = 0
        self.model_context_limit = int(self.model.config.text_config.max_position_embeddings)
        self.actor_max_tokens = max_tokens
        self.action_vocabulary = ActionVocabulary(self.processor.tokenizer)
        self.action_grammars = {}

    def generation_options(self, group, inputs):
        if not all('stop_allowed' in request for request, _, _ in group):
            return {}  # initial plan has its own non-action JSON schema
        grammar = ActionGrammar(self.action_vocabulary, [r['stop_allowed'] for r, _, _ in group], inputs['input_ids'].shape[1])
        for row, (_, folder, _) in enumerate(group):
            self.action_grammars[str(folder)] = (grammar, row)
        return {"logits_processor": LogitsProcessorList([grammar])}

    def parse_output(self, raw):
        return cp.parse(decision_json(raw), allowed_sources=('current_rgb', 'past_rgb',
            'current_depth', 'past_depth', 'self_odometry', 'recent_executed_motion',
            'own_initial_or_revised_plan'))

    def decide_batch(self, requests):
        cp.SYSTEM = SYSTEM
        cp.OUTPUT_TOKEN_LIMIT = self.actor_max_tokens
        self.action_grammars = {}
        records = super().decide_batch(requests)
        for (request, folder, attempt), record in zip(requests, records):
            if str(folder) in self.action_grammars:
                grammar, row = self.action_grammars[str(folder)]
                record['action_grammar'] = {'masked_argmax_tokens': grammar.masked_argmax[row],
                    'forbidden_stop_path_interceptions': grammar.stop_path_interceptions[row]}
            parsed = record.get("parsed")
            if parsed and parsed["action"] == "STOP" and not request["stop_allowed"]:
                record["rejected_stop"] = True
                record["parsed_before_stop_gate"] = parsed
                record["parsed"] = None
            dump(folder / "output.json", record)
        return records

    def plan_batch(self, trials, max_tokens):
        cp.SYSTEM = PLAN_SYSTEM
        cp.OUTPUT_TOKEN_LIMIT = max_tokens
        requests = []
        for trial in trials:
            request = cp.request_for(trial.frames[:1], "current", "rgbd")
            request["text"] += "\nPlan from this initial observation only; no later frame is available."
            requests.append((request, trial.folder / "initial_plan", 1))
        # The generic parser rejects this different schema; parse and log below.
        records = cp.Policy.decide_batch(self, requests)
        for trial, (_, folder, _), record in zip(trials, requests, records):
            plan, plan_format = initial_plan(record["raw_output"])
            trial.plan = plan
            trial.plan_valid = plan is not None
            record["plan_parsed"] = plan
            record["plan_format"] = plan_format
            dump(folder / "output.json", record)
        cp.SYSTEM = SYSTEM
        cp.OUTPUT_TOKEN_LIMIT = self.actor_max_tokens


class Trial(act.Trial):
    def __init__(self, setup, terrain, out, budget):
        self.sensor_poses = []
        self.plan = None
        self.plan_valid = False
        self.feedback = []
        self.budget = budget
        self.rejected_stops = 0
        self.reobserved = False
        self.reobservation_sample = None
        self.reappearance_predictions = []
        super().__init__(setup, terrain, out, "rgbd", "event4_odom_plan")
        self.origin = self.sensor_poses[0]
        self.approach_lost_frame = self.frames[-1]["frame_id"]
        self.autonomous = 0

    def observe(self):
        frame = super().observe()
        self.sensor_poses.append(pose(self.env.data))
        return frame

    def make_request(self):
        chosen = event_frames(self.frames)
        request = cp.request_for(chosen, "all_history", "rgbd")
        request["horizon"] = "event4_odom_plan"
        request["all_recorded_frame_count"] = len(self.frames)
        available = stop_allowed(self.frames[-1]["depth_sensor"])
        sensor = {"self_odometry_source": "ideal simulated base odometry, initial body frame; not SLAM",
                  "observation_poses": [{"frame_id": f["frame_id"], **odometry(self.sensor_poses[f["frame_id"]], self.origin)} for f in chosen],
                  "recent_executed_motion": self.feedback[-6:],
                  "own_initial_or_revised_plan": self.plan,
                  "stop_available": available,
                  "allowed_actions": [k for k in nt.ACTIONS if k != "STOP" or available]}
        request["text"] += "\nMeasured self-motion and own plan: " + json.dumps(sensor, ensure_ascii=True)
        request["stop_allowed"] = available
        request["input_description"] = "event4 RGB-D, RGB-derived range, ideal self-odometry, recent action effects and own visual plan; no target GT, map, terrain identity or oracle"
        return request

    def prepare(self):
        self.step += 1
        self.step_folder = self.folder / "steps" / f"{self.step:04d}"
        self.step_folder.mkdir(parents=True)
        self.before = self.cam.read(self.env.model, self.env.data)
        self.frame_before = self.frames[-1]["frame_id"]
        self.calls = []
        self.request = self.make_request()
        return self.request, self.step_folder / "inference_1", 1

    def complete(self):
        record = self.calls[-1]
        parsed = record.get("parsed")
        action = parsed["action"] if parsed else None
        terminal = record.get("termination")
        qbefore = self.env.data.qpos.copy()
        self_before = odometry(pose(self.env.data), self.origin)
        begin = time.perf_counter()
        rejected = sum(bool(c.get("rejected_stop")) for c in self.calls)
        self.rejected_stops += rejected
        if terminal is None:
            if action is None:
                terminal = "repeated_forbidden_stop" if rejected else "invalid_model_output"
            else:
                # Defense in depth: never execute an unconfirmed VLM STOP.
                assert action != "STOP" or stop_allowed(self.frames[-1]["depth_sensor"])
                self.executed.append(action)
                self.autonomous += 1
                terminal = self.physics.execute(action, self.step)
                if action == "STOP" and terminal is None:
                    terminal = "stop_not_confirmed_by_evaluator"
        after = self.cam.read(self.env.model, self.env.data)
        frame = self.observe() if action else self.frames[-1]
        if parsed:
            update = parsed.get("plan_update")
            if isinstance(update, str) and update.strip():
                self.plan = {"updated_at_frame": frame["frame_id"], "route": update[:1200]}
            prediction = parsed.get("expected_reappearance_m")
            if isinstance(prediction, (int, float)) and not isinstance(prediction, bool) and math.isfinite(prediction) and prediction >= 0:
                self.reappearance_predictions.append({"step": self.step, "distance_m": prediction,
                                                       "path_m": self.physics.path})
        delta = float(np.linalg.norm(self.env.data.qpos[:2]-qbefore[:2]))
        self_after = odometry(pose(self.env.data), self.origin)
        if action:
            yaw_change = self_after["yaw_rad"] - self_before["yaw_rad"]
            yaw_change = math.atan2(math.sin(yaw_change), math.cos(yaw_change))
            self.feedback.append({"action": action, "from_frame": self.frame_before,
                                  "to_frame": frame["frame_id"], "displacement_xy_m": round(delta, 4),
                                  "yaw_change_rad": round(yaw_change, 4),
                                  "low_translation": action in ("FORWARD", "BACKWARD", "STRAFE_LEFT", "STRAFE_RIGHT") and delta < .08})
        actual_calls = sum(not c.get("generation_skipped", False) for c in self.calls)
        self.inferences += actual_calls
        latency = sum(c["elapsed_s"] for c in self.calls)
        self.inference_latency += latency
        self.amortized += sum(c.get("batch_amortized_s", 0) for c in self.calls)
        if terminal is None and self.autonomous >= self.budget:
            terminal = "budget_exhausted"
        trace = {"step": self.step, "phase": "vlm_autonomous", "action_source": "VLM" if action else "none",
                 "action": action, "decision": parsed, "frame_before": self.frame_before,
                 "frame_after": frame["frame_id"], "before_evaluator": self.before,
                 "after_evaluator": after, "evaluator_ray": ray_check(self.cam),
                 "self_odometry_before": self_before, "self_odometry_after": self_after,
                 "actual_displacement_xy_m": delta, "inference_s": latency,
                 "execution_wall_s": time.perf_counter()-begin, "rejected_stop_count": rejected,
                 "context": json.loads((self.step_folder / "inference_1/context.json").read_text()),
                 "termination": terminal}
        self.write_transition(trace)
        dump(self.folder / "progress.json", {"autonomous_actions": self.autonomous,
             "frame_id": frame["frame_id"], "termination": terminal})
        print("STEP", self.name, self.autonomous, action, "range", after["median_target_surface_range_m"], terminal, flush=True)
        if terminal:
            self.finish(terminal, after)

    def finish(self, terminal, final=None):
        # Capture extra fields before the inherited writer closes file streams.
        super().finish(terminal, final)
        result_path = self.folder / "result.json"
        result = json.loads(result_path.read_text())
        result.update(protocol=VERSION, terrain_id=self.setup["terrain_id"],
                      visible_near_arrival=result["success"],
                      arrival_endpoint="automatic 3 consecutive evaluator samples; separate from VLM STOP",
                      legal_vlm_stop=result["success"] and result["model_stopped"],
                      rejected_stop_count=self.rejected_stops,
                      initial_plan_valid=self.plan_valid, reappearance_predictions=self.reappearance_predictions,
                      policy_action_counts=dict(collections.Counter(self.executed[self.prefix:])),
                      policy_action_sequence=self.executed[self.prefix:],
                      low_translation_actions=sum(f["low_translation"] for f in self.feedback),
                      autonomous_budget=self.budget)
        dump(result_path, result)


def run_status(root, active, expected, policy, begin, status="running"):
    values = [json.loads(p.read_text()) for p in root.glob("rgbd/*/trials/*/result.json")]
    row = {"status": status, "completed": len(values), "expected": expected,
           "arrivals": sum(v["success"] for v in values),
           "legal_vlm_stop": sum(v.get("legal_vlm_stop", False) for v in values),
           "terminations": dict(collections.Counter(v["termination"] for v in values)),
           "active": {t.name: t.autonomous for t in active},
           "generation_calls": policy.generation_calls if policy else 0,
           "wall_s": time.perf_counter()-begin}
    dump(root / "RUN_STATUS.json", row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--processor", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    p.add_argument("--terrains", nargs="*")
    p.add_argument("--action-budget", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--plan-max-new-tokens", type=int, default=640)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    if a.action_budget < 1 or a.batch_size < 1:
        p.error("positive budget and batch size required")
    root = a.out.resolve()
    setups, terrains = build_setups(a.admission.resolve(), a.terrains)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"version": VERSION, "model": str(a.model.resolve()),
                "processor": str(a.processor.resolve()), "scenes": [s["terrain_id"] for s in setups],
                "admission_manifest_sha256": sha(a.admission / "formal_manifest.json"),
                "system_prompt": SYSTEM, "plan_system_prompt": PLAN_SYSTEM,
                "action_budget": a.action_budget, "max_new_tokens": a.max_new_tokens,
                "plan_max_new_tokens": a.plan_max_new_tokens, "batch_size": a.batch_size,
                "max_format_or_stop_attempts": 2, "frame_area_cap": 65536,
                "event_selection": "first, last RGB-visible (>=20 px), latest visible-to-hidden transition, current; deduplicated sorted",
                "sensors": "ideal RGB-D and ideal simulated relative base odometry, not estimated SLAM",
                "stop_gate": "CURRENT RGB red >=20 pixels; finite 0<red-region surface range<1m",
                "action_constraint": "token-level JSON action-value legal-set mask; original relative legal-action logits preserved; no external action selection",
                "shared_output_adapter": "remove complete think block; parse final complete JSON or explicit canonical labeled fields; never infer action from prose; preserve raw output",
                "arrival_endpoint": "exact target RGB identity >=20 px, range<1m, upright/inbounds for 3 10Hz samples; automatic",
                "fixed_approach_included_in_autonomous_budget": False,
                "initial_plan": "one shared VLM call from initial frame only, before first autonomous action",
                "plan_update": "optional text produced by actor in action JSON; no external route correction",
                "return_to_origin": False, "automatic_motion_fallback": False,
                "privileged_target_in_policy": False, "same_base_processor_for_both_models": True,
                "seed": 7, "decoding": "greedy", "real_hardware": False,
                "source_sha256": sha(Path(__file__)),
                "torch": torch.__version__, "renderer": os.environ.get("MUJOCO_GL", "default"),
                "historical_comparability": "new recovered protocol, deleted Oct-17 script unavailable"}
    existing = root / "protocol.json"
    if existing.exists():
        if not a.resume_completed or json.loads(existing.read_text()) != protocol:
            raise RuntimeError("existing run requires exact protocol and --resume-completed")
    else:
        dump(existing, protocol)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    actions()
    act.MAX_STEPS = a.action_budget  # inherited metadata; actual budget handled above
    source = root / "source_snapshot"
    source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        saved = source / path.name
        if saved.exists() and saved.read_bytes() != path.read_bytes():
            raise RuntimeError("source snapshot drift; create a new run")
        saved.write_bytes(path.read_bytes())
    dump(root / "scene_setup.json", setups)
    todo = []
    for setup in setups:
        folder = root / "rgbd/event4_odom_plan/trials" / setup["scene_name"]
        if (folder / "result.json").exists():
            continue
        if folder.exists():
            raise RuntimeError("incomplete trial retained; use new run directory or verified replay")
        todo.append(setup)
    begin = time.perf_counter()
    active = []
    policy = None
    dump(root / "process.json", {"pid": os.getpid(), "started_unix": time.time(),
         "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    try:
        policy = Policy(a.model, a.processor, a.max_new_tokens)
        print("MODEL_READY", a.model, flush=True)
        tasks = iter(todo)
        while True:
            if (root / "STOP").exists():
                run_status(root, active, len(setups), policy, begin, "paused_at_action_boundary")
                return
            new = []
            while len(active) < a.batch_size:
                setup = next(tasks, None)
                if setup is None:
                    break
                trial = Trial(setup, terrains[setup["scene_name"]], root, a.action_budget)
                active.append(trial)
                new.append(trial)
            if not active:
                break
            if shutil.disk_usage(root).free < 8 * 1024**3:
                raise RuntimeError("disk-space guard before next inference")
            if new:
                policy.plan_batch(new, a.plan_max_new_tokens)
            run_status(root, active, len(setups), policy, begin)
            records = policy.decide_batch([t.prepare() for t in active])
            retry = []
            for trial, record in zip(active, records):
                trial.calls.append(record)
                if record["parsed"] is None and not record.get("termination"):
                    retry.append(trial)
            if retry:
                requests = []
                for trial in retry:
                    request = dict(trial.request)
                    request["text"] += "\nCorrection: return the complete valid JSON. STOP remains forbidden when stop_available=false; choose a legal motion."
                    requests.append((request, trial.step_folder / "inference_2", 2))
                records = policy.decide_batch(requests)
                for trial, record in zip(retry, records):
                    trial.calls.append(record)
            for trial in active:
                trial.complete()
            active = [t for t in active if not t.done]
            run_status(root, active, len(setups), policy, begin)
        run_status(root, active, len(setups), policy, begin, "complete_pending_analysis")
    except BaseException as exc:
        dump(root / "ERROR.json", {"type": type(exc).__name__, "message": str(exc)})
        run_status(root, active, len(setups), policy, begin, "error_incomplete_not_navigation_result")
        raise
    finally:
        for trial in active:
            if not trial.done:
                trial.close()


if __name__ == "__main__":
    main()
