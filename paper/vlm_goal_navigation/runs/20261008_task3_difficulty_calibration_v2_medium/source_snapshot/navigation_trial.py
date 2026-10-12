"""Auditable RGB-only VLM navigation. Evaluator data never enter model messages."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import api
import episode as ep
from camera_reacquisition_audit import Camera, bounds, mask_rgb
from controllers import Primitive
from go2_piper_camera import CAMERA_NAME
from vlm_benchmark import dump, sha

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'assets/model/Qwen3-VL-8B-Instruct'
ACTIONS = {
    'FORWARD': Primitive('forward', vx=1., duration_s=3.),
    'TURN_LEFT': Primitive('turn_left', wz=.75, duration_s=3.),
    'TURN_RIGHT': Primitive('turn_right', wz=-.65, duration_s=3.),
    'STOP': Primitive('stop', duration_s=.3),
}
MIN_PIXELS = 20
SUCCESS_RANGE_M = 1.
SUCCESS_FRAMES = 3
SYSTEM = '''You control a quadruped robot through its current forward-facing RGB camera.
Task: find the described target, approach it, and finish with the target still clearly visible and within approximately 1 metre of the camera.
Use careful visual reasoning before choosing an action. Report a concise assessment grounded in visible evidence, not a long chain of thought. Never invent an object hidden behind terrain. If uncertain about seeing the target, say so.
Actions: FORWARD moves forward for 3 seconds; TURN_LEFT and TURN_RIGHT rotate in place for 3 seconds; STOP ends the trial. Motion on rough terrain can differ from the command.
If the target is not visible, explore safely: you may turn to scan or advance through traversable terrain to seek a new viewpoint. Absence alone does not forbid forward motion. Do not claim arrival without seeing the target. If visible to one side, align towards it. Avoid walking into an obvious obstacle. Choose exploration according to the image; avoid repeatedly staying at the same viewpoint. Do not advance past a close target and lose it from the camera. STOP only when the target is visible and appears within 1 metre.
Every decision is followed by actual motion and a fresh camera image. You may make many decisions, but each response must choose exactly one action.
Return one JSON object, no markdown:
{"target_visible": true or false, "target_location": "left|centre|right|absent|uncertain", "proximity": "near|far|uncertain", "confidence": number from 0 to 1, "visual_assessment": "short description of visible target and obstacles", "decision_rationale": "brief reason for this action", "action": "FORWARD|TURN_LEFT|TURN_RIGHT|STOP"}'''
USER = 'Target: red sphere\nCurrent camera image:'


def parse_decision(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or value.get('action') not in ACTIONS:
        return None
    if not isinstance(value.get('target_visible'), bool):
        return None
    for key in ('visual_assessment', 'decision_rationale'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            return None
    if value.get('target_location') not in ('left', 'centre', 'right', 'absent', 'uncertain'):
        return None
    if value.get('proximity') not in ('near', 'far', 'uncertain'):
        return None
    confidence = value.get('confidence')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return None
    return value


def qualifies(pixels, median_range, upright, in_bounds):
    return bool(pixels >= MIN_PIXELS and median_range is not None
                and math.isfinite(median_range) and median_range < SUCCESS_RANGE_M
                and upright >= .25 and in_bounds)


class EvaluatorCamera(Camera):
    """Exact target identity via segmentation; RGB and depth remain isolated from policy."""
    def __init__(self, env, red_min=50, render_size=None):
        # Multisampled ID-color edges can produce invalid segmentation IDs.
        # Disable MSAA consistently for RGB, depth and identity evaluation.
        env.model.vis.quality.offsamples = 0
        super().__init__(env, red_min=red_min)
        if render_size is not None:
            self.renderer.close()
            self.renderer = mujoco.Renderer(env.model, width=render_size[0], height=render_size[1])

    def read(self, model, data):
        super().read(model, data)
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(self.render_data, camera=CAMERA_NAME)
            seg = self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()
        target_id = model.geom('task3_target_geom').id
        exact_mask = (seg[..., 0] == target_id) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        colour_mask = mask_rgb(self.rgb, red_min=50)
        visible_mask = exact_mask & colour_mask
        y, x = np.where(visible_mask)
        cid = model.camera(CAMERA_NAME).id
        height, width = self.depth.shape
        focal = height / (2 * math.tan(math.radians(float(model.cam_fovy[cid])) / 2))
        axial = self.depth[visible_mask]
        radial = axial * np.sqrt(1 + ((x + .5 - width/2) / focal)**2 + ((y + .5 - height/2) / focal)**2)
        valid = radial[np.isfinite(radial) & (radial > 0)]
        distance = float(np.median(valid)) if len(valid) else None
        upright = float(data.xmat[1].reshape(3, 3)[2, 2])
        in_bounds = bounds(self.env, data)
        self.visible_mask = visible_mask
        centre_range = float(np.linalg.norm(data.cam_xpos[cid] - data.geom_xpos[target_id]))
        self.measurement = {
            'sim_time_s': float(data.time), 'target_segment_pixels': int(exact_mask.sum()),
            'target_visible_rgb_pixels': int(visible_mask.sum()),
            'red_false_positive_pixels': int((colour_mask & ~exact_mask).sum()),
            'centroid_uv': [float(x.mean()/width), float(y.mean()/height)] if len(x) else None,
            'median_target_axial_depth_m': float(np.median(axial)) if len(axial) else None,
            'median_target_surface_range_m': distance, 'camera_to_target_centre_m': centre_range,
            'base_xy_target_distance_m_diagnostic': float(np.linalg.norm(data.qpos[:2]-data.geom_xpos[target_id, :2])),
            'camera_xyz_evaluator': data.cam_xpos[cid].tolist(),
            'target_centre_xyz_evaluator': data.geom_xpos[target_id].tolist(),
            'upright': upright, 'in_bounds': in_bounds,
            'near_and_visible': qualifies(len(x), distance, upright, in_bounds),
        }
        return self.measurement

    def save_evidence(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgb).save(folder/'rgb.png')
        Image.fromarray(self.visible_mask.astype(np.uint8)*255).save(folder/'target_mask.png')
        np.save(folder/'depth.npy', self.depth)
        dump(folder/'measurement.json', self.measurement)


class Policy:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODEL), dtype=torch.bfloat16, local_files_only=True,
            attn_implementation='sdpa').to('cuda:0').eval()

    def decide_batch(self, requests):
        """Batch independent observations; each sequence attends only to its own input."""
        batch = []
        for image_path, folder, attempt in requests:
            folder.mkdir(parents=True, exist_ok=True)
            text = USER if attempt == 1 else USER + '\nReturn the requested complete JSON object with one action and concise visual assessment and decision rationale.'
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [
                {'type': 'text', 'text': text}, {'type': 'image', 'image': Image.open(image_path).convert('RGB')}]}]
            dump(folder/'messages.json', {'messages': [messages[0], {'role': 'user', 'content': [
                {'type': 'text', 'text': text}, {'type': 'image', 'image_path': str(image_path), 'sha256': sha(image_path)}]}],
                'policy_inputs': 'only current RGB and target semantic description; no inter-trial sharing'})
            (folder/'rendered_prompt.txt').write_text(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            batch.append(messages)
        begin = time.perf_counter()
        inputs = self.processor.apply_chat_template(batch, tokenize=True, padding=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda:0')
        grids = inputs['image_grid_thw'].tolist()
        if len(grids) != len(requests):
            raise RuntimeError('Expected exactly one image for each independent batch sequence')
        pixels = inputs['pixel_values'].detach().cpu().float().numpy()
        offset = 0
        for i, (image_path, folder, attempt) in enumerate(requests):
            count = int(np.prod(grids[i]))
            image_pixels = pixels[offset:offset+count]
            offset += count
            dump(folder/'network_input.json', {'input_ids': inputs['input_ids'][i].tolist(),
                'attention_mask': inputs['attention_mask'][i].tolist(), 'image_grid_thw': grids[i],
                'pixel_values_shape': list(image_pixels.shape),
                'pixel_values_float32_sha256': hashlib.sha256(image_pixels.tobytes()).hexdigest(),
                'image_sha256': sha(image_path), 'batch_size': len(requests), 'independent_sequence': i})
        if offset != len(pixels):
            raise RuntimeError('Image patch accounting mismatch')
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                          return_dict_in_generate=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-begin
        records = []
        for i, (image_path, folder, attempt) in enumerate(requests):
            ids = outputs.sequences[i, inputs['input_ids'].shape[1]:].tolist()
            eos = self.model.generation_config.eos_token_id
            end_ids = eos if isinstance(eos, list) else [eos]
            for j, token in enumerate(ids):
                if token in end_ids:
                    ids = ids[:j+1]
                    break
            raw = self.processor.decode(ids, skip_special_tokens=True).strip()
            record = {'elapsed_s': elapsed, 'latency_definition': 'wall latency of independent batched model generation',
                'batch_size': len(requests), 'batch_amortized_s': elapsed/len(requests),
                'raw_output': raw, 'raw_output_with_special_tokens': self.processor.decode(ids, skip_special_tokens=False),
                'generated_token_ids': ids, 'generated_token_count': len(ids),
                'hit_token_limit': len(ids) >= 256, 'parsed': parse_decision(raw)}
            dump(folder/'output.json', record)
            (folder/'raw_output.txt').write_text(raw)
            records.append(record)
        return records

    def decide(self, image_path, folder, attempt):
        folder.mkdir(parents=True, exist_ok=True)
        text = USER if attempt == 1 else USER + '\nReturn the requested complete JSON object with one action and concise visual assessment and decision rationale.'
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [
            {'type': 'text', 'text': text}, {'type': 'image', 'image': Image.open(image_path).convert('RGB')}]}]
        dump(folder/'messages.json', {'messages': [messages[0], {'role': 'user', 'content': [
            {'type': 'text', 'text': text}, {'type': 'image', 'image_path': str(image_path), 'sha256': sha(image_path)}]}],
            'policy_inputs': 'current RGB and target semantic description only; no history, coordinates, depth or evaluator data'})
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        (folder/'rendered_prompt.txt').write_text(prompt)
        begin = time.perf_counter()
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to('cuda:0')
        pixels = inputs['pixel_values'].detach().cpu().float().numpy()
        dump(folder/'network_input.json', {'input_ids': inputs['input_ids'].tolist(),
            'image_grid_thw': inputs['image_grid_thw'].tolist(), 'pixel_values_shape': list(pixels.shape),
            'pixel_values_float32_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
            'image_sha256': sha(image_path)})
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                          return_dict_in_generate=True)
        torch.cuda.synchronize()
        ids = outputs.sequences[0, inputs['input_ids'].shape[1]:].tolist()
        raw = self.processor.decode(ids, skip_special_tokens=True).strip()
        record = {'elapsed_s': time.perf_counter()-begin, 'raw_output': raw,
                  'raw_output_with_special_tokens': self.processor.decode(ids, skip_special_tokens=False),
                  'generated_token_ids': ids, 'generated_token_count': len(ids),
                  'hit_token_limit': len(ids) >= 256, 'parsed': parse_decision(raw)}
        dump(folder/'output.json', record)
        (folder/'raw_output.txt').write_text(raw)
        return record


class Physics:
    def __init__(self, env, camera, stream, evaluator_stream, folder):
        self.env, self.camera, self.stream = env, camera, stream
        self.evaluator_stream, self.folder = evaluator_stream, folder
        self.ticks = 0
        self.angles = env.controller.default_angles.copy()
        self.path = 0.
        self.streak = 0
        self.samples = 0
        self.min_range = float('inf')
        self.min_upright = 1.

    def record(self, step, action):
        e, d = self.env, self.env.data
        measurement = self.camera.read(e.model, d)
        self.streak = self.streak + 1 if measurement['near_and_visible'] else 0
        measurement = dict(measurement, step=step, action=action, success_streak=self.streak)
        self.evaluator_stream.write(json.dumps(measurement)+'\n')
        self.samples += 1
        if measurement['median_target_surface_range_m'] is not None:
            self.min_range = min(self.min_range, measurement['median_target_surface_range_m'])
        self.min_upright = min(self.min_upright, measurement['upright'])
        state = {'step': step, 'action': action, 'sim_time_s': float(d.time),
                 'qpos': d.qpos.tolist(), 'qvel': d.qvel.tolist(), 'ctrl': d.ctrl.tolist(),
                 'actuator_force': d.actuator_force.tolist(), 'ncon': int(d.ncon),
                 'contacts': [{'geom1': int(d.contact[i].geom1), 'geom2': int(d.contact[i].geom2),
                               'distance_m': float(d.contact[i].dist)} for i in range(d.ncon)],
                 'evaluation': measurement}
        self.stream.write(json.dumps(state)+'\n')
        if measurement['near_and_visible']:
            self.camera.save_evidence(self.folder/'arrival_evidence'/f'sample_{self.samples:06d}')
        return state

    def execute(self, action, step):
        e, d, m, c = self.env, self.env.data, self.env.model, self.env.controller
        primitive = ACTIONS[action]
        previous = d.qpos[:2].copy()
        every = max(1, round(.1/c.timestep))
        terminal = None
        for _ in range(round(primitive.duration_s/c.timestep)):
            if self.ticks % c.control_decimation == 0:
                self.angles = c.act(d, [primitive.vx, primitive.vy, primitive.wz])
            d.ctrl[:] = np.clip((self.angles-d.qpos[7:])*c.kps-d.qvel[6:]*c.kds,
                               m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mujoco.mj_step(m, d)
            self.path += float(np.linalg.norm(d.qpos[:2]-previous))
            previous = d.qpos[:2].copy()
            self.ticks += 1
            upright = float(d.xmat[1].reshape(3, 3)[2, 2])
            if not np.isfinite(d.qpos).all() or upright < .25:
                terminal = 'fallen'
            elif not bounds(e, d):
                terminal = 'out_of_bounds'
            if self.ticks % every == 0 or terminal:
                self.record(step, action)
                if terminal is None and getattr(self, 'extra_terminal', None):
                    terminal = self.extra_terminal
                if terminal is None and self.streak >= SUCCESS_FRAMES:
                    terminal = 'success_visible_near'
            if terminal:
                break
        self.stream.flush()
        self.evaluator_stream.flush()
        return terminal


def protocol(out, names):
    dump(out/'protocol.json', {'version': 'qwen3vl-visible-near-trial/2', 'model': str(MODEL),
        'scenes': names, 'trial_count': len(names), 'budget_tiers': [1000, 2000, 3000],
        'max_inferences_per_action': 2, 'max_total_inferences_per_trial': 6000,
        'tier_mode': 'same continuous trial; expand only when budget reached; no retries after terminal',
        'system_prompt': SYSTEM, 'user_text': USER, 'max_new_tokens': 256,
        'temperature': 'greedy', 'seed': 7, 'thinking_mode': 'Instruct model; concise public decision rationale requested; no native Thinking switch',
        'action_commands': {k: vars(v) for k, v in ACTIONS.items()},
        'success': {'visible_pixels': MIN_PIXELS, 'range_m_strictly_less_than': 1.,
            'range_definition': 'median Euclidean camera-to-visible-target-surface range, axial rendered depth back-projected with intrinsics',
            'identity': 'target geom segmentation intersect RGB red pixels; evaluator only',
            'render_msaa_samples': 0,
            'consecutive_10hz_samples': SUCCESS_FRAMES, 'upright_min': .25, 'in_bounds': True},
        'input_fields': ['target semantic description', 'current RGB'],
        'initialization': 'scene.lost_xy, scene.yaw_rad, fresh physics/controller, no approach replay',
        'state_hz': 10, 'sim_pauses_for_inference': True, 'real_hardware': False,
        'runtime': {'torch': torch.__version__, 'mujoco': mujoco.__version__},
        'source_sha256': sha(Path(__file__))})
    snapshot = out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['navigation_trial.py', 'api.py', 'episode.py', 'benchmark.py', 'controllers.py',
                 'scenes.py', 'camera_reacquisition_audit.py', 'history_pairs.py', 'go2_piper_camera.py',
                 'experiment.py', 'terrain.py', 'slabs.py']:
        source = Path(__file__).with_name(name)
        (snapshot/name).write_bytes(source.read_bytes())
    descriptor = Path(__file__).with_name('generated')/'history_pairs.json'
    (out/'history_pairs.json').write_bytes(descriptor.read_bytes())


def run_trial(policy, name, scene, out):
    folder = out/'trials'/name
    if (folder/'result.json').exists():
        return
    if folder.exists():
        raise RuntimeError(f'Partial trial exists: {folder}; use a new run directory')
    folder.mkdir(parents=True)
    env = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table())
    cam = EvaluatorCamera(env, red_min=50)
    try:
        env.reset(replay_approach=False)
        initial_measurement = cam.read(env.model, env.data)
        dump(folder/'initialization.json', {'scene': vars(scene), 'qpos': env.data.qpos.tolist(),
            'qvel': env.data.qvel.tolist(), 'initial_camera_evaluator': initial_measurement,
            'target_anchor_xyz_evaluator': env.target.tolist()})
        cam.save_evidence(folder/'initial_evidence')
        begin = time.perf_counter()
        inference_count, total_inference, executed = 0, 0., []
        termination = 'budget_exhausted'
        with (folder/'state.jsonl').open('w') as sf, (folder/'evaluator.jsonl').open('w') as ef, (folder/'trace.jsonl').open('w') as tf:
            physics = Physics(env, cam, sf, ef, folder)
            physics.record(0, 'RESET')
            for step in range(1, 3001):
                step_folder = folder/'steps'/f'{step:04d}'
                step_folder.mkdir(parents=True)
                observation_begin = time.perf_counter()
                before = cam.read(env.model, env.data)
                image_path = step_folder/'input_rgb.png'
                Image.fromarray(cam.rgb).save(image_path)
                cam.save_evidence(step_folder/'before_evaluator')
                observation_s = time.perf_counter()-observation_begin
                calls = []
                parsed = None
                for attempt in (1, 2):
                    call = policy.decide(image_path, step_folder/f'inference_{attempt}', attempt)
                    calls.append(call)
                    inference_count += 1
                    total_inference += call['elapsed_s']
                    parsed = call['parsed']
                    if parsed is not None:
                        break
                action = parsed['action'] if parsed else None
                qpos_before = env.data.qpos.copy()
                action_begin = time.perf_counter()
                if action is None:
                    termination = 'invalid_model_output'
                else:
                    executed.append(action)
                    termination = physics.execute(action, step)
                    if action == 'STOP' and termination is None:
                        termination = 'premature_stop'
                after = cam.read(env.model, env.data)
                cam.save_evidence(step_folder/'after_evaluator')
                trace = {'step': step, 'input_image': str(image_path.relative_to(folder)),
                    'input_image_sha256': sha(image_path), 'inference_count_this_step': len(calls),
                    'inference_records': [str((step_folder/f'inference_{i+1}'/'output.json').relative_to(folder)) for i in range(len(calls))],
                    'model_output_raw': [c['raw_output'] for c in calls], 'decision': parsed,
                    'action': action, 'observation_s': observation_s,
                    'inference_s': sum(c['elapsed_s'] for c in calls),
                    'execution_wall_s': time.perf_counter()-action_begin,
                    'actual_displacement_xy_m': float(np.linalg.norm(env.data.qpos[:2]-qpos_before[:2])),
                    'before_evaluator': before, 'after_evaluator': after,
                    'success_streak': physics.streak, 'termination': termination}
                dump(step_folder/'transition.json', trace)
                tf.write(json.dumps(trace)+'\n')
                tf.flush()
                dump(folder/'progress.json', {'step': step, 'inference_count': inference_count,
                    'action': action, 'evaluator': after, 'termination': termination})
                distance = after['median_target_surface_range_m']
                print('STEP', name, step, action, 'visible_pixels', after['target_visible_rgb_pixels'],
                      'surface_range', None if distance is None else round(distance, 3), 'terminal', termination, flush=True)
                if termination:
                    break
                if step in (1000, 2000):
                    dump(folder/f'budget_{step}.json', {'step': step, 'next_budget': step+1000, 'evaluation': after})
            else:
                termination = 'budget_exhausted'
            # Record final state without fabricating another 10Hz success sample.
            final = cam.read(env.model, env.data)
            cam.save_evidence(folder/'final_evidence')
        result = {'scene': name, 'success': termination == 'success_visible_near',
            'termination': termination, 'inference_count': inference_count, 'action_steps': len(executed),
            'total_inference_s': total_inference, 'average_inference_s': total_inference/max(1, inference_count),
            'wall_s': time.perf_counter()-begin, 'simulated_s': float(env.data.time),
            'path_m': physics.path, 'min_visible_surface_range_m': None if not math.isfinite(physics.min_range) else physics.min_range,
            'min_upright': physics.min_upright, 'state_samples': physics.samples,
            'final_evaluator': final, 'success_streak': physics.streak,
            'model_stopped': bool(executed and executed[-1] == 'STOP'),
            'action_sequence': executed, 'action_counts': dict(collections.Counter(executed)), 'max_steps': 3000}
        dump(folder/'result.json', result)
        print('DONE', name, termination, flush=True)
    finally:
        cam.close()
        env.close()


class Trial:
    def __init__(self, name, scene, out, terrain=None, render_size=None):
        self.name = name
        self.folder = out/'trials'/name
        if self.folder.exists():
            raise RuntimeError(f'Trial already exists: {self.folder}; use fresh output directory')
        self.folder.mkdir(parents=True)
        self.env = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table(), terrain=terrain)
        self.cam = EvaluatorCamera(self.env, red_min=50, render_size=render_size)
        self.env.reset(replay_approach=False)
        initial = self.cam.read(self.env.model, self.env.data)
        dump(self.folder/'initialization.json', {'scene': vars(scene), 'qpos': self.env.data.qpos.tolist(),
            'qvel': self.env.data.qvel.tolist(), 'initial_camera_evaluator': initial,
            'target_anchor_xyz_evaluator': self.env.target.tolist()})
        self.cam.save_evidence(self.folder/'initial_evidence')
        dump(self.folder/'model_metadata.json', {'ngeom': self.env.model.ngeom,
            'geom_names': {str(i): mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
                           for i in range(self.env.model.ngeom)}, 'source': 'online model at reset'})
        self.sf = (self.folder/'state.jsonl').open('w')
        self.ef = (self.folder/'evaluator.jsonl').open('w')
        self.tf = (self.folder/'trace.jsonl').open('w')
        self.physics = Physics(self.env, self.cam, self.sf, self.ef, self.folder)
        self.physics.record(0, 'RESET')
        self.begin = time.perf_counter()
        self.step = 0
        self.inference_count = 0
        self.inference_latency = 0.
        self.amortized_inference = 0.
        self.executed = []
        self.own_active_wall = 0.
        self.done = False

    def prepare(self):
        begin = time.perf_counter()
        self.step += 1
        self.step_folder = self.folder/'steps'/f'{self.step:04d}'
        self.step_folder.mkdir(parents=True)
        self.before = self.cam.read(self.env.model, self.env.data)
        self.image_path = self.step_folder/'input_rgb.png'
        Image.fromarray(self.cam.rgb).save(self.image_path)
        self.cam.save_evidence(self.step_folder/'before_evaluator')
        self.observation_s = time.perf_counter()-begin
        self.calls = []
        return self.image_path, self.step_folder/'inference_1', 1

    def complete(self):
        parsed = self.calls[-1]['parsed']
        action = parsed['action'] if parsed else None
        qpos_before = self.env.data.qpos.copy()
        begin = time.perf_counter()
        if action is None:
            terminal = 'invalid_model_output'
        else:
            self.executed.append(action)
            terminal = self.physics.execute(action, self.step)
            if action == 'STOP' and terminal is None:
                terminal = 'premature_stop'
        after = self.cam.read(self.env.model, self.env.data)
        self.cam.save_evidence(self.step_folder/'after_evaluator')
        execution_s = time.perf_counter()-begin
        if terminal is None and self.step == 3000:
            terminal = 'budget_exhausted'
        self.inference_count += len(self.calls)
        latency = sum(c['elapsed_s'] for c in self.calls)
        amortized = sum(c['batch_amortized_s'] for c in self.calls)
        self.inference_latency += latency
        self.amortized_inference += amortized
        self.own_active_wall += self.observation_s + latency + execution_s
        trace = {'step': self.step, 'input_image': str(self.image_path.relative_to(self.folder)),
            'input_image_sha256': sha(self.image_path), 'inference_count_this_step': len(self.calls),
            'inference_records': [str((self.step_folder/f'inference_{i+1}'/'output.json').relative_to(self.folder)) for i in range(len(self.calls))],
            'model_output_raw': [c['raw_output'] for c in self.calls], 'decision': parsed, 'action': action,
            'observation_s': self.observation_s, 'inference_s': latency, 'amortized_inference_s': amortized,
            'execution_wall_s': execution_s,
            'actual_displacement_xy_m': float(np.linalg.norm(self.env.data.qpos[:2]-qpos_before[:2])),
            'before_evaluator': self.before, 'after_evaluator': after,
            'success_streak': self.physics.streak, 'termination': terminal}
        dump(self.step_folder/'transition.json', trace)
        self.tf.write(json.dumps(trace)+'\n')
        self.tf.flush()
        dump(self.folder/'progress.json', {'step': self.step, 'inference_count': self.inference_count,
            'action': action, 'evaluator': after, 'termination': terminal})
        distance = after['median_target_surface_range_m']
        print('STEP', self.name, self.step, action, 'visible_pixels', after['target_visible_rgb_pixels'],
              'surface_range', None if distance is None else round(distance, 3), 'terminal', terminal, flush=True)
        if self.step in (1000, 2000) and terminal is None:
            dump(self.folder/f'budget_{self.step}.json', {'step': self.step, 'next_budget': self.step+1000, 'evaluation': after})
        if terminal:
            self.cam.save_evidence(self.folder/'final_evidence')
            result = {'scene': self.name, 'success': terminal == 'success_visible_near', 'termination': terminal,
                'inference_count': self.inference_count, 'action_steps': len(self.executed),
                'total_inference_s': self.inference_latency, 'amortized_inference_s': self.amortized_inference,
                'average_inference_s': self.inference_latency/max(1, self.inference_count),
                'wall_s_including_scheduler_wait': time.perf_counter()-self.begin,
                'active_episode_wall_s': self.own_active_wall, 'simulated_s': float(self.env.data.time),
                'path_m': self.physics.path, 'min_visible_surface_range_m': None if not math.isfinite(self.physics.min_range) else self.physics.min_range,
                'min_upright': self.physics.min_upright, 'state_samples': self.physics.samples,
                'final_evaluator': after, 'success_streak': self.physics.streak,
                'model_stopped': bool(self.executed and self.executed[-1] == 'STOP'),
                'action_sequence': self.executed, 'action_counts': dict(collections.Counter(self.executed)), 'max_steps': 3000}
            dump(self.folder/'result.json', result)
            self.done = True
            self.close()
            print('DONE', self.name, terminal, flush=True)

    def close(self):
        self.sf.close()
        self.ef.close()
        self.tf.close()
        self.cam.close()
        self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenes', nargs='*')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8:
        parser.error('batch-size must be 1..8')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.set_grad_enabled(False)
    catalogue = api.scene_catalogue()
    names = args.scenes or list(catalogue)
    protocol(out, names)
    dump(out/'scheduler.json', {'batch_size': args.batch_size,
        'independence': 'separate simulator, image, messages, attention-mask sequence and trial logs; no cross-trial context',
        'inference_count': 'one independent image decision = one inference; physical generate calls counted separately',
        'timing': 'per-trial latency includes batch model wait; amortized GPU wall time also reported'})
    policy = Policy()
    print('MODEL_READY', flush=True)
    pending = iter(names)
    active = []
    generation_calls = 0
    start = time.perf_counter()
    try:
        while True:
            while len(active) < args.batch_size:
                name = next(pending, None)
                if name is None:
                    break
                active.append(Trial(name, catalogue[name], out))
            if not active:
                break
            records = policy.decide_batch([t.prepare() for t in active])
            generation_calls += 1
            retry = []
            for trial, record in zip(active, records):
                trial.calls.append(record)
                if record['parsed'] is None:
                    retry.append(trial)
            if retry:
                retry_records = policy.decide_batch([(t.image_path, t.step_folder/'inference_2', 2) for t in retry])
                generation_calls += 1
                for trial, record in zip(retry, retry_records):
                    trial.calls.append(record)
            for trial in active:
                trial.complete()
            active = [t for t in active if not t.done]
            results = [json.loads(p.read_text()) for p in (out/'trials').glob('*/result.json')]
            dump(out/'aggregate.json', {'completed': len(results), 'expected': len(names),
                'success_count': sum(r['success'] for r in results),
                'success_rate_completed': sum(r['success'] for r in results)/len(results) if results else None,
                'terminations': dict(collections.Counter(r['termination'] for r in results)),
                'model_generate_calls': generation_calls, 'run_wall_s': time.perf_counter()-start,
                'active_trials': {t.name: t.step for t in active}})
    finally:
        for trial in active:
            if not trial.done:
                trial.close()

if __name__ == '__main__':
    main()
