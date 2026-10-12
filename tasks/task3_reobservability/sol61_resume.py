"""Restore unfinished trials by replaying executed actions, with camera equality gates."""
import json
import time
from pathlib import Path

import all_terrain_context_trial as act
from vlm_benchmark import dump, sha


def restore(setup, terrain, root, mode, horizon, recovery_root):
    folder = root/mode/horizon/'trials'/setup['scene_name']
    traces = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    original_frames = json.loads((folder/'observations.json').read_text())
    if traces[-1].get('termination'):
        raise RuntimeError('Terminal trace without result; manual reconciliation required')
    trial = act.Trial(setup, terrain, recovery_root, mode, horizon)
    checks = []

    def verify_frame(index):
        original, replay = original_frames[index], trial.frames[index]
        for key in ('rgb_path', 'depth_path'):
            if sha(Path(original[key])) != sha(Path(replay[key])):
                raise RuntimeError(f'Replay camera mismatch: {trial.key}, frame {index}, {key}')
        if sha(Path(original['rgb_path']).parent/'depth.npy') != sha(Path(replay['rgb_path']).parent/'depth.npy'):
            raise RuntimeError(f'Replay metric depth mismatch: {trial.key}, frame {index}')
        if original['depth_sensor'] != replay['depth_sensor']:
            raise RuntimeError(f'Replay depth sensor measurement mismatch: {trial.key}, frame {index}')
        if abs(original['sim_time_s']-replay['sim_time_s']) > 1e-9:
            raise RuntimeError('Replay simulation time mismatch')
        checks.append(index)

    try:
        for index in range(len(trial.frames)):
            verify_frame(index)
        for trace in traces:
            if trace['phase'] != 'vlm_autonomous':
                continue
            trial.step = trace['step']
            action = trace['action']
            trial.executed.append(action)
            terminal = trial.physics.execute(action, trial.step)
            if terminal:
                raise RuntimeError(f'Replay unexpected terminal: {terminal}')
            trial.observe()
            verify_frame(trace['frame_after'])
        if len(trial.frames) != len(original_frames):
            raise RuntimeError('Replay frame count mismatch')
        dump(folder/'recovery_camera_check.json', {
            'verified_rgb_and_depth_frame_ids': checks, 'passed': True,
            'metric_depth_and_sensor_measurements_exact': True,
            'executed_actions_replayed': len(trial.executed),
            'restored_step': traces[-1]['step'], 'replay_evidence': str(trial.folder),
            'method': 'real physics replay of recorded actions; no target coordinates supplied to model'})
        for stream in (trial.sf, trial.ef, trial.tf):
            stream.close()
        trial.folder = folder
        trial.sf = (folder/'state.jsonl').open('a')
        trial.ef = (folder/'evaluator.jsonl').open('a')
        trial.tf = (folder/'trace.jsonl').open('a')
        trial.physics.stream = trial.sf
        trial.physics.evaluator_stream = trial.ef
        trial.physics.folder = folder
        trial.frames = original_frames
        trial.step = traces[-1]['step']
        trial.inferences = sum(t.get('actual_generation_calls', 0) for t in traces)
        trial.inference_latency = sum(t.get('inference_s', 0.) for t in traces)
        trial.amortized = trial.inference_latency
        trial.begin = time.perf_counter()
        trial.wall_time_scope = 'recovery session only; excludes original session, outage and replay'
        trial.resume_verified = True
        print('RESUME_CAMERA_MATCH', trial.key, 'step', trial.step, 'frames', len(checks), flush=True)
        return trial
    except Exception:
        trial.close()
        raise
