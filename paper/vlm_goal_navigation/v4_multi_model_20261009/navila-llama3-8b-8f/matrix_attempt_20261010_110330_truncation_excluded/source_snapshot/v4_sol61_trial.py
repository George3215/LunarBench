"""Run all V4 admitted terrains in the six camera-only conditions."""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import time
from pathlib import Path

import torch
import all_terrain_context_trial as act
from context_navigation_policy import HORIZONS, MODES, SYSTEM
from sol61_policy import Sol61Policy, MODEL
from v4_qwen3vl_trial import DEFAULT_ADMISSION, build_setups
from visible_start_trial import actions
from vlm_benchmark import dump


def analyse(root):
    results = [json.loads(p.read_text()) for p in root.glob('*/*/trials/*/result.json')]
    rows = []
    for mode in MODES:
        for horizon in HORIZONS:
            selected = [r for r in results if r['mode'] == mode and r['horizon'] == horizon]
            rows.append({'mode': mode, 'horizon': horizon, 'completed': len(selected),
                         'success': sum(r['success'] for r in selected),
                         'terminations': dict(collections.Counter(r['termination'] for r in selected))})
    dump(root / 'summary.json', rows)
    lines = ['# GPT-6.1 Sol V4 camera-only trials', '',
             '| Sensor | Frames | Completed | Success | Terminations |', '|---|---|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['horizon']} | {r['completed']} | {r['success']} | {r['terminations']} |")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--admission', type=Path, default=DEFAULT_ADMISSION)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--api-workers', type=int, default=1)
    p.add_argument('--trial-slots', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.batch_size < 1 or a.api_workers < 1:
        raise ValueError('batch-size and api-workers must be positive')
    if a.trial_slots < a.batch_size:
        raise ValueError('trial-slots must be at least batch-size')
    root = a.out.resolve()
    prior = None
    if a.resume:
        prior = json.loads((root/'protocol.json').read_text())
        if prior['model'] != MODEL or prior['max_steps'] != a.max_steps or prior['admission_source'] != str(a.admission.resolve()):
            raise RuntimeError('Resume protocol mismatch')
    elif (root / 'protocol.json').exists():
        raise RuntimeError('Run already exists; use a new output directory')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    actions()
    act.MAX_STEPS = a.max_steps
    setups, terrains = build_setups(a.admission.resolve(), None)
    total = len(setups) * len(MODES) * len(HORIZONS)
    protocol = {
        'model': MODEL, 'reasoning_effort': 'medium', 'version': 'v4-sol61-camera-only/1',
        'api_workers': a.api_workers,
        'admission_source': str(a.admission.resolve()), 'scene_count': len(setups), 'trial_count': total,
        'scenes': [s['terrain_id'] for s in setups], 'sensor_conditions': MODES, 'context_conditions': HORIZONS,
        'system_prompt': SYSTEM, 'max_steps': a.max_steps, 'budget_includes_controlled_approach': True,
        'raw_camera_resolution': [640, 480], 'image_area_cap': 65536,
        'max_output_tokens_including_reasoning': 4096, 'no_truncation_or_sampling': True,
        'history_definition': 'all action-boundary camera frames including common approach',
        'policy_tools': [], 'previous_response_id': None, 'past_outputs_or_actions_supplied': False,
        'depth': 'ideal aligned camera depth and RGB-threshold red-region range only',
        'ground_truth': 'isolated evaluator and fixed task initialization only; never sent to policy',
        'success': '>=20 visible target pixels and <1m surface range for three 10Hz samples; upright/in bounds',
        'comparison_limitations': 'Different image encoder, reasoning, token budget and decoding from Qwen; same camera area cap, task, action and step budget',
        'seed': 7}
    stamp = str(time.time_ns())
    if not a.resume:
        dump(root/'protocol.json', protocol)
    else:
        dump(root/f'resume_{stamp}.json', {'request_batch_size':a.batch_size,'api_workers':a.api_workers,
             'trial_slots':a.trial_slots,'method':'replay executed actions with exact RGB/depth hash equality; reuse matching saved decisions'})
        if (root/'ERROR.json').exists():
            history = root/'recovery'/f'previous_error_{stamp}.json'
            history.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/'ERROR.json',history)
        dump(root/'RECOVERY_STATUS.json', {'phase':'restoring', 'session':stamp,
             'request_batch_size':a.batch_size,'api_workers':a.api_workers,
             'historical_error_file':'ERROR.json; current state is RUN_STATUS.json',
             'wall_time_scope':'new session including replay; previous RUN_STATUS saved separately'})
        shutil.copy2(root/'RUN_STATUS.json',root/'recovery'/f'previous_status_{stamp}.json')
    snapshot = root / ('source_snapshot' if not a.resume else f'recovery/source_snapshot_{stamp}')
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
    dump(root / 'processes.json', {'engine_pid': os.getpid()})
    policy = Sol61Policy(a.api_workers)
    policy.stop_file = root/'STOP'
    if a.resume:
        policy.generation_calls = json.loads((root/'RUN_STATUS.json').read_text()).get('model_generate_calls',0)
    tasks_list = []
    for s in setups:
        for m in MODES:
            for h in HORIZONS:
                folder = root/m/h/'trials'/s['scene_name']
                if not (folder/'result.json').exists():tasks_list.append((s,m,h))
    tasks_list.sort(key=lambda t:not (root/t[1]/t[2]/'trials'/t[0]['scene_name']).exists())
    tasks = iter(tasks_list)
    active = []
    start = time.perf_counter()
    try:
        while True:
            while len(active) < a.trial_slots:
                item = next(tasks, None)
                if item is None:
                    break
                setup, mode, horizon = item
                folder = root/mode/horizon/'trials'/setup['scene_name']
                if folder.exists():
                    from sol61_resume import restore
                    trial = restore(setup,terrains[setup['scene_name']],root,mode,horizon,root/f'recovery/replay_{stamp}')
                else:
                    trial = act.Trial(setup, terrains[setup['scene_name']], root, mode, horizon)
                active.append(trial)
            if not active:
                break
            if a.resume:
                dump(root/'RECOVERY_STATUS.json', {'phase':'running','session':stamp,
                     'request_batch_size':a.batch_size,'api_workers':a.api_workers,
                     'restored_trials':[t.key for t in active if hasattr(t,'wall_time_scope')]})
            act.status(root, active, total, start, policy)
            if (root / 'STOP').exists():
                raise RuntimeError('User STOP marker; active trials left incomplete')
            if shutil.disk_usage(root).free < 5 * 1024**3:
                raise RuntimeError('Disk guard; active trials left incomplete')
            selected = active[:a.batch_size]
            records = policy.decide_batch([t.prepare() for t in selected])
            retry = []
            for trial, record in zip(selected, records):
                trial.calls.append(record)
                if record['parsed'] is None and not record.get('termination'):
                    retry.append(trial)
            if retry:
                records = policy.decide_batch([(t.request, t.step_folder / 'inference_2', 2) for t in retry])
                for trial, record in zip(retry, records):
                    trial.calls.append(record)
            for trial in selected:
                trial.complete()
            active = active[a.batch_size:] + [t for t in selected if not t.done]
            act.status(root, active, total, start, policy)
            analyse(root)
        value = act.status(root, active, total, start, policy)
        value['status'] = 'complete'
        dump(root / 'RUN_STATUS.json', value)
        if a.resume:
            dump(root/'RECOVERY_STATUS.json', {'phase':'complete','session':stamp})
        analyse(root)
    except Exception as exc:
        if a.resume:
            dump(root/'RECOVERY_STATUS.json', {'phase':'interrupted_incomplete','session':stamp,'error':str(exc)})
            dump(root/'recovery'/f'error_{stamp}.json', {'type':type(exc).__name__,'error':str(exc)})
        dump(root / 'ERROR.json', {'type': type(exc).__name__, 'error': str(exc)})
        value = act.status(root, active, total, start, policy)
        value['status'] = 'interrupted_incomplete'
        dump(root / 'RUN_STATUS.json', value)
        analyse(root)
        raise
    finally:
        for trial in active:
            if not trial.done:
                trial.close()


if __name__ == '__main__':
    main()
