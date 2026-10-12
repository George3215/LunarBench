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
    p.add_argument('--batch-size', type=int, default=6)
    a = p.parse_args()
    root = a.out.resolve()
    if (root / 'protocol.json').exists():
        raise RuntimeError('Run already exists; use a new output directory')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    actions()
    act.MAX_STEPS = a.max_steps
    setups, terrains = build_setups(a.admission.resolve(), None)
    total = len(setups) * len(MODES) * len(HORIZONS)
    dump(root / 'protocol.json', {
        'model': MODEL, 'reasoning_effort': 'medium', 'version': 'v4-sol61-camera-only/1',
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
        'seed': 7})
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
    dump(root / 'processes.json', {'engine_pid': os.getpid()})
    policy = Sol61Policy(a.batch_size)
    tasks = iter((s, m, h) for s in setups for m in MODES for h in HORIZONS)
    active = []
    start = time.perf_counter()
    try:
        while True:
            while len(active) < a.batch_size:
                item = next(tasks, None)
                if item is None:
                    break
                setup, mode, horizon = item
                active.append(act.Trial(setup, terrains[setup['scene_name']], root, mode, horizon))
            if not active:
                break
            act.status(root, active, total, start, policy)
            if (root / 'STOP').exists():
                raise RuntimeError('User STOP marker; active trials left incomplete')
            if shutil.disk_usage(root).free < 5 * 1024**3:
                raise RuntimeError('Disk guard; active trials left incomplete')
            records = policy.decide_batch([t.prepare() for t in active])
            retry = []
            for trial, record in zip(active, records):
                trial.calls.append(record)
                if record['parsed'] is None and not record.get('termination'):
                    retry.append(trial)
            if retry:
                records = policy.decide_batch([(t.request, t.step_folder / 'inference_2', 2) for t in retry])
                for trial, record in zip(retry, records):
                    trial.calls.append(record)
            for trial in active:
                trial.complete()
            active = [t for t in active if not t.done]
            act.status(root, active, total, start, policy)
            analyse(root)
        value = act.status(root, active, total, start, policy)
        value['status'] = 'complete'
        dump(root / 'RUN_STATUS.json', value)
        analyse(root)
    except Exception as exc:
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
