"""All 22 V4 terrains, same physical gate and six camera conditions."""
import argparse
import collections
import json
import shutil
import time
from pathlib import Path

import torch
import all_terrain_context_trial as act
from camera_service_policy import CameraServicePolicy
from context_navigation_policy import MODES, HORIZONS
from navila_policy import request_for
from v4_qwen3vl_trial import DEFAULT_ADMISSION, build_setups
from visible_start_trial import actions
from vlm_benchmark import dump


def summary(root):
    results = [json.loads(p.read_text()) for p in root.glob('*/*/trials/*/result.json')]
    rows = []
    for mode, horizon in sorted({(r['mode'], r['horizon']) for r in results}):
        values = [r for r in results if r['mode']==mode and r['horizon']==horizon]
        rows.append({'mode':mode, 'horizon':horizon, 'completed':len(values),
                     'success':sum(r['success'] for r in values),
                     'terminations':dict(collections.Counter(r['termination'] for r in values))})
    dump(root/'summary.json',rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--family',choices=['navila','taga'],required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8101')
    p.add_argument('--protocol',choices=['matrix','native8'],default='matrix')
    a=p.parse_args()
    root=a.out.resolve()
    if (root/'protocol.json').exists():raise RuntimeError('Existing run; use a new output directory')
    root.mkdir(parents=True,exist_ok=True)
    setups,terrains=build_setups(DEFAULT_ADMISSION,None)
    modes=MODES if a.protocol=='matrix' else ['rgb']
    horizons=HORIZONS if a.protocol=='matrix' else ['native8']
    total=len(setups)*len(modes)*len(horizons)
    dump(root/'protocol.json',{'model':a.model,'family':a.family,'protocol':a.protocol,
        'scenes':[s['terrain_id'] for s in setups],'scene_count':len(setups),'trial_count':total,
        'sensor_conditions':modes,'context_conditions':horizons,'max_steps':3000,
        'budget_includes_controlled_approach':True,'policy_inputs':'selected camera frames and RGB-derived depth measurement only',
        'no_ground_truth_to_policy':True,'no_prior_outputs_or_actions':True,
        'history_sampling':'none' if a.protocol=='matrix' else 'NaVILA native uniform sample/pad to 8 RGB images',
        'rgbd_adaptation':'aligned depth PNG plus selected-frame RGB-derived numeric depth; not a natively trained RGBD policy',
        'action_adapter':'NaVILA forward/turn/stop -> same 1s primitives; output distance/angle replaced by fixed primitive magnitudes' if a.family=='navila' else 'seven-action JSON with camera-only backbone; no topology or graph_sprels; not full native Taga method',
        'image_area_cap':65536 if a.protocol=='matrix' else None,
        'success':'same V4 camera evaluator: >=20 target pixels and <1m range for three samples; upright/in bounds',
        'resource_limits_are_not_navigation_failure_evidence':True})
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copy2(source,snapshot/source.name)
    torch.set_num_threads(1);torch.manual_seed(7);actions();act.MAX_STEPS=3000
    act.request_for=request_for
    policy=CameraServicePolicy(a.endpoint,a.model,a.protocol)
    tasks=iter((s,m,h) for s in setups for m in modes for h in horizons)
    active=[];start=time.perf_counter()
    try:
        while True:
            while len(active)<2:
                item=next(tasks,None)
                if item is None:break
                s,m,h=item;active.append(act.Trial(s,terrains[s['scene_name']],root,m,h))
            if not active:break
            act.status(root,active,total,start,policy)
            if (root/'STOP').exists():raise RuntimeError('STOP marker; incomplete trials retained')
            if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('disk guard')
            records=policy.decide_batch([t.prepare() for t in active]);retry=[]
            for trial,record in zip(active,records):
                trial.calls.append(record)
                if record['parsed'] is None and not record.get('termination'):retry.append(trial)
            if retry:
                records=policy.decide_batch([(t.request,t.step_folder/'inference_2',2) for t in retry])
                for trial,record in zip(retry,records):trial.calls.append(record)
            for trial in active:trial.complete()
            active=[t for t in active if not t.done];summary(root)
        value=act.status(root,active,total,start,policy);value['status']='complete'
        dump(root/'RUN_STATUS.json',value);summary(root)
    except Exception as exc:
        dump(root/'ERROR.json',{'type':type(exc).__name__,'error':str(exc)})
        value=act.status(root,active,total,start,policy);value['status']='interrupted_incomplete'
        dump(root/'RUN_STATUS.json',value);raise
    finally:
        for trial in active:
            if not trial.done:trial.close()


if __name__=='__main__':main()
