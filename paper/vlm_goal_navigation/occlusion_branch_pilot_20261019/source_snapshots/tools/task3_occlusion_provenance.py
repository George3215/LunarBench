"""Archive experiment sources, local checkpoint identities and rejection evidence."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
from PIL import Image
import torch

from task3_occlusion_branches import REPO,ROOT,Simulator,build_setups,DEFAULT_ADMISSION,actions,ray_check


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    sources=['tools/task3_occlusion_branches.py','tools/task3_occlusion_forecasts.py',
        'tools/task3_occlusion_analysis.py','tools/task3_occlusion_response.py','tools/task3_occlusion_provenance.py',
        'tools/task3_native_diagnostics.py']+['tasks/task3_reobservability/'+n+'.py' for n in
        ['navigation_trial','all_terrain_setup','scenes','visible_start_trial','v4_qwen3vl_trial',
         'camera_reacquisition_audit','api','controllers','go2_piper_camera','episode','vlm_benchmark']]
    manifest=[]
    for name in sources:
        path=REPO/name;dest=ROOT/'source_snapshots'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes());manifest.append(dict(path=name,sha256=digest(path)))
    assets=[REPO/'assets/robots/go2_rl_gym/policy/go2_moe_cts_high_slope_thre_164k_0.6715.pt']
    models={}
    for model in ['qwen3vl8b','spatial_interactor8b']:
        folder=REPO/'.local/models'/model
        paths=sorted(set(folder.glob('*.json'))|set(folder.glob('*.safetensors')))
        models[model]=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
    provenance=dict(git_parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
        ['torch','transformers','mujoco','numpy','Pillow','matplotlib']},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        render_backend=os.environ.get('MUJOCO_GL'),source_files=manifest,models=models,
        low_level_assets=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest(p)) for p in assets],
        model_generation='BF16, eval, SDPA, greedy seed7, max_new_tokens768; no training',
        incident='First Spatial-Interactor generation encountered parser TypeError before raw persistence; unchanged deterministic generation repeated. A second start failed configuration tuple/list comparison before generation. Logs and old source preserved; no prompt or label-based retry selection.')
    (ROOT/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    torch.set_num_threads(1);torch.manual_seed(7);np.random.seed(7);random.seed(7);actions()
    setups,terrains=build_setups(DEFAULT_ADMISSION,None)
    setup=next(x for x in setups if x['scene_name']=='basin_a1')
    sim=Simulator(setup,terrains['basin_a1']);rows=[]
    try:
        sim.prefix()
        def observe(obs):
            rows.append(dict(time_s=obs['time_s'],evaluation=obs['evaluation'],ray=ray_check(sim.cam)))
            return False
        sim.advance(setup['approach_action'],.4,observe)
        current=sim.observation();dest=ROOT/'states/S03_1'
        Image.fromarray(current['rgb']).save(dest/'excluded_current_rgb.png')
        record=dict(layout='basin_a1',state='S03_1',delayed_s=.4,
            purpose='Independent replay explaining pre-outcome eligibility exclusion; not scored or sent to model',
            samples=rows,current=current['evaluation'],ray=ray_check(sim.cam))
        (dest/'INELIGIBILITY_AUDIT_evaluator_only.json').write_text(json.dumps(record,indent=2)+'\n')
    finally:sim.close()
    print('Provenance and excluded-state audit saved',flush=True)


if __name__=='__main__':main()
