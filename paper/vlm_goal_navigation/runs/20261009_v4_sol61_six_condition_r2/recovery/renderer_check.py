import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0,'/home/lry/MoonUnrealEnv/MoonSim/tasks/task3_reobservability')
import all_terrain_context_trial as act
from v4_qwen3vl_trial import DEFAULT_ADMISSION,build_setups
from visible_start_trial import actions
from vlm_benchmark import sha
torch.set_num_threads(1);torch.manual_seed(7);actions()
setups,terrains=build_setups(DEFAULT_ADMISSION,None)
setup=next(s for s in setups if s['scene_name']=='barrier_a0')
trial=act.Trial(setup,terrains['barrier_a0'],Path('/tmp')/f'sol61_renderer_check_{time.time_ns()}','rgb','current')
try:
 from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
 print('RENDERER',glGetString(GL_RENDERER),'VENDOR',glGetString(GL_VENDOR),flush=True)
 original=Path('/home/lry/MoonUnrealEnv/MoonSim/paper/vlm_goal_navigation/runs/20261009_v4_sol61_six_condition_r2/rgb/current/trials/barrier_a0/observations.json')
 frames=json.loads(original.read_text())
 for i,frame in enumerate(trial.frames):
  same={k:sha(Path(frames[i][k]))==sha(Path(frame[k])) for k in ('rgb_path','depth_path')}
  same['metric_depth']=sha(Path(frames[i]['rgb_path']).parent/'depth.npy')==sha(Path(frame['rgb_path']).parent/'depth.npy')
  same['depth_sensor']=frames[i]['depth_sensor']==frame['depth_sensor']
  print('FRAME_MATCH',i,same,flush=True)
 print('OUTPUT',trial.folder,flush=True)
finally:trial.close()
