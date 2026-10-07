"""Camera-only VLM stress test on the unchanged 24 TASK3 worlds.
Prepare physically executed seen/lost prefixes, then run real model decisions and physics.
Privileged evaluator data are never included in LocalVLM inputs.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, platform, sys, time
from pathlib import Path
import mujoco, numpy as np, torch
from PIL import Image
import api, episode as ep
from benchmark import rollout_with_visibility
from camera_reacquisition_audit import Camera,bounds,ACTIONS as MOTION_ACTIONS
from controllers import Primitive
from local_vlm import LocalVLM,MODES

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'paper/vlm_goal_navigation/runs/20261004_qwen35'

def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(out,names):
 out.mkdir(parents=True,exist_ok=True)
 for name in names:
  folder=out/'prefixes'/name; meta=folder/'metadata.json'
  if meta.exists(): continue
  folder.mkdir(parents=True,exist_ok=True)
  scene=api.scene_catalogue()[name];env=api.MuJoCoEnvironment(scene,action_table=ep.protocol_action_table());cam=Camera(env,red_min=50)
  try:
   start=scene.history_xy if scene.history_xy is not None else scene.approach_xy
   env.reset(xy=start,replay_approach=False)
   dx,dy=np.asarray(scene.lost_xy)-np.asarray(start);c,s=math.cos(scene.yaw_rad),math.sin(scene.yaw_rad)
   vx,vy=c*dx+s*dy,-s*dx+c*dy;scale=.6/max(math.hypot(vx,vy),1e-6);vx*=scale;vy*=scale
   prim=Primitive('physical_approach',vx=float(vx),vy=float(vy),duration_s=12,yaw_hold=True)
   track={'seen':False,'seen_run':0,'lost_run':0,'snapshot':None,'frames':[],'last_frame_t':-100.,'sensors':[]}
   def observe(model,data):
    sensor=cam.read(model,data);sensor['in_bounds']=bounds(env,data)
    if track['snapshot'] is not None: return sensor
    now=float(data.time);track['sensors'].append({'time':now,'sensor':sensor})
    if now-track['last_frame_t']>=.49:
     fp=folder/f'prefix_{len(track["frames"]):03d}.png';Image.fromarray(cam.rgb).save(fp)
     track['frames'].append({'path':str(fp),'time':now,'sha256':sha(fp)});track['last_frame_t']=now
    if sensor['visible']:
     track['seen_run']+=1;track['lost_run']=0
     if track['seen_run']>=3: track['seen']=True
    else:
     track['seen_run']=0
     if track['seen']: track['lost_run']=track['lost_run']+1 if sensor['pixels']==0 else 0
     if track['lost_run']>=3 and sensor['in_bounds'] and float(data.xmat[1].reshape(3,3)[2,2])>=.25:
      spec,buf=ep.snapshot(env);np.savez_compressed(folder/'loss_state.npz',state=buf,spec=int(spec))
      track['snapshot']=True;track['loss_t']=now;track['loss_sensor']=sensor;track['loss_qpos']=data.qpos.tolist()
      Image.fromarray(cam.rgb).save(folder/'loss.png')
    return sensor
   observe(env.model,env.data)
   rollout_with_visibility(env.model,env.data,env.terrain,env.controller,prim,env.target,sample_observer=observe)
   metadata={'scene':name,'eligible':track['snapshot'] is not None,'prefix_frames':track['frames'],
    'sensors':track['sensors'],'approach_command':{'action':'approach','vx':float(vx),'vy':float(vy),'yaw_hold':True,
    'duration_s':track.get('loss_t',12)},'loss_t':track.get('loss_t'),'loss_sensor':track.get('loss_sensor'),
    'loss_qpos':track.get('loss_qpos'),'camera':'runtime injected 320x240 RGB; evaluator 10Hz RGB-D',
    'initialization':'unchanged strict_camera physical approach; scene anchors only used in initialization',
    'controller':'cold start at loss, then continuous within recovery','created':time.strftime('%Y-%m-%dT%H:%M:%S')}
   dump(meta,metadata);print('PREPARE',name,'eligible',metadata['eligible'],'loss',metadata['loss_t'],flush=True)
  finally: cam.close();env.close()

class Executor:
 """No target access in motion generation. Controller history persists across actions."""
 def __init__(self,env,cam):
  self.env,self.cam=env,cam;self.ticks=0;self.target_angles=env.controller.default_angles.copy()
  self.streak=0;self.reacquired=False;self.t_reacquired=None;self.path=0.;self.terminal=None
 def execute(self,action,seconds=3.):
  env,cam=self.env,self.cam;data,model,ctrl=env.data,env.model,env.controller
  prim=env.action_table[action];assert not prim.heading_hold and not prim.yaw_hold
  cmd=[prim.vx,prim.vy,prim.wz];lo,hi=model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]
  rows=[];prev=data.qpos[:2].copy();n=int(round(seconds/ctrl.timestep));every=int(round(.1/ctrl.timestep))
  for i in range(n):
   torque=(self.target_angles-data.qpos[7:])*ctrl.kps-data.qvel[6:]*ctrl.kds
   data.ctrl[:]=np.clip(torque,lo,hi);mujoco.mj_step(model,data)
   self.path+=float(np.linalg.norm(data.qpos[:2]-prev));prev=data.qpos[:2].copy()
   upright=float(data.xmat[1].reshape(3,3)[2,2]);in_bounds=bounds(env,data)
   if self.ticks%ctrl.control_decimation==0: self.target_angles=ctrl.act(data,cmd)
   if self.ticks%every==0:
    sensor=cam.read(model,data);self.streak=self.streak+1 if sensor['visible'] else 0
    if self.streak>=3 and upright>=.25 and in_bounds and not self.reacquired:
     self.reacquired=True;self.t_reacquired=float(data.time)
    # Target coordinates only used for logging/evaluation, not for actions or prompts.
    rows.append({'time':float(data.time),'sensor':sensor,'upright':upright,'in_bounds':in_bounds,
     'xy':data.qpos[:2].tolist(),'range_xy_evaluator':float(np.linalg.norm(data.qpos[:2]-env.target[:2]))})
   self.ticks+=1
   if not np.isfinite(data.qpos).all() or upright<.25: self.terminal='fallen';break
   if not in_bounds: self.terminal='out_of_bounds';break
  return rows

def select_history(frames,count=3):
 if len(frames)<=count: return [Path(f) for f in frames]
 return [Path(frames[i]) for i in np.linspace(0,len(frames)-1,count,dtype=int)]


def cached_env(name,prefix):
 # Cache only the physical model; restore a fresh physical and controller state
 # for every episode. No policy decision or outcome is cached here.
 from types import SimpleNamespace
 from controllers import load_controller
 binary=prefix/'compiled_scene.mjb';info=prefix/'compiled_scene.json'
 if binary.exists() and info.exists():
  d=json.loads(info.read_text(encoding='utf8'))
  if d['mujoco_version']!=mujoco.__version__: raise RuntimeError('model cache version mismatch')
  model=mujoco.MjModel.from_binary_path(str(binary))
  return SimpleNamespace(model=model,data=mujoco.MjData(model),controller=load_controller('go2rl'),
   terrain=SimpleNamespace(x=np.array(d['x_bounds']),y=np.array(d['y_bounds'])),
   target=np.array(d['target_xyz_evaluator']),action_table=ep.protocol_action_table(),close=lambda:None)
 env=api.MuJoCoEnvironment(api.scene_catalogue()[name],action_table=ep.protocol_action_table())
 mujoco.mj_saveModel(env.model,str(binary),None)
 dump(info,{'mujoco_version':mujoco.__version__,'x_bounds':[float(env.terrain.x[0]),float(env.terrain.x[-1])],
  'y_bounds':[float(env.terrain.y[0]),float(env.terrain.y[-1])],'target_xyz_evaluator':env.target.tolist(),
  'binary_sha256':sha(binary)})
 return env

def run(out,names,modes,model,max_steps,tokens,context,prefix_root=None):
 vlm=LocalVLM(model=model,max_tokens=tokens,context=context)
 (out/'source_snapshot').mkdir(parents=True,exist_ok=True)
 for source in [Path(__file__),Path(__file__).with_name('local_vlm.py'),Path(__file__).with_name('controllers.py')]:
  (out/'source_snapshot'/(sha(source)[:12]+'_'+source.name)).write_bytes(source.read_bytes())
 dump(out/'model_metadata.json',__import__('requests').post(vlm.endpoint+'/api/show',json={'model':model},timeout=20).json())
 for name in names:
  prefix=(prefix_root if prefix_root is not None else out/'prefixes')/name;meta=json.loads((prefix/'metadata.json').read_text(encoding='utf8'))
  if not meta['eligible']: continue
  env=cached_env(name,prefix);cam=Camera(env,red_min=50)
  try:
   for mode in modes:
    folder=out/'episodes'/model.replace(':','_')/mode/name; result_file=folder/'result.json'
    if result_file.exists(): continue
    folder.mkdir(parents=True,exist_ok=True)
    saved=np.load(prefix/'loss_state.npz');ep.restore(env,(int(saved['spec']),saved['state']))
    executor=Executor(env,cam);past=[r['path'] for r in meta['prefix_frames'] if r['time']<meta['loss_t']]
    commands=[meta['approach_command']];start_time=float(env.data.time);steps=[];termination='budget';arrival=False;beg=time.time()
    for k in range(max_steps):
     sensor=cam.read(env.model,env.data);current=folder/f'frame_{k:02d}.png';Image.fromarray(cam.rgb).save(current)
     try:
      parsed,log=vlm.decide(current,select_history(past),commands,mode,k)
      dump(folder/f'decision_{k:02d}.json',log)
     except Exception as e:
      dump(folder/f'error_{k:02d}.json',{'type':type(e).__name__,'error':str(e)});termination='model_error';break
     if parsed is None: termination='invalid_model_output';break
     action=parsed['action'];before=float(env.data.time)
     if action=='stop':
      # Supplemental stop-near marker criterion. Not a validated navigability metric.
      range_xy=float(np.linalg.norm(env.data.qpos[:2]-env.target[:2]))
      camera_depth=sensor['median_target_depth']
      arrival=bool(sensor['visible'] and range_xy<=1.0 and camera_depth is not None and camera_depth<=1.25 and float(env.data.xmat[1].reshape(3,3)[2,2])>=.25)
      termination='stop_near_target' if arrival else 'premature_stop'
      steps.append({'step':k+1,'action':action,'camera_sensor':sensor,'near_target':arrival,'range_xy_evaluator':range_xy,'model_s':log['elapsed_s']});break
     curve=executor.execute(action)
     steps.append({'step':k+1,'action':action,'model_s':log['elapsed_s'],'curve':curve,'reacquired_so_far':executor.reacquired})
     commands.append({'action':action,'duration_s':float(env.data.time)-before});past.append(str(current))
     dump(folder/'progress.json',{'scene':name,'mode':mode,'steps':steps,'reacquired':executor.reacquired})
     print('STEP',model,mode,name,k+1,action,'seen',executor.reacquired,'model_s',round(log['elapsed_s'],2),flush=True)
     if executor.terminal: termination=executor.terminal;break
    final_sensor=cam.read(env.model,env.data);Image.fromarray(cam.rgb).save(folder/'final.png')
    result={'scene':name,'group':name.rsplit('_',1)[0] if name.startswith('hist') else name,'model':model,'mode':mode,
     'max_steps':max_steps,'steps':steps,'termination':termination,'reacquired':executor.reacquired,
     't_reacquired_s':None if executor.t_reacquired is None else executor.t_reacquired-start_time,
     'arrival_stop_near_marker':arrival,'path_m':executor.path,'simulated_s':float(env.data.time)-start_time,
     'wall_s':time.time()-beg,'runner_sha256':sha(Path(__file__)),'client_sha256':sha(Path(__file__).with_name('local_vlm.py')),'final_sensor':final_sensor,'loss_state_sha256':sha(prefix/'loss_state.npz'),
     'limits':['current benchmark colored sphere','no new training or world model','single deterministic model seed',
      'synchronous simulation pauses for inference; latency recorded','supplemental arrival is stop plus distance and visual depth, not traversability-certified navigation success']}
    dump(result_file,result);print('DONE',model,mode,name,'reacquired',executor.reacquired,'arrival',arrival,termination,flush=True)
  finally: cam.close();env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--prepare',action='store_true');p.add_argument('--scenes',nargs='*');p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES));p.add_argument('--model',default='qwen3.5:4b');p.add_argument('--steps',type=int,default=10);p.add_argument('--tokens',type=int,default=1024);p.add_argument('--context',type=int,default=8192);p.add_argument('--prefix-root',type=Path);a=p.parse_args()
 torch.set_num_threads(1);torch.set_grad_enabled(False)
 names=a.scenes or list(api.scene_catalogue());a.out=a.out.resolve()
 if a.prepare: prepare(a.out,names)
 else:
  dump(a.out/('protocol_'+a.model.replace(':','_')+'.json'),{'model':a.model,'modes':a.modes,'scenes':names,'steps':a.steps,'duration_s':3,'tokens':a.tokens,'context':a.context,'history_images':3,'temperature':0,'seed':7,
   'task':'recover red sphere then approach and stop; reacquisition is primary unchanged-benchmark metric',
   'controller':'cold start at loss; continuous history thereafter','red_min':50,'reacquisition':'3 consecutive samples with >=3 matching pixels at 10Hz before failure',
   'policy_inputs':'RGB images and explicitly enabled command history only; no evaluator state, depth detector output, scene name or coordinates',
   'source_sha256':{f.name:sha(f) for f in [Path(__file__),Path(__file__).with_name('local_vlm.py'),Path(__file__).with_name('controllers.py')]}})
  run(a.out,names,a.modes,a.model,a.steps,a.tokens,a.context,a.prefix_root)
if __name__=='__main__': main()
