"""Auditable camera reacquisition supplement. No legacy result is overwritten.
Camera observations drive eligibility and outcomes; geometry is diagnostic only.
Run from repository root: python tasks/task3_reobservability/camera_reacquisition_audit.py
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import mujoco
import torch
import api
import episode as ep
from benchmark import rollout_with_visibility, _yaw
from controllers import Primitive
from go2_piper_camera import CAMERA_NAME

OUT = Path(__file__).resolve().parents[2] / 'paper/audit_20261004'
ACTIONS = ('forward','back','left','right','turn_left','turn_right')
THRESHOLDS = (1,3,10)

def mask_rgb(rgb, red_min=120):
    r,g,b = [rgb[...,i].astype(np.int16) for i in range(3)]
    return (r>red_min)&(r>1.5*g)&(r>1.5*b)

class Camera:
    def __init__(self, env, red_min=120):
        self.env=env
        self.red_min=red_min
        self.render_data=mujoco.MjData(env.model)
        self.renderer=mujoco.Renderer(env.model,height=240,width=320)
        self.rgb=None; self.depth=None
    def read(self, model, data):
        mujoco.mj_copyData(self.render_data,model,data)
        mujoco.mj_forward(model,self.render_data)
        self.renderer.update_scene(self.render_data,camera=CAMERA_NAME)
        self.rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.depth=self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        mask=mask_rgb(self.rgb,self.red_min); yy,xx=np.where(mask)
        visible=len(xx)>=3
        dep=np.clip(np.nan_to_num(self.depth,nan=20,posinf=20,neginf=0),0,20)
        grid=dep.reshape(4,60,4,80).mean(axis=(1,3)).ravel().tolist()
        # Only RGB/depth image measurements. No target position or simulator pose.
        return {'pixels':int(len(xx)), 'visible':visible,
                'centroid':[float(xx.mean()/320),float(yy.mean()/240)] if len(xx) else None,
                'median_target_depth':float(np.median(self.depth[mask])) if len(xx) else None,
                'depth_grid':grid, 'rgb_mean':(self.rgb.mean(axis=(0,1))/255).tolist()}
    def segmentation_check(self):
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.render_data,camera=CAMERA_NAME)
        seg=self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        target=(seg[...,0]==self.env.model.geom('task3_target_geom').id)&(seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        rgbmask=mask_rgb(self.rgb,self.red_min)
        return {'target_segmentation_pixels':int(target.sum()),'red_false_positive_pixels':int((rgbmask&~target).sum())}
    def close(self): self.renderer.close()

def bounds(env,data):
    x,y=data.qpos[:2]
    return bool(env.terrain.x[0]+.3<=x<=env.terrain.x[-1]-.3 and env.terrain.y[0]+.3<=y<=env.terrain.y[-1]-.3)

def sustained(curve,threshold=3):
    run=0
    for s in curve:
        if s['upright']<.25 or not s['sensor']['in_bounds']:
            return False
        run=run+1 if s['sensor']['pixels']>=threshold else 0
        if run>=3: return True
    return False

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='*');p.add_argument('--output',type=Path,default=OUT/'camera_rollouts.json');p.add_argument('--red-min',type=int,default=120);p.add_argument('--loss-max-pixels',type=int,default=2);a=p.parse_args()
    OUT=a.output.parent.resolve()
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'frames').mkdir(exist_ok=True)
    names=a.scenes or list(api.scene_catalogue())
    config={'actions':ACTIONS,'size':[320,240],'sample_hz':10,'action_duration_s':3,
            'red_min':a.red_min,'loss_max_pixels':a.loss_max_pixels,'thresholds':THRESHOLDS,'primary_threshold':3,'consecutive_frames':3,
            'approach_seconds':12,'approach_speed_cap':.6,'scenes':names,
            'eligibility':'physically executed approach; >=3 consecutive detections then >=3 frames at or below loss_max_pixels',
            'controller_state':'zero internal exported-policy history before each primitive; identical reset across branches','memory':'last RGB centroid and observed depth only','date':'2026-10-04',
            'note':'Local protocol fixed before this run; not external preregistration'}
    (a.output.with_suffix('.protocol.json')).write_text(json.dumps(config,indent=2),encoding='utf8')
    records=[];beg=time.time()
    for name in names:
        scene=api.scene_catalogue()[name]
        env=api.MuJoCoEnvironment(scene,action_table=ep.protocol_action_table())
        cam=Camera(env,red_min=a.red_min)
        try:
            env.reset(replay_approach=False)
            nominal=cam.read(env.model,env.data)
            nominal_check=cam.segmentation_check()
            nominal_oracle=env.privileged().target_visible
            from PIL import Image
            Image.fromarray(cam.rgb).save(OUT/'frames'/f'{name}_nominal.png')
            start=scene.history_xy if scene.history_xy is not None else scene.approach_xy
            env.reset(xy=start,replay_approach=False)
            dx,dy=np.array(scene.lost_xy)-np.array(start)
            c,s=math.cos(scene.yaw_rad),math.sin(scene.yaw_rad)
            vx,vy=c*dx+s*dy,-s*dx+c*dy
            scale=.6/max(math.hypot(vx,vy),1e-6);vx*=scale;vy*=scale
            approach=Primitive('physical_approach',vx=float(vx),vy=float(vy),duration_s=12,yaw_hold=True)
            track={'seen_run':0,'lost_run':0,'seen':False,'lost':None,'memory':None,'lost_image':None}
            def observe(model,data):
                sensor=cam.read(model,data);sensor['in_bounds']=bounds(env,data)
                if sensor['visible']:
                    track['seen_run']+=1;track['lost_run']=0
                    track['memory']={'centroid':sensor['centroid'],'depth':sensor['median_target_depth'],'time':float(data.time)}
                    if track['seen_run']>=3:
                        if not track['seen']: Image.fromarray(cam.rgb).save(OUT/'frames'/f'{name}_seen.png')
                        track['seen']=True
                else:
                    track['seen_run']=0
                    if track['seen']:
                        track['lost_run']=track['lost_run']+1 if sensor['pixels']<=a.loss_max_pixels else 0
                    if track['lost_run']>=3 and track['lost'] is None and sensor['in_bounds'] and float(data.xmat[1].reshape(3,3)[2,2])>=.25:
                        track['lost']=ep.snapshot(env)
                        track['lost_image']=cam.rgb.copy()
                        track['loss_sensor']=sensor.copy();track['loss_memory']=dict(track['memory'])
                        track['loss_time']=float(data.time)
                return sensor
            initial=observe(env.model,env.data)
            approach_result=rollout_with_visibility(env.model,env.data,env.terrain,env.controller,approach,env.target,sample_observer=observe)
            record={'scene':name,'group':name.rsplit('_',1)[0] if name.startswith('hist') else name,
                    'nominal_start':nominal,'nominal_oracle':bool(nominal_oracle),'nominal_detector_check':nominal_check,
                    'approach_start':list(start),'approach_velocity':[vx,vy],
                    'approach_initial':initial,'approach_curve':approach_result['curve'],
                    'eligible':track['lost'] is not None,'branches':[]}
            if track['lost'] is not None:
                Image.fromarray(track['lost_image']).save(OUT/'frames'/f'{name}_lost.png')
                record['loss_sensor']=track['loss_sensor'];record['loss_memory']=track['loss_memory'];record['loss_time']=track['loss_time']
                for action in ACTIONS:
                    ep.restore(env,track['lost'])
                    first_visible=[False]
                    def branch_observe(model,data):
                        sensor=cam.read(model,data);sensor['in_bounds']=bounds(env,data)
                        if sensor['visible'] and not first_visible[0]:
                            Image.fromarray(cam.rgb).save(OUT/'frames'/f'{name}_{action}_visible.png');first_visible[0]=True
                        return sensor
                    r=rollout_with_visibility(env.model,env.data,env.terrain,env.controller,env.action_table[action],env.target,sample_observer=branch_observe)
                    record['branches'].append({'action':action,'success':sustained(r['curve']),
                        'threshold_success':{str(t):sustained(r['curve'],t) for t in THRESHOLDS},
                        'oracle_any':r['rsr'],'fallen':r['fallen'],'cost_m':r['motion_cost_m'],
                        'curve':r['curve']})
            records.append(record)
            result={'protocol':config,'records':records,'elapsed_s':time.time()-beg,
                    'versions':{'python':platform.python_version(),'mujoco':mujoco.__version__,'torch':torch.__version__,'numpy':np.__version__},
                    'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(api.__file__),Path(ep.__file__),Path(__file__).with_name('benchmark.py'),Path(__file__).with_name('controllers.py')]}}
            a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
            print(name,'nominal pixels',nominal['pixels'],'eligible',record['eligible'],'successful',[b['action'] for b in record['branches'] if b['success']],f'elapsed {time.time()-beg:.1f}s',flush=True)
        finally: cam.close();env.close()
    print('wrote',a.output,flush=True)
if __name__=='__main__':main()
