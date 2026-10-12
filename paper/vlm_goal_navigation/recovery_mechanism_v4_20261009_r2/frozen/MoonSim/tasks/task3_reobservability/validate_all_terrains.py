"""Physical first-visible / moving-loss validation on every existing terrain."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import mujoco
import torch
import navigation_trial as nt
from visible_start_trial import actions, sensor_packet
from all_terrain_setup import originals,candidates,adjusted,place
from benchmark import _yaw
from vlm_benchmark import dump


def loss_kind(cam):
    env=cam.env;data=cam.render_data;cid=env.model.camera(nt.CAMERA_NAME).id
    eye=data.cam_xpos[cid];target=data.geom_xpos[env.model.geom('task3_target_geom').id]
    local=data.cam_xmat[cid].reshape(3,3).T@(target-eye)
    h,w=cam.depth.shape;vertical=math.tan(math.radians(float(env.model.cam_fovy[cid]))/2)
    fov=bool(local[2]<0 and abs(local[1]/local[2])<vertical and abs(local[0]/local[2])<vertical*w/h)
    hit=np.array([-1],dtype=np.int32);direction=(target-eye)/np.linalg.norm(target-eye)
    mujoco.mj_ray(env.model,data,eye,direction,None,1,-1,hit)
    geom=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(hit[0])) if hit[0]>=0 else None
    terrain_names={s.name for s in env.terrain.slabs} if hasattr(env.terrain,'slabs') else set()
    terrain_names.update(['navigation_added_occluder','task3_terrain','task3_slab','task3_apron','task3_landing'])
    terrain_hit=geom in terrain_names or bool(geom and geom.startswith('task3_step_'))
    if not fov:return 'fov_exit',geom
    if terrain_hit:return 'terrain_occlusion',geom
    return 'other_occlusion',geom


class Approach(nt.Physics):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.enabled=False;self.loss_run=0;self.loss=None
        self.start_xy=self.env.data.qpos[:2].copy();self.extra_terminal=None
    def record(self,step,action):
        state=super().record(step,action);m=state['evaluation']
        if self.enabled:
            valid=m['target_segment_pixels']<=2 and m['upright']>=.25 and m['in_bounds']
            moved=float(np.linalg.norm(self.env.data.qpos[:2]-self.start_xy))
            self.loss_run=self.loss_run+1 if valid and moved>=.25 else 0
            if self.loss_run>=3:
                kind,geom=loss_kind(self.camera)
                self.loss={'step':step,'measurement':m,'kind':kind,'blocking_geom':geom,'displacement_m':moved}
                self.extra_terminal='approach_loss'
        return state


def reference(env,cam,physics,target,max_actions=100):
    physics.enabled=False;physics.extra_terminal=None
    chosen=[];terminal=None
    obstacle=env.terrain.spec.get('navigation_added_occluder')
    points=[]
    if obstacle:
        side=-1 if env.data.qpos[1]<0 else 1
        bypass=side*(obstacle['half_y']+1.)
        after=obstacle['x']+obstacle['width']+.4 if obstacle.get('kind')=='cube' else obstacle['x']+.4
        points=[(obstacle['x']-.9,bypass),(after,bypass)]
    points.append((target[0]-.95,target[1]))
    for point in points:
        for _ in range(max_actions):
            delta=np.array(point)-env.data.qpos[:2]
            if np.linalg.norm(delta)<.35:break
            yaw=_yaw(env.data)
            if abs(yaw)>.4:action='TURN_RIGHT' if yaw>0 else 'TURN_LEFT'
            else:
                c,s=math.cos(yaw),math.sin(yaw)
                dx,dy=c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]
                if abs(dx)>abs(dy):action='FORWARD' if dx>0 else 'BACKWARD'
                else:action='STRAFE_LEFT' if dy>0 else 'STRAFE_RIGHT'
            chosen.append(action);terminal=physics.execute(action,100+len(chosen))
            if terminal:break
        if terminal:break
    # Aim camera at target when a positional waypoint is reached without visibility.
    if not terminal:
        for _ in range(12):
            delta=np.array(target)-env.data.qpos[:2];bearing=math.atan2(delta[1],delta[0])-_yaw(env.data)
            bearing=math.atan2(math.sin(bearing),math.cos(bearing))
            action=('TURN_LEFT' if bearing>0 else 'TURN_RIGHT') if abs(bearing)>.4 else 'FORWARD'
            chosen.append(action);terminal=physics.execute(action,100+len(chosen))
            if terminal:break
    return terminal or 'reference_limit',chosen


def validate(root,names=None,config_path=None):
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);actions()
    if config_path:
        import difficulty_terrain_setup as setup_module
        config=json.loads(config_path.read_text());catalogue=setup_module.catalogue(config)
        candidate_provider=lambda name,source,terrain:setup_module.candidates(name,source,terrain,config)
        adjust=lambda source,terrain,candidate:setup_module.adjusted(source,terrain,candidate,config)
        dump(root/'terrain_config.json',config)
    else:
        catalogue=originals();candidate_provider=candidates;adjust=adjusted
    accepted=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else [];failures=[]
    for name,(source,terrain) in catalogue.items():
        if names and name not in names:continue
        if any(item['scene_name']==name for item in accepted):
            print('RESUME_VALIDATED',name,flush=True);continue
        success=None
        for index,candidate in enumerate(candidate_provider(name,source,terrain)):
            scene,action=adjust(source,terrain,candidate)
            # The original collider representation is retained, even on legacy terrain.
            env=nt.api.MuJoCoEnvironment(scene,terrain=terrain);cam=nt.EvaluatorCamera(env,render_size=(640,480))
            place(env);folder=root/name/f'candidate_{index:02d}';folder.mkdir(parents=True,exist_ok=True)
            with (folder/'state.jsonl').open('w') as sf,(folder/'evaluator.jsonl').open('w') as ef:
                physics=Approach(env,cam,sf,ef,folder);physics.record(0,'RESET')
                stable=True
                for _ in range(6):
                    if physics.execute('STOP',0):stable=False;break
                initial=cam.read(env.model,env.data)
                cam.save_evidence(folder/'initial');depth=sensor_packet(cam,folder/'initial')
                valid_initial=stable and initial['target_visible_rgb_pixels']>=40 and not initial['near_and_visible']
                commands=[];terminal=None
                if valid_initial:
                    physics.start_xy=env.data.qpos[:2].copy();physics.enabled=True
                    for step in range(1,21):
                        before=cam.read(env.model,env.data);cam.save_evidence(folder/f'step_{step:02d}_before')
                        commands.append(action);terminal=physics.execute(action,step)
                        cam.read(env.model,env.data);cam.save_evidence(folder/f'step_{step:02d}_after')
                        if terminal:break
                if terminal=='approach_loss' and physics.loss['kind']=='terrain_occlusion':
                    loss=physics.loss;cam.save_evidence(folder/'lost')
                    loss_qpos=env.data.qpos.tolist()
                    ref_terminal,ref_actions=reference(env,cam,physics,scene.target_xy)
                    cam.read(env.model,env.data);cam.save_evidence(folder/'reference_final')
                    success={'scene_name':name,'source_scene':vars(source),'adjusted_scene':vars(scene),
                        'original_terrain_geometry_unchanged':terrain.spec.get('navigation_original_geometry_unchanged',True),'added_occluder':terrain.spec.get('navigation_added_occluder'),'terrain_spec':dict(terrain.spec),'initial':initial,'initial_depth':depth,
                        'approach_action':action,'approach_actions':len(commands),
                        'approach_end_sim_time_s':loss['measurement']['sim_time_s'],
                        'loss':loss,'loss_qpos':loss_qpos,'reference_termination':ref_terminal,
                        'reference_actions':ref_actions,'reference_privileged':True,
                        'accepted_candidate':index,'evidence_path':str(folder.resolve())}
                    # Solvability failures remain an explicit, separately reported stratum.
                    if ref_terminal=='success_visible_near':
                        dump(folder/'accepted.json',success);accepted.append(success);break
                    success=None
                failures.append({'scene':name,'candidate':index,'initial_pixels':initial['target_visible_rgb_pixels'],
                    'stable':stable,'terminal':terminal,'commands':len(commands)})
            cam.close();env.close()
        if success:
            cam.close();env.close()
            print('ACCEPT',name,'initial',success['initial']['target_visible_rgb_pixels'],'loss',success['loss']['kind'],
                  'steps',success['approach_actions'],'reference',success['reference_termination'],flush=True)
        else:print('NO_VALID_SETUP',name,flush=True)
        dump(root/'manifest.json',accepted);dump(root/'search_failures.json',failures)
        dump(root/'STATUS.json',{'expected':len(catalogue) if not names else len(names),'accepted':len(accepted),
            'all_original_terrain_names':list(catalogue),'accepted_names':[x['scene_name'] for x in accepted]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--scenes',nargs='*');p.add_argument('--terrain-config',type=Path);a=p.parse_args();validate(a.out,a.scenes,a.terrain_config)
