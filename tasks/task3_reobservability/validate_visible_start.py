"""Physical eligibility validation, never counted as VLM trials."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import mujoco
import navigation_trial as nt
from visible_start_scenes import catalogue
from vlm_benchmark import dump
from controllers import Primitive
from benchmark import _yaw


def in_fov(env, measurement):
    cid = env.model.camera(nt.CAMERA_NAME).id
    relative = env.data.geom_xpos[env.model.geom('task3_target_geom').id] - env.data.cam_xpos[cid]
    local = env.data.cam_xmat[cid].reshape(3,3).T @ relative
    vertical = math.tan(math.radians(float(env.model.cam_fovy[cid]))/2)
    return bool(local[2]<0 and abs(local[1]/local[2])<vertical and abs(local[0]/local[2])<vertical*320/240)


def blocked_by_wall(env):
    cid=env.model.camera(nt.CAMERA_NAME).id
    eye=env.data.cam_xpos[cid].copy()
    direction=env.data.geom_xpos[env.model.geom('task3_target_geom').id]-eye
    direction/=np.linalg.norm(direction)
    hit=np.array([-1],dtype=np.int32)
    mujoco.mj_ray(env.model,env.data,eye,direction,None,1,-1,hit)
    return bool(hit[0]==env.model.geom('task3_occluder_wall').id)


def validate(root, names=None):
    torch.set_num_threads(1)
    root.mkdir(parents=True,exist_ok=True)
    nt.ACTIONS['FORWARD']=Primitive('forward',vx=.6,duration_s=1.)
    results=[]
    for name,(scene,terrain) in catalogue().items():
        if names and name not in names: continue
        folder=root/name
        folder.mkdir(exist_ok=True)
        env=nt.api.MuJoCoEnvironment(scene,terrain=terrain)
        cam=nt.EvaluatorCamera(env,render_size=(480,360))
        env.reset(replay_approach=False)
        initial=cam.read(env.model,env.data)
        cam.save_evidence(folder/'initial')
        trace=[]
        with (folder/'state.jsonl').open('w') as sf,(folder/'evaluator.jsonl').open('w') as ef:
            phys=nt.Physics(env,cam,sf,ef,folder)
            phys.record(0,'RESET')
            loss=None
            consecutive=0
            for step in range(1,16):
                terminal=phys.execute('FORWARD',step)
                m=cam.read(env.model,env.data)
                fov=in_fov(env,m);blocked=blocked_by_wall(env)
                consecutive=consecutive+1 if m['target_segment_pixels']<=2 and fov and blocked else 0
                trace.append({'step':step,'measurement':m,'in_fov':fov,'blocked_by_wall':blocked,
                              'qpos':env.data.qpos.tolist()})
                cam.save_evidence(folder/f'approach_{step:02d}')
                if consecutive>=2:
                    loss={'step':step,'measurement':m,'qpos':env.data.qpos.tolist()}
                    break
                if terminal: break
            # Privileged waypoint steering is a feasibility reference only.
            reference_terminal=None
            reference_steps=0
            if loss:
                side=-1 if scene.lost_xy[1]<0 else 1
                bypass=side*(scene.terrain_params['wall_half_width_m']+1.0)
                waypoints=[(-.8,bypass),(.95,bypass),(scene.target_xy[0]-.65,scene.target_xy[1])]
                for point in waypoints:
                    for _ in range(220):
                        delta=np.array(point)-env.data.qpos[:2]
                        dist=np.linalg.norm(delta)
                        if dist<.16:break
                        bearing=math.atan2(delta[1],delta[0])-_yaw(env.data)
                        bearing=math.atan2(math.sin(bearing),math.cos(bearing))
                        nt.ACTIONS['REFERENCE']=Primitive('reference',vx=.5*max(0,math.cos(bearing)),
                            wz=float(np.clip(2*bearing,-1.2,1.2)),duration_s=.3)
                        reference_steps+=1
                        reference_terminal=phys.execute('REFERENCE',16+reference_steps)
                        if reference_terminal:break
                    if reference_terminal:break
                cam.read(env.model,env.data)
                cam.save_evidence(folder/'reference_final')
            eligible=initial['target_visible_rgb_pixels']>=40 and not initial['near_and_visible'] and loss is not None and reference_terminal=='success_visible_near'
            result={'scene':name,'eligible':eligible,'initial':initial,'loss':loss,
                    'reference_terminal':reference_terminal,'reference_steps':reference_steps,
                    'trace':trace,'reference_uses_privileged_waypoints':True,
                    'reference_not_a_vlm_result':True}
            dump(folder/'validation.json',result)
            results.append(result)
            dump(root/'validation.json',results)
            print(name,'initial',initial['target_visible_rgb_pixels'],'loss',None if loss is None else loss['step'],
                  'reference',reference_terminal,'eligible',eligible,flush=True)
        cam.close();env.close()
    dump(root/'VALIDATION_STATUS.json',{'count':len(results),'eligible':sum(r['eligible'] for r in results),
        'all_pass':len(results)==24 and all(r['eligible'] for r in results)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--scenes',nargs='*');a=p.parse_args()
    validate(a.out,a.scenes)
