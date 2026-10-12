"""Physical reachability reference using exactly the formal discrete action set.
Privileged waypoint selection, never part of VLM metrics or model inputs.
"""
import argparse
from pathlib import Path
import json
import math
import numpy as np
import torch
import navigation_trial as nt
from visible_start_trial import actions
from visible_start_scenes import catalogue
from benchmark import _yaw
from vlm_benchmark import dump


def run(root,names=None):
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);actions();results=[]
    for name,(scene,terrain) in catalogue().items():
        if names and name not in names:continue
        folder=root/name;folder.mkdir(exist_ok=True)
        env=nt.api.MuJoCoEnvironment(scene,terrain=terrain);cam=nt.EvaluatorCamera(env,render_size=(480,360))
        env.reset(replay_approach=False);cam.read(env.model,env.data);cam.save_evidence(folder/'initial')
        chosen=[];terminal=None
        with (folder/'state.jsonl').open('w') as sf,(folder/'evaluator.jsonl').open('w') as ef:
            physics=nt.Physics(env,cam,sf,ef,folder);physics.record(0,'RESET')
            for _ in range(7):
                chosen.append('FORWARD');terminal=physics.execute('FORWARD',len(chosen))
                if terminal:break
            cam.read(env.model,env.data);cam.save_evidence(folder/'loss')
            side=-1 if scene.lost_xy[1]<0 else 1
            bypass=side*(scene.terrain_params['wall_half_width_m']+1.0)
            points=[(-.8,bypass),(.95,bypass),(scene.target_xy[0]-.65,scene.target_xy[1])]
            for point in points:
                if terminal:break
                for _ in range(60):
                    delta=np.array(point)-env.data.qpos[:2]
                    if np.linalg.norm(delta)<.35:break
                    yaw=_yaw(env.data)
                    if abs(yaw)>.4:
                        action='TURN_RIGHT' if yaw>0 else 'TURN_LEFT'
                    else:
                        c,s=math.cos(yaw),math.sin(yaw)
                        dx,dy=c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]
                        if abs(dx)>abs(dy):action='FORWARD' if dx>0 else 'BACKWARD'
                        else:action='STRAFE_LEFT' if dy>0 else 'STRAFE_RIGHT'
                    chosen.append(action);terminal=physics.execute(action,len(chosen))
                    if terminal:break
            final=cam.read(env.model,env.data);cam.save_evidence(folder/'final')
            result={'scene':name,'success':terminal=='success_visible_near','termination':terminal or 'reference_action_limit',
                'total_actions':len(chosen),'autonomous_reference_actions':len(chosen)-7,'actions':chosen,
                'final':final,'same_action_space_and_duration_as_vlm':True,
                'privileged_selector':True,'not_a_vlm_result':True}
            dump(folder/'result.json',result);results.append(result);dump(root/'summary.json',results)
            print(name,result['termination'],len(chosen),final['median_target_surface_range_m'],flush=True)
        cam.close();env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--scenes',nargs='*');a=p.parse_args();run(a.out,a.scenes)
