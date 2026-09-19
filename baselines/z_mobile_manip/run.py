"""Standalone Z-Mobile baseline. Runtime inputs and outputs use bridge only."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from MoonSim.bridge.client import Client
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'third_party/z_mobile_manip'))
from .pipeline import Pipeline
from z_manip.control.visual_servo import VisualServoController, VisualServoConfig

OUTPUT=ROOT/'tasks/task1_collect/generated/bridge'
MAX_AGE_S=.9


def main():
    client=Client(timeout=1);pipeline=Pipeline();servo=VisualServoController(VisualServoConfig(max_forward_mps=.18,desired_depth_m=.42))
    seq=0;epoch=None;camera_seq=None;seen=None;last_log=0.;last_plan=-10.;reason='starting';frames=0
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print('Z_MOBILE_BASELINE_READY YOLOE + antipodal + robust IK + RRT; bridge only',flush=True)
    while True:
        started=time.monotonic();seq+=1
        try:
            robot=client.read('robot');camera=client.read('camera');lidar=client.read('lidar');selection=client.read('target')
            if not robot:time.sleep(.1);continue
            rm,r=robot;stamp=rm['sim_time_s'];action=r['hold_action'].copy();action[:3]=0.
            q=np.array([r['joint_position'][rm['joint_names'].index(f'joint{i}')] for i in range(1,7)])
            if epoch!=rm['epoch']:
                epoch=rm['epoch'];pipeline.reset();servo.reset();seen=None;camera_seq=None;last_plan=-10.
            fresh=bool(camera and lidar)
            if fresh:
                fresh=all(m['epoch']==epoch and (time.monotonic_ns()-m['sent_ns'])/1e9<MAX_AGE_S for m in [rm,camera[0],lidar[0]])
                fresh &= abs(camera[0]['sim_time_s']-lidar[0]['sim_time_s'])<.4
            reason='waiting_for_fresh_sensors'
            if rm['manual'] or rm['paused'] or rm['ended']:
                pipeline.reset();seen=None;reason='manual_or_paused'
            elif fresh:
                cm,c=camera;lm,l=lidar
                if pipeline.trajectory is not None:
                    reason=pipeline.execute(stamp,q,action)
                else:
                    if camera_seq!=cm['seq']:
                        seen=pipeline.observe(cm,c,selection[0] if selection else None);camera_seq=cm['seq'];frames+=1
                    if seen is None:
                        reason='search_no_detection'
                        search=np.array([.3*np.sin(stamp*.15),1.1,-1.4,0.,.8,0.])
                        action[3:9]+=np.clip(search-action[3:9],-.025,.025)
                    else:
                        points,scene,center=seen
                        distance=float(np.linalg.norm(center[:2]))
                        if distance>.44:
                            cmd=servo.update((-float(center[1]),0.,float(center[0])),stamp_s=cm['sim_time_s'])
                            action[0]=max(.12,cmd.linear_x) if cmd.linear_x>0 else cmd.linear_x
                            action[2]=cmd.angular_z;reason='approach_detected_rock'
                            box=pipeline.last_detection['bbox'];vertical_error=(box[1]+box[3])/(2*c['rgb'].shape[0])-.5
                            action[7]=np.clip(action[7]+np.clip(.12*vertical_error,-.02,.02),-1.2,1.2)
                            if action[7]>1.15 and vertical_error>.12:
                                action[4]=min(1.65,action[4]+.012)
                        else:reason='waiting_for_reachable_grasp'
                        # Ask IK inside the arm's working region; a failed grasp can re-approach.
                        if distance<.70 and time.monotonic()-last_plan>3:
                            last_plan=time.monotonic();action[:3]=0.
                            client.publish('action',{'source':'z_mobile_manip','seq':seq,'epoch':epoch,'sim_time_s':stamp,
                                'sent_ns':time.monotonic_ns(),'values':action.tolist(),'reason':'grasp_planning'})
                            seq+=1
                            try:pipeline.plan(points,scene,q);reason=pipeline.phase
                            except RuntimeError as e:reason='grasp_rejected: '+str(e)
                        T=np.array(lm['T_base_lidar']);pts=pipeline.environment_points(l['points']@T[:3,:3].T+T[:3,3],q)
                        blocked=(pts[:,0]>.36)&(pts[:,0]<.55)&(np.abs(pts[:,1])<.24)&(pts[:,2]>-.12)
                        if np.any(blocked) and action[0]>0:action[:3]=0.;reason='mid360_corridor_blocked'
                if abs(float(r['attitude_rpy'][0]))>.6 or abs(float(r['attitude_rpy'][1]))>.6:
                    pipeline.reset();action[:3]=0.;action[3:9]=q;reason='attitude_guard'
            else:pipeline.reset();seen=None
            meta={'source':'z_mobile_manip','seq':seq,'epoch':epoch,'sim_time_s':stamp,'sent_ns':time.monotonic_ns(),
                  'values':action.tolist(),'reason':reason,'camera_seq':camera[0]['seq'] if camera else None,
                  'lidar_seq':lidar[0]['seq'] if lidar else None}
            client.publish('action',meta)
            report={**meta,'camera_frames_processed':frames,**pipeline.stats,'detection':pipeline.last_detection,
                'input_topics':['camera','lidar','robot','target'],'grasp_backend':'upstream_antipodal',
                'tracker':'EdgeTAM-hf','detector':'yoloe-11s-seg.pt','upstream_commit':'db76d242cf9210fae6068d63ce6e6e50a7ab0069'}
            client.publish('baseline',report)
            if time.monotonic()-last_log>1:
                last_log=time.monotonic();(OUTPUT/'baseline.json').write_text(json.dumps(report,indent=2))
                with (OUTPUT/'baseline.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        except (OSError,ValueError,RuntimeError,KeyError) as e:
            pipeline.reset();seen=None
            if epoch:
                try:
                    action[:3]=0.;action[3:9]=q
                    client.publish('action',{'source':'z_mobile_manip','seq':seq+1,'epoch':epoch,'sim_time_s':stamp,
                        'sent_ns':time.monotonic_ns(),'values':action.tolist(),'reason':str(e)});seq+=1
                except OSError:pass
            if time.monotonic()-last_log>1:print('BASELINE_WAIT '+str(e),flush=True);last_log=time.monotonic()
        time.sleep(max(.001,.1-(time.monotonic()-started)))

if __name__=='__main__':main()
