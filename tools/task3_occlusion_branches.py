"""V4 fixed-state occlusion branches. Evaluator labels never enter public inputs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import mujoco
import numpy as np
from PIL import Image
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tasks/task3_reobservability'))
import navigation_trial as nt
from all_terrain_setup import place
from scenes import Scene
from visible_start_trial import actions,sensor_packet
from v4_qwen3vl_trial import build_setups,DEFAULT_ADMISSION,ray_check

ROOT=REPO/'paper/vlm_goal_navigation/occlusion_branch_pilot_20261019'
BRANCHES={
 'advance':['FORWARD']*6,
 'left_bypass':['STRAFE_LEFT']*2+['FORWARD']*4,
 'right_bypass':['STRAFE_RIGHT']*2+['FORWARD']*4,
 'retreat_scan':['BACKWARD']*2+['TURN_LEFT']*2+['TURN_RIGHT']*2,
}


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def clean_measurement(value):
    return json.loads(json.dumps(value,default=lambda x:x.tolist() if hasattr(x,'tolist') else float(x)))


class Simulator:
    def __init__(self,setup,terrain):
        self.setup=setup
        self.env=nt.api.MuJoCoEnvironment(Scene(**setup['adjusted_scene']),terrain=terrain)
        self.cam=nt.EvaluatorCamera(self.env,render_size=(640,480))
        place(self.env)
        self.ticks=0;self.angles=self.env.controller.default_angles.copy()
        self.last_seen=None
        self.origin=None

    def own_pose(self):
        d=self.env.data
        return dict(xyz=d.qpos[:3].copy(),rotation=d.xmat[1].reshape(3,3).copy())

    def odometry(self,value):
        r0=self.origin['rotation'];yaw0=math.atan2(r0[1,0],r0[0,0])
        r=value['rotation'];yaw=math.atan2(r[1,0],r[0,0])
        c,s=math.cos(yaw0),math.sin(yaw0);delta=value['xyz']-self.origin['xyz']
        return dict(forward_m=float(c*delta[0]+s*delta[1]),left_m=float(-s*delta[0]+c*delta[1]),
                    up_m=float(delta[2]),yaw_rad=math.atan2(math.sin(yaw-yaw0),math.cos(yaw-yaw0)))

    def observation(self):
        m=clean_measurement(self.cam.read(self.env.model,self.env.data))
        cid=self.env.model.camera(nt.CAMERA_NAME).id
        return dict(rgb=self.cam.rgb.copy(),depth=self.cam.depth.copy(),pose=self.own_pose(),
            camera_position=self.env.data.cam_xpos[cid].copy(),
            camera_rotation=self.env.data.cam_xmat[cid].reshape(3,3).copy(),
            time_s=float(self.env.data.time),evaluation=m)

    def advance(self,action,seconds=1.,observer=None):
        d=self.env.data;m=self.env.model;c=self.env.controller;p=nt.ACTIONS[action]
        rows=[];prev=d.qpos[:2].copy();prev_yaw=self.yaw()
        path=0.;turn=0.;terminal=None
        for _ in range(round(seconds/c.timestep)):
            if self.ticks%c.control_decimation==0:self.angles=c.act(d,[p.vx,p.vy,p.wz])
            d.ctrl[:]=np.clip((self.angles-d.qpos[7:25])*c.kps-d.qvel[6:24]*c.kds,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
            mujoco.mj_step(m,d);self.ticks+=1
            path+=float(np.linalg.norm(d.qpos[:2]-prev));prev=d.qpos[:2].copy()
            yaw=self.yaw();turn+=abs(math.atan2(math.sin(yaw-prev_yaw),math.cos(yaw-prev_yaw)));prev_yaw=yaw
            upright=float(d.xmat[1].reshape(3,3)[2,2])
            if not np.isfinite(d.qpos).all() or upright<.25:terminal='fallen'
            elif not nt.bounds(self.env,d):terminal='out_of_bounds'
            if self.ticks%round(.1/c.timestep)==0 or terminal:
                obs=self.observation()
                row=dict(action=action,time_s=obs['time_s'],own_qpos=d.qpos[:7].tolist(),
                         path_m=path,total_turn_rad=turn,measurement=obs['evaluation'])
                # Contact counts are diagnostic, not automatically collisions.
                row['contact_count']=int(d.ncon)
                rows.append(row)
                if observer is not None and observer(obs):break
            if terminal:break
        return dict(samples=rows,path_m=path,total_turn_rad=turn,termination=terminal)

    def yaw(self):
        r=self.env.data.xmat[1].reshape(3,3)
        return math.atan2(r[1,0],r[0,0])

    def snapshot(self):
        state=mujoco.MjData(self.env.model);mujoco.mj_copyData(state,self.env.model,self.env.data)
        return dict(data=state,history=self.env.controller.policy.history.clone(),
            controller_action=self.env.controller.action.copy(),ticks=self.ticks,angles=self.angles.copy(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state().clone())

    def restore(self,snap):
        mujoco.mj_copyData(self.env.data,self.env.model,snap['data'])
        self.env.controller.policy.history=snap['history'].clone()
        self.env.controller.action=snap['controller_action'].copy()
        self.ticks=snap['ticks'];self.angles=snap['angles'].copy()
        random.setstate(snap['python_rng']);np.random.set_state(snap['numpy_rng']);torch.set_rng_state(snap['torch_rng'])

    def state_hash(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        v=np.empty(mujoco.mj_stateSize(self.env.model,spec));mujoco.mj_getState(self.env.model,self.env.data,v,spec)
        obs=self.observation()
        return dict(integration=digest(v),controller_history=digest(self.env.controller.policy.history.numpy()),
            controller_action=digest(self.env.controller.action),angles=digest(self.angles),ticks=self.ticks,
            rgb=digest(obs['rgb']),depth=digest(obs['depth']))

    def prefix(self):
        for _ in range(6):
            r=self.advance('STOP',.3)
            if r['termination']:raise RuntimeError('initial stand '+r['termination'])
        self.initial=self.observation();self.origin=self.own_pose();self.last_seen=self.initial
        if self.initial['evaluation']['target_visible_rgb_pixels']<40:raise RuntimeError('initial target not visible')
        self.executed_prefix=[];self.loss_streak=0
        def observe(obs):
            if nt.mask_rgb(obs['rgb'],50).sum()>=20:self.last_seen=obs
            e=obs['evaluation'];ray=ray_check(self.cam)
            valid=e['target_segment_pixels']<=2 and e['upright']>=.5 and e['in_bounds'] and ray['entity_occlusion']
            moved=np.linalg.norm(self.env.data.qpos[:2]-self.origin['xyz'][:2])
            self.loss_streak=self.loss_streak+1 if valid and moved>=.25 else 0
            return self.loss_streak>=3
        for _ in range(self.setup['approach_actions']):
            before=float(self.env.data.time)
            r=self.advance(self.setup['approach_action'],observer=observe)
            self.executed_prefix.append(dict(action=self.setup['approach_action'],duration_s=float(self.env.data.time)-before))
            if r['termination']:raise RuntimeError('prefix '+r['termination'])
            if self.loss_streak>=3:break
        if self.loss_streak<3:raise RuntimeError('strict occlusion not reached')

    def save_observation(self,obs,dest):
        dest.mkdir(parents=True,exist_ok=True)
        Image.fromarray(obs['rgb']).save(dest/'rgb.png')
        finite=np.isfinite(obs['depth'])&(obs['depth']>0)
        grey=np.where(finite,255*(1-np.clip(obs['depth'],0,10)/10),0).astype('uint8')
        Image.fromarray(grey).convert('RGB').save(dest/'depth.png')
        np.save(dest/'depth_m.npy',obs['depth'])
        # Independent camera packet: sensor_packet reads ONLY these RGB/depth arrays.
        self.cam.rgb=obs['rgb'];self.cam.depth=obs['depth']
        reading=sensor_packet(self.cam,dest)
        # Never call save_evidence here: it writes target identity masks.
        return dict(time_s=obs['time_s']-self.initial['time_s'],
            rgb=str((dest/'rgb.png').relative_to(REPO)),depth=str((dest/'depth.png').relative_to(REPO)),
            rgb_sha256=hashlib.sha256((dest/'rgb.png').read_bytes()).hexdigest(),
            depth_sha256=hashlib.sha256((dest/'depth.png').read_bytes()).hexdigest(),
            self_odometry=self.odometry(obs['pose']),camera_measurement=reading)

    def target_belief(self,current):
        # Deproject an RGB-derived surface point and transform using OWN camera/base pose only.
        obs=self.last_seen;rgb=obs['rgb'];depth=obs['depth'];mask=nt.mask_rgb(rgb,50)
        yy,xx=np.where(mask);valid=np.isfinite(depth[mask])&(depth[mask]>0)&(depth[mask]<20)
        if valid.sum()<20:return dict(available=False)
        z=float(np.median(depth[mask][valid]));u=float(xx[valid].mean());v=float(yy[valid].mean())
        h,w=depth.shape;cid=self.env.model.camera(nt.CAMERA_NAME).id
        focal=h/(2*math.tan(math.radians(float(self.env.model.cam_fovy[cid]))/2))
        point=np.array([(u+.5-w/2)*z/focal,-(v+.5-h/2)*z/focal,-z])
        estimated_world=obs['camera_position']+obs['camera_rotation']@point
        relative=current['pose']['rotation'].T@(estimated_world-current['pose']['xyz'])
        return dict(available=True,source='last RGB red region + rendered camera depth + own extrinsics and odometry',
            forward_m=float(relative[0]),left_m=float(relative[1]),up_m=float(relative[2]),
            bearing_deg=math.degrees(math.atan2(relative[1],relative[0])),horizontal_range_m=float(np.linalg.norm(relative[:2])),
            estimate_kind='observed surface point, not hidden object centre',uncertainty_calibrated=False,
            uncertainty='Unknown target radius, partial-view bias and later motion error; no hidden-geometry inference.')

    def branch(self,name,dest=None):
        t0=float(self.env.data.time);path=0.;turn=0.;streak=0;first=None;arrival_streak=0;arrival=None;rows=[];terminal=None
        for action in BRANCHES[name]:
            result=self.advance(action)
            for row in result['samples']:
                e=row['measurement'];visible=e['target_visible_rgb_pixels']>=20 and e['upright']>=.5 and e['in_bounds']
                streak=streak+1 if visible else 0
                if streak>=3 and first is None:first=dict(time_s=row['time_s']-t0,path_m=path+row['path_m'])
                near=visible and e['median_target_surface_range_m'] is not None and 0<e['median_target_surface_range_m']<1
                arrival_streak=arrival_streak+1 if near else 0
                if arrival_streak>=3 and arrival is None:arrival=row['time_s']-t0
                row=dict(row,time_from_start_s=row['time_s']-t0,path_m=path+row['path_m'])
                rows.append(row)
            path+=result['path_m'];turn+=result['total_turn_rad']
            if dest is not None and first and not (dest/'first_recovery_rgb.png').exists():
                # Action-boundary evidence; exact recovery time comes from 10Hz rows.
                dest.mkdir(parents=True,exist_ok=True);Image.fromarray(self.cam.rgb).save(dest/'first_recovery_rgb.png')
            if result['termination']:terminal=result['termination'];break
        out=dict(branch=name,recovered=first is not None,first_recovery=first,arrival_time_s=arrival,
            executed_s=float(self.env.data.time)-t0,planned_horizon_s=6.,termination=terminal or 'horizon_exhausted',
            path_m=path,total_turn_rad=turn,samples=rows,endpoint_hash=self.state_hash())
        if dest is not None:
            dest.mkdir(parents=True,exist_ok=True);dump(dest/'label.json',out)
            Image.fromarray(self.cam.rgb).save(dest/'endpoint_rgb.png')
        return out

    def close(self):self.cam.close();self.env.close()


def collect(limit=None):
    torch.set_num_threads(1);torch.manual_seed(7);np.random.seed(7);random.seed(7);actions()
    ROOT.mkdir(parents=True,exist_ok=True)
    setups,terrains=build_setups(DEFAULT_ADMISSION,None)
    setups=sorted(setups,key=lambda x:x['scene_name'])
    delayed={x['scene_name'] for x in setups[:8]}
    protocol=dict(version='occlusion-branch-pilot/1',seed=7,branches=BRANCHES,horizon_s=6.,
        states='all 22 accepted V4 layouts at first strict loss; first 8 alphabetical layouts additionally after .4s continued approach',
        delayed_scenes=sorted(delayed),expected_states=30,expected_labelled_branches=120,
        duplicate_audit='repeat advance after all four candidates at every eligible state; additional rollouts, not extra data',
        replay='full mj_copyData + TorchScript history + wrapper action + PD targets + ticks + RNG',
        recovery='exact target RGB identity >=20px, upright>=.5 and in bounds, 3 consecutive 10Hz samples',
        partial_horizons='fall/out-of-bounds terminate physically; count no-recovery unless already recovered; never discard',
        independent_unit='physical layout, not individual state or branch',stage='collection only; no model tuning')
    if (ROOT/'protocol.json').exists():
        assert json.loads((ROOT/'protocol.json').read_text())==protocol,'protocol drift'
    else:dump(ROOT/'protocol.json',protocol)
    index=[];invalid=[]
    for si,setup in enumerate(setups):
        variants=2 if setup['scene_name'] in delayed else 1
        expected=[f'S{si:02d}_{j}' for j in range(variants)]
        # Completed layouts can be resumed only when all requested variants have records.
        if all((ROOT/'states'/key/'collection_done.json').exists() for key in expected):
            for key in expected:
                value=json.loads((ROOT/'states'/key/'collection_done.json').read_text())
                (index if value['eligible'] else invalid).append(value)
            continue
        sim=Simulator(setup,terrains[setup['scene_name']])
        try:
            sim.prefix()
            for variant,key in enumerate(expected):
                if limit is not None and len(index)>=limit:break
                folder=ROOT/'states'/key;folder.mkdir(parents=True,exist_ok=True)
                delayed_checks=[]
                if variant:
                    def delayed_observer(obs):
                        if nt.mask_rgb(obs['rgb'],50).sum()>=20:sim.last_seen=obs
                        e=obs['evaluation'];r=ray_check(sim.cam)
                        delayed_checks.append(e['target_segment_pixels']<=2 and r['entity_occlusion'] and e['upright']>=.5 and e['in_bounds'])
                        return False
                    continuation=sim.advance(setup['approach_action'],.4,delayed_observer)
                    if continuation['termination']:raise RuntimeError('delayed continuation '+continuation['termination'])
                current=sim.observation();check=ray_check(sim.cam);e=current['evaluation']
                if (folder/'collection_done.json').exists():
                    saved=json.loads((folder/'collection_done.json').read_text())
                    if saved['eligible']:
                        previous=json.loads((folder/'checkpoint_metadata.json').read_text())
                        assert sim.state_hash()==previous['own_state_hash'],'prefix replay differs from stored checkpoint'
                        assert hashlib.sha256((folder/'request.json').read_bytes()).hexdigest()==saved['request_sha256']
                        index.append(saved)
                    else:invalid.append(saved)
                    continue
                eligible=e['target_segment_pixels']<=2 and check['entity_occlusion'] and e['upright']>=.5 and e['in_bounds']
                if variant:eligible=eligible and len(delayed_checks)>=3 and all(delayed_checks[-3:])
                meta=dict(id=key,layout=setup['scene_name'],delayed_variant=bool(variant),eligible=bool(eligible))
                if not eligible:
                    dump(folder/'collection_done.json',meta);invalid.append(meta);continue
                checkpoint=sim.snapshot();initial_hash=sim.state_hash()
                np.savez_compressed(folder/'checkpoint.npz',
                    qpos=sim.env.data.qpos.copy(),qvel=sim.env.data.qvel.copy(),
                    controller_history=checkpoint['history'].numpy(),controller_action=checkpoint['controller_action'],
                    angles=checkpoint['angles'],torch_rng=checkpoint['torch_rng'].numpy(),numpy_rng_keys=checkpoint['numpy_rng'][1])
                # Archive the complete integration vector separately from convenience qpos/qvel.
                spec=mujoco.mjtState.mjSTATE_INTEGRATION
                vector=np.empty(mujoco.mj_stateSize(sim.env.model,spec));mujoco.mj_getState(sim.env.model,sim.env.data,vector,spec)
                np.save(folder/'mj_integration_state.npy',vector)
                dump(folder/'checkpoint_metadata.json',dict(state_spec=int(spec),ticks=checkpoint['ticks'],
                    t0_sim_s=float(sim.env.data.time),prefix=sim.executed_prefix,
                    delayed_continuation_s=.4 if variant else 0,
                    python_rng=checkpoint['python_rng'],numpy_rng_algorithm=checkpoint['numpy_rng'][0],
                    numpy_rng_position=checkpoint['numpy_rng'][2],numpy_rng_has_gauss=checkpoint['numpy_rng'][3],
                    numpy_rng_cached_gaussian=checkpoint['numpy_rng'][4],
                    own_state_hash=initial_hash,persistence='integration vector + controller tensors + deterministic prefix replay; in-memory mj_copyData used for all branches'))
                observations=[sim.save_observation(o,folder/'inputs'/role) for role,o in
                              [('initial',sim.initial),('last_visible',sim.last_seen),('current',current)]]
                names=list(BRANCHES);random.Random(700+si*2+variant).shuffle(names)
                request=dict(target='red sphere',observations=observations,coarse_target_belief=sim.target_belief(current),
                    candidate_actions=[dict(id=chr(65+i),actions=BRANCHES[name]) for i,name in enumerate(names)],
                    horizon_s=6.,coordinate_convention='body +x forward, +y left; positive yaw left',
                    allowed_sensors='RGB-D, own ideal odometry, own kinematic calibration; no prebuilt map')
                # Seal public request BEFORE generating the candidate outcome labels.
                dump(folder/'request.json',request)
                request_sha=hashlib.sha256((folder/'request.json').read_bytes()).hexdigest()
                dump(folder/'candidate_mapping_evaluator_only.json',dict(zip('ABCD',names)))
                dump(folder/'gate_evaluator_only.json',dict(current=e,ray=check,layout=setup['scene_name']))
                restore_checks=[];results={}
                for name in BRANCHES:
                    sim.restore(checkpoint);restored=sim.state_hash()
                    restore_checks.append(dict(branch=name,equal=restored==initial_hash,actual=restored))
                    if restored!=initial_hash:raise RuntimeError('state restoration mismatch')
                    results[name]=sim.branch(name,folder/'branches'/name)
                sim.restore(checkpoint)
                duplicate=sim.branch('advance',folder/'replay_audit')
                exact=results['advance']==duplicate
                dump(folder/'REPLAY_AUDIT.json',dict(start_hash=initial_hash,restore_checks=restore_checks,
                    repeat_after_other_candidates_equal=exact,request_sha256=request_sha))
                if not exact:raise RuntimeError('duplicate branch diverged')
                assert hashlib.sha256((folder/'request.json').read_bytes()).hexdigest()==request_sha
                sim.restore(checkpoint)
                meta.update(request_sha256=request_sha,branches=4,replay_passed=True)
                dump(folder/'collection_done.json',meta);index.append(meta)
                print(json.dumps(dict(state=key,layout=setup['scene_name'],recovery={k:v['recovered'] for k,v in results.items()},replay=True)),flush=True)
                dump(ROOT/'COLLECTION_STATUS.json',dict(phase='running',completed=len(index),ineligible=invalid,states=index))
            if limit is not None and len(index)>=limit:break
        finally:sim.close()
    dump(ROOT/'state_index.json',index)
    dump(ROOT/'COLLECTION_STATUS.json',dict(phase='smoke_complete' if limit else 'complete',completed=len(index),
        expected=30,ineligible=invalid,formal_branches=4*len(index),additional_replay_branches=len(index),states=index))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();collect(a.limit)
