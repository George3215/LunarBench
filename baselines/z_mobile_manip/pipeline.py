"""YOLOE + upstream cloud grasps, IK and RRT; observations arrive through bridge."""
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from scipy.spatial import cKDTree
from z_manip.models.antipodal_grasp import AntipodalGraspSource
from z_manip.models.grasp_source import GraspContext, GraspGenerationError
from z_manip.kinematics.robust_ik import RobustIKSolver, IKConfig, IKFailure
from z_manip.planning.rrt_connect import JointSpaceRRTConnect
from z_manip.planning.time_parameterization import retime_path
from z_manip.models.planner import PlanningError
from .kinematics import load_chain
from z_manip.perception.rgbd import backproject_depth
from z_manip.perception.edge_tam_backend import EdgeTamBackend, TrackingFailure

MODELS=Path(__file__).resolve().parents[3]/'.local/models/z_mobile_manip'

class Pipeline:
    def __init__(self):
        import torch
        from ultralytics import YOLOE
        torch.set_num_threads(2)
        self.detector=YOLOE(str(MODELS/'yoloe-11s-seg.pt'))
        prompts=torch.load(MODELS/'rock_text_embeddings.pt',map_location='cpu',weights_only=True)
        self.detector.set_classes(prompts['names'],prompts['embeddings'])
        self.edge=EdgeTamBackend(SimpleNamespace(model_id=str(MODELS/'edgetam'),device='cuda',vision_cache_frames=2,stream_history_frames=16,min_mask_pixels=20,min_score=.6))
        self.edge._load()
        self.chain=load_chain();self.ik=RobustIKSolver(self.chain,IKConfig(random_seeds=3,solve_timeout_s=.25,seed_timeout_s=.08,max_feasible_solutions=1))
        self.grasps=AntipodalGraspSource(max_aperture_m=.07,max_candidates=16)
        self.stats={'detections':0,'grasp_candidates':0,'ik_attempts':0,'ik_solutions':0,'planned_paths':0,'executed_paths':0}
        self.reset()

    def reset(self):
        self.trajectory=None;self.started=None;self.phase='search';self.goal=None;self.last_detection=None
        if getattr(self,'track_state',None) is not None:self.edge.reset(self.track_state)
        self.track_state=None;self.track_index=0;self.selection_seq=None;self.track_origin=None

    def observe(self,cm,c,selection=None):
        result=self.detector.predict(c['rgb'][...,::-1].copy(),conf=.15,imgsz=640,device=0,verbose=False,retina_masks=True)[0]
        self.stats['inference_ms']=sum(result.speed.values());self.stats['detections']=len(result.boxes)
        K=np.array(cm['K']);T=np.array(cm['T_base_camera']);depth=c['depth_m']
        xyz=backproject_depth(depth,K,T,min_depth_m=.12,max_depth_m=8.)
        valid=np.isfinite(xyz).all(axis=-1)
        scene=xyz[::4,::4][valid[::4,::4]]
        choices=[]
        proposals=[] if result.masks is None else list(zip(result.masks.data.cpu().numpy()>.5,result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()))
        mode='yoloe'
        mask=None;track_started=time.monotonic()
        try:
            if selection and selection['epoch']==cm['epoch'] and selection['seq']!=self.selection_seq:
                self.selection_seq=selection['seq']
                if abs(selection['sim_time_s']-cm['sim_time_s'])<2.:
                    if self.track_state is not None:self.edge.reset(self.track_state)
                    self.track_state,mask,score=self.edge.initialize(c['rgb'],tuple(selection['bbox']))
                    self.track_index=0;self.track_origin='operator_roi_edgetam'
            elif self.track_state is not None:
                self.track_index+=1
                mask,score=self.edge.update(self.track_state,c['rgb'],self.track_index)
            if mask is not None:
                ys,xs=np.nonzero(mask)
                if mask.sum()>depth.size*.12:raise TrackingFailure('mask expanded onto terrain')
                box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
                proposals=[(mask,box,score)];mode=self.track_origin
        except TrackingFailure as e:
            if self.track_state is not None:self.edge.reset(self.track_state)
            self.track_state=None;self.stats['tracking_status']=str(e)
        self.stats['tracking_ms']=(time.monotonic()-track_started)*1000
        self.stats['tracking_frames']=self.track_index
        for mask,box,score in proposals:
            if mask.sum()>depth.size*.12:continue
            points=xyz[mask&valid]
            if len(points)<20:continue
            center=np.median(points,axis=0)
            if center[0]<.1 or center[2]>.15:continue
            choices.append((np.linalg.norm(center[:2]),points,center,box,float(score)))
        if not choices:return None
        _,points,center,box,score=min(choices,key=lambda x:x[0])
        if mode=='yoloe' and self.track_state is None:
            try:
                self.track_state,_,_=self.edge.initialize(c['rgb'],tuple(int(x) for x in box))
                self.track_index=0;self.track_origin='yoloe_edgetam'
            except TrackingFailure:pass
        self.last_detection={'bbox':box.tolist(),'confidence':score,'center_base_m':center.tolist(),'camera_seq':cm['seq'],'target_source':mode}
        points=points[::max(1,int(np.ceil(len(points)/2000)))]
        self.stats['object_points']=len(points)
        try:
            candidates=self.grasps.generate(GraspContext(points,None,'base_link',np.eye(4),scene,lambda *_:None))
            self.stats['grasp_candidates']=len(candidates.grasps)
            self.stats['proposal_status']='candidates_ready'
        except GraspGenerationError as e:
            self.stats['grasp_candidates']=0;self.stats['proposal_status']=str(e)
        return points,scene,center

    def environment_points(self,points,current):
        frames=self.chain.link_transforms(current)
        own=np.zeros(len(points),bool)
        p=[frames[n][:3,3] for n in ['arm_base','link1','link2','link3','link4','link5','link6','grasp_tcp']]
        for a,b in zip(p[:-1],p[1:]):
            v=b-a;u=np.clip((points-a)@v/max(float(v@v),1e-10),0,1)
            own |= np.linalg.norm(points-a-u[:,None]*v,axis=1)<.065
        return points[~own]

    def plan(self,points,scene,current):
        context=GraspContext(points,None,'base_link',np.eye(4),scene,lambda *_:None)
        candidates=self.grasps.generate(context);self.stats['grasp_candidates']=len(candidates.grasps)
        # Known robot geometry + measured environment only. Occluded environment is unknown.
        scene=self.environment_points(scene,current)
        tree=cKDTree(scene) if len(scene) else None
        def valid(q):
            frames=self.chain.link_transforms(q)
            p=[frames[n][:3,3] for n in ['link2','link3','link4','link5','link6']]
            for a,b in zip(p[:-1],p[1:]):
                sample=np.linspace(a,b,12)
                # Simplified arm capsules; target contact only permitted at TCP
                if tree is not None and np.min(tree.query(sample)[0])<.025:return False
            return True
        planner=JointSpaceRRTConnect(joint_names=self.chain.joint_names,lower_limits=self.chain.lower_limits,upper_limits=self.chain.upper_limits,state_valid=valid)
        errors=[]
        for grasp in candidates.grasps[:4]:
            pre=grasp.copy();pre[:3,3]-=.08*grasp[:3,2]
            try:
                self.stats["ik_attempts"]+=1
                a=self.ik.solve(pre,current).joints
                self.stats["ik_attempts"]+=1
                b=self.ik.solve(grasp,a).joints
                self.stats['ik_solutions']+=2
                path=planner.plan_joint(current,a,timeout_s=.3)
                contact=planner.plan_joint(a,b,timeout_s=.3)
                waypoints=np.vstack((path.waypoints,contact.waypoints))
                self.trajectory=retime_path(waypoints,np.full(6,.4),np.full(6,.6))
                self.goal=b;self.started=None;self.phase='execute_reach';self.stats['planned_paths']+=1
                return
            except (IKFailure,PlanningError,RuntimeError) as e:errors.append(str(e))
        raise RuntimeError('no reachable clear grasp: '+str(errors[-1:] ))

    def execute(self,stamp,current,action):
        if self.started is None:self.started=stamp
        elapsed=stamp-self.started;t=self.trajectory
        action[:3]=0.;action[3:9]=[np.interp(elapsed,t.times_s,t.positions[:,j]) for j in range(6)];action[9]=.035
        if elapsed>=t.times_s[-1]:
            if np.max(np.abs(current-self.goal))<.08:
                action[9]=0.;self.stats['executed_paths']+=int(self.phase!='close_gripper');self.phase='close_gripper'
            elif elapsed>t.times_s[-1]+3:raise RuntimeError('arm tracking timeout')
        return self.phase
