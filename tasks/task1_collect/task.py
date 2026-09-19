"""TASK1 environment: robot adapters, sparse collection reward and explicit episode API.

reset(seed=None) -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)
physics_step() is the lower-level keyboard/viewer loop entry, not the policy API.
Observations are simulator ground truth, not camera perception.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import uuid
import numpy as np
import mujoco
try:
    from . import robot as robot_module
    from .scoring import CollectionScore
except ImportError:
    import robot as robot_module
    from scoring import CollectionScore

class Task1:
    def __init__(self,config,field,scene_path,metadata_path=None):
        self.config=copy.deepcopy(config);self.field=field
        self.scene_path=Path(scene_path);self.metadata_path=Path(metadata_path or self.scene_path.with_suffix('.json'))
        self.meta=json.loads(self.metadata_path.read_text())
        self.signature=hashlib.sha256(self.scene_path.read_bytes()+(self.scene_path.parent/'height.bin').read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
        self.robot=robot_module.load(config['robot'],self.scene_path,self.metadata_path)
        self.model=self.robot.model;self.data=self.robot.data
        self.zone_center=np.asarray(config['collection_zone']['center_m'],dtype=float)
        self.zone_size=np.asarray(config['collection_zone']['size_m'],dtype=float)
        self.rock_adr=[int(self.model.jnt_qposadr[self.model.joint(f'sample_{i}_joint').id]) for i in self.meta['rocks']]
        self.rock_home=[(self.model.body(f'sample_{i}').pos.copy(),self.model.body(f'sample_{i}').quat.copy()) for i in self.meta['rocks']]
        self.rock_bodies=[self.model.body(f'sample_{i}').id for i in self.meta['rocks']]
        self.control_steps=round(config['episode']['control_dt']/self.model.opt.timestep)
        if self.control_steps<1 or abs(self.control_steps*self.model.opt.timestep-config['episode']['control_dt'])>1e-9:
            raise ValueError('control_dt must be an integer multiple of physics timestep')
        root_body=self.model.body(self.robot.base_body).id
        def belongs_to_robot(body):
            while body:
                if body==root_body:return True
                body=int(self.model.body_parentid[body])
            return False
        self.visual=[i for i in range(self.model.ngeom) if belongs_to_robot(int(self.model.geom_bodyid[i]))
                     and (self.model.geom_group[i]==2 if self.robot.name=='go2' else self.model.geom_group[i]!=3)]
        self.visual+= [self.model.geom(f'sample_{i}').id for i in self.meta['rocks']]
        # Compare policies only when the physical environment fingerprint agrees.
        # Hash compiled geometry, not output paths or policy parameters.
        digest=hashlib.sha256()
        for key in ('field','rock','collection_zone','reward','episode','robot'):
            digest.update(json.dumps(self.config[key],sort_keys=True).encode())
        for key in ('mesh_vert','mesh_face','hfield_data','body_mass','body_inertia',
                    'body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_friction',
                    'actuator_gainprm','actuator_biasprm','jnt_range'):
            digest.update(np.asarray(getattr(self.model,key)).tobytes())
        digest.update(Path(__file__).with_name('scoring.py').read_bytes())
        digest.update(str(mujoco.__version__).encode())
        digest.update(np.r_[self.model.opt.gravity,self.model.opt.timestep].tobytes())
        self.environment_id=digest.hexdigest()
        layout=hashlib.sha256(self.field.crop.tobytes())
        for pos,quat in self.rock_home:layout.update(np.r_[pos,quat].tobytes())
        layout.update(np.asarray(self.meta['rock_scales']).tobytes())
        self.layout_id=layout.hexdigest()
        self.scoring=CollectionScore(self)
        self.reset()

    @property
    def motion(self):return self.robot.motion
    @property
    def steps(self):return self.robot.steps
    @property
    def paused(self):return self.robot.paused
    @property
    def fallen(self):return self.robot.fallen
    @property
    def boundary(self):return self.robot.boundary
    @property
    def action_spec(self):return copy.deepcopy(self.robot.action_spec)
    @property
    def base_position(self):return self.data.xpos[self.model.body(self.robot.base_body).id].copy()
    def yaw(self):return self.robot.yaw()

    def reset(self,seed=None):
        if seed is not None and int(seed)!=self.config['rock']['seed']:
            # Rebuild only the task-owned directory. Native source terrain is immutable.
            try:
                from .scene import build
            except ImportError:
                from scene import build
            self.config['rock']['seed']=int(seed)
            meta,field=build(self.config,self.scene_path.parent)
            self.__init__(self.config,field,self.scene_path,self.metadata_path)
            return self.observation(),self.info()
        self.epoch=uuid.uuid4().hex
        self.robot.reset();self.robot.place(self.field,getattr(self.robot,'spec',self.config['robot']))
        for adr,(pos,quat) in zip(self.rock_adr,self.rock_home):
            self.data.qpos[adr:adr+7]=np.r_[pos,quat]
        mujoco.mj_forward(self.model,self.data)
        self.collected=set();self.score=0.;self.last_reward=0.
        self.scoring.reset()
        self.terminated=False;self.truncated=False
        return self.observation(),self.info()

    def key(self,key):
        if key.lower()=='r':self.reset()
        else:self.robot.key(key)

    def _in_zone(self,x,y):
        return bool(np.all(np.abs(np.array([x,y])-self.zone_center)<=self.zone_size/2))

    def physics_step(self):
        if self.terminated or self.truncated or self.paused:return 0.
        self.robot.advance()
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data);mujoco.mj_comVel(self.model,self.data)
        self.robot.boundary=bool(np.any(np.abs(self.base_position[:2])>self.field.half))
        reward=self.scoring.update() if self.config['reward'].get('enabled',True) else 0.
        self.score+=reward;self.last_reward=reward
        self.terminated=bool(self.fallen or self.boundary or (self.rock_adr and len(self.collected)==len(self.rock_adr)))
        self.truncated=bool(not self.terminated and self.config['episode']['time_limit_s']>0 and self.data.time>=self.config['episode']['time_limit_s'])
        return reward

    def step(self,action):
        if self.terminated or self.truncated:raise RuntimeError('Episode ended; call reset()')
        if self.paused:raise RuntimeError('Environment paused by viewer; resume before step(action)')
        self.robot.apply_action(action);reward=0.
        terms={"collection":0.,"fall":0.}
        for _ in range(self.control_steps):
            reward+=self.physics_step()
            for key in terms:terms[key]+=self.scoring.terms[key]
            if self.terminated or self.truncated:break
        self.last_reward=reward
        self.scoring.terms=terms
        return self.observation(),reward,self.terminated,self.truncated,self.info()

    def observation(self):
        rocks=np.array([self.data.qpos[a:a+7].copy() for a in self.rock_adr],dtype=np.float64).reshape(-1,7)
        velocity=np.zeros((len(self.rock_adr),6))
        for i,bid in enumerate(self.rock_bodies):mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity[i],0)
        return {'robot':self.robot.observe(),'rocks':{'pose':rocks,'velocity_world':velocity,
                'lifted':self.scoring.lifted.copy(),
                'collected':np.array([i in self.collected for i in range(len(self.rock_adr))],dtype=bool)},
                'goal':{'center':self.zone_center.copy(),'size':self.zone_size.copy()},
                'time':float(self.data.time),'score':float(self.score)}

    def info(self):
        return {'schema_version':1,'task':'task1_collect','robot':self.robot.name,
                'observation_source':'simulator_ground_truth','coordinate_frame':'task_local_metres_z_up',
                'quaternion_order':'wxyz','velocity_order':'angular_xyz_then_linear_xyz',
                'action_spec':self.action_spec,'control_dt':self.config['episode']['control_dt'],
                'environment_id':self.environment_id,'layout_id':self.layout_id,
                'physics_dt':float(self.model.opt.timestep),'seed':self.config['rock']['seed'],
                'success':bool(self.rock_adr and len(self.collected)==len(self.rock_adr)),
                'collected_count':len(self.collected),'rocks_total':len(self.rock_adr),
                'reward_terms':dict(self.scoring.terms), 'lifted_count':int(self.scoring.lifted.sum()),
                'termination_reason':('fallen' if self.fallen else 'boundary' if self.boundary else 'success' if self.terminated else 'time_limit' if self.truncated else None),
                'fallen':self.fallen,'boundary':self.boundary,'paused':self.paused}

    def get_state(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        physics=np.empty(mujoco.mj_stateSize(self.model,spec));mujoco.mj_getState(self.model,self.data,physics,spec)
        # 休眠标记必须一起存。mj_getState 的任何一个 spec 都不含它（mjNSTATE 里根本没有这一类），
        # 而休眠的刚体树不参与积分——不存它，重放就会从"这棵树睡不睡"开始分岔，而且每步都在
        # 重算这个数组，光看 qpos 完全看不出问题。实测不存：go2_piper 1.9e-06、rover_piper
        # 2.0e-05；存了并还原：两者都是 0.0。
        return {'schema_version':3,'collection_state':self.scoring.state(),'model_signature':self.signature,'physics':physics,
                'tree_asleep':self.data.tree_asleep.copy(),
                'robot_control':self.robot.control_state(),'collected':sorted(self.collected),
                'score':self.score,'last_reward':self.last_reward,'terminated':self.terminated,'truncated':self.truncated}

    def set_state(self,state):
        if state['model_signature']!=self.signature:raise ValueError('State belongs to a different scene/configuration')
        values=np.asarray(state['physics'],dtype=float)
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        if values.shape!=(mujoco.mj_stateSize(self.model,spec),) or not np.isfinite(values).all():raise ValueError('Invalid physics state')
        asleep=np.asarray(state.get('tree_asleep',self.data.tree_asleep),dtype=np.int32)
        if asleep.shape!=self.data.tree_asleep.shape:raise ValueError('Invalid sleep state')
        mujoco.mj_setState(self.model,self.data,values,spec)
        self.robot.restore_control(state['robot_control'])
        self.scoring.restore(state['collection_state'])
        self.collected=set(state['collected'])
        for k in ('score','last_reward','terminated','truncated'):setattr(self,k,copy.deepcopy(state[k]))
        mujoco.mj_forward(self.model,self.data)
        # 休眠标记要放在 mj_forward **之后**：mj_forward 自己会按当前速度重算一遍这个数组，
        # 先设会被它盖掉，而它盖掉之后算出来的接触与约束就和原来那一集不一致了。实测
        # go2_piper 误差 1.9e-06、rover_piper 2.0e-05，放到最后是 0.0。
        self.data.tree_asleep[:]=asleep

    def packet(self):
        geoms=[]
        for i in self.visual:
            name=self.model.geom(i).name
            if name.startswith('sample_'):
                bid=self.model.geom_bodyid[i];pos=self.data.xpos[bid];q=self.data.xquat[bid]
            else:
                pos=self.data.geom_xpos[i];q=np.empty(4);mujoco.mju_mat2Quat(q,self.data.geom_xmat[i])
            p=self.field.to_world(*pos)*100
            geoms.append([i,*p.tolist(),-float(q[1]),float(q[2]),-float(q[3]),float(q[0])])
        rotation=self.data.xmat[self.model.body(self.robot.base_body).id].reshape(3,3)
        sid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'d435_color_optical')
        sensor=None
        if sid>=0:
            optical=self.data.site_xmat[sid].reshape(3,3)
            position=self.data.site_xpos[sid]
            T=np.eye(4);T[:3,:3]=rotation.T@optical
            T[:3,3]=rotation.T@(position-self.base_position)
            sensor={'position_cm':(self.field.to_world(*position)*100).tolist(),
                    'forward':(optical[:,2]*[1,-1,1]).tolist(),
                    'up':(-optical[:,1]*[1,-1,1]).tolist(),'T_base_camera':T.tolist()}
        return {'epoch':self.epoch,**({'sensor_camera':sensor} if sensor else {}),'v':1,'task':'task1_collect','sent_ns':time.monotonic_ns(),'time':float(self.data.time),
                'geoms':geoms,'origin_ue_m':self.field.origin.tolist(),'base_position_m':self.base_position.tolist(),
                'score':self.score,'collected_count':len(self.collected),'rocks_total':len(self.rock_adr),
                'gear':self.motion.gear+1,'moving':self.motion.active,'paused':self.paused,
                'fallen':self.fallen,'boundary':self.boundary,
                'remaining_m':float(np.linalg.norm(self.motion.goal-self.base_position[:2]))}

    @property
    def observation_spec(self):
        def describe(value):
            if isinstance(value,dict):return {k:describe(v) for k,v in value.items()}
            if isinstance(value,str):return {'type':'string'}
            array=np.asarray(value)
            return {'shape':list(array.shape),'dtype':str(array.dtype)}
        return describe(self.observation())

    def close(self):
        temporary=getattr(self,'_temporary_directory',None)
        if temporary:temporary.cleanup();self._temporary_directory=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
