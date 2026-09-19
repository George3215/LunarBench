"""Concrete robot adapters. Tasks own goals/reward; adapters own low-level control.

Go2 reuses its existing locomotion controller; no new policy is integrated.
The authored reference rover and external MJCF adapter use native actuators.
"""
from pathlib import Path
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from go2 import Simulation
from model import scene_tree
from motion import Motion,wrap

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/robots'

ARM_HOME={'joint1':0.,'joint2':1.1,'joint3':-1.4,'joint4':0.,'joint5':.3,'joint6':0.,'joint7':.02,'joint8':.02}
PROFILES={
    'piper_arm':{'base_body':'arm_base','base_height_m':.20,'end_effector_sites':['tool_tip'],
                 'initial_joints':ARM_HOME,'initial_action':[0.,1.1,-1.4,0.,.3,0.,.02],
                 'action_units':['rad']*6+['m'],
                 'action_groups':{'arm':list(range(6)),'gripper':[6]}},
    'rover_piper':{'base_body':'base_link','base_height_m':.27,'end_effector_sites':['tool_tip'],
                   'initial_joints':ARM_HOME,'initial_action':[0.,0.,0.,1.1,-1.4,0.,.3,0.,.02],
                   'action_units':['rad/s']*2+['rad']*6+['m'],
                   'action_groups':{'base':[0,1],'arm':list(range(2,8)),'gripper':[8]}},
    'go2_piper':{'base_body':'base_link','base_height_m':.37,'end_effector_sites':['tool_tip'],
                 'initial_joints':ARM_HOME,'initial_action':[0.,0.,0.,0.,1.1,-1.4,0.,.3,0.,.02],
                 'action_units':['m/s','m/s','rad/s']+['rad']*6+['m'],
                 'action_groups':{'base':[0,1,2],'arm':list(range(3,9)),'gripper':[9]}}
}

def resolve_spec(spec):return {**copy.deepcopy(PROFILES.get(spec['name'],{})),**copy.deepcopy(spec)}

def model_path(spec):
    if spec['name']=='go2':return ASSETS/'go2_demo/go2/go2.xml'
    if spec['name']=='reference_rover':return ASSETS/'reference_rover/rover.xml'
    if spec['name']=='piper_arm':return ASSETS/'piper/arm.xml'
    if spec['name']=='rover_piper':return ASSETS/'piper/rover_piper.xml'
    if spec['name']=='go2_piper':return ASSETS/'go2_piper/go2_piper.xml'
    path=Path(spec['model_path']).expanduser()
    return path if path.is_absolute() else ROOT/path

def model_tree(spec):
    path=model_path(spec).resolve();tree=scene_tree(path);root=tree.getroot()
    # Included files remain relative to their source, not generated/.
    for element in root.iter('include'):
        element.set('file',str((path.parent/element.get('file')).resolve()))
    option=root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    option.set('timestep','0.002');option.set('gravity','0 0 -9.81');option.set('iterations','30')
    for key in list(root.findall('keyframe')):root.remove(key)
    if root.find('asset') is None:ET.SubElement(root,'asset')
    return root

class RobotInterface:
    """All arrays are copied; actions are finite and bounded in documented SI units."""
    def apply_action(self,action):
        if isinstance(action,dict):
            groups=self.action_spec.get('groups',{})
            if set(action)!=set(groups):raise ValueError(f'Expected action groups {list(groups)}')
            merged=np.empty(self.action_spec['shape'][0])
            for name,indices in groups.items():
                values=np.asarray(action[name],dtype=float)
                if values.shape!=(len(indices),):raise ValueError(f'Invalid shape for action group {name}')
                merged[indices]=values
            action=merged
        value=np.asarray(action,dtype=np.float64)
        spec=self.action_spec
        if value.shape!=tuple(spec['shape']) or not np.isfinite(value).all():
            raise ValueError(f'Expected finite action {spec}, got {action!r}')
        if np.any(value<spec['low']) or np.any(value>spec['high']):
            raise ValueError('Action outside action_spec bounds (no silent clipping)')
        self.last_action=value.copy();self.manual=False
        return value

    def observe(self):
        m,d=self.model,self.data;bid=m.body(self.base_body).id
        effectors={}
        for name in getattr(self,'spec',{}).get('end_effector_sites',[]):
            site=m.site(name).id;q=np.empty(4);mujoco.mju_mat2Quat(q,d.site_xmat[site])
            velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,site,velocity,0)
            jacp=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jacp,jacr,site)
            effectors[name]={'position':d.site_xpos[site].copy(),'quaternion_wxyz':q,
                             'velocity_world':velocity,'jacobian_linear':jacp[:,self.robot_qvel].copy(),
                             'jacobian_angular':jacr[:,self.robot_qvel].copy()}
        return {'end_effectors':effectors,'base_fixed':self.base_qadr is None,'name':self.name,'position':d.xpos[bid].copy(),'quaternion_wxyz':d.xquat[bid].copy(),
                'joint_position':d.qpos[self.robot_qpos].copy(),
                'joint_velocity':d.qvel[self.robot_qvel].copy(),
                'body_velocity_world':self.body_velocity(),'sensors':d.sensordata.copy(),
                'last_action':self.last_action.copy()}

    def body_velocity(self):
        v=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,
                               self.model.body(self.base_body).id,v,0)
        return v  # angular xyz, linear xyz, world frame

    def bind_indices(self):
        m=self.model
        joints=[j for j in range(m.njnt) if not m.joint(j).name.startswith('sample_')]
        self.robot_qpos=[];self.robot_qvel=[]
        for j in joints:
            typ=int(m.jnt_type[j]);nq=7 if typ==mujoco.mjtJoint.mjJNT_FREE else 4 if typ==mujoco.mjtJoint.mjJNT_BALL else 1
            nv=6 if typ==mujoco.mjtJoint.mjJNT_FREE else 3 if typ==mujoco.mjtJoint.mjJNT_BALL else 1
            self.robot_qpos.extend(range(int(m.jnt_qposadr[j]),int(m.jnt_qposadr[j])+nq))
            self.robot_qvel.extend(range(int(m.jnt_dofadr[j]),int(m.jnt_dofadr[j])+nv))
        self.robot_qpos=np.array(self.robot_qpos);self.robot_qvel=np.array(self.robot_qvel)
        bid=m.body(self.base_body).id
        if m.body_parentid[bid]!=0:raise ValueError('base_body must be a direct child of worldbody')
        j=int(m.body_jntadr[bid])
        if m.body_jntnum[bid]==0:self.base_qadr=None
        elif int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_FREE):self.base_qadr=int(m.jnt_qposadr[j])
        else:raise ValueError('Root base must be fixed or have a free joint')

    def place(self,field,spec):
        spawn=np.asarray(spec['spawn_m']);yaw=math.radians(spec['yaw_deg']);a=self.base_qadr
        position=[*spawn,float(field.height_local(*spawn))+spec.get('base_height_m',.37 if self.name=='go2' else .27)]
        quaternion=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        if a is None:
            bid=self.model.body(self.base_body).id
            self.model.body_pos[bid]=position;self.model.body_quat[bid]=quaternion
            # 固定基座没关节，MuJoCo 把它当成"焊死在世界"的刚体：mj_kinematics **不刷新**它的
            # xpos，而 mj_resetData 只在调用当时按 body_pos 播一次种子。place() 是在上一次
            # resetData **之后**改 body_pos 的，于是 data 里留着旧位姿，臂其实还长在原地——
            # piper_arm 因此一直站在场地中心，而不是 robot.spawn_m。这里把 data 侧一并写上，
            # mj_forward 之后子链就会跟着走到正确位置。
            self.data.xpos[bid]=position;self.data.xquat[bid]=quaternion
        else:
            self.data.qpos[a:a+3]=position;self.data.qpos[a+3:a+7]=quaternion
        mujoco.mj_forward(self.model,self.data)
        self.motion.bounds=np.array([[-field.half,-field.half],[field.half,field.half]])
        self.motion.stop(spawn,yaw)

    def control_state(self):
        state=copy.deepcopy({k:v for k,v in self.__dict__.items() if k in
            ('last_action','manual','steps','paused','fallen','boundary','command','action','target',
             'blend','hold','holding_goal','external_command','motion')})
        state['motion']=copy.deepcopy(self.motion.__dict__)
        return state

    def restore_control(self,state):
        for k,v in copy.deepcopy(state).items():
            if k=='motion':self.motion.__dict__.update(v)
            else:setattr(self,k,v)

class Go2Robot(Simulation,RobotInterface):
    name='go2';base_body='base_link'
    action_spec={'mode':'body_velocity','shape':[3],'low':[-.5,-.3,-.65],'high':[.5,.3,.65],
                 'units':['m/s','m/s','rad/s'],'axes':['forward','left','yaw_rate']}
    def __init__(self,scene_path,metadata_path,spec):
        super().__init__(scene_path=scene_path,metadata_path=metadata_path)
        self.bind_indices();self.last_action=np.zeros(3);self.manual=True
    def reset(self):
        super().reset();self.external_command=None;self.last_action=np.zeros(3);self.manual=True
    def apply_action(self,action):
        value=RobotInterface.apply_action(self,action)
        self.external_command=value.copy();self.holding_goal=False
    def key(self,key):
        self.external_command=None;self.manual=True;super().key(key)
    def advance(self):super().step()

class Go2PiperRobot(Simulation,RobotInterface):
    """Quadruped locomotion policy plus a position-servoed Piper arm on the back.

    The arm is not part of the upstream whole-body policy; it is commanded through the task
    action vector and written directly to its own actuators. Only the mounting transform is
    taken from the upstream GO2-PIPER platform.
    """
    name='go2_piper';base_body='base_link'
    arm_act=arm_qadr=None  # bound after the model exists; reset() may run before that
    def __init__(self,scene_path,metadata_path,spec):
        spec=resolve_spec(spec);self.spec=spec
        super().__init__(scene_path=scene_path,metadata_path=metadata_path)
        m=self.model
        self.arm_act=np.array([next(i for i in range(m.nu) if m.actuator_trnid[i,0]==m.joint(j).id)
                               for j in ('joint1','joint2','joint3','joint4','joint5','joint6','joint7')])
        self.arm_qadr=np.array([int(m.jnt_qposadr[m.joint(f'joint{i}').id]) for i in range(1,9)])
        groups=spec['action_groups'];width=3+len(self.arm_act)
        if sorted(i for g in groups.values() for i in g)!=list(range(width)) or len(groups['arm'])!=6:
            raise ValueError(f'go2_piper action_groups must cover 3 base + 6 arm + 1 gripper, got {groups}')
        self.action_spec={'mode':'body_velocity_and_arm_position','shape':[width],
                          'low':[-.5,-.3,-.65,*m.actuator_ctrlrange[self.arm_act,0].tolist()],
                          'high':[.5,.3,.65,*m.actuator_ctrlrange[self.arm_act,1].tolist()],
                          'units':spec['action_units'],
                          'axes':['forward','left','yaw_rate']+[m.joint(m.actuator_trnid[i,0]).name for i in self.arm_act],
                          'groups':groups}
        if len(self.action_spec['units'])!=width:raise ValueError('go2_piper action_units must match the action width')
        self.bind_indices();self.reset()
    def reset(self):
        super().reset()
        self.external_command=None;self.holding_goal=False;self.manual=True
        self.last_action=np.asarray(self.spec['initial_action'],dtype=float).copy()
        if self.arm_act is None:return
        home=self.spec.get('initial_joints',{});missing=[f'joint{i}' for i in range(1,9) if f'joint{i}' not in home]
        if missing:raise ValueError(f'go2_piper initial_joints missing {missing}')
        self.data.qpos[self.arm_qadr]=[home[f'joint{i}'] for i in range(1,9)]
        # A <position> servo at ctrl=0 drives to zero; joint2 would fold onto its lower limit.
        self.data.ctrl[self.arm_act]=self.last_action[3:]
        mujoco.mj_forward(self.model,self.data)
    def apply_action(self,action):
        value=RobotInterface.apply_action(self,action)
        self.external_command=value[:3].copy();self.holding_goal=False
        # Simulation.step() rewrites only the twelve leg motors, so this arm target persists.
        self.data.ctrl[self.arm_act]=value[3:]
    def key(self,key):
        self.external_command=None;self.manual=True;super().key(key)
    def advance(self):super().step()

class ActuatorRobot(RobotInterface):
    def __init__(self,scene_path,metadata_path,spec):
        spec=resolve_spec(spec);self.spec=spec;self.name=spec['name'];self.base_body=spec.get('base_body','base_link')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(scene_tree(scene_path).getroot(),encoding='unicode'))
        self.data=mujoco.MjData(self.model);self.motion=Motion();self.bind_indices()
        m=self.model
        if self.name=='reference_rover':
            self.action_spec={'mode':'wheel_velocity','shape':[2],'low':[-8.,-8.],'high':[8.,8.],
                              'units':['rad/s','rad/s'],'axes':['left','right']}
        else:
            if not m.nu or not m.actuator_ctrllimited.all():
                raise ValueError('External MJCF requires explicit bounded actuator ctrlrange')
            self.action_spec={'mode':'actuator_control','shape':[m.nu],
                              'low':m.actuator_ctrlrange[:,0].tolist(),'high':m.actuator_ctrlrange[:,1].tolist(),
                              'units':spec.get('action_units',['model-defined']*m.nu),
                              'axes':[m.actuator(i).name for i in range(m.nu)]}
        groups=spec.get('action_groups',{'all':list(range(m.nu))})
        indices=[i for group in groups.values() for i in group]
        if sorted(indices)!=list(range(m.nu)):raise ValueError('action_groups must partition all actuators exactly once')
        self.action_spec['groups']=groups
        if len(self.action_spec['units'])!=m.nu:raise ValueError('action_units must match actuator count')
        self.reset()
    def reset(self):
        mujoco.mj_resetData(self.model,self.data);self.steps=0
        self.paused=self.boundary=self.fallen=False;self.manual=True
        for name,value in self.spec.get('initial_joints',{}).items():
            j=self.model.joint(name).id
            if int(self.model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)):raise ValueError('initial_joints supports scalar joints')
            self.data.qpos[int(self.model.jnt_qposadr[j])]=value
        self.last_action=np.asarray(self.spec.get('initial_action',np.zeros(self.model.nu)),dtype=float).copy()
        if self.last_action.shape!=(self.model.nu,) or not np.isfinite(self.last_action).all():raise ValueError('Invalid initial_action')
        if np.any(self.last_action<self.action_spec['low']) or np.any(self.last_action>self.action_spec['high']):raise ValueError('initial_action outside bounds')
        self.data.ctrl[:]=self.last_action;self.command=np.zeros(3)
        self.motion.stop(np.zeros(2),0.);mujoco.mj_forward(self.model,self.data)
    def yaw(self):
        q=self.data.xquat[self.model.body(self.base_body).id]
        return float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))
    def key(self,key):
        key=key.lower();self.manual=True
        if key=='p':self.paused=not self.paused
        elif self.name in ('reference_rover','rover_piper'):self.motion.key(key,self.data.xpos[self.model.body(self.base_body).id,:2],self.yaw())
    def advance(self):
        if self.paused:return
        if self.manual and self.name in ('reference_rover','rover_piper'):
            bid=self.model.body(self.base_body).id
            delta=self.motion.goal-self.data.xpos[bid,:2];distance=float(np.linalg.norm(delta))
            heading=float(np.arctan2(delta[1],delta[0])) if distance>.03 else self.motion.yaw
            error=wrap(heading-self.yaw())
            if self.motion.active and distance<.03 and abs(wrap(self.motion.yaw-self.yaw()))<.07:
                self.motion.active=False
            v=min(self.motion.speeds[self.motion.gear],1.7*distance)*max(0.,np.cos(error)) if distance>.03 else 0.
            w=float(np.clip(2.5*error,-.65,.65))
            if not self.motion.active:v=w=0.
            self.command=np.array([v,0.,w]);self.last_action[:2]=np.clip([(v-w*.245)/.16,(v+w*.245)/.16],-8,8)
        self.data.ctrl[:]=self.last_action;mujoco.mj_step(self.model,self.data);self.steps+=1
        bid=self.model.body(self.base_body).id
        self.fallen=bool(self.base_qadr is not None and self.data.xmat[bid,8]<.25)

def load(spec,scene_path,metadata_path):
    kinds={'go2':Go2Robot,'go2_piper':Go2PiperRobot}
    return kinds.get(spec['name'],ActuatorRobot)(scene_path,metadata_path,spec)
