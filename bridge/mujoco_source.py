"""Engine-owned ideal LiDAR, encoders and IMU. Only measurements leave the producer."""
import threading
import copy
import time
import numpy as np
import mujoco
from .client import Client

LIDAR_HZ=10.
PROPRIO_HZ=20.
MAX_RANGE=40.
MIN_RANGE=.1
BEAMS=720


class MuJoCoSource:
    def __init__(self,task):
        self.task=task;self.epoch=None;self.last_lidar=-1.;self.last_robot=-1.
        self.latest={};self.snapshot=None;self.action=None;self.error=None;self.closed=False;self.seq=0;self.applied_seq=None
        self.lock=threading.Lock();m=task.model
        self.directions=np.zeros((BEAMS,3))
        self.distance=np.zeros(BEAMS);self.ids=np.zeros(BEAMS,np.int32)
        self.sensor=m.site('mid360_scan').id
        self.sensor_body=int(m.site_bodyid[self.sensor])
        self.base=m.body(task.robot.base_body).id
        self.exclude=[]
        for g in range(m.ngeom):
            b=int(m.geom_bodyid[g])
            while b and b!=self.base:b=int(m.body_parentid[b])
            if int(m.geom_bodyid[g])==self.sensor_body or (m.geom_contype[g]==0 and m.geom_conaffinity[g]==0):self.exclude.append(g)
        self.joints=[i for i in range(m.njnt) if int(m.jnt_type[i]) in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE))]
        self.qadr=np.array([m.jnt_qposadr[i] for i in self.joints],dtype=np.int32);self.vadr=np.array([m.jnt_dofadr[i] for i in self.joints],dtype=np.int32)
        if not self.joints:raise ValueError(f"No scalar joints in robot model: {m.jnt_type}")
        self.ray_model=copy.copy(m);self.ray_model.geom_group[self.exclude]=5
        self.ray_data=mujoco.MjData(self.ray_model)
        self.thread=threading.Thread(target=self._transport,daemon=True);self.thread.start()
        self.ray_thread=threading.Thread(target=self._scan,daemon=True);self.ray_thread.start()

    def _transport(self):
        client=Client()
        while not self.closed:
            with self.lock:frames=self.latest;self.latest={}
            try:
                for topic,(meta,arrays) in frames.items():client.publish(topic,meta,arrays)
                value=client.read('action')
                if value:
                    with self.lock:self.action=value[0]
                self.error=None
            except Exception as e:self.error=str(e)
            time.sleep(.025)
        client.close()

    def update(self):
        t=self.task;m,d=t.model,t.data;stamp=float(d.time)
        if self.epoch!=t.epoch:
            self.epoch=t.epoch;self.last_lidar=self.last_robot=-1.
            with self.lock:self.action=None;self.latest={}
        now=time.monotonic()
        if now-self.last_robot>=1/PROPRIO_HZ:
            self.last_robot=now;self.seq+=1
            rotation=d.xmat[self.base].reshape(3,3)
            roll=float(np.arctan2(rotation[2,1],rotation[2,2]));pitch=float(np.arcsin(np.clip(-rotation[2,0],-1,1)))
            velocity=np.empty(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.base,velocity,1)
            meta={'source':'mujoco_proprioception','seq':self.seq,'epoch':t.epoch,'sim_time_s':stamp,
                  'sent_ns':time.monotonic_ns(),'frame_id':'base_link',
                  'joint_names':[m.joint(i).name for i in self.joints],
                  'applied_action_seq':self.applied_seq,'manual':t.robot.manual,'paused':t.paused,'ended':t.terminated or t.truncated,
                  'linear_speed_mps':float(velocity[3]),'yaw_rate_rps':float(velocity[2])}
            arrays={'joint_position':d.qpos[self.qadr].copy(),'joint_velocity':d.qvel[self.vadr].copy(),
                    'attitude_rpy':np.array([roll,pitch,0.]),'angular_velocity':velocity[:3].copy(),
                    'hold_action':t.robot.last_action.copy()}
            with self.lock:
                self.latest['robot']=(meta,arrays)
                self.snapshot=(d.qpos.copy(),stamp,t.epoch,time.monotonic_ns())

    def _scan(self):
        m,d=self.ray_model,self.ray_data;sequence=0
        while not self.closed:
            started=time.monotonic()
            with self.lock:snapshot=self.snapshot
            if snapshot is None:time.sleep(.02);continue
            positions,stamp,epoch,sent_ns=snapshot
            d.qpos[:]=positions;mujoco.mj_forward(m,d)
            rotation=d.site_xmat[self.sensor].reshape(3,3)
            origin=d.site_xpos[self.sensor]
            # Low-discrepancy non-repeating coverage, not a spinning multi-ring LiDAR.
            index=np.arange(BEAMS)+sequence*BEAMS
            az=2*np.pi*np.mod(index*.6180339887498949,1.)
            el=np.deg2rad(-7.+59.*np.mod(index*.4142135623730951,1.))
            self.directions[:]=np.column_stack((np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)))
            base_rotation=d.xmat[self.base].reshape(3,3)
            transform=np.eye(4);transform[:3,:3]=base_rotation.T@rotation
            transform[:3,3]=base_rotation.T@(origin-d.xpos[self.base])
            world=np.ascontiguousarray(self.directions@rotation.T)
            mujoco.mj_multiRay(m,d,origin,world.ravel(),np.array([1,1,1,1,1,0],np.uint8),True,-1,
                              self.ids,self.distance,None,len(self.distance),MAX_RANGE)
            valid=(self.distance>=MIN_RANGE)&(self.distance<=MAX_RANGE)
            ranges=np.where(valid,self.distance,0).astype('<f4')
            points=(self.directions[valid]*self.distance[valid,None]).astype('<f4');sequence+=1
            meta={'source':'mujoco_raycast','seq':sequence,'epoch':epoch,'sim_time_s':stamp,
                  'sent_ns':sent_ns,'frame_id':'mid360_scan','beams':len(ranges),
                  'range_max_m':MAX_RANGE,'range_min_m':MIN_RANGE,'horizontal_fov_deg':360.,
                  'vertical_fov_deg':[-7.,52.],'T_base_lidar':transform.tolist(),
                  'raycast_ms':(time.monotonic()-started)*1000,
                  'sensor_model':'Livox Mid-360','self_occlusion':True,'configured_hz':LIDAR_HZ,
                  'native_point_rate_hz':200000,'simulated_point_rate_budget_hz':int(LIDAR_HZ*BEAMS),
                  'model':'ideal low-discrepancy nonrepeating snapshot; 7200 rays/s budget, not vendor scan pattern; no reflectivity/noise/motion distortion'}
            with self.lock:
                if epoch==self.epoch:self.latest['lidar']=(meta,{'points':points,'ranges_m':ranges})
            time.sleep(max(.001,1/LIDAR_HZ-(time.monotonic()-started)))

    def command(self):
        with self.lock:action=self.action
        if not action or action.get('epoch')!=self.task.epoch:return None
        if (time.monotonic_ns()-action['sent_ns'])/1e9>.6:return None
        values=np.asarray(action.get('values',[]),dtype=float)
        if values.shape!=tuple(self.task.action_spec['shape']) or not np.isfinite(values).all():return None
        self.applied_seq=action['seq']
        return values

    def close(self):
        self.closed=True;self.thread.join(timeout=2);self.ray_thread.join(timeout=2)
