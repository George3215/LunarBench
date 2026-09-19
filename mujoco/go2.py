"""Go2 policy + local MuJoCo collisions. UDP is localhost-only, UE is visual-only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,socket,time,select,sys,termios,tty
import numpy as np
import mujoco
from motion import Motion
from model import scene_tree
import xml.etree.ElementTree as ET
from camera import CameraSync

ROOT=Path(__file__).resolve().parent
JOINTS=[f'{leg}_{part}_joint' for leg in ('FR','FL','RR','RL') for part in ('hip','thigh','calf')]
DEFAULT=np.tile([0.,.8,-1.5],4)
SCALE=np.tile([.125,.25,.25],4)

class Simulation:
    def __init__(self,flat=False,gravity=-9.81,stream=False,scene_path=None,metadata_path=None,policy_path=None):
        self.scene_path=Path(scene_path or ROOT/'generated/scene.xml')
        self.meta=json.loads(Path(metadata_path or ROOT/'generated/scene.json').read_text())
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(scene_tree(self.scene_path).getroot(),encoding='unicode'))
        m=self.model
        m.opt.gravity[2]=gravity
        if flat:
            m.hfield_data[:]=0
            m.geom_pos[m.geom('terrain').id,2]=0
            for i in range(m.ngeom):
                if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('rock_'):
                    m.geom_contype[i]=m.geom_conaffinity[i]=0
        self.data=mujoco.MjData(m)
        self.qa=np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
        self.va=np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINTS])
        self.act=np.array([next(i for i in range(m.nu) if m.actuator_trnid[i,0]==m.joint(j).id) for j in JOINTS])
        self.weights=dict(np.load(policy_path or ROOT.parent/'assets/robots/go2_demo/policy/policy.npz'))
        self.visual=[i for i in range(m.ngeom) if m.geom_group[i]==2 and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
        self.command=np.zeros(3);self.action=np.zeros(12);self.target=DEFAULT.copy()
        self.steps=0;self.paused=False;self.boundary=False;self.fallen=False
        self.motion=Motion();self.blend=0.;self.hold=DEFAULT.copy()
        self.visual_ids=list(self.visual)
        self.windows=None
        if stream and not flat:
            from streaming import CollisionWindows
            self.windows=CollisionWindows(self.meta, self.scene_path)
            self.motion.bounds=self.windows.bounds
        self.holding_goal=False
        self.external_command=None
        self.reset()

    def reset(self):
        if self.windows is not None:
            gravity=self.model.opt.gravity.copy()
            if self.windows.pending is not None:
                self.windows.pending.result();self.windows.pending=None
            self.model,self.windows.center,self.meta['rocks'],_=self.windows.build([0,0])
            self.model.opt.gravity[:]=gravity
            self.data=mujoco.MjData(self.model)
            self.visual=[i for i in range(self.model.ngeom) if self.model.geom_group[i]==2 and self.model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:3]=[0,0,.37]
        self.data.qpos[3:7]=[1,0,0,0]
        self.data.qpos[self.qa]=DEFAULT
        self.command[:]=0;self.action[:]=0;self.target=DEFAULT.copy()
        self.steps=0;self.fallen=False;self.boundary=False;self.paused=False
        self.motion.stop(np.zeros(2),0.);self.blend=0.;self.hold=DEFAULT.copy()
        self.holding_goal=False
        mujoco.mj_forward(self.model,self.data)

    def yaw(self):
        q=self.data.qpos[3:7]
        return float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))

    def key(self,key):
        key=key.lower()
        if key=='r':self.reset()
        elif key=='p':self.paused=not self.paused
        else:self.motion.key(key,self.data.qpos[:2],self.yaw())
        if key==' ':self.command[:]=0
        if key==' ':self.hold=self.data.qpos[self.qa].copy();self.blend=0.
        if key in 'wasdqe':self.holding_goal=True
        if key==' ':self.holding_goal=False
        print(f'KEY {key!r} goal={self.motion.goal.tolist()} gear={self.motion.gear+1} active={self.motion.active}',flush=True)

    def step(self):
        if self.paused:return
        if self.windows is not None and not self.windows.update(self):return
        m,d=self.model,self.data
        self.boundary=bool(np.any(d.qpos[:2]<self.motion.bounds[0]) or np.any(d.qpos[:2]>self.motion.bounds[1]))
        if self.boundary:
            self.command[:]=0;self.motion.stop(d.qpos[:2],self.yaw())
        if self.boundary:
            self.paused=True;return
        if self.steps%10==0:
            if self.holding_goal and not self.motion.active and np.linalg.norm(self.motion.goal-d.qpos[:2])>.06:
                self.motion.active=True;self.motion.settled=0.
            was_active=self.motion.active
            if self.external_command is None:
                self.command=self.motion.update(d.qpos[:2],self.yaw(),d.qvel[:2])
            else:
                self.command=self.external_command.copy()
                self.motion.active=bool(np.linalg.norm(self.command)>1e-8)
            if was_active and not self.motion.active:
                self.hold=d.qpos[self.qa].copy();self.blend=0.
            mat=np.empty(9);mujoco.mju_quat2Mat(mat,d.qpos[3:7])
            gravity=mat.reshape(3,3).T@np.array([0.,0.,-1.])
            if gravity[2]>-.25 or not np.isfinite(d.qpos).all():
                self.fallen=True;self.paused=True;self.command[:]=0;return
            obs=np.concatenate([d.qvel[3:6]*.25,gravity,self.command,
                d.qpos[self.qa]-DEFAULT,d.qvel[self.va]*.05,self.action]).astype(np.float32)
            x=np.clip(obs,-100,100)
            for layer in (0,2,4,6):
                x=self.weights[f'actor.{layer}.weight']@x+self.weights[f'actor.{layer}.bias']
                if layer!=6:x=np.where(x>0,x,np.expm1(np.minimum(x,0)))
            self.action=np.clip(x,-100,100)
            self.target=DEFAULT+self.action*SCALE
        self.blend=float(np.clip(self.blend+(1 if self.motion.active else -1)*m.opt.timestep/.3,0,1))
        target=self.hold+(self.target-self.hold)*self.blend
        torque=(35-15*self.blend)*(target-d.qpos[self.qa])-(1.5-self.blend)*d.qvel[self.va]
        d.ctrl[self.act]=np.clip(torque,-23.5,23.5)
        mujoco.mj_step(m,d);self.steps+=1

    def packet(self):
        mujoco.mj_forward(self.model,self.data)
        geoms=[];origin=np.array(self.meta['origin_ue_m'])
        for ident,i in zip(self.visual_ids,self.visual):
            p=self.data.geom_xpos[i]*[1,-1,1]+origin
            q=np.empty(4);mujoco.mju_mat2Quat(q,self.data.geom_xmat[i])
            geoms.append([ident,*(p*100).tolist(),-float(q[1]),float(q[2]),-float(q[3]),float(q[0])])
        return {'v':1,'sent_ns':time.monotonic_ns(),'time':self.data.time,'geoms':geoms,'command':self.command.tolist(),
                'paused':self.paused,'boundary':self.boundary,'fallen':self.fallen,
                'gear':self.motion.gear+1,'speed_limit':self.motion.speeds[self.motion.gear],
                'goal':self.motion.goal.tolist(),'remaining_m':float(np.linalg.norm(self.motion.goal-self.data.qpos[:2])),
                'moving':self.motion.active,'message':self.motion.message}

def open_viewer(sim,interactive=True):
    from viewer import Viewer
    callback=(lambda k:sim.key(chr(k)) if k<128 else None) if interactive else None
    return Viewer(sim.model,sim.data,callback),set()


def close_viewer(handle,threads):
    handle.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--viewer',action='store_true',help='Show MuJoCo alongside UE; keep the same window across collision swaps')
    ap.add_argument('--headless',action='store_true');ap.add_argument('--seconds',type=float,default=0)
    ap.add_argument('--flat',action='store_true');ap.add_argument('--distance',type=float,default=0)
    ap.add_argument('--gravity',type=float,default=-9.81)
    ap.add_argument('--report',type=Path);ap.add_argument('--terminal',action='store_true')
    ap.add_argument('--no-viewer',action='store_true',help='Real-time physics for UE, without MuJoCo GUI')
    ap.add_argument('--ue',action='store_true',help='Stop movement if focused UE viewport heartbeat disappears')
    ap.add_argument('--stream',action='store_true',help='Dynamic local collision windows, also for headless tests')
    args=ap.parse_args()
    if args.ue:
        sys.stdout=open(ROOT/'generated/viewport_physics.log','a',buffering=1);sys.stderr=sys.stdout
    sim=Simulation(args.flat,args.gravity,stream=args.ue or args.stream)
    tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind(('127.0.0.1',19401));rx.setblocking(False)
    print(f'Go2: {len(sim.meta["rocks"])} rocks, 21x21 hfield, policy 50 Hz, physics 500 Hz',flush=True)
    print('W/S/A/D: 0.5m displacement; Q/E: 15deg; H/L: lower/higher gear; SPACE stop; R reset; P pause.',flush=True)
    viewer=None;viewer_threads=set();old=None
    if not args.headless and not args.terminal and (args.viewer or (not args.no_viewer and sim.windows is None)):
        viewer,viewer_threads=open_viewer(sim,interactive=not args.ue)
        viewer.cam.lookat[:]=[0,0,.2];viewer.cam.distance=5;viewer.cam.elevation=-25
    if args.terminal and sys.stdin.isatty():
        old=termios.tcgetattr(sys.stdin);tty.setcbreak(sys.stdin.fileno())
    camera_sync=CameraSync(sim,viewer,receive=args.ue)
    started=time.monotonic();next_step=started;last_metrics=started;last_sim=0.;packet_seq=0;render_ms=0.;packet_ms=0.;sent_bytes=0
    last_frame=0.;compute=0.;max_contact=0;min_up=1.;count=0;issued=False;heartbeat=started
    try:
        while (not args.seconds or sim.data.time<args.seconds) and (viewer is None or viewer.is_running()):
            tick=time.monotonic()
            while True:
                try:
                    payload,_=rx.recvfrom(1024);key=payload.decode('ascii',errors='ignore')[:1]
                    if key=='_':heartbeat=time.monotonic()
                    elif key:sim.key(key)
                except BlockingIOError:break
            if args.ue and time.monotonic()-heartbeat>.75 and sim.motion.active:sim.key(' ')
            if old and select.select([sys.stdin],[],[],0)[0]:
                key=sys.stdin.read(1)
                if key=='\x1b':break
                sim.key(key);print('command',sim.command,'paused',sim.paused,flush=True)
            if args.distance and sim.data.time>1 and not issued:
                sim.motion.goal=sim.data.qpos[:2]+[args.distance,0];sim.motion.yaw=sim.yaw();sim.motion.active=True;sim.holding_goal=True;issued=True
            old_model=sim.model
            sim.step();count+=1
            if viewer and sim.model is not old_model:
                viewer.replace_model(sim.model,sim.data)
            max_contact=max(max_contact,sim.data.ncon)
            min_up=min(min_up,float(sim.data.xmat[sim.model.body('base_link').id,8]))
            compute+=time.monotonic()-tick
            if time.monotonic()-last_frame>1/50:
                send_start=time.monotonic()
                camera_frame=camera_sync.update(sim)
                packet=sim.packet();packet_seq+=1
                packet.update(seq=packet_seq,camera=camera_frame,camera_ack=camera_sync.ack)
                payload=json.dumps(packet,separators=(',',':'),allow_nan=False).encode()
                tx.sendto(payload,('127.0.0.1',int(os.environ.get('LUNARBENCH_STATE_PORT', '19400'))))
                sent_bytes+=len(payload);packet_ms=(time.monotonic()-send_start)*1000
                render_start=time.monotonic()
                if viewer:viewer.sync()
                render_ms=(time.monotonic()-render_start)*1000
                last_frame=send_start
            if args.headless and args.seconds and sim.paused:break
            if not args.headless:
                # Accumulate deadlines so rendering/sleep overhead does not slow physics.
                next_step+=sim.model.opt.timestep
                now=time.monotonic()
                if sim.paused or now-next_step>.25:next_step=now
                delay=next_step-now
                if delay>0:time.sleep(delay)
                if now-last_metrics>=1:
                    metrics={'wall':now,'sim_time_s':float(sim.data.time),
                             'real_time_factor':(sim.data.time-last_sim)/(now-last_metrics),
                             'packet_encode_send_ms':packet_ms,'viewer_sync_ms':render_ms,
                             'bytes_per_s':sent_bytes/(now-last_metrics),'packets_sent':packet_seq,
                             'camera_edits_received':camera_sync.edits_received,
                             'viewer_model_reload_ms':viewer.reload_ms if viewer else 0.,
                             'collision_swaps':sim.windows.swaps if sim.windows else 0}
                    (ROOT/'generated/runtime_metrics.json').write_text(json.dumps(metrics))
                    with (ROOT/'generated/runtime_metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                    last_metrics=now;last_sim=sim.data.time;sent_bytes=0
    finally:
        if old:termios.tcsetattr(sys.stdin,termios.TCSADRAIN,old)
        if viewer:close_viewer(viewer,viewer_threads)
        camera_sync.close()
        tx.close();rx.close()
        if sim.windows:sim.windows.close()
    report={'sim_seconds':float(sim.data.time),'wall_seconds':time.monotonic()-started,
            'compute_seconds':compute,'physics_steps':sim.steps,'steps_per_compute_second':sim.steps/max(compute,1e-9),
            'position_m':sim.data.qpos[:3].tolist(),'max_contacts':int(max_contact),'min_base_up_z':min_up,
            'fallen':sim.fallen,'boundary':sim.boundary,'finite':bool(np.isfinite(sim.data.qpos).all()),'rocks':len(sim.meta['rocks']),
            'collision_swaps':sim.windows.swaps if sim.windows else 0}
    print(json.dumps(report,indent=2))
    if args.report:args.report.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
