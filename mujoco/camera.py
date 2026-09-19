"""Demo camera synchronization: same eye/basis/projection convention as ATEC_UE_sim.

Keep this engine detail local: UE sends user camera edits; MuJoCo publishes the
canonical camera alongside robot poses. No lockstep or generic middleware.
"""
import json
import math
import socket
import numpy as np
import mujoco


class CameraSync:
    def __init__(self, sim, viewer, receive=False):
        self.viewer=viewer
        self.camera=mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:]=[0,0,.3]
        self.camera.distance=4.568369
        self.camera.azimuth=-52.594643
        self.camera.elevation=-20.499
        self.scene=mujoco.MjvScene(sim.model,maxgeom=1)
        self.last_robot=(sim.base_position if hasattr(sim,'base_position') else sim.data.qpos[:3]).copy()
        self.ack=0
        self.edits_received=0
        self.socket=None
        self.source='mujoco'
        self.vfov=float(sim.model.vis.global_.fovy)
        if receive:
            self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            self.socket.bind(('127.0.0.1',19402));self.socket.setblocking(False)
        self.attach(viewer)

    def attach(self, viewer):
        self.viewer=viewer
        if viewer:
            with viewer.lock():
                for name in ('lookat','distance','azimuth','elevation'):
                    if name=='lookat':viewer.cam.lookat[:]=self.camera.lookat
                    else:setattr(viewer.cam,name,getattr(self.camera,name))

    def update(self, sim):
        if self.viewer:
            with self.viewer.lock():
                self.camera.lookat[:]=self.viewer.cam.lookat
                for name in ('distance','azimuth','elevation'):
                    setattr(self.camera,name,getattr(self.viewer.cam,name))
        delta=(sim.base_position if hasattr(sim,'base_position') else sim.data.qpos[:3])-self.last_robot
        self.last_robot=(sim.base_position if hasattr(sim,'base_position') else sim.data.qpos[:3]).copy()
        self.camera.lookat[:]+=delta
        latest=None
        if self.socket:
            for _ in range(256):
                try:latest,_=self.socket.recvfrom(4096)
                except BlockingIOError:break
        if latest:
            try:
                packet=json.loads(latest)
                frame=np.asarray(packet['camera'],dtype=float)
                seq=int(packet['seq'])
                if frame.shape!=(11,) or not np.isfinite(frame).all() or not 1<frame[9]<170:
                    raise ValueError('Invalid camera')
                forward=frame[3:6];norm=np.linalg.norm(forward)
                if norm<1e-8:raise ValueError('Invalid direction')
                if seq>self.ack:
                    forward=forward/norm
                    self.camera.lookat[:]=frame[:3]+forward*self.camera.distance
                    self.camera.azimuth=math.degrees(math.atan2(forward[1],forward[0]))
                    self.camera.elevation=math.degrees(math.asin(float(np.clip(forward[2],-1,1))))
                    self.vfov=float(frame[9])
                    self.ack=seq;self.edits_received+=1;self.source='ue'
            except (ValueError,KeyError,TypeError):
                pass
        sim.model.vis.global_.fovy=self.vfov
        self.attach(self.viewer)
        # Like ATEC: update only camera transforms, never rebuild geometry for this.
        mujoco.mjv_updateCamera(sim.model,sim.data,self.camera,self.scene)
        a,b=self.scene.camera
        eye=(a.pos+b.pos)*.5;forward=(a.forward+b.forward)*.5;up=(a.up+b.up)*.5
        aspect=16/9
        if self.viewer:
            rect=self.viewer.viewport
            if rect and rect.height>0:aspect=rect.width/rect.height
        return [*map(float,eye),*map(float,forward),*map(float,up),float(sim.model.vis.global_.fovy),aspect]

    def close(self):
        if self.socket:self.socket.close()
        self.scene=None
