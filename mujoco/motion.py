"""Finite relative-displacement commands; velocity is only the inner policy reference."""
import numpy as np

def wrap(a):return (a+np.pi)%(2*np.pi)-np.pi

class Motion:
    speeds=(.15,.30,.50)
    def __init__(self):
        self.gear=1;self.active=False;self.goal=np.zeros(2);self.yaw=0.;self.settled=0.;self.message='Ready';self.integral=np.zeros(2);self.yaw_integral=0.
        self.bounds=np.array([[-3.8,-3.8],[3.8,3.8]])

    def stop(self,xy,yaw):
        self.goal=np.array(xy).copy();self.yaw=yaw;self.active=False;self.settled=0.;self.message='Standing'
        self.integral[:]=0.;self.yaw_integral=0.

    def key(self,key,xy,yaw):
        if key in ('h','l'):
            self.gear=int(np.clip(self.gear+(-1 if key=='h' else 1),0,2));return
        if key==' ':
            self.stop(xy,yaw);return
        if key not in 'wasdqe':return
        if not self.active:self.goal=np.array(xy).copy();self.yaw=yaw
        angle=self.yaw
        direction={'w':(.5,0),'s':(-.5,0),'a':(0,.5),'d':(0,-.5)}
        if key in direction:
            dx,dy=direction[key]
            candidate=self.goal+np.array([np.cos(angle)*dx-np.sin(angle)*dy,np.sin(angle)*dx+np.cos(angle)*dy])
            if np.any(candidate<self.bounds[0]) or np.any(candidate>self.bounds[1]):
                self.message='Target outside local patch';return
            self.goal=candidate
        else:self.yaw=wrap(self.yaw+(1 if key=='q' else -1)*np.pi/12)
        self.active=True;self.settled=0.;self.message='Moving to displacement target'

    def update(self,xy,yaw,velocity,dt=.02):
        if not self.active:return np.zeros(3)
        delta=self.goal-xy;distance=np.linalg.norm(delta);angle=wrap(self.yaw-yaw)
        self.integral=np.clip(self.integral+delta*dt,-.3,.3)
        self.yaw_integral=float(np.clip(self.yaw_integral+angle*dt,-.3,.3))
        desired=1.7*delta+.7*self.integral-.25*velocity
        limit=self.speeds[self.gear]
        desired*=min(1.,limit/max(np.linalg.norm(desired),1e-9))
        body=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@desired
        command=np.array([body[0],body[1],np.clip(2.5*angle+.8*self.yaw_integral,-.65,.65)])
        tolerance=(.02,.045,.045)[self.gear]
        if distance<tolerance and abs(angle)<.07:self.settled+=dt
        else:self.settled=0.
        if self.settled>.10:
            self.active=False;self.message='Target reached';return np.zeros(3)
        return command
