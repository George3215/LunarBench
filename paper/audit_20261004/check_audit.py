"""Focused regression checks for the audit's changes, independent of old scores."""
from pathlib import Path
import sys,math,types,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tasks/task3_reobservability'))
import api
from camera_reacquisition_audit import mask_rgb,sustained
from benchmark import _yaw
checks=[]
for yaw in [-math.pi,-1.3,0,.7,math.pi/2]:
    q=np.zeros(25);q[3]=math.cos(yaw/2);q[6]=math.sin(yaw/2)
    obj=types.SimpleNamespace(data=types.SimpleNamespace(qpos=q))
    got=api.MuJoCoEnvironment._pose(obj).yaw
    err=math.atan2(math.sin(got-yaw),math.cos(got-yaw))
    assert abs(err)<1e-10 and abs(got-_yaw(obj.data))<1e-10
checks.append('API yaw matches analytic pure-yaw rotations and rollout convention')
assert len(api.scene_catalogue())==24
checks.append('full scene catalogue has 24 worlds')
a=np.array([[[255,0,0],[100,0,0],[255,255,255],[0,0,0]]],dtype=np.uint8)
assert mask_rgb(a).tolist()==[[True,False,False,False]]
checks.append('RGB detector does not mistake white or subthreshold red for target')
def c(p,u=1,b=True):return {'sensor':{'pixels':p,'in_bounds':b},'upright':u}
assert sustained([c(3),c(3),c(3)])
assert not sustained([c(3),c(0),c(3)])
assert not sustained([c(3),c(3,u=0),c(3)])
assert not sustained([c(3),c(3,b=False),c(3)])
assert not sustained([c(0,b=False),c(3),c(3),c(3)])
checks.append('camera success requires three consecutive valid frames before first termination')
Path(__file__).with_name('regression_checks.json').write_text(json.dumps({'passed':checks},indent=2),encoding='utf8')
print('\n'.join('PASS '+s for s in checks))
