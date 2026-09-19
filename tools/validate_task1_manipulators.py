"""Physical action, gripper, fixed/mobile base and replay checks in full TASK1."""
from pathlib import Path
import sys,json,tempfile
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from MoonSim.tasks.task1_collect import make
from MoonSim.tasks.task1_collect.robot import PROFILES
ROOT=Path(__file__).resolve().parents[1]

def check(name,config=None):
    with make(config,robot=name) as env:
        a=env.robot.last_action.copy();groups=env.action_spec['groups']
        def advance(a,n):
            for _ in range(n):
                o,_,t,tr,_=env.step({k:a[v] for k,v in groups.items()})
                assert not t and not tr
                assert np.isfinite(env.data.qpos).all()
            return o
        initial=env.base_position.copy();o=advance(a,30)
        tip=o['robot']['end_effectors']['tool_tip'];before=tip['position'].copy()
        a[groups['arm'][0]]+=.15;o=advance(a,25)
        distance=float(np.linalg.norm(o['robot']['end_effectors']['tool_tip']['position']-before))
        joint=env.model.joint('joint1').qposadr[0]
        assert abs(env.data.qpos[joint]-.15)<.04,(name,env.data.qpos[joint])
        assert distance>.005,(name,distance)
        a[groups['gripper']]=.032;advance(a,20)
        ids=[int(env.model.joint(f'joint{i}').qposadr[0]) for i in (7,8)]
        opened=env.data.qpos[ids].copy()
        a[groups['gripper']]=.004;advance(a,20);closed=env.data.qpos[ids].copy()
        assert np.all(opened-closed>.02),(opened,closed)
        assert abs(closed[0]-closed[1])<.002
        base_distance=0.
        if 'base' in groups:
            # Full-scale base command, clipped to whatever this robot's base group actually accepts
            # (wheel speed rad/s for the rovers, body velocity m/s for go2_piper).
            lo=np.asarray(env.action_spec['low']);hi=np.asarray(env.action_spec['high']);idx=groups['base']
            before=env.base_position.copy();a[idx]=np.clip(1.,lo[idx],hi[idx]);advance(a,25)
            base_distance=float(np.linalg.norm(env.base_position[:2]-before[:2]));assert base_distance>.005
        else:assert np.array_equal(env.base_position,initial)
        saved=env.get_state();advance(a,3);expected=env.data.qpos.copy()
        env.set_state(saved);advance(a,3);error=float(np.max(abs(expected-env.data.qpos)));assert error<1e-10
        assert tip['jacobian_linear'].shape==(3,len(env.robot.robot_qvel))
        result={'passed':True,'actions':env.model.nu,'base_fixed':env.robot.base_qadr is None,
                'arm_tip_displacement_m':distance,'finger_open_m':opened.tolist(),'finger_closed_m':closed.tolist(),
                'base_displacement_m':base_distance,'replay_error':error,'jacobian_shape':list(tip['jacobian_linear'].shape)}
        print(name,result,flush=True);return result

def main():
    result={name:check(name) for name in ('piper_arm','rover_piper','go2_piper')}
    with tempfile.TemporaryDirectory() as tmp:
        config=Path(tmp)/'external.yaml'
        config.write_text(yaml.safe_dump({'robot':{**PROFILES['piper_arm'],'name':'mjcf','model_path':str(ROOT/'assets/robots/piper/arm.xml')}}))
        result['external_fixed_mjcf']=check('mjcf',config)
    out=ROOT.parent/'.local/validation/task1_manipulators.json';out.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
