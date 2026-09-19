"""Physics factory for scene tooling. Runtime policies live in separate bridge clients.

The returned Task object is engine-internal and is not a baseline observation API.
Launch task1.sh for UE sensors and the standalone baseline.
"""
from pathlib import Path
import sys
# Existing engine code is a script collection, not a package named 'mujoco'
# (that name is reserved for the upstream MuJoCo Python dependency).
_engine=str(Path(__file__).resolve().parents[2]/'mujoco')
if _engine not in sys.path:sys.path.insert(0,_engine)

def make(config_path=None, *, robot=None, seed=None, output_dir=None, policy=None):
    from .config import load
    from .scene import ensure,GENERATED
    from .task import Task1
    cfg=load(config_path)
    if robot:cfg['robot']['name']=robot
    if seed is not None:cfg['rock']['seed']=int(seed)
    if policy is not None:cfg['policy']['name']=policy
    import tempfile
    temporary=tempfile.TemporaryDirectory(prefix='moonsim-task1-') if output_dir is None else None
    out=Path(output_dir or temporary.name)
    _,field=ensure(cfg,out)
    env=Task1(cfg,field,out/'scene.xml',out/'scene.json')
    env._temporary_directory=temporary
    _attach_policy(env,cfg)
    return env

def _attach_policy(env,cfg):
    # Policies are external bridge clients, never constructed with an engine handle.
    env.policy=None
