"""UE process boundary. Machine-specific paths live in environment variables."""
import os
from pathlib import Path
import subprocess
def workspace():
    return Path(__file__).resolve().parents[1]


DEFAULT_EDITOR = '/home/lry/UE/UE_5.7.4/Engine/Binaries/Linux/UnrealEditor'
DEFAULT_PROJECT = '/home/lry/文档/Unreal Projects/Moon/Moon.uproject'
# 默认仍是原 demo 地图。换皮地图是它的副本，用 LUNARBENCH_UE_MAP 切换，
# 这个变量同时会被 C++ 侧读取（MoonPaths::IsGo2Map），用来决定该地图是否由插件接管。
DEFAULT_MAP = '/Game/MoonGo2/Maps/MoonTerrain_Go2_FullRender'


def launch(*, engine="both", extra_args=(), extra_env=None):
    editor = Path(os.environ.get('LUNARBENCH_UE_EDITOR', DEFAULT_EDITOR))
    project = Path(os.environ.get('LUNARBENCH_UE_PROJECT', DEFAULT_PROJECT))
    env = os.environ.copy()
    env.update(extra_env or {})
    env['LUNARBENCH_WORKSPACE'] = str(workspace())
    env['LUNARBENCH_STATE_PORT'] = '19400'
    python = env.get('LUNARBENCH_PYTHON', '/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python')
    env['LUNARBENCH_PYTHON'] = python
    env['PYTHONPATH'] = ''
    env['LUNARBENCH_MUJOCO_VIEWER'] = '1' if engine == 'both' else '0'
    # 必须在 env 里回写：C++ 侧靠这个变量判断当前世界是不是它该接管的地图。
    ue_map = env.get('LUNARBENCH_UE_MAP') or DEFAULT_MAP
    env['LUNARBENCH_UE_MAP'] = ue_map
    if engine == 'mujoco':
        return subprocess.Popen([python, str(workspace() / 'mujoco/run.py'), *extra_args], cwd=workspace(), env=env)
    for path in (editor, project):
        if not path.is_file():
            raise FileNotFoundError(f'{path}; set LUNARBENCH_UE_EDITOR / LUNARBENCH_UE_PROJECT')
    for comm in Path('/proc').glob('[0-9]*/comm'):
        try:
            if comm.read_text().strip() == 'UnrealEditor':
                cmd = (comm.parent / 'cmdline').read_bytes().split(b'\0')
                if str(project).encode() in cmd:
                    raise RuntimeError('UE project already open; close it before changing physics ownership')
        except (OSError, PermissionError):
            continue
    env['LUNARBENCH_EXTERNAL_PHYSICS'] = '1' if engine == 'ue' else '0'
    args = [str(editor), str(project), ue_map, '-NoSplash', *extra_args]
    return subprocess.Popen(args, cwd=workspace(), env=env)
