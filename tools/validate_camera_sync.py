"""Real-window bidirectional camera checks, using the demo's actual UE projection."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from validate_demo_gui import launch,workspace,physics_processes,focus_and_walk

root=workspace();out=root/'.local/validation';generated=root/'MoonSim/mujoco/generated'
report={'started_wall':time.monotonic()}
if physics_processes():raise SystemExit('Close existing demo physics before validation')
log=out/'camera_live_ue.log';log.write_text('')
process=launch(engine='both',extra_args=[f'-Abslog={log}','-NoSound'])
owned=set()


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def wait_for(predicate, timeout=45):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        owned.update(physics_processes())
        value=read(generated/'latency_current.json')
        if value.get('wall',0)>report['started_wall'] and predicate(value):return value
        if process.poll() is not None:raise RuntimeError(f'UE exited: {process.returncode}')
        time.sleep(.2)
    raise TimeoutError('Camera validation condition not met')


def frame():
    return read(generated/'viewport_last_state.json')['camera']


try:
    initial=wait_for(lambda v:v.get('frames',0)>180 and v.get('camera_projection_error_at_1280x720_px',1)<.1)
    report['before']=initial
    start=frame()
    # Native UE right-mouse drag changes its piloted transient camera.
    focus_and_walk(process.pid,drag_button=3)
    ue=wait_for(lambda v:v.get('camera_edits_sent',0)>initial.get('camera_edits_sent',0)
                and v.get('camera_ack',0)>=v.get('camera_edits_sent',0)
                and v.get('camera_projection_error_at_1280x720_px',1)<.1)
    time.sleep(1)
    changed=frame();ue_delta=math.dist(start[:9],changed[:9])
    assert ue_delta>.02,(start,changed)
    report['ue_to_mujoco']={'camera_delta':ue_delta,'metrics':ue}
    # Native MuJoCo left drag changes its free camera. UE follows the canonical packet.
    start=changed
    physics=physics_processes();assert len(physics)==1,physics
    before=time.monotonic()
    focus_and_walk(physics[0],drag_button=1)
    mj=wait_for(lambda v:v['wall']>before+2 and v.get('camera_projection_error_at_1280x720_px',1)<.1)
    time.sleep(1)
    changed=frame();mj_delta=math.dist(start[:9],changed[:9])
    assert mj_delta>.02,(start,changed)
    report['mujoco_to_ue']={'camera_delta':mj_delta,'metrics':mj}
    subprocess.run(['import','-window','root',str(out/'camera_sync_windows.png')],check=True,timeout=15)
    report['physics']=read(generated/'runtime_metrics.json')
    report['passed']=True
    (out/'camera_sync_live.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
finally:
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
    for pid in owned:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
