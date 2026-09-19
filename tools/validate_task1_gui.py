"""Real TASK1 windows: independent robot/camera readback, both mouse directions, 20 rocks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import math
import argparse
from validate_demo_gui import focus_and_walk
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'.local/validation/task1_gui'
OUT.mkdir(parents=True,exist_ok=True)

def read(name):
    try:return json.loads((OUT/name).read_text())
    except (OSError,ValueError):return {}

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--robot',choices=['go2','go2_piper','reference_rover','piper_arm','rover_piper'],default='go2');args=ap.parse_args()
    if args.robot!='go2':OUT=OUT.parent/('task1_gui_'+args.robot);OUT.mkdir(parents=True,exist_ok=True)
    report={'started_wall':time.monotonic(),'robot':args.robot}
    with (OUT/'supervisor.log').open('w') as log:
        code=("from MoonSim.tasks.task1_collect.run import main; "
              f"main(engine='both',robot={args.robot!r},output={str(OUT)!r},wall_seconds=60,report={str(OUT/'process.json')!r})")
        p=subprocess.Popen([sys.executable,'-c',code],cwd=ROOT.parent,stdout=log,stderr=log,
                           env={**os.environ,'PYTHONPATH':'','OPENBLAS_NUM_THREADS':'1'})
        def wait(predicate,timeout=150):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                v=read('latency_current.json')
                if v.get('wall',0)>report['started_wall'] and predicate(v):return v
                if p.poll() is not None:raise RuntimeError(f'TASK1 exited early: {p.returncode}; see {OUT}/supervisor.log')
                time.sleep(.2)
            raise TimeoutError('TASK1 GUI condition timed out')
        try:
            initial=wait(lambda v:v.get('frames',0)>20 and v.get('display_rocks')==read('scene.json').get('config',{}).get('rock',{}).get('count'))
            ues=[]
            for comm in Path('/proc').glob('[0-9]*/comm'):
                try:
                    if comm.read_text().strip()=='UnrealEditor' and int((comm.parent/'stat').read_text().split(')')[1].split()[1])==p.pid:
                        ues.append(int(comm.parent.name))
                except OSError:pass
            assert len(ues)==1,ues
            def camera():return read('viewport_last_state.json')['camera']
            before=camera();focus_and_walk(ues[0],drag_button=3)
            ue=wait(lambda v:v.get('camera_edits_sent',0)>initial.get('camera_edits_sent',0) and v.get('camera_ack',0)>=v['camera_edits_sent'])
            time.sleep(2);after=camera();assert math.dist(before[3:9],after[3:9])>.01
            report['ue_mouse']=ue
            start=time.monotonic();before=after;focus_and_walk(p.pid,drag_button=1)
            mj=wait(lambda v:v['wall']>start+4)
            time.sleep(2);after=camera();assert math.dist(before[3:9],after[3:9])>.01
            report['mujoco_mouse']=mj
            for v in (ue,mj):
                assert v['camera_robot_offset_error_m']<1e-5
                assert v['camera_projection_error_at_1280x720_px']<.01
                assert v['actor_position_error_cm']<1e-3 and v['display_rocks']==read('scene.json')['config']['rock']['count']
            report['mujoco_mouse']=mj;report['runtime']=read('runtime_metrics.json')
            subprocess.run(['import','-window','root',str(OUT/'windows.png')],check=True,timeout=15)
            focus_and_walk(ues[0],drag_button=0)
            subprocess.run(['import','-window','root',str(OUT/'ue_view.png')],check=True,timeout=15)
            report['passed']=True
            (OUT/'camera_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        finally:
            (OUT/'camera_validation.json').write_text(json.dumps(report,indent=2))
            if p.poll() is None:
                p.send_signal(__import__('signal').SIGINT)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)

if __name__=='__main__':main()
