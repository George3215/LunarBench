"""Run eligible local models now, or optionally wait for a prerequisite run."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import requests

WORKSPACE = Path(__file__).resolve().parents[3]
TASK = WORKSPACE/'MoonSim/tasks/task3_reobservability/v4_camera_model_trial.py'
PYTHON = '/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python'
SERVER = WORKSPACE/'third_party/navigation_inference/camera_server.py'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--after',type=Path)
    p.add_argument('--batch-size',type=int,default=6)
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.batch_size<1:p.error('batch-size must be positive')
    plan=json.loads((root/'models.json').read_text())
    status={'phase':'waiting_for_sol61' if a.after else 'starting_local_models',
            'prerequisite':str(a.after.resolve()) if a.after else None,'request_count_limit':None,
            'pid':os.getpid(),'models':{v['id']:{'state':v['state'],'reason':v.get('reason')} for v in plan}}
    while a.after is not None:
        if (root/'STOP').exists():
            status['phase']='stopped_before_local_models';save(root/'STATUS.json',status);return
        try:
            prior=json.loads(a.after.read_text())
            status['sol61_completed']=prior['completed'];status['sol61_expected']=prior['expected']
            if prior['status']=='complete' and prior['completed']==prior['expected']==132:break
            if prior['status']=='interrupted_incomplete':
                status['phase']='waiting_for_sol61_recovery'
        except (FileNotFoundError,json.JSONDecodeError):pass
        save(root/'STATUS.json',status);time.sleep(30)
    env=os.environ.copy()
    env.update(PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',
               HF_HOME=str(WORKSPACE/'.hf-home'),XDG_CACHE_HOME=str(WORKSPACE/'.cache'),
               TRITON_CACHE_DIR=str(WORKSPACE/'.cache/triton'),
               TRITON_AUTOTUNE_DIR=str(WORKSPACE/'.cache/triton/autotune'),HF_HUB_OFFLINE='1',
               NAVILA_SINGLE_GPU_INFERENCE='1',NO_PROXY='127.0.0.1,localhost')
    # Match the persisted Sol camera observations; CUDA model inference remains available.
    env.update(__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
               MUJOCO_EGL_DEVICE_ID='2',LIBGL_ALWAYS_SOFTWARE='1')
    for item in plan:
        if item['state']!='queued':continue
        if (root/'STOP').exists():
            status['phase']='stopped_between_models';save(root/'STATUS.json',status);return
        model_root=root/item['id'];model_root.mkdir(exist_ok=True)
        server=None
        try:
            status['phase']='loading_model';status['active_model']=item['id']
            status['models'][item['id']]={'state':'loading'};save(root/'STATUS.json',status)
            command=[item['python'],str(SERVER),'--family',item['family'],'--repo',item['repo'],
                     '--checkpoint',item['checkpoint'],'--name',item['id'],'--port','8101']
            if item.get('vision'):command+=['--vision',item['vision']]
            with (model_root/'server.log').open('a') as log:
                server=subprocess.Popen(command,cwd=WORKSPACE,env=env,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+600
            while True:
                if server.poll() is not None:raise RuntimeError(f'Model server exited: {server.returncode}; see server.log')
                if time.monotonic()>deadline:raise RuntimeError('Model server readiness timeout')
                try:
                    health=requests.get('http://127.0.0.1:8101/health',timeout=2).json()
                    if health.get('ready') and health.get('model')==item['id']:break
                except (requests.RequestException,ValueError):pass
                time.sleep(3)
            for protocol in item['protocols']:
                out=model_root/protocol
                status['phase']='running';status['active_protocol']=protocol
                status['models'][item['id']]={'state':'running','protocol':protocol};save(root/'STATUS.json',status)
                cmd=[PYTHON,str(TASK),'--out',str(out),'--model',item['id'],'--family',item['family'],'--protocol',protocol,'--batch-size',str(a.batch_size)]
                with (model_root/(protocol+'.log')).open('a') as log:
                    process=subprocess.Popen(cmd,cwd=WORKSPACE,env=env,stdout=log,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        if (root/'STOP').exists():
                            out.mkdir(exist_ok=True);(out/'STOP').touch()
                        try:
                            status['active_trial_status']=json.loads((out/'RUN_STATUS.json').read_text())
                            save(root/'STATUS.json',status)
                        except (FileNotFoundError,ValueError):pass
                        time.sleep(5)
                if process.returncode:raise RuntimeError(f'{protocol} runner exited {process.returncode}; see {protocol}.log')
                done=json.loads((out/'RUN_STATUS.json').read_text())
                if done['status']!='complete' or done['completed']!=done['expected']:
                    raise RuntimeError('Runner exited without completing all planned episodes')
            status['models'][item['id']]={'state':'complete','protocols':item['protocols']}
        except Exception as exc:
            status['models'][item['id']]={'state':'incomplete_error','reason':str(exc)}
            save(model_root/'ISSUE.json',status['models'][item['id']])
        finally:
            if server and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=20)
                except subprocess.TimeoutExpired:server.kill();server.wait()
            save(root/'STATUS.json',status)
    status['phase']='queue_finished'
    status.pop('active_model',None);status.pop('active_protocol',None);status.pop('active_trial_status',None)
    save(root/'STATUS.json',status)


if __name__=='__main__':main()
