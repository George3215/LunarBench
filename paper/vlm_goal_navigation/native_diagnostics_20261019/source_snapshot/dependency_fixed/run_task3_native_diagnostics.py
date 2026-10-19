"""Sequential, resumable one-GPU diagnostic queue; never retries errors silently."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'paper/vlm_goal_navigation/native_diagnostics_20261019'
PY='/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python'
MODELS=['robobrain25_8b_nv','spatialreasoner','spatialrgpt','qwen3vl8b','spatial_interactor8b']


def main():
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             MPLCONFIGDIR='/tmp/task3_native_mpl',PYTHONUNBUFFERED='1')
    state={'phase':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':{}}
    def save():
        (ROOT/'QUEUE_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    for model in MODELS:
        state['active_model']=model;state['models'][model]={'phase':'running'};save()
        python=str(REPO/'.local/envs/spatialrgpt/bin/python') if model=='spatialrgpt' else PY
        with (ROOT/f'{model}.log').open('a') as log:
            code=subprocess.call([python,'-u','tools/task3_native_diagnostics.py','run','--model',model],
                                 cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['models'][model]={'phase':'complete' if code==0 else 'error','exit_code':code};save()
        subprocess.run([PY,'tools/task3_native_diagnostics.py','analyse'],cwd=REPO,env=env,
                       stdout=subprocess.DEVNULL,check=True)
    state['phase']='complete' if all(x['exit_code']==0 for x in state['models'].values()) else 'complete_with_errors'
    state['active_model']=None;state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
