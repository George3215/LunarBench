"""Authorized single-GPU action-interface experiment with auditable backups."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_task3_candidates_queue import push_backup, read, save

REPO=Path(__file__).resolve().parents[1]
KEYS=('spatialreasoner','spatial_interactor8b','spatialrgpt','qwen3vl8b','robobrain25_8b_nv')
BRANCH='experiments/task3-spatial-transfer-20261018'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--push',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    status_file=root/'QUEUE_STATUS.json'
    if status_file.exists():
        raise RuntimeError('Queue evidence exists; use a fresh root or review exact resume')
    state=dict(pid=os.getpid(),phase='preflight',started_unix=time.time(),models={},backups=[])
    save(status_file,state)
    env=dict(os.environ,MUJOCO_GL='egl',MPLCONFIGDIR='/tmp/task3_mpl_cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

    def analyse(plots=False):
        subprocess.run([sys.executable,str(REPO/'tools/analyse_task3_action_only.py'),'--root',str(root)]+(['--plots'] if plots else []),cwd=REPO,env=env,check=True)

    def backup(reason,plots=False):
        analyse(plots)
        if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!=BRANCH:
            raise RuntimeError('Backup branch changed')
        paths=[str(root.relative_to(REPO)), 'tools/task3_action_only.py',
            'tools/task3_candidate_driver.py','tools/task3_spatialrgpt_worker.py',
            'tools/analyse_task3_action_only.py','tools/run_task3_action_only_queue.py']
        subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
        subprocess.run(['git','diff','--cached','--check','--',*paths],cwd=REPO,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=REPO).returncode:
            subprocess.run(['git','commit','-m','TASK3 action-only: '+reason,'--only','--',*paths],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        pushed,error=push_backup(BRANCH) if args.push else (False,None)
        state['backups'].append(dict(reason=reason,commit=commit,pushed=pushed,push_error=error))
        save(status_file,state)

    def stop_requested():
        return (root/'STOP').exists()

    def wait_gpu():
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            if stop_requested():return False
            state.update(phase='waiting_for_gpu',updated_unix=time.time());save(status_file,state);time.sleep(5)
        return not stop_requested()

    def run(key,smoke):
        if not wait_gpu():return None
        out=root/('smoke' if smoke else 'runs')/key
        if out.exists():raise RuntimeError(f'Existing output must be reviewed: {out}')
        backend='spatialrgpt' if key=='spatialrgpt' else 'native_qwen'
        processor='qwen3vl8b' if key=='spatial_interactor8b' else key
        cmd=[sys.executable,str(REPO/'tools/task3_candidate_driver.py'),'--backend',backend,
            '--model',str(REPO/'.local/models'/key),'--processor',str(REPO/'.local/models'/processor),
            '--action-interface','action_only','--max-new-tokens','16','--out',str(out)]
        if smoke:cmd+=['--action-budget','4','--terrains','terrain_design_v4/barrier_a0','terrain_design_v4/pit_a0']
        state.update(phase=('smoke_' if smoke else 'running_')+key,current={})
        state['models'][key]={'state':'smoke_running' if smoke else 'running'};save(status_file,state)
        with (root/(key+('_smoke.log' if smoke else '_full.log'))).open('w') as log:
            child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['current_pid']=child.pid;save(status_file,state)
            milestone=0
            while child.poll() is None:
                if stop_requested() and out.exists():(out/'STOP').touch()
                current=read(out/'RUN_STATUS.json')
                state.update(current=current,updated_unix=time.time());save(status_file,state)
                completed=current.get('completed',0)
                if not smoke and completed//4>milestone:
                    milestone=completed//4;backup(f'{key} {completed}/22')
                time.sleep(5)
        result=read(out/'RUN_STATUS.json')
        state['current']=result
        return dict(exit_code=child.returncode,status=result)

    try:
        backup('predeclared protocol and interface checks')
        eligible=[]
        for key in KEYS:
            response=run(key,True)
            if response is None:break
            results=[read(p) for p in (root/'smoke'/key).glob('rgbd/*/trials/*/result.json')]
            passed=response['exit_code']==0 and len(results)==2 and all(r['autonomous_actions']>0 for r in results) and not any(r['termination'] in ('invalid_model_output','context_limit','gpu_memory_limit','repeated_forbidden_stop') for r in results)
            audit_subprocess=[sys.executable,str(REPO/'tools/analyse_task3_action_only.py'),'--root',str(root)]
            subprocess.run(audit_subprocess,cwd=REPO,env=env,check=True,stdout=subprocess.DEVNULL)
            smoke_audit=read(root/'metrics.json')['smoke_not_formal_SR'][key]['input_audit']
            passed=passed and not smoke_audit['forbidden_field_or_role_violations']
            state['models'][key]=dict(state='smoke_passed_queued' if passed else 'smoke_requires_review',smoke=response)
            save(status_file,state);backup(key+' action-only interface diagnostic')
            if passed:eligible.append(key)
            if stop_requested():break
        # A fresh Qwen baseline comes first; never rank against a changed prompt alone.
        order=[key for key in ('qwen3vl8b','spatialreasoner','spatial_interactor8b','spatialrgpt','robobrain25_8b_nv') if key in eligible]
        for key in order:
            response=run(key,False)
            if response is None:break
            complete=response['exit_code']==0 and response['status'].get('completed')==22
            state['models'][key]=dict(state='complete' if complete else 'incomplete_requires_review',formal=response)
            save(status_file,state);backup(key+' final',plots=True)
            if stop_requested():break
        state.update(phase='paused_at_action_boundary' if stop_requested() else 'all_scheduled_models_processed',finished_unix=time.time())
        save(status_file,state);backup('queue final analysis',plots=True)
    except BaseException as exc:
        state.update(phase='stopped_with_error',error=repr(exc),updated_unix=time.time());save(status_file,state);raise


if __name__=='__main__':main()
