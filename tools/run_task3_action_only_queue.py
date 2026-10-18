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
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--only',nargs='+',choices=KEYS)
    parser.add_argument('--retry-incomplete-smoke',action='store_true')
    parser.add_argument('--after-queue-pid',type=int)
    args=parser.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    status_file=root/'QUEUE_STATUS.json'
    if args.after_queue_pid:
        pending=root/'FOLLOWUP_STATUS.json'
        while Path(f'/proc/{args.after_queue_pid}').exists():
            try:
                if Path(f'/proc/{args.after_queue_pid}/stat').read_text().split()[2]=='Z':break
            except FileNotFoundError:
                break
            if (root/'STOP').exists():
                save(pending,dict(pid=os.getpid(),phase='cancelled_by_queue_STOP'));return
            save(pending,dict(pid=os.getpid(),phase='waiting_for_previous_queue',previous_pid=args.after_queue_pid,only=args.only,updated_unix=time.time()))
            time.sleep(5)
        if read(status_file).get('phase')!='all_scheduled_models_processed':
            save(pending,dict(pid=os.getpid(),phase='preceding_queue_requires_review'));return
        save(pending,dict(pid=os.getpid(),phase='starting_followup',only=args.only))
    previous={}
    if status_file.exists():
        previous=read(status_file)
        if not args.resume or previous.get('phase') not in ('stopped_with_error','paused_at_action_boundary','all_scheduled_models_processed'):
            raise RuntimeError('Existing queue is not explicitly resumable')
        try:
            os.kill(previous['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Previous queue process is still alive')
        history=root/'queue_history';history.mkdir(exist_ok=True)
        save(history/f"status_before_resume_{time.time_ns()}.json",previous)
    state=dict(pid=os.getpid(),phase='preflight',started_unix=time.time(),
        models=previous.get('models',{}),backups=previous.get('backups',[]),only=args.only)
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
        # Preserve inherited source snapshots byte-for-byte, including upstream
        # whitespace; check our maintained code/docs and derived records normally.
        exclude=':(exclude)'+str(root.relative_to(REPO))+'/**/source_snapshot/**'
        subprocess.run(['git','diff','--cached','--check','--',*paths,exclude],cwd=REPO,check=True)
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
        if out.exists():
            completed=read(out/'RUN_STATUS.json')
            expected=2 if smoke else 22
            if args.resume and completed.get('status')=='complete_pending_analysis' and completed.get('completed')==expected:
                return dict(exit_code=0,status=completed,reused_completed_run=True)
            if smoke and args.retry_incomplete_smoke and args.resume:
                history=root/'interface_failures';history.mkdir(exist_ok=True)
                out.rename(history/f'{key}_before_retry_{time.time_ns()}')
                log=root/(key+'_smoke.log')
                if log.exists():log.rename(history/f'{key}_before_retry_{time.time_ns()}.log')
            else:
                raise RuntimeError(f'Existing incomplete output must be reviewed: {out}')
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
            if args.only and key not in args.only:continue
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
