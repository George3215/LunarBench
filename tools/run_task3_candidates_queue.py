"""Continue authorized additional checkpoint tests on one GPU; persist milestones."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--push',action='store_true')
    parser.add_argument('--only',nargs='+',choices=['spatialreasoner','spatialrgpt','robobrain25_8b_nv'])
    parser.add_argument('--after-status',type=Path)
    parser.add_argument('--status-name',default='QUEUE_STATUS.json')
    parser.add_argument('--continue-format-failed-smoke',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    old = REPO/'paper/vlm_goal_navigation/spatial_transfer_v2_20261018'
    state = dict(pid=os.getpid(),phase='waiting_for_existing_frozen_run',
        models={},backups=[],robobrain_source='official ModelScope; same NV weight SHA256 as HF',
        only=args.only,continue_format_failed_smoke=args.continue_format_failed_smoke)
    status_file = root/args.status_name
    save(status_file,state)
    env=dict(os.environ,MUJOCO_GL='egl',MPLCONFIGDIR='/tmp/task3_mpl_cache',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    branch='experiments/task3-spatial-transfer-20261018'

    def stopped():
        if (root/'STOP').exists():
            raise RuntimeError('queue STOP; no next experiment launched')

    def analyse(plots=False):
        cmd=[sys.executable,str(REPO/'tools/analyse_task3_candidates.py'),'--root',str(root)]
        subprocess.run(cmd+(['--plots'] if plots else []),cwd=REPO,env=env,check=True)

    def backup(reason):
        analyse()
        paths=[str(root.relative_to(REPO)),'tools/fetch_task3_extra_models.py',
            'tools/task3_candidate_driver.py','tools/task3_spatialrgpt_worker.py',
            'tools/analyse_task3_candidates.py','tools/run_task3_candidates_queue.py']
        if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!=branch:
            raise RuntimeError('backup branch changed')
        subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
        subprocess.run(['git','diff','--cached','--check','--',*paths],cwd=REPO,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=REPO).returncode:
            subprocess.run(['git','commit','-m','Record TASK3 candidates: '+reason,'--only','--',*paths],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        if args.push:
            subprocess.run(['git','-c','http.proxy=http://127.0.0.1:7890','push','origin',branch],cwd=REPO,check=True,timeout=120)
        state['backups'].append(dict(reason=reason,commit=commit,pushed=args.push))
        save(status_file,state)

    def wait_gpu():
        while True:
            stopped()
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not processes:
                return
            state['phase']='waiting_for_gpu_owner_exit'
            save(status_file,state)
            time.sleep(10)

    try:
        if args.after_status:
            state['phase']='waiting_for_preceding_candidate_queue'
            save(status_file,state)
            while read(args.after_status).get('phase')!='all_candidates_processed':
                stopped()
                if read(args.after_status).get('phase')=='stopped_with_error':
                    raise RuntimeError('preceding candidate queue requires review')
                time.sleep(10)
        while read(old/'QUEUE_STATUS.json').get('phase')!='complete':
            stopped()
            phase=read(old/'QUEUE_STATUS.json').get('phase')
            if phase=='stopped_with_error':
                raise RuntimeError('existing frozen queue requires review before using GPU')
            time.sleep(10)
        candidates = []
        for key,backend in [('spatialreasoner','native_qwen'),('spatialrgpt','spatialrgpt'),('robobrain25_8b_nv','native_qwen')]:
            if args.only and key not in args.only:
                continue
            while read(root/(key+'_download.json')).get('status')!='complete':
                stopped()
                state['phase']='waiting_for_verified_download_'+key
                save(status_file,state)
                if read(root/(key+'_download.json')).get('status')=='blocked_or_error':
                    break
                time.sleep(10)
            if read(root/(key+'_download.json')).get('status')!='complete':
                state['models'][key]={'state':'download_error_not_evaluated'}
                continue
            wait_gpu()
            state['phase']='smoke_'+key
            save(status_file,state)
            command=[sys.executable,str(REPO/'tools/task3_candidate_driver.py'),'--backend',backend,
                '--model',str(REPO/'.local/models'/key),'--processor',str(REPO/'.local/models'/key)]
            smoke=root/'smoke'/key
            if not (smoke/'RUN_STATUS.json').exists():
                with (root/(key+'_smoke.log')).open('w') as log:
                    rc=subprocess.run(command+['--out',str(smoke),'--action-budget','2','--terrains',
                        'terrain_design_v4/barrier_a0','terrain_design_v4/pit_a0'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            else:
                rc=0
            results=[read(p) for p in smoke.glob('rgbd/*/trials/*/result.json')]
            format_failed=not any(r['autonomous_actions'] for r in results)
            if rc or len(results)!=2 or (format_failed and not args.continue_format_failed_smoke) or any(r['termination'] in ('context_limit','gpu_memory_limit','repeated_forbidden_stop') for r in results):
                state['models'][key]={'state':'smoke_requires_review','completed_smoke_cases':len(results),'exit_code':rc}
                save(status_file,state)
                backup(key+' interface diagnostic')
                continue
            state['models'][key]={'state':'format_failed_formal_diagnostic_queued' if format_failed else 'smoke_passed_queued','completed_smoke_cases':len(results)}
            save(status_file,state)
            backup(key+(' format failure retained; formal diagnostic queued' if format_failed else ' interface passed'))
            candidates.append((key,command))
        for key,command in candidates:
            wait_gpu()
            run=root/'runs'/key
            state['phase']='running_'+key
            state['models'][key]={'state':'running'}
            save(status_file,state)
            with (root/(key+'_full.log')).open('w') as log:
                child=subprocess.Popen(command+['--out',str(run)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['current_pid']=child.pid
                milestone=-1
                while child.poll() is None:
                    if (root/'STOP').exists():
                        run.mkdir(exist_ok=True)
                        (run/'STOP').touch()
                    current=read(run/'RUN_STATUS.json')
                    state['current']=current
                    state['updated_unix']=time.time()
                    save(status_file,state)
                    completed=current.get('completed',0)
                    if completed and completed//4>milestone:
                        milestone=completed//4
                        backup(f'{key} {completed} of 22')
                    time.sleep(10)
            status=read(run/'RUN_STATUS.json')
            state['models'][key]={'state':'complete' if status.get('completed')==22 else 'incomplete_requires_review','exit_code':child.returncode,'status':status}
            save(status_file,state)
            backup(key+' final status')
        analyse(plots=True)
        state['phase']='all_candidates_processed'
        save(status_file,state)
        backup('available-model evaluation and related work')
    except BaseException as exc:
        state.update(phase='stopped_with_error',error=repr(exc))
        save(status_file,state)
        raise


if __name__=='__main__':
    main()
