"""Complete -> classify -> materialise relief/cube revision -> fully evaluate.
Never tunes terrain from incomplete success rates. At most two finite revisions.
"""
import argparse,json,os,shutil,subprocess,sys,time,traceback,hashlib,fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from compact_trial_browser import build,physical_gallery

PYTHON='/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python'
PLOT_PYTHON='/home/lry/miniconda3/bin/python3'
ROOT=Path('/home/lry/MoonUnrealEnv/MoonSim')

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def finalize_frozen(root,analysis):
    with (root/'finalization.log').open('w') as log:
        result=subprocess.run([PYTHON,str(analysis/'finalize_difficulty.py'),str(root)],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Full-suite finalization failed: {root}/finalization.log')
    return read(root/'evaluation_metrics.json')
def freeze(workflow):
    target=workflow/'frozen/MoonSim';tasks=target/'tasks/task3_reobservability';tasks.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'tasks/task3_reobservability').glob('*.py'):shutil.copy2(p,tasks/p.name)
    generated=tasks/'generated';generated.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'tasks/task3_reobservability/generated/history_pairs.json',generated/'history_pairs.json')
    for directory in ('assets','third_party','paper'):
        link=target/directory
        if not link.exists():link.symlink_to(ROOT/directory,target_is_directory=True)
    analysis=workflow/'analysis_snapshot';analysis.mkdir(exist_ok=True)
    for name in ('analyse_all_terrain_context.py','watch_all_terrain_context.py','depth_archive.py','compact_trial_browser.py','finalize_difficulty.py','analyse_task_quality.py','plot_all_terrain_context.py','build_scene_review.py','calibrate_task3_suite.py'):
        shutil.copy2(ROOT/'paper/vlm_goal_navigation'/name,analysis/name)
    paths=list(tasks.glob('*.py'))+list(analysis.glob('*.py'))+[generated/'history_pairs.json']
    write(workflow/'source_hashes.json',{str(p.relative_to(workflow)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return tasks,analysis

def process_running(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def await_complete(root,workflow,phase):
    last_signature=None;last_print=0.
    while True:
        status=read(root/'RUN_STATUS.json',{});analysis=read(root/'analysis.json',{})
        signature=((root/'analysis.json').stat().st_mtime_ns if (root/'analysis.json').exists() else None)
        if signature!=last_signature and analysis:
            build(root);physical_gallery(root);last_signature=signature
        value={'phase':phase,'updated_at':stamp(),'current_run':str(root),'completed':status.get('completed',0),'expected':status.get('expected',186),
            'active_trials':status.get('active_trials',{}),'final_success_rate_published':False,'terrain_revised_from_partial_results':False}
        write(workflow/'WORKFLOW_STATUS.json',value)
        if time.monotonic()-last_print>120:
            print('WAIT',phase,value['completed'],value['expected'],value['active_trials'],flush=True);last_print=time.monotonic()
        if status.get('status')=='complete' and analysis.get('all_complete'):
            pid=read(root/'processes.json',{}).get('engine_pid')
            if not pid or not process_running(pid):return
        if status.get('status') in ('audit_failed','interrupted') or (root/'ERROR.json').exists():
            raise RuntimeError(f'Cannot calibrate incomplete or failed suite: {root}; status={status.get("status")}')
        time.sleep(30)

def revision_level(level,direction):
    levels=['intro','easy','medium','hard','very_hard'];index=levels.index(level)
    if direction=='lower':return levels[max(0,index-1)]
    if direction=='higher':return levels[min(len(levels)-1,index+1)]
    return level

def run_revision(root,validation,config_path,tasks,analysis,workflow):
    env=dict(os.environ,MUJOCO_GL='egl',PYTHONPATH='',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/task3_matplotlib')
    validation.mkdir(parents=True,exist_ok=True)
    write(workflow/'WORKFLOW_STATUS.json',{'phase':'validating_revised_terrains','updated_at':stamp(),'validation':str(validation),'configuration':read(config_path)})
    with (validation/'validation.log').open('w') as log:
        result=subprocess.run([PYTHON,str(tasks/'validate_all_terrains.py'),'--out',str(validation),'--terrain-config',str(config_path)],env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Revised physical validation exited {result.returncode}; see {validation}')
    gate=read(validation/'STATUS.json',{});setups=read(validation/'manifest.json',[])
    if gate.get('accepted')!=31 or len(setups)!=31:raise RuntimeError(f'Revised terrain eligibility incomplete: {gate.get("accepted")}/31; no partial suite started')
    if any(s['added_occluder'] and s['added_occluder'].get('kind')!='cube' for s in setups):raise RuntimeError('Non-cube added occluder rejected')
    subprocess.run([PLOT_PYTHON,str(analysis/'build_scene_review.py'),str(validation)],env=env,check=True)
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(analysis,root/'analysis_snapshot')
    with (root/'run.log').open('w') as log,(root/'analysis_watch.log').open('w') as watchlog,(root/'plot_watch.log').open('w') as plotlog:
        engine=subprocess.Popen([PYTHON,str(tasks/'all_terrain_context_trial.py'),'--out',str(root),'--validation',str(validation),'--terrain-config',str(config_path)],env=env,stdout=log,stderr=subprocess.STDOUT)
        watcher=subprocess.Popen([PYTHON,str(root/'analysis_snapshot/watch_all_terrain_context.py'),str(root)],env=env,stdout=watchlog,stderr=subprocess.STDOUT)
        plotter=subprocess.Popen([PLOT_PYTHON,str(root/'analysis_snapshot/plot_all_terrain_context.py'),str(root),'--watch'],env=env,stdout=plotlog,stderr=subprocess.STDOUT)
        write(root/'orchestration_processes.json',{'engine_pid':engine.pid,'watcher_pid':watcher.pid,'plotter_pid':plotter.pid,'calibration_pid':os.getpid()})
        try:
            await_complete(root,workflow,'revised_suite_running')
            if engine.wait()!=0:raise RuntimeError('Revised trial engine failed')
            if watcher.wait()!=0:raise RuntimeError('Revised analysis watcher failed')
            plotter.wait()
        except Exception:
            # Preserve all data; no automatic repeated trial attempts.
            if engine.poll() is not None:
                for p in (watcher,plotter):
                    if p.poll() is None:p.terminate()
            raise
    return finalize_frozen(root,analysis)

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--workflow',type=Path,required=True);p.add_argument('--max-revisions',type=int,default=2);p.add_argument('--resume-wait',action='store_true');a=p.parse_args()
    if not 1<=a.max_revisions<=2:raise RuntimeError('At most two adaptive revisions')
    lock_stream=(ROOT/'paper/vlm_goal_navigation/.calibration.lock').open('a+')
    try:fcntl.flock(lock_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A calibration workflow is already active; no duplicate GPU suite')
    workflow=a.workflow.resolve();workflow.mkdir(parents=True,exist_ok=True)
    if (workflow/'process.json').exists():
        previous=read(workflow/'process.json');status=read(workflow/'WORKFLOW_STATUS.json',{})
        if not a.resume_wait or process_running(previous['pid']) or status.get('phase')!='baseline_suite_running' or list(workflow.glob('terrain_config_v*.json')):
            raise RuntimeError('Existing workflow; only an inactive baseline-wait process may be explicitly resumed, no duplicate trial attempts')
        history=read(workflow/'process_history.json',[]);history.append(previous);write(workflow/'process_history.json',history)
    write(workflow/'process.json',{'pid':os.getpid(),'started_at':stamp(),'baseline':str(a.baseline.resolve()),'max_revisions':a.max_revisions})
    tasks,analysis=freeze(workflow);runs=[]
    try:
        baseline=a.baseline.resolve();await_complete(baseline,workflow,'baseline_suite_running')
        value=finalize_frozen(baseline,analysis);build(baseline);physical_gallery(baseline);runs.append({'run':str(baseline),'metrics':value})
        decision=value['calibration_decision'];level='easy' if decision['direction']=='lower' else 'hard' if decision['direction']=='higher' else 'medium'
        for revision in range(1,a.max_revisions+1):
            # Difficulty labels are saved before ANY new terrain is constructed.
            write(workflow/f'classification_before_revision_{revision}.json',{'previous_run':runs[-1]['run'],'decision':decision,'classification':read(Path(runs[-1]['run'])/'difficulty_classification.json')})
            config={'version':revision+1,'level':level,'seed':23,'max_native_candidates':24,
                'native_relief_priority':True,'added_occluder_type':'cube','walls_allowed_only_for_window_structure':True,
                'loss_requirement':'entity_line_of_sight','calibration_source':runs[-1]['run'],'calibration_decision':decision,
                'max_actions_per_trial':3000,'conditions':['current','first_current','all_history'],'sensor_conditions':['rgb','rgbd'],
                'no_prompt_or_image_resolution_change_during_geometry_calibration':True}
            config_path=workflow/f'terrain_config_v{revision+1}.json';write(config_path,config)
            validation=workflow/f'validation_v{revision+1}_{level}';run=ROOT/'paper/vlm_goal_navigation/runs'/f'{workflow.name}_v{revision+1}_{level}'
            value=run_revision(run,validation,config_path,tasks,analysis,workflow);build(run);physical_gallery(run);runs.append({'run':str(run),'metrics':value});decision=value['calibration_decision']
            if decision['direction'] not in ('lower','higher'):break
            next_level=revision_level(level,decision['direction'])
            if next_level==level:break
            level=next_level
        write(workflow/'FINAL_RESULTS.json',{'status':'complete','completed_at':stamp(),'runs':runs,'final_calibration':decision,'finite_revision_limit':a.max_revisions})
        write(workflow/'WORKFLOW_STATUS.json',{'phase':'complete','updated_at':stamp(),'completed_runs':len(runs),'final_run':runs[-1]['run'],'final_success_rate':runs[-1]['metrics']['overall']['success_rate'],
            'final_calibration_direction':decision['direction'],'all_runs_completed_and_audited':True,'unbounded_adaptation':False})
        lines=['# Task3 完整评测与难度校准','', '|版本|成功/总数|成功率|地形等权成功率|下一难度决策|','|---|---:|---:|---:|---|']
        for item in runs:
            m=item['metrics'];o=m['overall'];lines.append(f"|{Path(item['run']).name}|{o['success_count']}/{o['trial_count']}|{100*o['success_rate']:.2f}%|{100*m['family_balanced_success_rate']:.2f}%|{m['calibration_decision']['direction']}|")
        lines+=['','全部指标与每轮31场景的难度划分见对应目录。每轮186trial均完成并通过审计，版本之间用于任务设计校准，不能作为独立泛化测试。','', '如果最后仍在5%–40%之外，记录校准边界，不无限增加版本或重复trial。']
        (workflow/'FINAL_REPORT.md').write_text('\n'.join(lines));print('WORKFLOW_COMPLETE',flush=True)
    except Exception as exc:
        write(workflow/'WORKFLOW_ERROR.json',{'error':str(exc),'type':type(exc).__name__,'at':stamp(),'completed_runs':len(runs),'no_unbounded_retry':True})
        write(workflow/'WORKFLOW_STATUS.json',{'phase':'error','updated_at':stamp(),'error':str(exc),'completed_runs':len(runs)})
        traceback.print_exc();raise
if __name__=='__main__':main()
