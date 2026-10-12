"""All original terrain scenes x three exact image histories x RGB/RGB-D."""
import argparse
import collections
import json
import math
import os
import shutil
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from scenes import Scene
import navigation_trial as nt
from visible_start_trial import actions,sensor_packet
from all_terrain_setup import originals,place
from validate_all_terrains import Approach,loss_kind
from context_navigation_policy import Policy,HORIZONS,MODES,SYSTEM,request_for,INPUT_TOKEN_LIMIT,OUTPUT_TOKEN_LIMIT
from vlm_benchmark import dump,sha

VALIDATION=nt.ROOT/'paper/vlm_goal_navigation/all_terrain_validation'
# Autonomous VLM action budget per trial. Default unchanged at 3000 for all
# previously recorded runs; callers may lower it before constructing trials.
MAX_STEPS=3000


class Trial:
    def __init__(self,setup,terrain,out,mode,horizon):
        self.setup=setup;self.mode=mode;self.horizon=horizon
        self.scene=Scene(**setup['adjusted_scene']);self.name=self.scene.name
        self.key=f'{mode}/{horizon}/{self.name}'
        self.folder=out/mode/horizon/'trials'/self.name
        if self.folder.exists():raise RuntimeError(f'Existing trial: {self.folder}')
        self.folder.mkdir(parents=True)
        self.env=nt.api.MuJoCoEnvironment(self.scene,terrain=terrain)
        self.cam=nt.EvaluatorCamera(self.env,render_size=(640,480));place(self.env)
        self.sf=(self.folder/'state.jsonl').open('w');self.ef=(self.folder/'evaluator.jsonl').open('w');self.tf=(self.folder/'trace.jsonl').open('w')
        self.physics=Approach(self.env,self.cam,self.sf,self.ef,self.folder)
        self.frames=[];self.executed=[];self.step=0;self.inferences=0;self.inference_latency=0.;self.amortized=0.;self.done=False;self.begin=time.perf_counter()
        self.physics.record(0,'RESET')
        for _ in range(6):
            if self.physics.execute('STOP',0):raise RuntimeError(f'Initial stand failed: {self.key}')
        initial=self.cam.read(self.env.model,self.env.data)
        if initial['target_visible_rgb_pixels']<40 or initial['near_and_visible']:raise RuntimeError(f'Initial visibility failed: {self.key}')
        frame=self.observe()
        dump(self.folder/'initialization.json',{'scene':vars(self.scene),'initial_evaluator':initial,
            'qpos':self.env.data.qpos.tolist(),'qvel':self.env.data.qvel.tolist(),
            'initial_depth_camera_range_m':frame['depth_sensor']['red_region_median_surface_range_m'],
            'initial_stand_s':1.8,'scene_setup':setup})
        dump(self.folder/'model_metadata.json',{'ngeom':self.env.model.ngeom,
            'geom_names':{str(i):mujoco.mj_id2name(self.env.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(self.env.model.ngeom)}})
        self.physics.start_xy=self.env.data.qpos[:2].copy();self.physics.enabled=True
        for _ in range(setup['approach_actions']):
            self.step+=1;before=self.cam.read(self.env.model,self.env.data);qpos=self.env.data.qpos.copy()
            terminal=self.physics.execute(setup['approach_action'],self.step)
            self.executed.append(setup['approach_action']);after=self.cam.read(self.env.model,self.env.data)
            frame=self.observe()
            self.write_transition({'step':self.step,'phase':'controlled_approach','action_source':'fixed task setup, not VLM',
                'action':setup['approach_action'],'before_evaluator':before,'after_evaluator':after,
                'actual_displacement_xy_m':float(np.linalg.norm(self.env.data.qpos[:2]-qpos[:2])),
                'frame_before':frame['frame_id']-1,'frame_after':frame['frame_id'],
                'decision':None,'model_output_raw':[],'inference_records':[],'inference_s':0.,'termination':None})
            if terminal:
                if terminal!='approach_loss':raise RuntimeError(f'Approach ended incorrectly: {self.key}: {terminal}')
                break
        after=self.cam.read(self.env.model,self.env.data);kind,geom=loss_kind(self.cam)
        if kind!='terrain_occlusion' or after['target_segment_pixels']>2 or not self.physics.loss:
            raise RuntimeError(f'Physical occlusion gate failed: {self.key}')
        self.prefix=self.step;self.physics.enabled=False;self.physics.extra_terminal=None
        dump(self.folder/'approach_gate.json',{'initial':initial,'lost':after,'loss_kind':kind,'blocking_geom':geom,
            'prefix_actions':self.prefix,'loss_qpos':self.env.data.qpos.tolist(),
            'initial_rgb_sha256':sha(Path(self.frames[0]['rgb_path'])),
            'lost_rgb_sha256':sha(Path(self.frames[-1]['rgb_path'])),
            'initial_depth_sha256':sha(Path(self.frames[0]['depth_path'])),
            'added_occluder':setup['added_occluder'],'original_terrain_geometry_unchanged':setup['original_terrain_geometry_unchanged']})

    def observe(self):
        self.cam.read(self.env.model,self.env.data)
        index=len(self.frames);folder=self.folder/'observations'/f'{index:05d}'
        self.cam.save_evidence(folder);sensor=sensor_packet(self.cam,folder)
        # The sensor helper duplicated the exact same float32 depth; keep one copy.
        duplicate=folder/'input_depth_m.npy'
        if not np.array_equal(np.load(duplicate),np.load(folder/'depth.npy')):raise RuntimeError('Unexpected depth copy mismatch')
        duplicate.unlink()
        frame={'frame_id':index,'sim_time_s':float(self.env.data.time),'rgb_path':str((folder/'rgb.png').resolve()),
            'depth_path':str((folder/'input_depth.png').resolve()),'depth_sensor':sensor}
        self.frames.append(frame);dump(self.folder/'observations.json',self.frames)
        return frame

    def write_transition(self,trace):
        folder=self.folder/'steps'/f"{trace['step']:04d}"
        dump(folder/'transition.json',trace);self.tf.write(json.dumps(trace)+'\n');self.tf.flush()

    def prepare(self):
        self.step+=1;self.step_folder=self.folder/'steps'/f'{self.step:04d}';self.step_folder.mkdir(parents=True)
        self.before=self.cam.read(self.env.model,self.env.data)
        self.frame_before=self.frames[-1]['frame_id'];self.calls=[]
        self.request=request_for(self.frames,self.horizon,self.mode)
        return self.request,self.step_folder/'inference_1',1

    def complete(self):
        record=self.calls[-1];parsed=record.get('parsed');action=parsed['action'] if parsed else None
        terminal=record.get('termination');qpos=self.env.data.qpos.copy();start=time.perf_counter()
        if terminal is None:
            if action is None:terminal='invalid_model_output'
            else:
                self.executed.append(action);terminal=self.physics.execute(action,self.step)
                if action=='STOP' and terminal is None:terminal='premature_stop'
        after=self.cam.read(self.env.model,self.env.data)
        if action is not None:frame=self.observe()
        else:frame=self.frames[-1]
        actual_calls=sum(not c.get('generation_skipped',False) for c in self.calls);self.inferences+=actual_calls
        latency=sum(c['elapsed_s'] for c in self.calls);self.inference_latency+=latency;self.amortized+=sum(c.get('batch_amortized_s',0.) for c in self.calls)
        if terminal is None and self.step>=MAX_STEPS:terminal='budget_exhausted'
        trace={'step':self.step,'phase':'vlm_autonomous','action_source':'VLM' if action else 'no executed action',
            'action':action,'decision':parsed,'frame_before':self.frame_before,'frame_after':frame['frame_id'],
            'model_output_raw':[c['raw_output'] for c in self.calls],
            'inference_records':[str((self.step_folder/f'inference_{i+1}'/'output.json').relative_to(self.folder)) for i in range(len(self.calls))],
            'context':json.loads((self.step_folder/'inference_1/context.json').read_text()),
            'actual_generation_calls':actual_calls,'inference_s':latency,'execution_wall_s':time.perf_counter()-start,
            'actual_displacement_xy_m':float(np.linalg.norm(self.env.data.qpos[:2]-qpos[:2])),
            'before_evaluator':self.before,'after_evaluator':after,'success_streak':self.physics.streak,'termination':terminal}
        self.write_transition(trace)
        dump(self.folder/'progress.json',{'step':self.step,'autonomous_actions':len(self.executed)-self.prefix,
            'evaluator':after,'termination':terminal,'context':trace['context']})
        print('STEP',self.key,self.step,action,'frames',trace['context']['input_frame_count'],
              'tokens',trace['context']['input_tokens'],'range',after['median_target_surface_range_m'],'terminal',terminal,flush=True)
        if self.step in (1000,2000) and terminal is None:dump(self.folder/f'budget_{self.step}.json',{'next_budget':self.step+1000})
        if terminal:self.finish(terminal,after)

    def finish(self,terminal,final=None):
        if self.done:return
        final=final or self.cam.read(self.env.model,self.env.data);self.cam.save_evidence(self.folder/'final_evidence')
        result={'scene':self.name,'mode':self.mode,'horizon':self.horizon,'success':terminal=='success_visible_near',
            'termination':terminal,'action_steps':len(self.executed),'controlled_approach_actions':self.prefix,
            'autonomous_actions':len(self.executed)-self.prefix,'inference_count':self.inferences,
            'total_inference_s':self.inference_latency,'amortized_inference_s':self.amortized,
            'wall_s_including_scheduler_wait':time.perf_counter()-self.begin,'simulated_s':float(self.env.data.time),
            'path_m':self.physics.path,'state_samples':self.physics.samples,'min_upright':self.physics.min_upright,
            'final_evaluator':final,'success_streak':self.physics.streak,
            'model_stopped':bool(self.executed and self.executed[-1]=='STOP'),
            'action_sequence':self.executed,'action_counts':dict(collections.Counter(self.executed)),
            'max_steps':MAX_STEPS,'observation_frame_count':len(self.frames),
            'resource_limit':terminal in ('context_limit','gpu_memory_limit','disk_space_limit')}
        if hasattr(self, 'wall_time_scope'):
            result['wall_time_scope'] = self.wall_time_scope
        dump(self.folder/'result.json',result);self.done=True;self.close();print('DONE',self.key,terminal,flush=True)

    def close(self):
        for stream in (self.sf,self.ef,self.tf):stream.close()
        self.cam.close();self.env.close()


def status(root,active,total,start,policy):
    results=[json.loads(p.read_text()) for p in root.glob('*/*/trials/*/result.json')]
    value={'status':'running','completed':len(results),'expected':total,'success_count':sum(r['success'] for r in results),
        'terminations':dict(collections.Counter(r['termination'] for r in results)),
        'active_trials':{t.key:t.step for t in active},'model_generate_calls':policy.generation_calls,
        'wall_s':time.perf_counter()-start,'max_actions_per_trial':MAX_STEPS}
    dump(root/'RUN_STATUS.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--validation',type=Path,default=VALIDATION);p.add_argument('--terrain-config',type=Path);a=p.parse_args()
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(7);actions()
    setups=json.loads((a.validation/'manifest.json').read_text())
    if a.terrain_config:
        import difficulty_terrain_setup
        source_catalogue=difficulty_terrain_setup.catalogue(json.loads(a.terrain_config.read_text()))
    else:source_catalogue=originals()
    if len(setups)!=len(source_catalogue) or set(s['scene_name'] for s in setups)!=set(source_catalogue):raise RuntimeError('All original terrain scenes must pass before trials')
    if not all(s['loss']['kind']=='terrain_occlusion' and s['reference_termination']=='success_visible_near' for s in setups):raise RuntimeError('Eligibility/solvability gate failed')
    terrains={}
    for setup in setups:
        terrain=source_catalogue[setup['scene_name']][1];terrain.spec.update(setup['terrain_spec']);terrains[setup['scene_name']]=terrain
    total=len(setups)*len(MODES)*len(HORIZONS)
    dump(root/'protocol.json',{'version':'relief-cube-context-depth/2' if a.terrain_config else 'all-original-terrains-context-depth/1','scenes':[s['scene_name'] for s in setups],
        'scene_count':len(setups),'trial_count':total,'sensor_conditions':MODES,'context_conditions':HORIZONS,
        'system_prompt':SYSTEM,'model':str(nt.MODEL),'model_max_context_tokens':262144,
        'runtime_input_token_limit':INPUT_TOKEN_LIMIT,'max_new_tokens':OUTPUT_TOKEN_LIMIT,'frame_preprocessing_area':65536,
        'raw_camera_resolution':[640,480],'budget_tiers':[1000,2000,3000],'max_format_attempts':2,
        'no_truncation_or_sampling':True,'history_definition':'all action-boundary camera observations from settled initial frame through current, including common approach; not 10Hz evaluator samples',
        'current':'current frame only, no previous outputs/actions/depth/images',
        'first_current':'first frame always retained + current frame, no previous outputs/actions',
        'all_history':'every recorded camera observation, no summaries/frame dropping or prior assistant outputs',
        'depth':'aligned ideal rendered image and RGB-derived red-region metric range for selected frames only; no GT',
        'initial':'six 0.3s actual stand actions before first task image','approach':'common physically executed approach ending after 3 occluded samples; not VLM decisions; included in 3000 budget',
        'original_terrain_geometry_retained':all(s['original_terrain_geometry_unchanged'] for s in setups),'added_occluders':'explicit per-scene specification in scene_setup.json',
        'success':'visible target >=20 pixels and median 3D surface range <1m for 3 consecutive 10Hz samples; upright/in bounds',
        'resource_limits_are_not_navigation_failure_evidence':True,'real_hardware':False,'seed':7,'decoding':'greedy'})
    if a.terrain_config:dump(root/'terrain_config.json',json.loads(a.terrain_config.read_text()))
    dump(root/'scene_setup.json',setups);snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):(snapshot/path.name).write_bytes(path.read_bytes())
    dump(root/'processes.json',{'engine_pid':os.getpid()})
    policy=Policy();print('MODEL_READY',flush=True)
    tasks=iter([(s,m,h) for s in setups for m in MODES for h in HORIZONS]);active=[];start=time.perf_counter()
    try:
        while True:
            while len(active)<a.batch_size:
                task=next(tasks,None)
                if task is None:break
                setup,mode,horizon=task;active.append(Trial(setup,terrains[setup['scene_name']],root,mode,horizon))
            if not active:break
            status(root,active,total,start,policy)
            if shutil.disk_usage(root).free<5*1024**3:
                for trial in active:trial.finish('disk_space_limit')
                raise RuntimeError('Explicit disk-space guard; remaining tasks not started, no repeated attempts')
            records=policy.decide_batch([t.prepare() for t in active]);retry=[]
            for trial,record in zip(active,records):
                trial.calls.append(record)
                if record['parsed'] is None and not record.get('termination'):retry.append(trial)
            if retry:
                records=policy.decide_batch([(t.request,t.step_folder/'inference_2',2) for t in retry])
                for trial,record in zip(retry,records):trial.calls.append(record)
            for trial in active:trial.complete()
            active=[t for t in active if not t.done];value=status(root,active,total,start,policy)
        value['status']='trials_complete_pending_analysis';dump(root/'RUN_STATUS.json',value)
    except Exception as exc:
        dump(root/'ERROR.json',{'type':type(exc).__name__,'error':str(exc)});raise
    finally:
        for trial in active:
            if not trial.done:trial.close()

if __name__=='__main__':main()
