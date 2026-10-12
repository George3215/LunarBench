"""Independent strict G1-G4 physical admission; never runs a VLM.

Freezes task implementation and thresholds before candidate search. Rejected
candidates remain evidence. Only four true gates admit a formal scene.
"""
import argparse, hashlib, json, math, re, shutil, sys, time
from pathlib import Path
from dataclasses import replace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'paper/vlm_goal_navigation/recovery_benchmark_20261009'


def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2));temp.replace(p)


def freeze(out):
    frozen=out/'frozen/MoonSim';tasks=frozen/'tasks/task3_reobservability';tasks.mkdir(parents=True,exist_ok=True)
    if not (out/'source_hashes.json').exists():
        for p in (ROOT/'tasks/task3_reobservability').glob('*.py'):shutil.copy2(p,tasks/p.name)
        shutil.copytree(ROOT/'tasks/task3_reobservability/generated',tasks/'generated',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.png','*.npy','*.npz','*.mp4','*.jsonl'))
        for name in ('assets','third_party','paper'):
            p=frozen/name
            if not p.exists():p.symlink_to(ROOT/name,target_is_directory=True)
        shutil.copy2(Path(__file__),out/'strict_recovery_gate.py')
        paths=list(tasks.glob('*.py'))+[out/'strict_recovery_gate.py',out/'PROTOCOL.json']
        write(out/'source_hashes.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    hashes=json.loads((out/'source_hashes.json').read_text())
    if any(hashlib.sha256((out/k).read_bytes()).hexdigest()!=v for k,v in hashes.items()):raise RuntimeError('frozen source/protocol changed')
    sys.path.insert(0,str(tasks))


def terrain_class(kind):
    if kind in ('stairs_up','stairs_down','stairs','staircase10'):return 'A'
    if kind in ('pit','basin','crater','trench','gullies'):return 'B'
    if kind in ('barrier','block_maze','movable_rubble'):return 'C'
    if kind in ('window','window_wall'):return 'E'
    return 'D'


def main(out, design_module=None):
    out.mkdir(parents=True,exist_ok=True);freeze(out);P=json.loads((out/'PROTOCOL.json').read_text())
    import mujoco,torch
    import navigation_trial as nt
    from visible_start_trial import actions
    from all_terrain_setup import place
    from difficulty_terrain_setup import catalogue,geometric_visible
    from terrain_design_v3 import designs,build_terrain_v3,scene_for_config
    from scenes import Scene
    torch.set_num_threads(1);actions()
    legacy=ROOT/'paper/vlm_goal_navigation/runs/20261008_task3_difficulty_calibration_v2_medium'
    old_setups=json.loads((legacy/'scene_setup.json').read_text());old_config=json.loads((legacy/'terrain_config.json').read_text())
    inventory=[{'terrain_id':'v2/'+s['scene_name'],'terrain_class':terrain_class(s['source_scene']['terrain_profile']),'source':'v2','seed':old_config['seed'],'setup':s} for s in old_setups]
    inventory += [{'terrain_id':'design_v3/'+c['name'],'terrain_class':terrain_class(c['kind']),'source':'design_v3','seed':c['seed'],'config':c} for c in designs()]
    if design_module:
        import importlib
        module=importlib.import_module(design_module)
        designs,build_terrain_v3,scene_for_config=module.designs,module.build_terrain_v3,module.scene_for_config
        inventory=[{'terrain_id':design_module+'/'+c['name'],'terrain_class':terrain_class(c['kind']),'source':'custom_design','seed':c['seed'],'config':c} for c in designs()]
    write(out/'candidate_inventory.json',inventory)
    accepted=[];records=[]
    def ray(camera,terrain):
        env=camera.env;data=camera.render_data;cid=env.model.camera(nt.CAMERA_NAME).id;eye=data.cam_xpos[cid];target=data.geom_xpos[env.model.geom('task3_target_geom').id];local=data.cam_xmat[cid].reshape(3,3).T@(target-eye)
        h,w=camera.depth.shape;v=math.tan(math.radians(float(env.model.cam_fovy[cid]))/2)
        fov=bool(local[2]<0 and abs(local[1]/local[2])<v and abs(local[0]/local[2])<v*w/h)
        hit=np.array([-1],np.int32);direction=(target-eye)/np.linalg.norm(target-eye)
        distance=mujoco.mj_ray(env.model,data,eye,direction,None,1,-1,hit);name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(hit[0])) if hit[0]>=0 else None
        names={s.name for s in terrain.slabs} if hasattr(terrain,'slabs') else set()
        names.update(re.findall(r'<geom\s+name="([^"]+)"',terrain.spec.get('navigation_extra_geom_xml','')+terrain.spec.get('navigation_dynamic_body_xml','')))
        names.update(['task3_terrain','task3_slab','task3_apron','task3_landing'])
        blocked=name in names or bool(name and name.startswith('task3_step_'))
        return {'target_centre_in_fov':fov,'blocking_geom':name,'hit_distance_m':float(distance),'entity_occlusion':bool(fov and blocked and distance>=0 and distance<np.linalg.norm(target-eye))}
    class GatePhysics(nt.Physics):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.phase='stand';self.loss_run=0;self.recover_run=0;self.extra_terminal=None;self.loss=None;self.recovery=None;self.start_xy=self.env.data.qpos[:2].copy()
        def record(self,step,action):
            s=super().record(step,action);m=s['evaluation'];safe=m['upright']>=P['upright_minimum'] and m['in_bounds']
            if not safe:self.extra_terminal='unsafe';return s
            if self.phase=='approach':
                moved=float(np.linalg.norm(self.env.data.qpos[:2]-self.start_xy));check=ray(self.camera,self.env.terrain)
                valid=m['target_segment_pixels']<=P['lost_segmentation_max_pixels'] and moved>=P['minimum_movement_m']
                self.loss_run=self.loss_run+1 if valid and check['entity_occlusion'] else 0
                if self.loss_run>=P['lost_consecutive_samples']:
                    self.loss={'measurement':m,'ray':check,'lost_frames':self.loss_run,'displacement_m':moved,'qpos':self.env.data.qpos.tolist()};self.extra_terminal='strict_loss'
            if self.phase=='oracle':
                self.recover_run=self.recover_run+1 if m['target_visible_rgb_pixels']>=P['recovery_rgb_min_pixels'] else 0
                if self.recover_run>=P['recovery_consecutive_samples']:
                    self.recovery={'measurement':m,'qpos':self.env.data.qpos.tolist(),'consecutive_frames':self.recover_run};self.extra_terminal='strict_recovery'
            return s
    def candidates(item):
        if item['source']=='v2':
            setup=item['setup'];t=catalogue(old_config)[setup['scene_name']][1];t.spec.clear();t.spec.update(setup['terrain_spec']);yield Scene(**setup['adjusted_scene']),t,setup['approach_action'];return
        c=item['config'];t=build_terrain_v3(c);base=scene_for_config(c)
        if c['kind']=='window':
            cy=c['center_y_m']
            for side in [-1,1]:
                for shift in [.5,.8,1.1]:
                    start=(-3.,cy+side*shift);target=(3.,cy-side*shift);yaw=math.atan2(target[1]-start[1],target[0]-start[0])
                    yield replace(base,lost_xy=start,approach_xy=start,target_xy=target,yaw_rad=yaw),t,('STRAFE_RIGHT' if side>0 else 'STRAFE_LEFT')
        elif c['kind'] in ('barrier','block_maze','movable_rubble'):
            for sy in [-2.,2.,-3.,3.]:
                for ty in [0.,-sy*.25]:
                    start=(-3.8,sy);target=(3.,ty);yaw=math.atan2(ty-sy,6.8)
                    yield replace(base,lost_xy=start,approach_xy=start,target_xy=target,yaw_rad=yaw),t,('STRAFE_LEFT' if sy<0 else 'STRAFE_RIGHT')
        else:
            ranked=[]
            for sy in [0.,-2.,2.]:
                for sx in np.arange(-6.,2.,.5):
                    for distance in [3.,4.5,6.]:
                        tx=sx+distance
                        if tx>7.:continue
                        for height in [.35,.55,.8]:
                            start=(float(sx),sy);target=(float(tx),sy)
                            if geometric_visible(t,start,target,height)<3:continue
                            loss=next((float(d) for d in np.arange(.4,min(distance-1.,3.6),.2) if geometric_visible(t,(sx+d,sy),target,height)==0),None)
                            if loss is not None:ranked.append((distance,loss,start,target,height))
            ranked.sort(key=lambda v:v[:2])
            for _,_,start,target,height in ranked[:P['candidate_limit_per_terrain']]:yield replace(base,lost_xy=start,approach_xy=start,target_xy=target,target_height_m=height),t,'FORWARD'
    for item in inventory:
        id=item['terrain_id'];dest=out/'validation'/id;prior=dest/'admission.json'
        if prior.exists():
            result=json.loads(prior.read_text());records.append(result)
            if result['accepted']:accepted.append(result)
            continue
        attempts=[];winner=None
        for index,(scene,terrain,approach) in enumerate(candidates(item)):
            if index>=P['candidate_limit_per_terrain']:break
            folder=dest/f'candidate_{index:02d}';folder.mkdir(parents=True,exist_ok=True);env=nt.api.MuJoCoEnvironment(scene,terrain=terrain);camera=nt.EvaluatorCamera(env,render_size=(640,480));place(env)
            try:
                with (folder/'state.jsonl').open('w') as sf,(folder/'evaluator.jsonl').open('w') as ef:
                    physics=GatePhysics(env,camera,sf,ef,folder);physics.record(0,'RESET')
                    stable=True
                    for _ in range(6):
                        if physics.execute('STOP',0):stable=False;break
                    initial=camera.read(env.model,env.data);camera.save_evidence(folder/'initial');shutil.copy2(folder/'initial/rgb.png',folder/'initial_rgb.png')
                    init_pose=env.data.qpos.tolist();g1=stable and initial['target_visible_rgb_pixels']>=P['initial_rgb_min_pixels'] and not initial['near_and_visible'];commands=[]
                    physics.phase='approach';physics.extra_terminal=None;physics.start_xy=env.data.qpos[:2].copy()
                    if g1:
                        for step in range(1,P['approach_action_limit']+1):
                            commands.append(approach);term=physics.execute(approach,step)
                            if term:break
                    loss=physics.loss;g2=bool(loss);g3=bool(loss and loss['ray']['entity_occlusion']);g4=False;oracle=[];searched=0
                    if g2 and g3:
                        camera.save_evidence(folder/'lost');shutil.copy2(folder/'lost/rgb.png',folder/'lost_rgb.png');ss=mujoco.mjtState.mjSTATE_INTEGRATION;buf=np.empty(mujoco.mj_stateSize(env.model,ss));mujoco.mj_getState(env.model,env.data,buf,ss);angles=physics.angles.copy();ticks=physics.ticks
                        reverse={'FORWARD':'BACKWARD','STRAFE_LEFT':'STRAFE_RIGHT','STRAFE_RIGHT':'STRAFE_LEFT'}[approach]
                        templates=[[reverse]*min(len(commands)+4,24)]+[[a]*8 for a in ['STRAFE_LEFT','STRAFE_RIGHT','BACKWARD','FORWARD','TURN_LEFT','TURN_RIGHT']]
                        for branch,seq in enumerate(templates):
                            mujoco.mj_setState(env.model,env.data,buf,ss);mujoco.mj_forward(env.model,env.data);env.controller.reset();physics.angles=angles.copy();physics.ticks=ticks;physics.phase='oracle';physics.extra_terminal=None;physics.recover_run=0;physics.recovery=None
                            executed=[]
                            for action in seq:
                                if searched>=P['oracle_action_budget_total_per_candidate']:break
                                searched+=1;executed.append(action);term=physics.execute(action,1000+searched)
                                if term:break
                            oracle.append({'branch':branch,'actions':executed,'recovery':physics.recovery,'termination':term})
                            if physics.recovery:
                                g4=True;camera.save_evidence(folder/'recovered');break
                            if searched>=P['oracle_action_budget_total_per_candidate']:break
                    gates={'G1':bool(g1),'G2':g2,'G3':g3,'G4':g4};result={'terrain_id':id,'terrain_class':item['terrain_class'],'seed':item['seed'],'gates':gates,'accepted':all(gates.values()),'initial_target_pixels':initial['target_visible_rgb_pixels'],'initial_pose':init_pose,'initial_evaluator':initial,'loss':loss,'approach_actions':commands,'oracle':oracle,'oracle_searched_actions':searched,'scene':vars(scene),'terrain_spec':terrain.spec,'evidence_path':str(folder.resolve()),'history_necessity_claim':False}
                    write(folder/'gates.json',result);attempts.append({'candidate':index,'gates':gates,'evidence_path':str(folder.resolve())})
                    if result['accepted']:winner=result
            finally:camera.close();env.close()
            if winner:break
        result=winner or {'terrain_id':id,'terrain_class':item['terrain_class'],'seed':item['seed'],'accepted':False,'gates':{'G1':False,'G2':False,'G3':False,'G4':False},'reason':'bounded candidates exhausted or no geometric candidate','attempts':attempts}
        result['attempts']=attempts;write(prior,result);records.append(result)
        if result['accepted']:accepted.append(result)
        write(out/'formal_manifest.json',accepted);write(out/'admission_records.json',records)
        write(out/'STATUS.json',{'status':'running','processed':len(records),'expected':len(inventory),'accepted':len(accepted),'all_formal_gates_pass':all(all(r['gates'].values()) for r in accepted)})
        print('GATE',id,result['accepted'],result['gates'],len(records),len(inventory),flush=True)
    write(out/'formal_manifest.json',accepted);write(out/'admission_records.json',records);write(out/'STATUS.json',{'status':'complete','processed':len(records),'expected':len(inventory),'accepted':len(accepted),'all_formal_gates_pass':all(all(r['gates'].values()) for r in accepted)})
    print('STRICT_GATE_COMPLETE',len(accepted),len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--design-module');a=p.parse_args()
    try:main(a.out.resolve(),a.design_module)
    except Exception as e:write(a.out/'ERROR.json',{'error':str(e),'type':type(e).__name__});raise
