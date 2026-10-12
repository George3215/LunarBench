"""Paired RGB/RGB-D reacquisition trials after an identical physical approach.
No simulator coordinates, segmentation or waypoint oracle enter model inputs.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from PIL import Image
import navigation_trial as nt
from controllers import Primitive
from visible_start_scenes import catalogue
from validate_visible_start import in_fov, blocked_by_wall
from vlm_benchmark import dump, sha

VALIDATION = nt.ROOT/'paper/vlm_goal_navigation/visible_start_validation'
MODES = ('rgb', 'rgbd')
SYSTEM = '''You control a quadruped with a fixed forward-facing camera. Find and approach the red sphere, finishing with it clearly visible and within 1 metre of the camera.
The initial image shows the target before a standardized physical FORWARD approach. The approach made the terrain/obstacle occlude the target. You now control every further action. You receive that initial visual memory, the current RGB image, and recent executed actions. Recover sight, navigate around obstacles, and approach carefully. No target world coordinates or ground-truth object identity are provided.
Actions each last 1 second: FORWARD walks at commanded 0.6 m/s; BACKWARD at -0.5 m/s; STRAFE_LEFT at +0.5 m/s sideways; STRAFE_RIGHT at -0.5 m/s sideways; TURN_LEFT rotates at +0.6 rad/s; TURN_RIGHT at -0.6 rad/s. STOP ends the trial after a short stand. Actual motion may differ on terrain or against obstacles.
Use brief visual reasoning grounded in these images. Avoid walking into a wall repeatedly. Consider lateral bypass or changing heading when the target is hidden. Preserve target visibility near arrival. STOP only when visible and reliably within 1 metre; confidence alone is not a distance measurement.
RGB-D trials additionally supply aligned depth images and sensor-derived measurements. A grayscale depth image uses a fixed linear mapping: white=0m axial depth, black=10m or further; invalid readings are black. These are camera measurements, not target coordinates. Numeric red-region range uses RGB red pixels and depth back-projection; it can include false positives, and becomes null when no reliable red region is visible. Never treat a null range as near. RGB-only trials receive no metric depth information.
Return one JSON object, no markdown. Give concise public assessment and decision rationale, not a long chain of thought:
{"target_visible": true or false, "target_location": "left|centre|right|absent|uncertain", "proximity": "near|far|uncertain", "confidence": number from 0 to 1, "visual_assessment": "brief visible evidence", "decision_rationale": "brief reason", "action": "FORWARD|BACKWARD|STRAFE_LEFT|STRAFE_RIGHT|TURN_LEFT|TURN_RIGHT|STOP"}'''


def actions():
    nt.ACTIONS.clear()
    nt.ACTIONS.update({
        'FORWARD':Primitive('forward',vx=.6,duration_s=1.),
        'BACKWARD':Primitive('backward',vx=-.5,duration_s=1.),
        'STRAFE_LEFT':Primitive('strafe_left',vy=.5,duration_s=1.),
        'STRAFE_RIGHT':Primitive('strafe_right',vy=-.5,duration_s=1.),
        'TURN_LEFT':Primitive('turn_left',wz=.6,duration_s=1.),
        'TURN_RIGHT':Primitive('turn_right',wz=-.6,duration_s=1.),
        'STOP':Primitive('stop',duration_s=.3)})


def sensor_packet(cam, folder):
    """Uses RGB and axial depth only, never cam.measurement or segmentation."""
    mask=nt.mask_rgb(cam.rgb,red_min=50)
    y,x=np.where(mask)
    height,width=cam.depth.shape
    cid=cam.env.model.camera(nt.CAMERA_NAME).id
    fovy=float(cam.env.model.cam_fovy[cid])
    focal=height/(2*math.tan(math.radians(fovy)/2))
    axial=cam.depth[mask]
    radial=axial*np.sqrt(1+((x+.5-width/2)/focal)**2+((y+.5-height/2)/focal)**2)
    valid=np.isfinite(radial)&(radial>0)&(radial<20)
    reading={'source':'aligned rendered depth + RGB red threshold; no target geom/pose/segmentation',
        'red_pixel_count':len(x),'valid_red_depth_count':int(valid.sum()),
        'red_region_median_axial_depth_m':float(np.median(axial[valid])) if valid.sum()>=20 else None,
        'red_region_median_surface_range_m':float(np.median(radial[valid])) if valid.sum()>=20 else None,
        'centroid_uv':[float(x.mean()/width),float(y.mean()/height)] if len(x)>=20 else None,
        'image_width':width,'image_height':height,'vertical_fov_deg':fovy,
        'depth_encoding':'fixed linear grayscale white=0m axial, black=10m+ or invalid',
        'sensor_simulation':'ideal MuJoCo rendered metric depth; no physical-camera noise model'}
    valid_full=np.isfinite(cam.depth)&(cam.depth>0)
    grey=np.where(valid_full,255*(1-np.clip(cam.depth,0,10)/10),0).astype(np.uint8)
    Image.fromarray(grey).convert('RGB').save(folder/'input_depth.png')
    np.save(folder/'input_depth_m.npy',cam.depth)
    dump(folder/'depth_sensor.json',reading)
    return reading


class Policy(nt.Policy):
    def decide_batch(self,requests):
        batch=[]
        for request,folder,attempt in requests:
            folder.mkdir(parents=True,exist_ok=True)
            content=[];saved=[]
            text=request['text']+ ('\nReturn the complete valid JSON schema.' if attempt>1 else '')
            content.append({'type':'text','text':text});saved.append({'type':'text','text':text})
            for label,path in request['images']:
                content.extend([{'type':'text','text':label},{'type':'image','image':Image.open(path).convert('RGB')}])
                saved.extend([{'type':'text','text':label},{'type':'image','image_path':str(path),'sha256':sha(path)}])
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':content}]
            dump(folder/'messages.json',{'messages':[messages[0],{'role':'user','content':saved}],
                'mode':request['mode'],'policy_inputs':request['input_description'],
                'forbidden_inputs':'no segmentation, geom identity, world pose, waypoints or GT target coordinates'})
            (folder/'rendered_prompt.txt').write_text(self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
            batch.append(messages)
        start=time.perf_counter()
        inputs=self.processor.apply_chat_template(batch,tokenize=True,padding=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt').to('cuda:0')
        grids=inputs['image_grid_thw'].tolist()
        pixels=inputs['pixel_values'].detach().cpu().float().numpy()
        offset=0;grid_offset=0
        for i,(request,folder,attempt) in enumerate(requests):
            image_info=[]
            for label,path in request['images']:
                grid=grids[grid_offset];grid_offset+=1
                count=int(np.prod(grid));chunk=pixels[offset:offset+count];offset+=count
                image_info.append({'label':label,'image_path':str(path),'image_sha256':sha(path),
                    'image_grid_thw':grid,'pixel_values_shape':list(chunk.shape),
                    'pixel_values_float32_sha256':hashlib.sha256(chunk.tobytes()).hexdigest()})
            dump(folder/'network_input.json',{'input_ids':inputs['input_ids'][i].tolist(),
                'attention_mask':inputs['attention_mask'][i].tolist(),'images':image_info,
                'batch_size':len(requests),'independent_sequence':i})
        if offset!=len(pixels) or grid_offset!=len(grids):raise RuntimeError('Multi-image patch accounting mismatch')
        with torch.inference_mode():
            outputs=self.model.generate(**inputs,max_new_tokens=224,do_sample=False,return_dict_in_generate=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        records=[]
        eos=self.model.generation_config.eos_token_id
        end_ids=eos if isinstance(eos,list) else [eos]
        for i,(request,folder,attempt) in enumerate(requests):
            ids=outputs.sequences[i,inputs['input_ids'].shape[1]:].tolist()
            for j,token in enumerate(ids):
                if token in end_ids:ids=ids[:j+1];break
            raw=self.processor.decode(ids,skip_special_tokens=True).strip()
            record={'elapsed_s':elapsed,'batch_size':len(requests),'batch_amortized_s':elapsed/len(requests),
                'raw_output':raw,'raw_output_with_special_tokens':self.processor.decode(ids,skip_special_tokens=False),
                'generated_token_ids':ids,'generated_token_count':len(ids),'hit_token_limit':len(ids)>=224,
                'parsed':nt.parse_decision(raw)}
            dump(folder/'output.json',record);(folder/'raw_output.txt').write_text(raw);records.append(record)
        return records


class Trial(nt.Trial):
    def __init__(self,name,scene,terrain,out,mode,validation):
        super().__init__(name,scene,out/mode,terrain=terrain,render_size=(480,360))
        self.mode=mode;self.scene_name=name;self.name=f'{mode}/{name}'
        self.initial_depth=sensor_packet(self.cam,self.folder/'initial_evidence')
        initial=self.cam.measurement
        if initial['target_visible_rgb_pixels']<40 or initial['near_and_visible']:
            raise RuntimeError(f'Initial visibility gate failed: {self.name}')
        self.prefix_actions=validation['loss']['step']
        self.prefix_trace=[]
        # Real physical motion; no state teleport and no controller reset after prefix.
        for step in range(1,self.prefix_actions+1):
            step_folder=self.folder/'steps'/f'{step:04d}';step_folder.mkdir(parents=True)
            before=self.cam.read(self.env.model,self.env.data)
            image_path=step_folder/'input_rgb.png';Image.fromarray(self.cam.rgb).save(image_path)
            self.cam.save_evidence(step_folder/'before_evaluator')
            sensor_packet(self.cam,step_folder)
            qpos=self.env.data.qpos.copy();start=time.perf_counter()
            terminal=self.physics.execute('FORWARD',step)
            after=self.cam.read(self.env.model,self.env.data)
            self.cam.save_evidence(step_folder/'after_evaluator')
            if terminal:raise RuntimeError(f'Controlled approach terminated: {self.name}/{terminal}')
            trace={'step':step,'phase':'controlled_approach','input_image':str(image_path.relative_to(self.folder)),
                'input_image_sha256':sha(image_path),'inference_count_this_step':0,'inference_records':[],
                'model_output_raw':[],'decision':None,'action':'FORWARD',
                'action_source':'fixed task setup, not VLM','inference_s':0.,
                'execution_wall_s':time.perf_counter()-start,
                'actual_displacement_xy_m':float(np.linalg.norm(self.env.data.qpos[:2]-qpos[:2])),
                'before_evaluator':before,'after_evaluator':after,'termination':None,'success_streak':0}
            dump(step_folder/'transition.json',trace);self.tf.write(json.dumps(trace)+'\n');self.tf.flush()
            self.executed.append('FORWARD');self.prefix_trace.append(trace)
        final=self.cam.read(self.env.model,self.env.data)
        if final['target_segment_pixels']>2 or not in_fov(self.env,final) or not blocked_by_wall(self.env):
            raise RuntimeError(f'Physical occlusion gate failed: {self.name}')
        dump(self.folder/'approach_gate.json',{'initial_visible':True,'geometric_occlusion':True,
            'prefix_actions':self.prefix_actions,'initial':initial,'lost':final,
            'lost_qpos':self.env.data.qpos.tolist(),'rgb_sha256':sha(self.folder/'steps'/f'{self.prefix_actions:04d}'/'after_evaluator/rgb.png'),
            'not_vlm_decisions':True,'prefix_included_in_total_3000_action_budget':True})
        self.cam.save_evidence(self.folder/'lost_evidence')
        self.step=self.prefix_actions
        self.recent=[{'action':'FORWARD','source':'controlled_approach'} for _ in range(min(4,self.prefix_actions))]
        self.initial_image=self.folder/'initial_evidence/rgb.png'
        self.request=None

    def prepare(self):
        super().prepare()
        self.sensor=sensor_packet(self.cam,self.step_folder)
        images=[('Initial visible RGB memory',self.initial_image),('Current RGB camera',self.image_path)]
        text=f'Target: red sphere. Input condition: {self.mode}.\nRecent executed actions: '+json.dumps(self.recent[-4:])
        if self.mode=='rgbd':
            images=[('Initial visible RGB memory',self.initial_image),
                ('Initial aligned depth, fixed grayscale 0..10 m',self.folder/'initial_evidence/input_depth.png'),
                ('Current RGB camera',self.image_path),('Current aligned depth, fixed grayscale 0..10 m',self.step_folder/'input_depth.png')]
            text+='\nInitial depth sensor reading: '+json.dumps(self.initial_depth)+'\nCurrent depth sensor reading: '+json.dumps(self.sensor)
        self.request={'mode':self.mode,'text':text,'images':images,
            'input_description':'initial RGB memory + current RGB + recent own actions'+(' + aligned initial/current depth images + RGB-derived red-region metric depth' if self.mode=='rgbd' else '')}
        return self.request,self.step_folder/'inference_1',1

    def complete(self):
        super().complete()
        path=self.step_folder/'transition.json'
        trace=json.loads(path.read_text());trace['phase']='vlm_autonomous';trace['action_source']='VLM'
        if self.mode=='rgbd':trace['policy_depth_sensor']=self.sensor
        dump(path,trace)
        self.recent.append({'action':trace['action'],'model_assessment':trace['decision']})
        # Base trace remains sufficient; phase is derivable from prefix gate and step number.
        if self.done:
            result_path=self.folder/'result.json';result=json.loads(result_path.read_text())
            result.update(mode=self.mode,scene=self.scene_name,controlled_approach_actions=self.prefix_actions,
                autonomous_actions=result['action_steps']-self.prefix_actions,
                initial_visible=True,physical_occlusion_verified=True,
                initial_depth_camera_range_m=self.initial_depth['red_region_median_surface_range_m'],
                success_definition='initial visible + physical approach occlusion + autonomous visible/near arrival')
            dump(result_path,result)


def status(root,completed,active,total,start,generation_calls):
    results=[json.loads(p.read_text()) for mode in MODES for p in (root/mode/'trials').glob('*/result.json')]
    value={'status':'running','completed':len(results),'expected':total,
        'success_count':sum(r['success'] for r in results),
        'by_mode':{mode:{'completed':sum(r['mode']==mode for r in results),
            'success_count':sum(r['success'] and r['mode']==mode for r in results)} for mode in MODES},
        'active_trials':{t.name:t.step for t in active},'model_generate_calls':generation_calls,
        'run_wall_s':time.perf_counter()-start,'max_total_actions_per_trial':3000}
    dump(root/'RUN_STATUS.json',value)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.batch_size<=4:p.error('batch-size must be 1..4 for multi-image GPU memory')
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
    validations=json.loads((VALIDATION/'validation.json').read_text())
    if len(validations)!=24 or not all(v['eligible'] for v in validations):raise RuntimeError('All 24 physical eligibility gates must pass before model trials')
    validation={v['scene']:v for v in validations};scenes=catalogue();actions()
    torch.set_num_threads(1);torch.manual_seed(7);torch.set_grad_enabled(False)
    protocol={'version':'visible-start-rgb-depth-paired/1','model':str(nt.MODEL),'scene_count':24,'trial_count':48,
        'conditions':list(MODES),'system_prompt':SYSTEM,'seed':7,'decoding':'greedy','max_new_tokens':224,
        'render_size':[480,360],'action_commands':{k:vars(v) for k,v in nt.ACTIONS.items()},
        'budget_tiers':[1000,2000,3000],'budget_includes_controlled_approach':True,'max_inferences_per_action':2,
        'initial_gate':'>=40 visible RGB target pixels; not already near',
        'loss_gate':'<=2 target segmentation pixels while target centre in FOV and centre ray hits finite wall; physical motion',
        'approach':'identical physically executed forward prefix per paired scene, no VLM decisions during task setup',
        'policy_start':'verified occluded state; initial RGB memory included in first and every network input',
        'success':'autonomous arrival: >=20 visible pixels and median camera-to-visible-surface range <1m for three 10Hz samples; upright/in bounds',
        'depth_input':'aligned ideal rendered axial depth image + RGB red-threshold region back-projected radial range; no GT identity/coordinates',
        'real_hardware':False,'timing':'simulation pauses during VLM inference; no real-time control claim',
        'scene_definition':'new controlled descent + finite barrier, 8 geometry variants x 3 offsets; replaces original incompatible scenes',
        'validation_source':str(VALIDATION),'source_sha256':sha(Path(__file__))}
    dump(root/'protocol.json',protocol)
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):(snapshot/path.name).write_bytes(path.read_bytes())
    dump(root/'scene_manifest.json',[{'scene':vars(scene),'terrain':terrain.spec,
         'slabs':[vars(s) for s in terrain.slabs]} for scene,terrain in scenes.values()])
    (root/'validation.json').write_bytes((VALIDATION/'validation.json').read_bytes())
    dump(root/'processes.json',{'engine_pid':os.getpid()})
    policy=Policy();print('MODEL_READY',flush=True)
    pending=iter([(mode,name) for name in scenes for mode in MODES]);active=[]
    start=time.perf_counter();generation_calls=0
    try:
        while True:
            while len(active)<a.batch_size:
                item=next(pending,None)
                if item is None:break
                mode,name=item;scene,terrain=scenes[name]
                active.append(Trial(name,scene,terrain,root,mode,validation[name]))
            if not active:break
            status(root,None,active,48,start,generation_calls)
            records=policy.decide_batch([t.prepare() for t in active]);generation_calls+=1
            retry=[]
            for trial,record in zip(active,records):
                trial.calls.append(record)
                if record['parsed'] is None:retry.append(trial)
            if retry:
                records=policy.decide_batch([(t.request,t.step_folder/'inference_2',2) for t in retry]);generation_calls+=1
                for trial,record in zip(retry,records):trial.calls.append(record)
            for trial in active:trial.complete()
            active=[t for t in active if not t.done]
            state=status(root,None,active,48,start,generation_calls)
        state['status']='trials_complete_pending_analysis';dump(root/'RUN_STATUS.json',state)
    except Exception as exc:
        dump(root/'ERROR.json',{'type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        for trial in active:
            if not trial.done:trial.close()

if __name__=='__main__':main()
