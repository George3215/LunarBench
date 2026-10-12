"""Explicit image-history conditions; auditable public decision explanations."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import navigation_trial as nt
from vlm_benchmark import dump,sha

HORIZONS=('current','first_current','all_history')
MODES=('rgb','rgbd')
INPUT_TOKEN_LIMIT=16384
OUTPUT_TOKEN_LIMIT=224
SYSTEM='''Control the quadruped to find the red sphere, approach it, and finish with the sphere clearly visible and within 1 metre of the camera. Use only the observations supplied in this request. No hidden earlier frames, target coordinates, scene name, or previous assistant messages are available.
Actions last 1 second: FORWARD vx=0.6m/s, BACKWARD vx=-0.5m/s, STRAFE_LEFT vy=0.5m/s, STRAFE_RIGHT vy=-0.5m/s, TURN_LEFT wz=0.6rad/s, TURN_RIGHT wz=-0.6rad/s. STOP ends the trial after a short stand. Actual motion can differ on terrain or against obstacles. Look for a traversable route, avoid repeated collisions, recover sight when the target is occluded, and avoid overshooting. STOP only when the current target is visible and reliably within 1 metre.
Each supplied RGB image is tagged with an observation frame ID. In RGB-D requests it is followed by aligned ideal camera depth: fixed grayscale white=0m axial depth, black=10m or farther/invalid. A numeric red-region range is calculated from RGB red pixels and depth back-projection, without ground-truth object identity; null means no reliable red-region depth, not arrival. Historical depth belongs to its historical frame and is not the current range.
For each action, give a brief public explanation of the strategy and cite the supplied frame IDs and evidence channels that support it. Explain whether you use current observations or remembered supplied images/depth. Do not provide a long internal chain of thought. If there is no earlier frame in the request, do not claim to have observed one.
Return one JSON object, no markdown:
{"target_visible":true or false,"target_location":"left|centre|right|absent|uncertain","proximity":"near|far|uncertain","confidence":number from 0 to 1,"visual_assessment":"brief visible evidence","decision_rationale":"brief strategy and reason for this action","evidence_frames":[integer frame IDs],"evidence_sources":["current_rgb|past_rgb|current_depth|past_depth"],"action":"FORWARD|BACKWARD|STRAFE_LEFT|STRAFE_RIGHT|TURN_LEFT|TURN_RIGHT|STOP"}'''


def select_frames(frames,horizon):
    if horizon=='current':return [frames[-1]]
    if horizon=='first_current':return [frames[0],frames[-1]] if len(frames)>1 else [frames[0]]
    if horizon=='all_history':return list(frames)
    raise ValueError(horizon)


def request_for(frames,horizon,mode):
    chosen=select_frames(frames,horizon);images=[]
    text=f'Target: red sphere. Available image context: {horizon}. Sensor condition: {mode}.\n'
    for frame in chosen:
        label=f"Observation frame {frame['frame_id']}, simulation time {frame['sim_time_s']:.3f}s"
        images.append((label+' RGB',Path(frame['rgb_path'])))
        if mode=='rgbd':
            images.append((label+' aligned depth (white 0m, black 10m+)',Path(frame['depth_path'])))
            d=frame['depth_sensor'];reading={'frame_id':frame['frame_id'],'red_pixels':d['red_pixel_count'],
                'red_region_axial_depth_m':d['red_region_median_axial_depth_m'],
                'red_region_surface_range_m':d['red_region_median_surface_range_m']}
            text+='Camera measurement: '+json.dumps(reading)+'\n'
    return {'mode':mode,'horizon':horizon,'text':text,'images':images,
        'frame_ids':[f['frame_id'] for f in chosen],'current_frame_id':frames[-1]['frame_id'],
        'first_frame_in_context':chosen[0]['frame_id']==0,
        'history_span_sim_s':frames[-1]['sim_time_s']-chosen[0]['sim_time_s'],
        'all_recorded_frame_count':len(frames),'input_description':f'{horizon} frames, {mode}; no past outputs/actions/GT'}


def parse(raw):
    value=nt.parse_decision(raw)
    if value is None:return None
    refs=value.get('evidence_frames');sources=value.get('evidence_sources')
    if not isinstance(refs,list) or not refs or not all(isinstance(i,int) and not isinstance(i,bool) for i in refs):return None
    if not isinstance(sources,list) or not sources or not all(s in ('current_rgb','past_rgb','current_depth','past_depth') for s in sources):return None
    return value


class Policy(nt.Policy):
    def __init__(self):
        super().__init__()
        # Identical per-frame area cap across all history/sensor conditions.
        self.processor.image_processor.size={'shortest_edge':65536,'longest_edge':65536}
        self.generation_calls=0
        self.model_context_limit=int(self.model.config.text_config.max_position_embeddings)

    def messages(self,request,attempt):
        text=request['text']+ ('Return the full requested JSON schema, including evidence_frames and evidence_sources.' if attempt>1 else '')
        content=[{'type':'text','text':text}];saved=[{'type':'text','text':text}]
        for label,path in request['images']:
            content.extend([{'type':'text','text':label},{'type':'image','image':Image.open(path).convert('RGB')}])
            saved.extend([{'type':'text','text':label},{'type':'image','image_path':str(path),'sha256':sha(path)}])
        return [{'role':'system','content':SYSTEM},{'role':'user','content':content}], [{'role':'system','content':SYSTEM},{'role':'user','content':saved}]

    def encode(self,batch):
        prompts=[self.processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in batch]
        images=[block['image'] for m in batch for block in m[1]['content'] if block['type']=='image']
        return self.processor(text=prompts,images=images,padding=True,return_tensors='pt'),prompts

    def save_inputs(self,requests,inputs,prompts):
        grids=inputs['image_grid_thw'].tolist();pixels=inputs['pixel_values'].detach().cpu().float().numpy()
        offset=0;grid_index=0
        for i,(request,folder,attempt) in enumerate(requests):
            tokens=int(inputs['attention_mask'][i].sum());ids=inputs['input_ids'][i].tolist();mask=inputs['attention_mask'][i].tolist()
            image_info=[]
            for label,path in request['images']:
                grid=grids[grid_index];grid_index+=1;count=int(np.prod(grid));chunk=pixels[offset:offset+count];offset+=count
                image_info.append({'label':label,'image_path':str(path),'image_sha256':sha(path),'image_grid_thw':grid,
                    'pixel_values_shape':list(chunk.shape),'pixel_values_float32_sha256':hashlib.sha256(chunk.tobytes()).hexdigest()})
            context={'horizon':request['horizon'],'mode':request['mode'],'input_frame_ids':request['frame_ids'],
                'input_frame_count':len(request['frame_ids']),'input_image_count':len(image_info),
                'all_recorded_frame_count':request['all_recorded_frame_count'],'first_frame_in_context':request['first_frame_in_context'],
                'history_span_sim_s':request['history_span_sim_s'],'input_tokens':tokens,
                'vision_placeholder_tokens':sum(v==self.model.config.image_token_id for v,m in zip(ids,mask) if m),
                'model_max_position_embeddings':self.model_context_limit,'runtime_input_token_limit':INPUT_TOKEN_LIMIT,
                'output_token_limit':OUTPUT_TOKEN_LIMIT,'truncated':False,'past_assistant_outputs_in_context':False,
                'past_action_history_in_context':False}
            dump(folder/'context.json',context)
            dump(folder/'network_input.json',{'input_ids':ids,'attention_mask':mask,'images':image_info,
                'context':context,'batch_size':len(requests),'independent_sequence':i})
            (folder/'rendered_prompt.txt').write_text(prompts[i])
        if offset!=len(pixels) or grid_index!=len(grids):raise RuntimeError('Image patch accounting mismatch')

    def decide_batch(self,requests):
        results=[None]*len(requests);allowed=[];messages=[];lengths=[]
        for index,(request,folder,attempt) in enumerate(requests):
            folder.mkdir(parents=True,exist_ok=True)
            msg,saved=self.messages(request,attempt)
            dump(folder/'messages.json',{'messages':saved,'mode':request['mode'],'horizon':request['horizon'],
                'policy_inputs':request['input_description']})
            inputs,prompts=self.encode([msg]);self.save_inputs([(request,folder,attempt)],inputs,prompts)
            count=int(inputs['attention_mask'].sum())
            if count>INPUT_TOKEN_LIMIT or count+OUTPUT_TOKEN_LIMIT>self.model_context_limit:
                record={'parsed':None,'raw_output':'','raw_output_with_special_tokens':'','generated_token_ids':[],
                    'generated_token_count':0,'hit_token_limit':False,'elapsed_s':0.,'batch_amortized_s':0.,
                    'generation_skipped':True,'termination':'context_limit','input_tokens':count,
                    'explanation':'All history retained; no truncation or frame sampling. Explicit runtime context boundary.'}
                dump(folder/'output.json',record);(folder/'raw_output.txt').write_text('');results[index]=record
            else:allowed.append(index);messages.append(msg);lengths.append(count)
        position=0
        while position<len(allowed):
            end=position+1
            while end<len(allowed) and end-position<4 and max(lengths[position:end+1])*(end-position+1)<=INPUT_TOKEN_LIMIT:end+=1
            indices=allowed[position:end];batch=messages[position:end]
            group=[requests[i] for i in indices];inputs,prompts=self.encode(batch);self.save_inputs(group,inputs,prompts)
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats();self.generation_calls+=1
            try:
                inputs=inputs.to('cuda:0')
                with torch.inference_mode():outputs=self.model.generate(**inputs,max_new_tokens=OUTPUT_TOKEN_LIMIT,do_sample=False,return_dict_in_generate=True)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                eos=self.model.generation_config.eos_token_id;end_ids=eos if isinstance(eos,list) else [eos]
                for row,index in enumerate(indices):
                    request,folder,attempt=requests[index]
                    ids=outputs.sequences[row,inputs['input_ids'].shape[1]:].tolist()
                    for j,token in enumerate(ids):
                        if token in end_ids:ids=ids[:j+1];break
                    raw=self.processor.decode(ids,skip_special_tokens=True).strip();parsed=parse(raw)
                    record={'raw_output':raw,'raw_output_with_special_tokens':self.processor.decode(ids,skip_special_tokens=False),
                        'parsed':parsed,'generated_token_ids':ids,'generated_token_count':len(ids),'hit_token_limit':len(ids)>=OUTPUT_TOKEN_LIMIT,
                        'elapsed_s':elapsed,'batch_amortized_s':elapsed/len(indices),'batch_size':len(indices),
                        'generation_skipped':False,'generation_call_id':self.generation_calls,
                        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
                    if parsed:
                        available=set(request['frame_ids']);current=request['current_frame_id']
                        record['unsupported_evidence_frame_ids']=[f for f in parsed['evidence_frames'] if f not in available]
                        record['unsupported_depth_claim']=request['mode']=='rgb' and any('depth' in s for s in parsed['evidence_sources'])
                        record['unsupported_past_claim']=len(available)==1 and any(s.startswith('past_') for s in parsed['evidence_sources'])
                    dump(folder/'output.json',record);(folder/'raw_output.txt').write_text(raw);results[index]=record
            except torch.cuda.OutOfMemoryError:
                for index in indices:
                    folder=requests[index][1]
                    record={'raw_output':'','parsed':None,'generation_skipped':False,'generation_failed':True,
                        'termination':'gpu_memory_limit','elapsed_s':time.perf_counter()-start,'batch_amortized_s':0.,
                        'generated_token_ids':[],'generated_token_count':0,'hit_token_limit':False}
                    dump(folder/'output.json',record);(folder/'raw_output.txt').write_text('');results[index]=record
                torch.cuda.empty_cache()
            inputs=None
            outputs=None
            position=end
        return results
