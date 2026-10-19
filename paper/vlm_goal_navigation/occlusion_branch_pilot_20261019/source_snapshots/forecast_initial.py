"""Blind native Qwen-family forecasting. This module never reads branch labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'paper/vlm_goal_navigation/occlusion_branch_pilot_20261019'
sys.path.insert(0,str(REPO/'tools'))
from task3_native_diagnostics import load_qwen

SYSTEM='''You are predicting the consequences of short robot motion plans from incomplete visual evidence.
The goal is to regain a view of the red sphere. You receive only prior/current camera RGB-D,
own measured odometry, an uncertain surface-point memory computed from those observations,
and candidate action sequences. No prebuilt map, hidden geometry, target ground-truth position,
or future branch outcome is available. Body +x is forward, +y is left, positive yaw turns left.
Each action lasts 1 second. Commands: FORWARD vx=0.6m/s, BACKWARD vx=-0.5m/s,
STRAFE_LEFT vy=0.5m/s, STRAFE_RIGHT vy=-0.5m/s, TURN_LEFT wz=0.6rad/s,
TURN_RIGHT wz=-0.6rad/s. These are commands, not guaranteed actual displacement.
Aligned grayscale depth is white=0m axial, black=10m+ or invalid.
For each FULL six-second candidate, estimate the probability that the target becomes
stably visible (at least .3s) within the horizon. Also estimate elapsed seconds until
stable visibility, conditional on it occurring, or null if you cannot estimate it.
Account for occlusion, observed obstacles, locomotion feasibility and uncertainty.
Choose the candidate you would execute to recover sight while avoiding ineffective motion.
Return only one JSON object using this exact compact format:
{"choice":"A","forecasts":{"A":{"p":0.5,"time_s":null},"B":{"p":0.5,"time_s":null},"C":{"p":0.5,"time_s":null},"D":{"p":0.5,"time_s":null}}}
The values above are format examples, not predictions. Probabilities must be in [0,1];
non-null times must be in (0,6]. Do not add reasoning text or use a zero time for failure.
'''


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def parse(raw):
    if '</think>' in raw:raw=raw.rsplit('</think>',1)[1]
    if '<answer>' in raw:raw=raw.split('<answer>',1)[1].split('</answer>',1)[0]
    decoder=json.JSONDecoder()
    for i,c in enumerate(raw):
        if c!='{':continue
        try:value,_=decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:continue
        if not isinstance(value,dict) or value.get('choice') not in 'ABCD' or len(value['choice'])!=1:continue
        forecasts=value.get('forecasts')
        if not isinstance(forecasts,dict) or set(forecasts)!=set('ABCD'):continue
        valid=True
        for prediction in forecasts.values():
            if not isinstance(prediction,dict):valid=False;break
            p=prediction.get('p');t=prediction.get('time_s')
            if isinstance(p,bool) or not isinstance(p,(float,int)) or not 0<=p<=1:valid=False
            if t is not None and (isinstance(t,bool) or not isinstance(t,(float,int)) or not 0<t<=6):valid=False
        if valid:return value
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['qwen3vl8b','spatial_interactor8b'])
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(7)
    # Only collection manifests and PUBLIC requests are opened here, never evaluator files.
    states=json.loads((ROOT/'state_index.json').read_text())
    if args.limit:states=states[:args.limit]
    model,processor=load_qwen(args.model)
    folder=ROOT/'models'/args.model
    protocol=dict(system=SYSTEM,max_new_tokens=768,do_sample=False,grammar=None,seed=7,
        input_scope='same public RGB-D/own-odometry/sensor-derived goal belief and candidates',
        image_processor=processor.image_processor.to_dict(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (folder/'protocol.json').exists():assert json.loads((folder/'protocol.json').read_text())==protocol,'inference protocol drift'
    else:dump(folder/'protocol.json',protocol)
    for state in states:
        key=state['id'];dest=folder/key
        source=ROOT/'states'/key/'request.json'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==state['request_sha256']
        request=json.loads(source.read_text())
        if (dest/'result.json').exists():continue
        assert set(request)=={'target','observations','coarse_target_belief','candidate_actions','horizon_s','coordinate_convention','allowed_sensors'}
        text={k:v for k,v in request.items() if k!='observations'}
        text['observations']=[{k:v for k,v in obs.items() if k not in ['rgb','depth','rgb_sha256','depth_sha256']} for obs in request['observations']]
        content=[dict(type='text',text=json.dumps(text,ensure_ascii=False))]
        saved=[dict(type='text',text=content[0]['text'])];image_audit=[]
        # Keep initial, last-visible, current order. Six images, no future frame.
        for role,obs in zip(['initial','last_visible','current'],request['observations']):
            for kind in ['rgb','depth']:
                path=REPO/obs[kind]
                h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==obs[kind+'_sha256']
                label=f'{role} {kind}, observation time {obs["time_s"]:.3f}s'
                content.extend([dict(type='text',text=label),dict(type='image',image=Image.open(path).convert('RGB'))])
                saved.extend([dict(type='text',text=label),dict(type='image',path=obs[kind],sha256=h)])
                image_audit.append(dict(path=obs[kind],sha256=h))
        messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)]
        dump(dest/'serialized_request.json',dict(messages=[messages[0],dict(role='user',content=saved)],
            public_request_sha256=state['request_sha256']))
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt').to('cuda')
        dump(dest/'input_audit.json',dict(rendered_prompt=prompt,input_ids=inputs['input_ids'][0].tolist(),
            image_grid_thw=inputs['image_grid_thw'].tolist(),images=image_audit,
            pixel_values_shape=list(inputs['pixel_values'].shape),
            pixel_values_sha256=hashlib.sha256(inputs['pixel_values'].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=768,do_sample=False,use_cache=True)
        ids=out[0,inputs['input_ids'].shape[1]:].tolist();raw=processor.tokenizer.decode(ids,skip_special_tokens=True)
        value=dict(id=key,raw=raw,parsed=parse(raw),generated_ids=ids,generated_tokens=len(ids),
            hit_token_limit=len(ids)>=768,elapsed_s=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        dump(dest/'result.json',value)
        print(json.dumps(dict(model=args.model,state=key,parsed=value['parsed'],tokens=len(ids))),flush=True)
        del inputs,out


if __name__=='__main__':main()
