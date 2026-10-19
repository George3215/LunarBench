"""Bounded native-format QA diagnostics on archived TASK3 observations.

This is a custom diagnostic, not an official benchmark or navigation episode.
Labels are stored separately and never passed to the inference functions.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'paper/vlm_goal_navigation/native_diagnostics_20261019'
MODELS = ['qwen3vl8b', 'spatial_interactor8b', 'spatialreasoner',
          'robobrain25_8b_nv', 'spatialrgpt']
SR_SYSTEM = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant '
    'first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning '
    'process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think><answer> answer here </answer>')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    import numpy as np
    from PIL import Image, ImageDraw
    if (ROOT / 'inputs.json').exists():
        raise RuntimeError('Dataset already exists; do not overwrite a frozen diagnostic.')
    source = REPO / 'paper/vlm_goal_navigation/action_only_transfer_v3_20261018/runs/qwen3vl8b/rgbd/event4_odom_plan/trials'
    records, labels, frames = [], {}, []

    def add(key, task, images, question, answer, models=None, **extra):
        records.append(dict(id=key, task=task, images=images, question=question,
                            models=models or MODELS, **extra))
        labels[key] = answer

    def mcq(key, task, images, question, options, correct, models=None):
        # Rotate the correct letter to avoid an all-A visibility/temporal diagnostic.
        n = len(records) % len(options)
        options = options[n:] + options[:n]
        letter = 'ABCD'[options.index(correct)]
        text = 'Question: ' + question + '\nOptions:\n' + '\n'.join(
            f'{"ABCD"[i]}. {v}' for i, v in enumerate(options))
        text += '\nPlease select the correct answer from the options above.'
        add(key, task, images, text, dict(kind='mcq', answer=letter, meaning=correct), models)

    multi = ['qwen3vl8b', 'spatial_interactor8b', 'robobrain25_8b_nv']
    for si, scene in enumerate(['barrier_a0', 'pit_a0', 'window_a0']):
        obs = json.loads((source / scene / 'observations.json').read_text())
        first = obs[0]
        lost = next(x for x in obs if x['depth_sensor']['red_pixel_count'] == 0)
        visible = [x for x in obs if x['depth_sensor']['red_pixel_count'] >= 64
                   and x['depth_sensor']['centroid_uv']]
        # A deterministic image-based sampling rule; no trial outcome is consulted.
        desired = .2 if si == 0 else .8
        offset = min(visible, key=lambda x: abs(x['depth_sensor']['centroid_uv'][0]-desired))
        local = {}
        for role, ob in [('first', first), ('lost', lost), ('offset', offset)]:
            dest = ROOT / 'assets' / f'{scene}_{role}.png'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(ob['rgb_path']).read_bytes())
            p = str(dest.relative_to(REPO))
            local[role] = p
            image = np.array(Image.open(dest).convert('RGB'))
            red = ((image[:, :, 0] > 100) & (image[:, :, 0] > 1.5*image[:, :, 1])
                   & (image[:, :, 0] > 1.5*image[:, :, 2]))
            ys, xs = np.where(red)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None
            frames.append(dict(scene=scene, role=role, source_frame=ob['frame_id'],
                image=p, sha256=sha(dest), red_pixels=int(red.sum()), bbox=bbox,
                centroid=[float(xs.mean()), float(ys.mean())] if len(xs) else None))
            mcq(f'{scene}_{role}_visibility', 'visibility', [p],
                'Is any part of the red target sphere visible in this image?',
                ['Yes, it is visible.', 'No, it is not visible.'],
                'Yes, it is visible.' if len(xs) else 'No, it is not visible.')
            if role != 'lost':
                u = xs.mean()/image.shape[1]
                correct = 'left third' if u < 1/3 else 'right third' if u > 2/3 else 'middle third'
                mcq(f'{scene}_{role}_bearing', 'bearing', [p],
                    'Which horizontal third of the image contains the center of the visible red target sphere?',
                    ['left third', 'middle third', 'right third'], correct)
            if role == 'first':
                truth = dict(kind='point', centroid=[float(xs.mean()), float(ys.mean())], bbox=bbox,
                             width=image.shape[1], height=image.shape[0], coordinate_scale=1000)
                add(f'{scene}_pointing', 'pointing', [p],
                    'Identify a spot within the red sphere. Please provide its 2D coordinates. '
                    'Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple '
                    'contains the x and y coordinates of a point satisfying the conditions above.',
                    truth, ['robobrain25_8b_nv'])
                add(f'{scene}_grounding', 'grounding', [p],
                    'Please provide the bounding box coordinate of the region this sentence describes: the red sphere.',
                    dict(truth, kind='box'), ['robobrain25_8b_nv'])
                # Native SpatialRGPT region branch. Regions come from RGB threshold and a fixed image patch.
                h, w = red.shape
                near = np.zeros((h,w), dtype=np.uint8)
                x0 = int(w*(.12 if si%2 == 0 else .76))
                near[int(.83*h):int(.93*h), x0:x0+int(.1*w)] = 1
                for ri, mask in enumerate([red.astype('uint8'), near]):
                    Image.fromarray(mask*255).save(ROOT/'assets'/f'{scene}_region{ri}.png')
                masks = [str((ROOT/'assets'/f'{scene}_region{ri}.png').relative_to(REPO)) for ri in range(2)]
                add(f'{scene}_region_left', 'region_left', [p],
                    '<image>\nIs region 0 <mask> to the left of region 1 <mask>?',
                    dict(kind='yesno', answer=bool(xs.mean() < np.where(near)[1].mean())),
                    ['spatialrgpt'], masks=masks)
                # Controlled sensor-depth variant: match the relative near-white depth branch,
                # but explicitly NOT a reproduction of the author's DepthAnything pipeline.
                encoded = np.array(Image.open(ob['depth_path']).convert('L'), dtype=float)
                metric = 10*(1-encoded/255)
                inv = 1/np.maximum(metric, .1)
                inv = (inv-inv.min())/max(float(inv.max()-inv.min()), 1e-8)
                dp = ROOT/'assets'/f'{scene}_relative_sensor_depth.png'
                Image.fromarray((inv*255).astype('uint8')).convert('RGB').save(dp)
                closer = 0 if np.median(metric[red]) < np.median(metric[near.astype(bool)]) else 1
                add(f'{scene}_region_depth', 'region_depth_sensor_variant', [p],
                    '<image>\nWhich is closer to the camera, region 0 <mask> or region 1 <mask>?',
                    dict(kind='region', answer=closer, observed_median_depths_m=[float(np.median(metric[red])),
                         float(np.median(metric[near.astype(bool)]))]), ['spatialrgpt'], masks=masks,
                    depth=str(dp.relative_to(REPO)))
        mcq(f'{scene}_visibility_change', 'temporal_visibility', [local['first'], local['lost']],
            'The images are in chronological order. How does visibility of the red target sphere change from the first image to the second?',
            ['visible to not visible', 'not visible to visible', 'visible in both', 'not visible in either'],
            'visible to not visible', multi)
        u0=first['depth_sensor']['centroid_uv'][0]; u1=offset['depth_sensor']['centroid_uv'][0]
        mcq(f'{scene}_image_motion', 'temporal_image_motion', [local['first'], local['offset']],
            'The images are in chronological order. In which horizontal image direction does the red sphere move from the first image to the second?',
            ['toward the left', 'toward the right', 'no horizontal change'],
            'toward the left' if u1<u0 else 'toward the right', multi)
    dump(ROOT/'inputs.json', records)
    dump(ROOT/'labels_evaluator_only.json', labels)
    dump(ROOT/'frames.json', frames)
    montage=Image.new('RGB',(960,720))
    for i,f in enumerate(frames):
        im=Image.open(REPO/f['image']).resize((320,240))
        ImageDraw.Draw(im).text((4,4),f["scene"]+' '+f['role'],fill='yellow')
        montage.paste(im,((i%3)*320,(i//3)*240))
    montage.save(ROOT/'assets/contact_sheet.png')
    dump(ROOT/'DATASET_AUDIT.json',dict(version=1, input_records=len(records), image_records=len(frames),
         per_model={m:sum(m in r['models'] for r in records) for m in MODELS},
         selection='three fixed scenes; first, first zero-red frame, nearest eligible visible frame to u=.2/.8/.8',
         permitted_inputs='camera RGB; RGB-derived region masks; explicitly named sensor-depth variant',
         no_simulator_target_pose=True, no_labels_in_inference=True,
         evaluation_scope='native-format custom diagnostics, not an official benchmark or navigation trial'))


def load_qwen(name):
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
    path=REPO/'.local/models'/name
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    if config.model_type=='qwen2_5_vl':
        from transformers import Qwen2_5_VLProcessor,Qwen2VLImageProcessor,Qwen2VLVideoProcessor,AutoTokenizer
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        processor=Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor.from_pretrained(path,local_files_only=True),
            video_processor=Qwen2VLVideoProcessor(),tokenizer=tok,chat_template=tok.chat_template)
        cls=Qwen2_5_VLForConditionalGeneration
    else:
        processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        cls=Qwen3VLForConditionalGeneration
    model=cls.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    model.generation_config.use_cache=True
    return model,processor


def run_qwen(name, items):
    import torch
    from qwen_vl_utils import process_vision_info
    model,processor=load_qwen(name)
    dump(ROOT/'runs'/name/'environment.json',dict(torch=torch.__version__,
        processor=type(processor).__name__, image_processor_config=processor.image_processor.to_dict(),
        native_checkpoint_image_resolution=True, logits_processors=None, seed=7,
        spatialreasoner_system=SR_SYSTEM if name=='spatialreasoner' else None))
    for row in items:
        dest=ROOT/'runs'/name/row['id']
        if (dest/'result.json').exists():
            continue
        messages=[]
        if name=='spatialreasoner': messages.append(dict(role='system',content=SR_SYSTEM))
        messages.append(dict(role='user',content=[*[dict(type='image',image=str(REPO/p)) for p in row['images']],
                        dict(type='text',text=row['question'])]))
        dump(dest/'serialized_request.json',dict(messages=messages,task=row['task']))
        try:
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            images,videos=process_vision_info(messages)
            inputs=processor(text=[prompt],images=images,videos=videos,padding=True,return_tensors='pt').to('cuda')
            limit=512 if name=='spatialreasoner' else 768 if name=='robobrain25_8b_nv' else 512
            audit=dict(input_ids=inputs['input_ids'][0].tolist(), rendered_prompt=prompt,
                images=[dict(path=p,sha256=sha(REPO/p)) for p in row['images']],
                input_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},
                image_grid_thw=inputs.get('image_grid_thw').tolist(),max_new_tokens=limit,
                generation=dict(do_sample=False,use_cache=True,grammar=None),
                pixel_values_sha256=hashlib.sha256(inputs['pixel_values'].cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            dump(dest/'input_audit.json',audit)
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                out=model.generate(**inputs,max_new_tokens=limit,do_sample=False,use_cache=True)
            generated=out[0,inputs['input_ids'].shape[1]:].tolist()
            raw=processor.tokenizer.decode(generated,skip_special_tokens=True)
            dump(dest/'result.json',dict(id=row['id'],task=row['task'],raw=raw,generated_ids=generated,
                generated_tokens=len(generated),hit_token_limit=len(generated)>=limit,
                elapsed_s=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            print(json.dumps(dict(model=name,id=row['id'],tokens=len(generated),answer=raw[-180:])),flush=True)
            del inputs,out
        except Exception:
            dump(dest/'error.json',dict(traceback=traceback.format_exc()))
            raise


def run_rgpt(items):
    import torch
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(REPO/'.local/vendor/SpatialRGPT'))
    from llava.model.language_model.llava_llama import LlavaLlamaModel,LlavaLlamaConfig
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token,process_images
    path=REPO/'.local/models/spatialrgpt'
    config=LlavaLlamaConfig.from_pretrained(path,local_files_only=True)
    config._name_or_path=str(path);config.resume_path=str(path);config.model_dtype='torch.bfloat16'
    model=LlavaLlamaModel(config=config,low_cpu_mem_usage=True,attn_implementation='sdpa',model_max_length=8192).to('cuda').eval()
    model.llm.config.tokenizer_model_max_length=None
    tok=model.tokenizer;proc=model.get_vision_tower().image_processor
    maskproc=copy.deepcopy(proc);maskproc.do_normalize=False;maskproc.do_convert_rgb=False;maskproc.rescale_factor=1.
    dump(ROOT/'runs/spatialrgpt/environment.json',dict(torch=torch.__version__,native_conv='llama_3',
        native_region_branch=True, depth_variant='simulated RGB-D inverse range, normalized per image; NOT author DepthAnything',
        image_processor_config=proc.to_dict(),max_new_tokens=128,do_sample=False,grammar=None))
    for row in items:
        dest=ROOT/'runs/spatialrgpt'/row['id']
        if (dest/'result.json').exists(): continue
        conv=conv_templates['llama_3'].copy()
        text=row['question']
        if '<image>' not in text:text='<image>\n'+text
        if row.get('depth'):text=text.replace('<mask>','<mask> <depth>')
        conv.append_message(conv.roles[0],text);conv.append_message(conv.roles[1],None)
        prompt=conv.get_prompt()
        dump(dest/'serialized_request.json',dict(question=text,images=row['images'],masks=row.get('masks'),depth=row.get('depth')))
        try:
            ids=tokenizer_image_token(prompt,tok,-200,return_tensors='pt').unsqueeze(0).cuda()
            images=process_images([Image.open(REPO/p).convert('RGB') for p in row['images']],proc,model.config).to('cuda',dtype=torch.bfloat16)
            masks=None;depths=None
            if row.get('masks'):
                ms=[]
                for p in row['masks']:
                    m=(np.array(Image.open(REPO/p).convert('L'))>0).astype('uint8')
                    ms.append(maskproc.preprocess(m[None,...],return_tensors='pt')['pixel_values'][0])
                masks=[torch.vstack(ms).to('cuda',dtype=torch.bfloat16)]
            if row.get('depth'):
                depths=process_images([Image.open(REPO/row['depth']).convert('RGB')],proc,model.config).to('cuda',dtype=torch.bfloat16)
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                _,_,att,_,embeds,_=model.prepare_inputs_labels_for_multimodal(ids,None,torch.ones_like(ids,dtype=torch.bool),None,None,images,masks,depths)
                dump(dest/'input_audit.json',dict(rendered_prompt=prompt,input_ids=ids[0].tolist(),
                    expanded_tokens=embeds.shape[1],image_shape=list(images.shape),
                    mask_shapes=[list(m.shape) for m in masks] if masks else None,
                    depth_shape=list(depths.shape) if depths is not None else None,
                    assets=[dict(path=p,sha256=sha(REPO/p)) for p in row['images']+row.get('masks',[])+([row['depth']] if row.get('depth') else [])],
                    packed_embeddings_sha256=hashlib.sha256(embeds.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
                out=model.llm.generate(inputs_embeds=embeds,attention_mask=att,max_new_tokens=128,
                    do_sample=False,use_cache=True,eos_token_id=[tok.eos_token_id,tok.convert_tokens_to_ids('<|eot_id|>')],pad_token_id=tok.eos_token_id)
            generated=out[0].tolist();raw=tok.decode(generated,skip_special_tokens=True).strip()
            # Transformers 4.37 inputs_embeds generation prepends a BOS.
            tokens=len(generated)-(1 if generated and generated[0]==tok.bos_token_id else 0)
            dump(dest/'result.json',dict(id=row['id'],task=row['task'],raw=raw,generated_ids=generated,
                generated_tokens=tokens,hit_token_limit=tokens>=128,elapsed_s=time.perf_counter()-start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            print(json.dumps(dict(model='spatialrgpt',id=row['id'],answer=raw[-180:])),flush=True)
            del embeds,images,ids,out,masks,depths
        except Exception:
            dump(dest/'error.json',dict(traceback=traceback.format_exc()));raise


def analyse():
    items=json.loads((ROOT/'inputs.json').read_text());labels=json.loads((ROOT/'labels_evaluator_only.json').read_text())
    all_scores=[]
    for m in MODELS:
        for row in items:
            if m not in row['models']:continue
            p=ROOT/'runs'/m/row['id']/'result.json'
            if not p.exists():continue
            r=json.loads(p.read_text());gt=labels[row['id']];raw=r['raw'];pred=None;correct=False;extra={}
            if gt['kind']=='mcq':
                # Final answer only; do not count a letter occurring inside reasoning.
                tagged=re.findall(r'<answer>(.*?)</answer>',raw,re.S|re.I)
                answer=tagged[-1].strip() if tagged else raw.strip()
                match=re.match(r'^\s*(?:\*\*)?\(?([A-D])(?:[\s\).:,]|$)',answer)
                if not match:
                    match=re.search(r'(?:answer is|answer:|correct (?:answer|option) is)\s*(?:\*\*)?\(?([A-D])(?:[\s\).:]|$)',answer,re.I)
                pred=match.group(1).upper() if match else None
                correct=pred==gt['answer']
            elif gt['kind'] in ['point','box']:
                nums=re.findall(r'-?\d+(?:\.\d+)?',raw)
                need=2 if gt['kind']=='point' else 4
                if len(nums)==need:
                    pred=list(map(float,nums));v=pred
                    if gt['kind']=='point':
                        x=v[0]/1000*gt['width'];y=v[1]/1000*gt['height'];b=gt['bbox']
                        correct=b[0]<=x<b[2] and b[1]<=y<b[3]
                        extra['center_error_pixels']=((x-gt['centroid'][0])**2+(y-gt['centroid'][1])**2)**.5
                    else:
                        b=[v[0]/1000*gt['width'],v[1]/1000*gt['height'],v[2]/1000*gt['width'],v[3]/1000*gt['height']];g=gt['bbox']
                        inter=max(0,min(b[2],g[2])-max(b[0],g[0]))*max(0,min(b[3],g[3])-max(b[1],g[1]))
                        union=max(0,b[2]-b[0])*max(0,b[3]-b[1])+(g[2]-g[0])*(g[3]-g[1])-inter
                        extra['iou']=inter/union if union>0 else 0;correct=extra['iou']>=.5
            elif gt['kind']=='yesno':
                match=re.match(r'^\s*(yes|no)\b',raw,re.I)
                pred=match.group(1).lower()=='yes' if match else None;correct=pred is not None and pred==gt['answer']
            elif gt['kind']=='region':
                match=re.match(r'^\s*(?:the\s+)?region\s*([01])\b',raw,re.I)
                pred=int(match.group(1)) if match else None;correct=pred==gt['answer']
            all_scores.append(dict(model=m,id=row['id'],task=row['task'],prediction=pred,
                correct=bool(correct),parseable=pred is not None,hit_token_limit=r['hit_token_limit'],**extra))
    metrics={}
    for m in MODELS:
        rr=[x for x in all_scores if x['model']==m]
        metrics[m]={t:dict(completed=len(a),correct=sum(x['correct'] for x in a),
            parseable=sum(x['parseable'] for x in a),token_limits=sum(x['hit_token_limit'] for x in a))
            for t in sorted(set(x['task'] for x in rr)) if (a:=[x for x in rr if x['task']==t])}
    dump(ROOT/'scores.json',all_scores);dump(ROOT/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['build','run','analyse']);parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    if args.command=='build':build()
    elif args.command=='analyse':analyse()
    else:
        import torch
        torch.set_num_threads(1);torch.manual_seed(7)
        rows=[x for x in json.loads((ROOT/'inputs.json').read_text()) if args.model in x['models']]
        (run_rgpt if args.model=='spatialrgpt' else lambda rows:run_qwen(args.model,rows))(rows)
