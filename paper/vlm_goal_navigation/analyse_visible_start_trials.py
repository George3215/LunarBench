"""Paired depth comparison, source audits and self-contained trial browser."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(root):
    protocol=read(root/'protocol.json')
    summaries=[];browser=[];audits=[]
    for mode in ('rgb','rgbd'):
        for folder in sorted((root/mode/'trials').glob('*')):
            if not folder.is_dir():continue
            gate=read(folder/'approach_gate.json')
            init=read(folder/'initialization.json')
            if not gate or not init:continue
            trace=[read(path) for path in sorted((folder/'steps').glob('*/transition.json'))]
            states=rows(folder/'state.jsonl');result=read(folder/'result.json')
            prefix=[t for t in trace if t['step']<=gate['prefix_actions']]
            autonomous=[t for t in trace if t['step']>gate['prefix_actions']]
            outputs=[read(folder/p) for t in trace for p in t['inference_records']]
            decisions=[t for t in autonomous if t['decision']]
            sensor=read(folder/'initial_evidence/depth_sensor.json')
            recover=next((t['step'] for t in autonomous if t['after_evaluator']['target_visible_rgb_pixels']>=20),None)
            final=result['final_evaluator'] if result else (trace[-1]['after_evaluator'] if trace else gate['lost'])
            metadata=read(folder/'model_metadata.json',{})
            wall_ids={int(k) for k,v in metadata.get('geom_names',{}).items() if v=='task3_occluder_wall'}
            prefix_states=[s for s in states if 1<=s['step']<=gate['prefix_actions']]
            summary={'scene':folder.name,'mode':mode,'completed':bool(result),'success':result['success'] if result else None,
                'termination':result['termination'] if result else 'running',
                'controlled_approach_actions':len(prefix),'autonomous_actions':sum(t['action'] is not None for t in autonomous),
                'autonomous_decision_rounds':len(autonomous),
                'total_actions':sum(t['action'] is not None for t in trace),'inferences':len(outputs),
                'initial_visible_pixels':gate['initial']['target_visible_rgb_pixels'],
                'initial_depth_camera_range_m':sensor['red_region_median_surface_range_m'],
                'initial_axial_depth_m':sensor['red_region_median_axial_depth_m'],
                'occluded_start_pixels':gate['lost']['target_segment_pixels'],
                'first_recovered_action':recover-gate['prefix_actions'] if recover else None,
                'final_visible_pixels':final['target_visible_rgb_pixels'],
                'final_visible_surface_range_m':final['median_target_surface_range_m'],
                'false_near_claims':sum(t['decision']['proximity']=='near' and not t['before_evaluator']['near_and_visible'] for t in decisions),
                'false_visible_zero_pixels':sum(t['decision']['target_visible'] and t['before_evaluator']['target_segment_pixels']==0 for t in decisions),
                'forward_stalls_below_5cm':sum(t['action']=='FORWARD' and t['actual_displacement_xy_m']<.05 for t in autonomous),
                'action_counts':dict(collections.Counter(t['action'] for t in autonomous)),
                'state_samples':len(states),
                'path_m_500hz':result['path_m'] if result else None,
                'wall_contact_10hz_samples':sum(any(c['geom1'] in wall_ids or c['geom2'] in wall_ids for c in s['contacts']) for s in states),
                'max_abs_actuator_force_10hz':max((max(abs(v) for v in s['actuator_force']) for s in states),default=None),
                'max_base_linear_speed_10hz_m_s':max((math.sqrt(sum(v*v for v in s['qvel'][:3])) for s in states),default=None),
                'model_stopped':bool(result and result['model_stopped']),
                'min_upright':min((s['evaluation']['upright'] for s in states),default=None)}
            summaries.append(summary)
            if result:
                metadata=read(folder/'model_metadata.json')
                checks={'trial':f'{mode}/{folder.name}',
                    'initial_visible':gate['initial']['target_visible_rgb_pixels']>=40,
                    'physical_loss_gate':gate['geometric_occlusion'] and gate['lost']['target_segment_pixels']<=2,
                    'sustained_loss_last_three_prefix_samples':len(prefix_states)>=3 and all(s['evaluation']['target_segment_pixels']<=2 and s['evaluation']['upright']>=.25 and s['evaluation']['in_bounds'] for s in prefix_states[-3:]),
                    'source_actions_correct':all(t['action_source']=='fixed task setup, not VLM' and not t['inference_records'] for t in prefix) and all(t['action_source']=='VLM' for t in autonomous),
                    'input_hashes':all(sha(folder/t['input_image'])==t['input_image_sha256'] for t in trace),
                    'raw_output_matches':all(read(folder/p)['raw_output']==raw for t in trace for p,raw in zip(t['inference_records'],t['model_output_raw'])),
                    'network_input_image_hashes':all(sha(Path(im['image_path']))==im['image_sha256'] for t in autonomous for p in t['inference_records'] for im in read((folder/p).with_name('network_input.json'))['images']),
                    'network_mode_correct':all(read((folder/p).with_name('messages.json'))['mode']==mode for t in autonomous for p in t['inference_records']),
                    'no_depth_in_rgb_input':all('depth_sensor' not in json.dumps(read((folder/p).with_name('messages.json'))['messages'][1]) and all('depth.png' not in im['image_path'] for im in read((folder/p).with_name('network_input.json'))['images']) for t in autonomous for p in t['inference_records']) if mode=='rgb' else True,
                    'depth_images_in_rgbd_input':all(len(read((folder/p).with_name('network_input.json'))['images'])==4 for t in autonomous for p in t['inference_records']) if mode=='rgbd' else True,
                    'action_counts':sum(t['action'] is not None for t in trace)==result['action_steps'] and sum(t['action'] is not None for t in autonomous)==result['autonomous_actions'],
                    'inference_counts':len(outputs)==result['inference_count'],
                    'state_counts':len(states)==result['state_samples'],
                    'finite_physics':all(all(math.isfinite(v) for v in s['qpos']+s['qvel']+s['ctrl']+s['actuator_force']) for s in states),
                    'contact_ids':all(0<=c['geom1']<metadata['ngeom'] and 0<=c['geom2']<metadata['ngeom'] for s in states for c in s['contacts']),
                    'budget':len(trace)<=3000 and len(outputs)<=2*len(autonomous),
                    'success':not result['success'] or (final['near_and_visible'] and result['success_streak']>=3)}
                audits.append(checks)
            steps=[]
            for t in trace:
                sf=folder/'steps'/f"{t['step']:04d}"
                steps.append(dict(t,
                    input_image=str((folder/t['input_image']).relative_to(root)),
                    after_image=str((sf/'after_evaluator/rgb.png').relative_to(root)),
                    depth_image=str((sf/'input_depth.png').relative_to(root)),
                    depth_sensor=read(sf/'depth_sensor.json'),
                    record_links=[str((folder/p).relative_to(root)) for p in t['inference_records']]))
            browser.append({'scene':f'{mode} / {folder.name}','summary':summary,'steps':steps,
                'initial_image':str((folder/'initial_evidence/rgb.png').relative_to(root)),
                'initial_depth_image':str((folder/'initial_evidence/input_depth.png').relative_to(root)),
                'initial_sensor':sensor})
    pairs=[]
    pair_checks=[]
    scene_names=sorted({s['scene'] for s in summaries})
    for name in scene_names:
        records={s['mode']:s for s in summaries if s['scene']==name}
        if len(records)!=2:continue
        a,b=[root/m/'trials'/name for m in ('rgb','rgbd')]
        ga,gb=read(a/'approach_gate.json'),read(b/'approach_gate.json')
        pair_checks.append({'scene':name,'identical_initial_rgb':sha(a/'initial_evidence/rgb.png')==sha(b/'initial_evidence/rgb.png'),
            'identical_initial_depth':sha(a/'initial_evidence/input_depth_m.npy')==sha(b/'initial_evidence/input_depth_m.npy'),
            'identical_occluded_rgb':ga['rgb_sha256']==gb['rgb_sha256'],
            'identical_occluded_qpos':ga['lost_qpos']==gb['lost_qpos'],
            'identical_approach_actions':ga['prefix_actions']==gb['prefix_actions']})
        if all(s['completed'] for s in records.values()):
            pairs.append({'scene':name,'rgb_success':records['rgb']['success'],'rgbd_success':records['rgbd']['success'],
                'rgb_autonomous_actions':records['rgb']['autonomous_actions'],'rgbd_autonomous_actions':records['rgbd']['autonomous_actions'],
                'rgb_false_near':records['rgb']['false_near_claims'],'rgbd_false_near':records['rgbd']['false_near_claims']})
    reproduction=read(root/'network_input_reproduction_audit.json',[])
    metrics={'completed':sum(s['completed'] for s in summaries),'expected':48,
        'by_mode':{},'paired_completed':len(pairs),'pairs':pairs,
        'paired_rgb_success_rate':sum(p['rgb_success'] for p in pairs)/len(pairs) if pairs else None,
        'paired_rgbd_success_rate':sum(p['rgbd_success'] for p in pairs)/len(pairs) if pairs else None,
        'paired_success_rate_difference':sum(int(p['rgbd_success'])-int(p['rgb_success']) for p in pairs)/len(pairs) if pairs else None,
        'rgbd_only_success':sum(p['rgbd_success'] and not p['rgb_success'] for p in pairs),
        'rgb_only_success':sum(p['rgb_success'] and not p['rgbd_success'] for p in pairs),
        'both_success':sum(p['rgb_success'] and p['rgbd_success'] for p in pairs),
        'neither_success':sum(not p['rgb_success'] and not p['rgbd_success'] for p in pairs),
        'all_artifact_checks_pass':bool(audits) and all(all(v for k,v in a.items() if k!='trial') for a in audits),
        'network_input_reproduction_checks_pass':bool(reproduction) and all(c['unpadded_token_ids_match'] and c['all_image_preprocessing_matches'] for c in reproduction),
        'all_available_pair_checks_pass':bool(pair_checks) and all(all(v for k,v in a.items() if k!='scene') for a in pair_checks)}
    for mode in ('rgb','rgbd'):
        completed=[s for s in summaries if s['mode']==mode and s['completed']]
        count=len(completed);successes=sum(s['success'] for s in completed)
        metrics['by_mode'][mode]={'completed':count,'expected':24,'success_count':successes,
            'success_rate_completed':successes/count if count else None,
            'terminations':dict(collections.Counter(s['termination'] for s in completed)),
            'recovered_target_count':sum(s['first_recovered_action'] is not None for s in completed),
            'false_near_claims':sum(s['false_near_claims'] for s in completed),
            'forward_stalls':sum(s['forward_stalls_below_5cm'] for s in completed)}
    references=read(root/'discrete_physical_reference/summary.json',[])
    metrics['same_discrete_action_reference_completed']=len(references)
    metrics['same_discrete_action_reference_success_count']=sum(r['success'] for r in references)
    for filename,value in [('run_metrics.json',metrics),('artifact_audit.json',audits),('paired_initial_state_audit.json',pair_checks),('trial_summary.json',summaries)]:
        (root/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False))
    if summaries:
        with (root/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,list(summaries[0]));w.writeheader();w.writerows(summaries)
    text=['# 初始可见—遮挡—自主重新接近：RGB / RGB-D 配对实验','',
        f'当前完成 {metrics["completed"]}/48；完成配对 {len(pairs)}/24。未完成 trial 不记作失败，未完成的配对不用于深度效果比较。','',
        '全部24个新场景已通过实际物理准入：初始>=40红球可见像素；前进后球在相机FOV内被实体墙遮挡；独立参考绕行实际满足近距成功。新场景替换旧场景，不与旧结果混算。',
        '每个trial包含相同固定FORWARD接近段（任务设置，不是模型动作），接近后连续保留实际机器人和步态控制状态。VLM从确认遮挡位置开始自主控制，接收初始可见RGB记忆、当前RGB、近期自身动作。两组起点/接近段/动作空间/预算/解码一致。',
        'RGB-D额外输入初始与当前对齐深度图，以及由RGB红色区域与深度计算的公制表面距离；不输入分割真值或目标坐标。这比较的是深度图+传感器距离辅助的整体效果，不能单独归因于深度图编码。',
        '初始距离来自深度相机的红色区域，轴向深度经过内参反投影得到三维欧氏表面距离。深度为MuJoCo理想渲染，未模拟实体深度相机噪声。',
        '成功：目标可见像素>=20且可见表面到相机的三维距离中位数<1m，连续三个10Hz样本满足，机器人未跌倒/越界。相机深度而非身体XY距离判断成功。STOP不满足条件则提前停止。',
        '累计动作预算1000→2000→3000，含固定接近段；终止后不重试。每动作最多一次格式修复。保存每次公开原始输出和简短决策说明；不是隐藏内部思维。机器人记录为MuJoCo实际物理状态，不是实体机械狗。','',
        '|条件|已完成|完整成功|已完成成功率|重新看见|提前STOP|越界|','|---|---:|---:|---:|---:|---:|---:|']
    for mode,m in metrics['by_mode'].items():
        rate=f"{m['success_rate_completed']:.1%}" if m['completed'] else '待完成'
        text.append(f"|{mode}|{m['completed']}/24|{m['success_count']}|{rate}|{m['recovered_target_count']}|{m['terminations'].get('premature_stop',0)}|{m['terminations'].get('out_of_bounds',0)}|")
    text+=['',f'已完成配对：深度组独有成功 {metrics["rgbd_only_success"]}；RGB独有成功 {metrics["rgb_only_success"]}；两组成功 {metrics["both_success"]}；两组失败 {metrics["neither_success"]}。',
        '实际状态分析包括500Hz积分路径长度、10Hz采样的站立稳定性、基座速度、实际执行器力和墙体接触次数；10Hz峰值只是采样峰值，不保证捕捉所有瞬时峰值。success由评估器首次到达终止，不能把它等同于模型主动STOP；提前STOP则独立报告。',
        '固定seed和每场景每条件一次运行只支持描述性配对结果，不能声称已证明统计显著的普遍提升。完整48个trial及记录审计通过后才能给最终结果。24个场景属于同一受控下降遮挡机制的几何变化，不能代表原有全部地形机制或真实世界导航难度。',
        '深度原始值使用逐位无损字节重排压缩和内容去重：步骤内原.npy文件由.npy.json引用共享depth_archive/*.npz。运行root/depth_archive.py中的load_depth(original_npy_path)可恢复float32数组；归档前已验证恢复后.npy字节SHA256与原文件一致。PNG和模型网络输入保持原样。',
        'proximity=near但未满足到达条件的计数只是标签与评估判据的差异；不能把每次near标签都解释为明确声称小于1m。提前STOP时不满足距离/可见条件才是明确的终止判断失败。模型confidence没有统计校准保证。',
        f'同正式离散动作空间的物理参考：已完成 {len(references)}/24，实际成功 {sum(r["success"] for r in references)}。参考选择器使用特权路点，结果只验证可行性，不输入VLM、不计入VLM成功率。',
        '', '|场景|条件|初始深度表面距离m|固定接近动作|自主动作|结果|near标签未满足到达|','|---|---|---:|---:|---:|---|---:|']
    for s in summaries:
        d=s['initial_depth_camera_range_m'];text.append(f"|{s['scene']}|{s['mode']}|{d:.3f}|{s['controlled_approach_actions']}|{s['autonomous_actions']}|{s['termination']}|{s['false_near_claims']}|")
    text+=['','记录入口：TRIALS.html逐步浏览；steps/*/input_rgb.png、input_depth.png、input_depth_m.npy、depth_sensor.json；inference_*/messages.json、rendered_prompt.txt、network_input.json、output.json；state.jsonl保存实际qpos/qvel/ctrl/actuator_force/contacts；paired_initial_state_audit.json核对两组相同起点和遮挡状态。']
    (root/'REPORT.md').write_text('\n'.join(text))
    make_browser(root,browser,protocol)
    from plot_visible_start_trials import plot
    plot(root)
    return metrics


def make_browser(root,data,protocol):
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    prompt=json.dumps(protocol['system_prompt'],ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><meta charset="utf-8"><title>Qwen3-VL 初始可见 / RGB-D 配对 Trial</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#182238;margin:24px}select,button,input{font:inherit;padding:8px}header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.row{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}.panel{background:white;border-radius:10px;padding:16px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px ui-monospace}h1{font-size:26px}#range{flex:1;min-width:200px}.note{color:#536176}a{color:#175ac4}@media(max-width:800px){.row{grid-template-columns:1fr}}</style>
<h1>Qwen3-VL · 初始可见 → 实体遮挡 → 自主寻找并靠近</h1>
<p class="note">相同24个场景分别运行RGB与RGB-D。固定前进接近段是任务设置，随后由模型自主控制；初始可见图像保留在网络输入中。深度组额外输入对齐深度图和RGB红色区域的深度测量。评估真值不输入模型。记录为MuJoCo实际运动。</p>
<header><select id="scene"></select><button id="prev">上一步</button><input id="range" type="range" min="0" value="0"><button id="next">下一步</button><span id="label"></span></header>
<details class="panel" open><summary>初始可见图像与深度相机距离</summary><div class="row"><div><img id="initial"><p>初始RGB（作为模型视觉记忆）</p></div><div><img id="initialDepth"><p>初始深度，仅RGB-D送入模型；RGB组只保存作分析。</p></div></div><pre id="initialMeasurement"></pre></details>
<div class="row"><div class="panel"><h3>动作前相机RGB</h3><img id="before"></div><div class="panel"><h3>动作执行后RGB</h3><img id="after"></div></div>
<div class="row"><div class="panel"><h3>当前对齐深度</h3><img id="depth"><p id="depthNote"></p><pre id="sensor"></pre></div><div class="panel"><h3>模型原始输出（含格式重试）</h3><pre id="raw"></pre><div id="links"></div></div></div>
<div class="panel"><h3>决策与真实执行对照（评估字段不输入模型）</h3><pre id="evaluation"></pre></div>
<details class="panel"><summary>固定System Prompt</summary><pre id="prompt"></pre></details><details class="panel"><summary>当前Trial汇总</summary><pre id="summary"></pre></details>
<script>const trials=PAYLOAD;const prompt=PROMPT;const scene=document.getElementById('scene'),range=document.getElementById('range');
for(let i=0;i<trials.length;i++){const o=document.createElement('option');o.value=i;o.textContent=trials[i].scene;scene.appendChild(o)}document.getElementById('prompt').textContent=prompt;
function draw(){const t=trials[Number(scene.value)];if(!t)return;const s=t.steps[Number(range.value)];document.getElementById('summary').textContent=JSON.stringify(t.summary,null,2);document.getElementById('initial').src=t.initial_image;document.getElementById('initialDepth').src=t.initial_depth_image;document.getElementById('initialMeasurement').textContent=JSON.stringify(t.initial_sensor,null,2);if(!s)return;
document.getElementById('label').textContent=`步骤 ${s.step} · ${s.phase==='controlled_approach'?'固定接近段（非VLM）':'VLM自主段'} · ${s.action}`;document.getElementById('before').src=s.input_image;document.getElementById('after').src=s.after_image;document.getElementById('depth').src=s.depth_image;document.getElementById('depthNote').textContent=t.summary.mode==='rgbd'?'RGB-D：自主段中这张深度图实际送入网络。白色=0m，黑色=10m以上或无效。':'RGB：此深度仅用于记录和评估，不送入网络。';document.getElementById('sensor').textContent=JSON.stringify(s.depth_sensor,null,2);
document.getElementById('raw').textContent=s.model_output_raw.length?s.model_output_raw.map((r,i)=>`Inference ${i+1}\\n${r}`).join('\\n\\n'):'固定接近动作，无模型推理，不伪造模型输出。';
document.getElementById('evaluation').textContent=JSON.stringify({phase:s.phase,action_source:s.action_source,action:s.action,actual_displacement_xy_m:s.actual_displacement_xy_m,inference_s:s.inference_s,decision:s.decision,before_evaluator:s.before_evaluator,after_evaluator:s.after_evaluator,termination:s.termination},null,2);
const links=document.getElementById('links');links.replaceChildren();for(const path of s.record_links){const a=document.createElement('a');a.href=path;a.textContent='完整模型输出 JSON ';links.appendChild(a)}}
scene.onchange=()=>{range.max=Math.max(0,(trials[Number(scene.value)]?.steps.length||0)-1);range.value=0;draw()};range.oninput=draw;document.getElementById('prev').onclick=()=>{range.value=Math.max(0,Number(range.value)-1);draw()};document.getElementById('next').onclick=()=>{range.value=Math.min(Number(range.max),Number(range.value)+1);draw()};if(trials.length)scene.onchange();</script>'''
    (root/'TRIALS.html').write_text(html.replace('PAYLOAD',payload).replace('PROMPT',prompt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(analyse(a.root.resolve()),indent=2,ensure_ascii=False))
