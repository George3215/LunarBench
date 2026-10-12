"""Audit six paired observation conditions and publish a local trial browser."""
import json, math, hashlib, collections, argparse, os
from pathlib import Path

def read(p,default=None):
    return json.loads(p.read_text()) if p.exists() else default

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyse(root):
    protocol=read(root/'protocol.json',{});summaries=[];checks=[];trials=[];pairs=collections.defaultdict(list)
    def url(p):return os.path.relpath(p,root)
    for folder in sorted(root.glob('*/*/trials/*')):
        gate=read(folder/'approach_gate.json');frames=read(folder/'observations.json',[])
        if not gate:continue
        mode,horizon=folder.relative_to(root).parts[:2]
        traces=[read(p) for p in sorted(folder.glob('steps/*/transition.json'))];result=read(folder/'result.json')
        autonomous=[t for t in traces if t['phase']=='vlm_autonomous'];inferences=[];steps=[]
        audit={'trial':f'{mode}/{horizon}/{folder.name}','initial_visible':gate['initial']['target_visible_rgb_pixels']>=40,
            'entity_occlusion':gate['loss_kind']=='terrain_occlusion' and gate['lost']['target_segment_pixels']<=2,
            'context_selection':True,'image_hashes':True,'token_counts':True,'raw_outputs':True,'no_truncation':True,
            'budget':not result or result['action_steps']<=3000,'success_criterion':True}
        for t in autonomous:
            records=[]
            for rel,raw in zip(t['inference_records'],t['model_output_raw']):
                path=folder/rel;output=read(path);context=read(path.parent/'context.json');network=read(path.parent/'network_input.json');messages=read(path.parent/'messages.json')
                last=t['frame_before'];expected=[last] if horizon=='current' else ([0,last] if last else [0]) if horizon=='first_current' else list(range(last+1))
                audit['context_selection'] &= context['input_frame_ids']==expected and context['all_recorded_frame_count']==last+1
                audit['no_truncation'] &= context['truncated'] is False
                audit['token_counts'] &= sum(network['attention_mask'])==context['input_tokens']
                audit['image_hashes'] &= all(sha(Path(i['image_path']))==i['image_sha256'] for i in network['images'])
                audit['raw_outputs'] &= output['raw_output']==raw and (path.parent/'raw_output.txt').read_text()==raw
                inferences.append(output)
                records.append({'context':context,'output':output,'images':[{'label':i['label'],'url':url(Path(i['image_path']))} for i in network['images']],
                    'messages':messages,'network_url':url(path.parent/'network_input.json'),'prompt_url':url(path.parent/'rendered_prompt.txt')})
            steps.append({'transition':t,'records':records,'rgb':url(Path(frames[t['frame_before']]['rgb_path'])),'depth':url(Path(frames[t['frame_before']]['depth_path']))})
        if result and result['success']:
            f=result['final_evaluator'];audit['success_criterion']=f['near_and_visible'] and f['target_visible_rgb_pixels']>=20 and f['median_target_surface_range_m']<1 and result['success_streak']>=3
        state_path=folder/'state.jsonl';state_count=0;force_peak=0.;speed_peak=0.;upright=1.;wall_contacts=0
        metadata=read(folder/'model_metadata.json',{});wall_ids={int(k) for k,v in metadata.get('geom_names',{}).items() if v=='navigation_added_occluder' or v.endswith(('_wall','_wall_low','_wall_high','_sill'))}
        if state_path.exists():
            prefix_tail=[];finite_states=True
            with state_path.open() as stream:
                for line in stream:
                    s=json.loads(line);state_count+=1
                    finite_states &= all(math.isfinite(v) for key in ('qpos','qvel','ctrl','actuator_force') for v in s[key])
                    if 1<=s['step']<=gate['prefix_actions']:
                        prefix_tail=(prefix_tail+[s['evaluation']])[-3:]
                    force_peak=max(force_peak,max(map(abs,s['actuator_force']),default=0.));speed_peak=max(speed_peak,math.sqrt(sum(v*v for v in s['qvel'][:3])))
                    upright=min(upright,s['evaluation']['upright']);wall_contacts+=any(c['geom1'] in wall_ids or c['geom2'] in wall_ids for c in s['contacts'])
            audit['finite_physics_states']=finite_states
            audit['sustained_loss']=len(prefix_tail)==3 and all(s['target_segment_pixels']<=2 and s['in_bounds'] and s['upright']>=.25 for s in prefix_tail)
        summary={'scene':folder.name,'mode':mode,'horizon':horizon,'completed':bool(result),'success':result['success'] if result else None,
            'termination':result['termination'] if result else 'running','resource_limit':result.get('resource_limit',False) if result else False,
            'actions':result['action_steps'] if result else len(traces),'autonomous_actions':result['autonomous_actions'] if result else len(autonomous),
            'initial_depth_camera_range_m':read(folder/'initialization.json')['initial_depth_camera_range_m'],
            'first_recovered_action':next((t['step']-gate['prefix_actions'] for t in autonomous if t['after_evaluator']['target_visible_rgb_pixels']>=20),None),
            'max_context_tokens':max((s['records'][0]['context']['input_tokens'] for s in steps),default=0),
            'max_input_frames':max((s['records'][0]['context']['input_frame_count'] for s in steps),default=0),
            'unsupported_evidence_claims':sum(bool(o.get('unsupported_evidence_frame_ids')) or o.get('unsupported_depth_claim',False) or o.get('unsupported_past_claim',False) for o in inferences),
            'false_visible_claims':sum(t['decision']['target_visible'] and t['before_evaluator']['target_segment_pixels']==0 for t in autonomous if t['decision']),
            'stalled_forward_actions':sum(t['action']=='FORWARD' and t['actual_displacement_xy_m']<.05 for t in autonomous),
            'state_samples':state_count,'sampled_actuator_force_peak':force_peak,'sampled_base_speed_peak_m_s':speed_peak,
            'min_upright':upright,'wall_contact_samples':wall_contacts,'path_m':result['path_m'] if result else None}
        summaries.append(summary);checks.append(audit);pairs[folder.name].append(gate)
        trials.append({'id':audit['trial'],'summary':summary,'gate':gate,'initial_rgb':url(Path(frames[0]['rgb_path'])),
            'initial_depth':url(Path(frames[0]['depth_path'])),'steps':steps,'result':result})
    pairing=[]
    for scene,gates in pairs.items():
        fields=('initial_rgb_sha256','lost_rgb_sha256','initial_depth_sha256','loss_qpos','prefix_actions')
        pairing.append({'scene':scene,'available_conditions':len(gates),'matching_initial_and_loss':all(all(g[k]==gates[0][k] for k in fields) for g in gates)})
    aggregates=[]
    for mode in ('rgb','rgbd'):
        for horizon in ('current','first_current','all_history'):
            selected=[s for s in summaries if s['mode']==mode and s['horizon']==horizon and s['completed']]
            success=sum(bool(s['success']) for s in selected);resource=sum(s['resource_limit'] for s in selected)
            aggregates.append({'mode':mode,'horizon':horizon,'completed':len(selected),'success':success,'resource_limits':resource,
                'observed_completion_fraction':success/len(selected) if selected else None,'terminations':dict(collections.Counter(s['termination'] for s in selected))})
    audit_pass=all(all(v for k,v in c.items() if k!='trial') for c in checks) and all(p['matching_initial_and_loss'] for p in pairing)
    completed=sum(s['completed'] for s in summaries);expected=protocol.get('trial_count',186)
    report={'expected':expected,'completed':completed,'audit_pass':audit_pass,'all_complete':completed==expected and audit_pass,
        'conditions':aggregates,'trials':summaries,'pairing':pairing,
        'limitations':['One deterministic trial per scene and condition; descriptive paired comparison, no statistical reliability claim.',
            '31 scenarios cover 15 terrain families plus 16 window parameter variants; variants are not independent terrain types. Adjusted window targets are on the near side, so this is not the original H2 benchmark.',
            'Policy evidence citations are public self-reports, not causal explanations of internal computation.',
            'MuJoCo Go2 physical state; no physical hardware dog was connected.',
            'All history means every saved action-boundary observation; evaluator 10Hz frames are not model inputs.',
            'Explicit runtime context/GPU limits are reported separately and cannot establish VLM navigation inability.']}
    (root/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(root/'audit.json').write_text(json.dumps({'pass':audit_pass,'trials':checks,'pairing':pairing},indent=2))
    data=json.dumps({'report':report,'trials':trials},ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>Task3 六条件 Trial 浏览</title><style>body{font:15px sans-serif;background:#10151d;color:#eee;margin:24px}select,button,input{font:inherit;padding:8px}img{max-width:420px;width:45%;margin:8px}pre{white-space:pre-wrap;background:#202938;padding:12px;max-height:450px;overflow:auto}a{color:#9cd0ff}.frames img{width:160px}.frames figure{display:inline-block;margin:5px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #555}</style><h1>Task3：全部地形 × 图像上下文 × 深度</h1><p>首帧 → 实体遮挡 → 自主找回并靠近。成功要求目标在相机中可见且表面三维距离 &lt;1 m，连续三个物理样本成立。模型说明是公开策略自述，不代表内部思维或因果解释。</p><div id="overview"></div><select id="trial"></select><p id="label"></p><div id="initial"></div><input id="step" type="range" min="0" value="0"><button onclick="move(-1)">上一步</button><button onclick="move(1)">下一步</button><div id="content"></div><script>const D=__DATA__;const $=x=>document.getElementById(x);function esc(x){return String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}function pre(x){return '<pre>'+esc(JSON.stringify(x,null,2))+'</pre>'}function pic(p){return '<img loading="lazy" src="'+esc(p)+'">'}$('overview').innerHTML=pre(D.report.conditions)+'<p>完成 '+D.report.completed+'/'+D.report.expected+'；记录审计 '+D.report.audit_pass+'。资源终止单独报告；样本仅为描述性比较。</p>';D.trials.forEach((t,i)=>{let o=document.createElement('option');o.value=i;o.textContent=t.id+' — '+t.summary.termination;$('trial').appendChild(o)});function show(){let t=D.trials[$('trial').value];if(!t)return;$('label').textContent=JSON.stringify(t.summary);$('initial').innerHTML='<h3>评估浏览用首帧（current 组模型未接收此首帧）</h3>'+pic(t.initial_rgb)+pic(t.initial_depth);$('step').max=Math.max(0,t.steps.length-1);let s=t.steps[+$('step').value];if(!s){$('content').innerHTML='<p>等待自主推理记录</p>';return}let h='<h3>动作 '+s.transition.step+'，当前观测</h3>'+pic(s.rgb)+pic(s.depth)+pre(s.transition);for(let r of s.records){h+='<h3>本次真实网络输入：全部帧</h3>'+pre(r.context)+'<div class="frames">';for(let im of r.images)h+='<figure>'+pic(im.url)+'<figcaption>'+esc(im.label)+'</figcaption></figure>';h+='</div><a href="'+r.prompt_url+'">完整渲染提示词</a> | <a href="'+r.network_url+'">网络 token / 图像 hash</a>'+pre(r.messages)+'<h3>完整输出与公开策略说明</h3>'+pre(r.output)}$('content').innerHTML=h}function move(n){$('step').value=Math.max(0,Math.min(+$('step').max,+$('step').value+n));show()}$('trial').onchange=()=>{$('step').value=0;show()};$('step').oninput=show;show();</script>'''
    (root/'trial_browser.html').write_text(page.replace('__DATA__',data))
    lines=['# Task3 全地形六条件实验','',f'完成 {completed}/{expected}；记录审计通过：{audit_pass}。','', '|传感器|上下文|完成|成功|资源终止|','|---|---|---:|---:|---:|']
    lines += [f"|{g['mode']}|{g['horizon']}|{g['completed']}|{g['success']}|{g['resource_limits']}|" for g in aggregates]
    lines+=['','上述比例是实际完成任务的描述性比例；资源终止不能解释为模型导航能力失败。','']+['- '+x for x in report['limitations']]
    (root/'REPORT.md').write_text('\n'.join(lines));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=analyse(a.root.resolve());print(json.dumps({k:r[k] for k in ('completed','expected','audit_pass','all_complete')}))
