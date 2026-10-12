"""Read-only trial audit, physical analysis and offline image/output browser."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    if not path.exists():
        return []
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip() and line.endswith('}')]


def analyse(root):
    protocol = json.loads((root/'protocol.json').read_text())
    summaries, audits, browser = [], [], []
    all_results = []
    fig, axes = plt.subplots(6, 4, figsize=(18, 22))
    fig2, axes2 = plt.subplots(6, 4, figsize=(18, 22))
    folders = sorted((root/'trials').iterdir())
    for index, folder in enumerate(folders):
        trace = rows(folder/'trace.jsonl')
        states = rows(folder/'state.jsonl')
        if not states:
            continue
        result_path = folder/'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        if result:
            all_results.append(result)
        init = json.loads((folder/'initialization.json').read_text())
        assessments = [t for t in trace if t['decision'] is not None]
        seen_before = sum(t['before_evaluator']['target_visible_rgb_pixels'] >= 20 for t in trace)
        hallucinated = sum(t['decision']['target_visible'] and t['before_evaluator']['target_segment_pixels'] == 0 for t in assessments)
        missed = sum(not t['decision']['target_visible'] and t['before_evaluator']['target_visible_rgb_pixels'] >= 20 for t in assessments)
        positive_gt = sum(t['before_evaluator']['target_visible_rgb_pixels'] >= 20 for t in assessments)
        matched_visible = sum(t['decision']['target_visible'] and t['before_evaluator']['target_visible_rgb_pixels'] >= 20 for t in assessments)
        false_near = sum(t['decision']['proximity'] == 'near' and not t['before_evaluator']['near_and_visible'] for t in assessments)
        actions = [t['action'] for t in trace if t['action']]
        turns = sum(a in ('TURN_LEFT', 'TURN_RIGHT') for a in actions)
        reversals = sum(a != b and a in ('TURN_LEFT', 'TURN_RIGHT') and b in ('TURN_LEFT', 'TURN_RIGHT') for a, b in zip(actions, actions[1:]))
        stalls = sum(t['action'] == 'FORWARD' and t['actual_displacement_xy_m'] < .05 for t in trace)
        available_near = sum(s['evaluation']['near_and_visible'] for s in states)
        raw_calls = [json.loads((folder/p).read_text()) for t in trace for p in t['inference_records']]
        metadata_path = folder/'model_metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        names = metadata.get('geom_names', {})
        wall_ids = {int(k) for k,v in names.items() if v.endswith(('_sill','_wall_low','_wall_high','_wall'))}
        wall_contacts = sum(any(c['geom1'] in wall_ids or c['geom2'] in wall_ids for c in s['contacts']) for s in states)
        max_actuator_force = max(max(abs(v) for v in s['actuator_force']) for s in states)
        final = result['final_evaluator'] if result else states[-1]['evaluation']
        summary = {'scene': folder.name, 'completed': result is not None,
            'success': result['success'] if result else None,
            'termination': result['termination'] if result else 'running',
            'action_steps': len(actions), 'inference_count': len(raw_calls),
            'valid_first_outputs': sum(t['inference_count_this_step'] == 1 and t['decision'] is not None for t in trace),
            'repair_calls': sum(t['inference_count_this_step'] == 2 for t in trace),
            'token_limit_calls': sum(c['hit_token_limit'] for c in raw_calls),
            'target_visible_input_count': seen_before, 'input_count': len(trace),
            'model_visible_recall_under_20px_rule': matched_visible/positive_gt if positive_gt else None,
            'visible_gt_inputs': positive_gt, 'missed_visible_inputs': missed,
            'claimed_visible_with_zero_target_pixels': hallucinated, 'claimed_near_but_evaluator_not_near': false_near,
            'near_visible_state_samples': available_near,
            'final_visible_pixels': final['target_visible_rgb_pixels'],
            'final_surface_range_m': final['median_target_surface_range_m'],
            'final_camera_to_target_centre_m': final['camera_to_target_centre_m'],
            'min_upright': min(s['evaluation']['upright'] for s in states),
            'turn_fraction': turns/max(1, len(actions)), 'opposite_turn_switches': reversals,
            'forward_net_displacement_below_5cm': stalls,
            'path_m': result['path_m'] if result else None,
            'wall_or_sill_contact_state_samples': wall_contacts,
            'max_abs_actuator_force': max_actuator_force,
            'model_stopped': bool(result and result['model_stopped']),
            'active_episode_wall_s': result['active_episode_wall_s'] if result else None,
            'mean_inference_latency_s': result['average_inference_s'] if result else None}
        summaries.append(summary)
        if result:
            checks = {'scene': folder.name,
                'image_hashes_match': all(hashlib.sha256((folder/t['input_image']).read_bytes()).hexdigest() == t['input_image_sha256'] for t in trace),
                'raw_output_matches_trace': all(c['raw_output'] == raw for t in trace for c, raw in zip([json.loads((folder/p).read_text()) for p in t['inference_records']], t['model_output_raw'])),
                'inference_count_matches': len(raw_calls) == result['inference_count'],
                'action_count_matches': len(actions) == result['action_steps'],
                'state_count_matches': len(states) == result['state_samples'],
                'contact_ids_valid_for_reconstructed_model': bool(metadata) and all(0 <= c['geom1'] < metadata['ngeom'] and 0 <= c['geom2'] < metadata['ngeom'] for s in states for c in s['contacts']),
                'input_image_matches_saved_camera_evidence': all(hashlib.sha256((folder/t['input_image']).read_bytes()).hexdigest() == hashlib.sha256((folder/'steps'/f"{t['step']:04d}"/'before_evaluator/rgb.png').read_bytes()).hexdigest() for t in trace),
                'finite_physical_states': all(all(math.isfinite(v) for v in s['qpos']+s['qvel']) for s in states),
                'budget_bounded': len(actions) <= 3000 and len(raw_calls) <= 6000,
                'success_valid': not result['success'] or (final['near_and_visible'] and result['success_streak'] >= 3 and states[-1]['evaluation']['success_streak'] >= 3),
                'all_calls_have_network_inputs_and_prompts': all((folder/p).with_name('network_input.json').exists() and (folder/p).with_name('messages.json').exists() for t in trace for p in t['inference_records'])}
            audits.append(checks)
        browser.append({'scene': folder.name, 'summary': summary, 'steps': [dict(t,
            input_image=str((folder/t['input_image']).relative_to(root)),
            after_image=str((folder/'steps'/f"{t['step']:04d}"/'after_evaluator/rgb.png').relative_to(root)),
            record_links=[str((folder/p).relative_to(root)) for p in t['inference_records']]) for t in trace]})
        ax, ax2 = axes.flat[index], axes2.flat[index]
        xyz = np.array([s['qpos'][:3] for s in states])
        target = init['initial_camera_evaluator']['target_centre_xyz_evaluator']
        ax.plot(xyz[:, 0], xyz[:, 1], lw=1)
        ax.scatter(*xyz[0, :2], marker='o', label='start')
        ax.scatter(*xyz[-1, :2], marker='x', label='end')
        ax.scatter(*target[:2], marker='*', color='red', label='target (XY diagnostic)')
        ax.set_title(f"{folder.name}: {summary['termination']} ({len(actions)})")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
        times = [s['sim_time_s'] for s in states]
        distances = [s['evaluation']['median_target_surface_range_m'] if s['evaluation']['median_target_surface_range_m'] is not None else float('nan') for s in states]
        ax2.plot(times, distances, label='visible target surface range (m)')
        ax2.plot(times, [s['evaluation']['upright'] for s in states], label='upright')
        ax2.axhline(1, color='red', ls='--', lw=.6)
        ax2.set_title(folder.name)
        ax2.grid(alpha=.2)
    for index in range(len(folders), 24):
        axes.flat[index].set_visible(False)
        axes2.flat[index].set_visible(False)
    axes.flat[0].legend(fontsize=7)
    axes2.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root/'physical_trajectories.png', dpi=130)
    plt.close(fig)
    fig2.tight_layout()
    fig2.savefig(root/'visible_distance_upright.png', dpi=130)
    plt.close(fig2)
    (root/'physical_analysis.json').write_text(json.dumps(summaries, indent=2))
    (root/'artifact_audit.json').write_text(json.dumps(audits, indent=2))
    if summaries:
        with (root/'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    complete = len(all_results)
    successes = sum(r['success'] for r in all_results)
    completed_summaries = [s for s in summaries if s['completed']]
    metrics = {'completed': complete, 'expected': protocol['trial_count'], 'success_count': successes,
        'success_rate_completed': successes/complete if complete else None,
        'terminations': dict(collections.Counter(r['termination'] for r in all_results)),
        'action_count': sum(r['action_steps'] for r in all_results),
        'inference_count': sum(r['inference_count'] for r in all_results),
        'state_samples': sum(r['state_samples'] for r in all_results),
        'model_stop_count': sum(r['action_counts'].get('STOP', 0) for r in all_results),
        'successful_model_stop_count': sum(r['success'] and r['model_stopped'] for r in all_results),
        'reached_1000': sum(r['action_steps'] >= 1000 for r in all_results),
        'reached_2000': sum(r['action_steps'] >= 2000 for r in all_results),
        'reached_3000': sum(r['action_steps'] >= 3000 for r in all_results),
        'sr_at_k': {str(k): sum(r['success'] and r['action_steps'] <= k for r in all_results)/complete if complete else None for k in [1,3,5,10,25,50,100,1000,2000,3000]},
        'zero_target_pixel_visibility_claims': sum(s['claimed_visible_with_zero_target_pixels'] for s in completed_summaries),
        'missed_visible_inputs': sum(s['missed_visible_inputs'] for s in completed_summaries),
        'visible_gt_inputs': sum(s['visible_gt_inputs'] for s in completed_summaries),
        'all_artifact_checks_pass': bool(audits) and all(all(v for k,v in a.items() if k != 'scene') for a in audits)}
    (root/'run_metrics.json').write_text(json.dumps(metrics, indent=2))
    text = ['# Qwen3-VL-8B 可见且近距导航 Trial 报告', '',
        f'已完成 {complete}/{protocol["trial_count"]}；满足完整判据 {successes}/{complete}。未完成场景不计为失败，也不用于全场景成功率。', '',
        '成功必须同时满足：当前相机中目标身份正确（分割与RGB红色像素交集≥20像素）；可见目标表面到相机的三维距离中位数严格小于1m；连续3个10Hz物理样本满足；机器人未跌倒、未越界。身体XY距离不是成功条件。',
        '模型STOP若不满足完整判据，记为premature_stop；到达完整判据由评估器终止，并单独报告模型是否主动STOP。',
        '输入仅目标语义和当前RGB；每次动作后重新拍照、重新推理。保存的是模型公开输出的图像判断和简短决策说明，不是隐藏内部思维。该Instruct模型没有原生Thinking开关。',
        '预算依次累计开放1000/2000/3000；终止后不重试；格式错误最多再推理1次，因此每个trial最多6000次独立图像推理。',
        '记录为MuJoCo实际物理状态，不是实体机械狗。仿真在推理和其他trial调度时暂停；active_episode_wall_s排除其他trial物理执行等待，另保存包含调度等待的墙钟时间。批内观测、历史和attention sequence独立。', '',
        '|场景|结果|动作|推理|最终可见像素|最终可见表面距离m|低位移前进次数|转向比例|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        distance = '不可见' if s['final_surface_range_m'] is None else f"{s['final_surface_range_m']:.3f}"
        text.append(f"|{s['scene']}|{s['termination']}|{s['action_steps']}|{s['inference_count']}|{s['final_visible_pixels']}|{distance}|{s['forward_net_displacement_below_5cm']}|{s['turn_fraction']:.1%}|")
    text += ['', '## 证据与分析边界', '',
        f'已完成trial中，共 {metrics["action_count"]} 个动作、{metrics["inference_count"]} 次独立推理、{metrics["state_samples"]} 条实际状态。',
        f'模型在目标分割像素为0的输入上声称看见目标 {metrics["zero_target_pixel_visibility_claims"]} 次；在满足≥20像素可见判据的输入上漏报 {metrics["missed_visible_inputs"]}/{metrics["visible_gt_inputs"]} 次。少于20像素的可见碎片不直接等同于幻觉。',
        '连续左右转向与前进净位移低于5cm是轨迹描述，不能单独证明推理或碰撞原因；应结合对应图像、原始输出和接触状态分析。',
        '单一模型、固定seed、单次trial不能证明任务本身不可解；在可行性未验证之前，整体失败率不应用来判断模型能力或任务固有难度。还需同动作空间的物理可达性参考和重复trial。',
        '候选提升点需按记录验证：目标漏检/错误定位；无历史带来的搜索振荡；3秒动作粒度导致过冲；前进指令与实际位移不一致；RGB缺少目标尺寸/深度尺度导致近距离判断不准。不能把输出中的解释当作真实因果证据。',
        '当前场景从固定lost_xy/yaw_rad开始，无approach历史；16个配对场景由当前源码seed=7生成并冻结在history_pairs.json，与旧归档编号不直接配对。',
        'steps/每步/input_rgb.png：实际输入图片；inference_*/messages.json、rendered_prompt.txt、network_input.json、output.json、raw_output.txt：每次提示词、网络输入与原文；transition.json：模型决策和实际执行；before_evaluator/after_evaluator：RGB、目标mask、深度与判据；trace.jsonl/state.jsonl/evaluator.jsonl：完整trial轨迹。',
        '打开TRIALS.html可离线逐步查看输入、输出和执行结果。']
    feasibility_path = root/'nominal_camera_feasibility.json'
    if feasibility_path.exists():
        feasibility = json.loads(feasibility_path.read_text())
        absent = [r for r in feasibility if not r['nominal_near_and_vertical_fov_candidate_count']]
        exists = [r for r in feasibility if r['nominal_near_and_vertical_fov_candidate_count']]
        text += ['', '## 原始场景的近距观察条件', '',
            f'用户明确选择保持原目标高度和24场景。正常站立相机的静态几何采样中，{len(absent)}/24个场景未找到近距与垂直视野兼容候选；{len(exists)}个存在候选。',
            '空候选是正常站立高度下的离散几何诊断，不是任意动态姿态的不可达证明；候选存在也不能证明从起点物理可达。',
            '16个配对场景原为重新观察设计：3m高墙贯穿整个地形宽度，窗口下沿约0.9–1.15m，目标位于墙后。原任务保证重新看见的侧向视点，不保证走到目标旁边。',
            '因此原始24场景的0或低成功率不能直接用于判断VLM能力；应按可行性问题和可观察的感知/动作问题分别分析。',
            '详见nominal_camera_feasibility.json和GEOMETRY_NOTES.md。']
        metrics['nominal_camera_candidate_absent_count'] = len(absent)
        metrics['nominal_camera_candidate_exists_count'] = len(exists)
        metrics['overall_sr_is_not_a_pure_vlm_capability_metric'] = True
        (root/'run_metrics.json').write_text(json.dumps(metrics, indent=2))
    reference_path = root/'fixed_forward_physical_reference/summary.json'
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        proven = [r['scene'] for r in reference['cases'] if r['success']]
        text += ['', '## 固定前进物理对照（不计入VLM结果）', '',
            '在8个普通地形上，从同一固定起点执行最多10个预定FORWARD动作，不根据目标坐标或图像生成动作；成功使用同一完整相机判据。',
            f'实际对照中{reference["success_count"]}/8成功，成功场景：{", ".join(proven)}。对照成功证明这些原场景存在实际可执行的近距到达序列；对照失败不证明场景不可达。',
            '|场景|固定前进结果|动作数|最终可见像素|相机表面距离m|',
            '|---|---|---:|---:|---:|']
        for r in reference['cases']:
            final_ref = r['final_evaluator']
            distance_ref = '不可见' if final_ref['median_target_surface_range_m'] is None else f"{final_ref['median_target_surface_range_m']:.3f}"
            text.append(f"|{r['scene']}|{r['termination']}|{r['action_steps']}|{final_ref['target_visible_rgb_pixels']}|{distance_ref}|")
        text += ['', '至少4个场景只需2–3个预定前进动作即可达标，因此不能将整个任务统一称为困难。若VLM在这些已证明可达的场景失败，其探索/动作选择是实际改进对象；完整24场景仍包含近距和通行条件问题。',
            '这是诊断性的单次物理对照，不是新的主成功率、不是VLM自主导航证据，也不证明其他场景不可解。']
        metrics['fixed_forward_proven_reachable_scenes'] = proven
        metrics['vlm_results_on_reference_proven_scenes'] = {r['scene']: r['success'] for r in all_results if r['scene'] in proven}
        (root/'run_metrics.json').write_text(json.dumps(metrics, indent=2))
    (root/'REPORT.md').write_text('\n'.join(text))
    make_browser(root, browser, protocol)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def make_browser(root, data, protocol):
    payload = json.dumps(data, ensure_ascii=False).replace('<','\\u003c')
    prompt = json.dumps(protocol['system_prompt'], ensure_ascii=False).replace('<','\\u003c')
    html = '''<!doctype html><meta charset="utf-8"><title>Qwen3-VL Task3 Trial</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#182238;margin:24px}select,button,input{font:inherit;padding:8px}header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.row{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}.panel{background:white;border-radius:10px;padding:16px}img{width:100%;image-rendering:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px ui-monospace}h1{font-size:26px}#range{flex:1;min-width:200px}.note{color:#536176}a{color:#175ac4}@media(max-width:800px){.row{grid-template-columns:1fr}}</style>
<h1>Qwen3-VL · Task3 完整 Trial</h1><p class="note">左侧为实际送入网络的当前相机图片，右侧为动作执行后图片。模型只接收目标语义与当前RGB；下方GT测量仅用于评估。记录为MuJoCo物理执行。</p>
<header><select id="scene"></select><button id="prev">上一步</button><input id="range" type="range" min="0" value="0"><button id="next">下一步</button><span id="label"></span></header>
<div class="row"><div class="panel"><h3>模型输入相机 RGB</h3><img id="before"></div><div class="panel"><h3>动作执行后的相机 RGB</h3><img id="after"></div></div>
<div class="row"><div class="panel"><h3>模型原始输出（含所有格式重试）</h3><pre id="raw"></pre><div id="links"></div></div><div class="panel"><h3>决策与实际执行对照</h3><pre id="evaluation"></pre></div></div>
<details class="panel"><summary>本轮固定 System Prompt</summary><pre id="prompt"></pre></details><details class="panel"><summary>当前 Trial 汇总</summary><pre id="summary"></pre></details>
<script>const trials=PAYLOAD;const prompt=PROMPT;const scene=document.getElementById('scene'),range=document.getElementById('range');
for(let i=0;i<trials.length;i++){const o=document.createElement('option');o.value=i;o.textContent=trials[i].scene;scene.appendChild(o)}
document.getElementById('prompt').textContent=prompt;
function draw(){const t=trials[Number(scene.value)],i=Number(range.value),s=t.steps[i];document.getElementById('summary').textContent=JSON.stringify(t.summary,null,2);if(!s)return;
document.getElementById('label').textContent=`步骤 ${s.step} / ${t.steps.length} · ${s.action}`;document.getElementById('before').src=s.input_image;document.getElementById('after').src=s.after_image;
document.getElementById('raw').textContent=s.model_output_raw.map((r,i)=>`Inference ${i+1}\\n${r}`).join('\\n\\n');
document.getElementById('evaluation').textContent=JSON.stringify({action:s.action,actual_displacement_xy_m:s.actual_displacement_xy_m,inference_s:s.inference_s,decision:s.decision,before_evaluator:s.before_evaluator,after_evaluator:s.after_evaluator,success_streak:s.success_streak,termination:s.termination},null,2);
const links=document.getElementById('links');links.replaceChildren();for(const path of s.record_links){const a=document.createElement('a');a.href=path;a.textContent='完整输出 JSON ';links.appendChild(a)}}
scene.onchange=()=>{range.max=Math.max(0,trials[Number(scene.value)].steps.length-1);range.value=0;draw()};range.oninput=draw;document.getElementById('prev').onclick=()=>{range.value=Math.max(0,Number(range.value)-1);draw()};document.getElementById('next').onclick=()=>{range.value=Math.min(Number(range.max),Number(range.value)+1);draw()};scene.onchange();</script>'''
    (root/'TRIALS.html').write_text(html.replace('PAYLOAD',payload).replace('PROMPT',prompt))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    args=p.parse_args()
    analyse(args.root.resolve())
