"""Offline scoring/plots. This evaluator may read labels; inference must not."""
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from task3_occlusion_response import extract,valid as schema_valid

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'paper/vlm_goal_navigation/occlusion_branch_pilot_20261019'
MODELS=['qwen3vl8b','spatial_interactor8b']
BRANCHES=['advance','left_bypass','right_bypass','retreat_scan']


def read(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(v):return float(np.mean(v)) if len(v) else None


def cluster_ci(values,layouts):
    """Layout block bootstrap: all sibling states stay together."""
    keys=sorted(set(layouts));rng=np.random.default_rng(71019)
    sums=np.array([sum(v for v,k in zip(values,layouts) if k==key) for key in keys])
    counts=np.array([sum(k==key for k in layouts) for key in keys])
    draws=rng.integers(0,len(keys),size=(10000,len(keys)))
    samples=sums[draws].sum(1)/counts[draws].sum(1)
    return [float(v) for v in np.quantile(samples,[.025,.975])]


def main():
    states=read(ROOT/'state_index.json');status=read(ROOT/'COLLECTION_STATUS.json')
    table=[];by_state={};audits=[];parity=[]
    for s in states:
        key=s['id'];folder=ROOT/'states'/key
        request=read(folder/'request.json')
        assert hashlib.sha256((folder/'request.json').read_bytes()).hexdigest()==s['request_sha256']
        mapping=read(folder/'candidate_mapping_evaluator_only.json')
        labels={b:read(folder/'branches'/b/'label.json') for b in BRANCHES}
        replay=read(folder/'REPLAY_AUDIT.json')
        assert replay['repeat_after_other_candidates_equal'] and all(c['equal'] for c in replay['restore_checks'])
        # Independently recompute stable event and its timestamp from the 10 Hz rows.
        for b,label in labels.items():
            streak=0;first=None
            for row in label['samples']:
                e=row['measurement'];ok=e['target_visible_rgb_pixels']>=20 and e['upright']>=.5 and e['in_bounds']
                streak=streak+1 if ok else 0
                if streak>=3 and first is None:first=row['time_from_start_s']
            assert (first is not None)==label['recovered']
            if first is not None:assert abs(first-label['first_recovery']['time_s'])<1e-8
        by_state[key]=dict(state=s,request=request,mapping=mapping,labels=labels,
            count=sum(l['recovered'] for l in labels.values()),models={})
        for model in MODELS:
            dest=ROOT/'models'/model/key
            result=read(dest/'result.json') if (dest/'result.json').exists() else None
            assert result is not None, f'missing inference {model} {key}'
            serialized=read(dest/'serialized_request.json')
            assert serialized['public_request_sha256']==s['request_sha256']
            for block in serialized['messages'][1]['content']:
                if block['type']=='image':assert hashlib.sha256((REPO/block['path']).read_bytes()).hexdigest()==block['sha256']
            pred,parse_method=extract(result['raw'],check_time=False)
            if result['parsed'] is not None:assert pred==result['parsed']
            result=dict(result,parsed=pred,parse_method=parse_method)
            by_state[key]['models'][model]=result
            audits.append(dict(model=model,state=key,public_input_hash_ok=True,parse_ok=pred is not None,
                token_limit=result['hit_token_limit'],parse_method=parse_method,schema_valid=schema_valid(pred)))
            for letter,b in mapping.items():
                l=labels[b];p=pred['forecasts'][letter] if pred else None
                table.append(dict(model=model,state=key,layout=s['layout'],delayed=s['delayed_variant'],
                    letter=letter,branch=b,recovered=l['recovered'],
                    actual_recovery_s=l['first_recovery']['time_s'] if l['recovered'] else None,
                    predicted_p=p['p'] if p else None,predicted_time_s=p['time_s'] if p else None,
                    chosen=pred['choice']==letter if pred else False,format_valid=schema_valid(pred),
                    probability_valid=pred is not None,
                    time_valid=p is not None and (p['time_s'] is None or (isinstance(p['time_s'],(float,int)) and not isinstance(p['time_s'],bool) and 0<p['time_s']<=6)),parse_method=parse_method,
                    path_m=l['path_m'],turn_rad=l['total_turn_rad'],termination=l['termination']))
        model_dirs=[ROOT/'models'/model/key for model in MODELS]
        packets=[read(d/'serialized_request.json') for d in model_dirs]
        inputs=[read(d/'input_audit.json') for d in model_dirs]
        check=dict(state=key,serialized_messages_equal=packets[0]['messages']==packets[1]['messages'],
            image_grid_equal=inputs[0]['image_grid_thw']==inputs[1]['image_grid_thw'],
            processed_pixels_equal=inputs[0]['pixel_values_sha256']==inputs[1]['pixel_values_sha256'],
            token_ids_equal=inputs[0]['input_ids']==inputs[1]['input_ids'])
        assert check['serialized_messages_equal']
        parity.append(check)
    with (ROOT/'forecast_scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    values=list(by_state.values());n=len(values);layouts=[v['state']['layout'] for v in values]
    oracle=[int(v['count']>0) for v in values];discriminative=[0<v['count']<4 for v in values]
    summary=dict(states=n,layouts=len(set(layouts)),formal_branches=n*4,additional_replay_branches=n,
        ineligible=status['ineligible'],replay_checks_passed=n,input_audit_count=len(audits),
        all_failure_states=sum(v['count']==0 for v in values),all_success_states=sum(v['count']==4 for v in values),
        discriminative_states=sum(discriminative),oracle_covered=sum(oracle),oracle_rate=mean(oracle),
        branch_event_prevalence=mean([r['recovered'] for r in table if r['model']==MODELS[0]]),
        constant_half_brier=.25,baselines={},models={},differences={})
    for b in BRANCHES:
        y=[int(v['labels'][b]['recovered']) for v in values]
        summary['baselines'][b]=dict(recovered=sum(y),rate=mean(y),
            rate_layout_bootstrap95=cluster_ci(y,layouts),
            discriminative_recovered=sum(a for a,d in zip(y,discriminative) if d))
    summary['baselines']['uniform_random']=dict(expected_recovered=sum(v['count']/4 for v in values),
        expected_rate=mean([v['count']/4 for v in values]),
        discriminative_expected_recovered=sum(v['count']/4 for v,d in zip(values,discriminative) if d))
    prevalence=summary['branch_event_prevalence']
    summary['posthoc_prevalence_constant_brier']=prevalence*(1-prevalence)
    summary['all_branch_terminations']={k:sum(v['labels'][b]['termination']==k for v in values for b in BRANCHES)
        for k in ['horizon_exhausted','fallen','out_of_bounds']}
    selected={}
    for model in MODELS:
        rows=[r for r in table if r['model']==model];valid=[r for r in rows if r['probability_valid']]
        time_rows=[r for r in valid if r['recovered'] and r['time_valid'] and r['predicted_time_s'] is not None]
        ys=[];choices=[];contradictions=0
        for v in values:
            pred=v['models'][model]['parsed'];choice=v['mapping'][pred['choice']] if pred else None
            choices.append(choice);ys.append(int(v['labels'][choice]['recovered']) if choice else 0)
            if pred and pred['forecasts'][pred['choice']]['p']<max(x['p'] for x in pred['forecasts'].values()):contradictions+=1
        selected[model]=ys
        chosen=[v['labels'][c] for v,c in zip(values,choices) if c]
        recovered=[l for l in chosen if l['recovered']]
        briers=[mean([(r['predicted_p']-r['recovered'])**2 for r in rows if r['state']==v['state']['id'] and r['probability_valid']]) for v in values]
        model_summary=dict(valid_states=len(valid)//4,total_states=n,
            schema_valid_states=sum(schema_valid(v['models'][model]['parsed']) for v in values),
            invalid_time_fields=sum(not r['time_valid'] for r in valid),
            strict_json_states=sum(v['models'][model]['parse_method']=='strict_json' for v in values),
            syntax_repaired_states=sum(v['models'][model]['parse_method'].startswith('append_') for v in values),
            selected_recovered=sum(ys),selected_rate=mean(ys),
            selected_rate_layout_bootstrap95=cluster_ci(ys,layouts),
            selected_rate_on_oracle_covered=mean([y for y,o in zip(ys,oracle) if o]),
            discriminative_recovered=sum(y for y,d in zip(ys,discriminative) if d),discriminative_total=sum(discriminative),
            brier=mean([(r['predicted_p']-r['recovered'])**2 for r in valid]),brier_denominator=len(valid),
            time_mae_s=mean([abs(r['predicted_time_s']-r['actual_recovery_s']) for r in time_rows]),
            time_prediction_count_on_true_recoveries=len(time_rows),true_recovery_branches=sum(r['recovered'] for r in rows),
            chosen_recovery_time_s=mean([l['first_recovery']['time_s'] for l in recovered]),
            chosen_path_m=mean([l['path_m'] for l in chosen]),chosen_turn_rad=mean([l['total_turn_rad'] for l in chosen]),
            chosen_falls=sum(l['termination']=='fallen' for l in chosen),chosen_oob=sum(l['termination']=='out_of_bounds' for l in chosen),
            choice_not_argmax_probability=contradictions,token_limit_states=sum(v['models'][model]['hit_token_limit'] for v in values),
            selected_branch_counts={b:choices.count(b) for b in BRANCHES},state_briers=briers)
        model_summary['choice_letter_counts']={c:sum(v['models'][model]['parsed'] is not None and v['models'][model]['parsed']['choice']==c for v in values) for c in 'ABCD'}
        model_summary['constant_probability_states']=sum(v['models'][model]['parsed'] is not None and
            len({f['p'] for f in v['models'][model]['parsed']['forecasts'].values()})==1 for v in values)
        base=[i for i,v in enumerate(values) if not v['state']['delayed_variant']]
        model_summary['one_base_state_per_layout']=dict(states=len(base),selected_recovered=sum(ys[i] for i in base),
            rate=mean([ys[i] for i in base]),brier=mean([briers[i] for i in base if briers[i] is not None]))
        model_summary['per_family']={}
        for family in sorted({k.rsplit('_a',1)[0] for k in layouts}):
            subset=[i for i,k in enumerate(layouts) if k.rsplit('_a',1)[0]==family]
            model_summary['per_family'][family]=dict(states=len(subset),selected_recovered=sum(ys[i] for i in subset))
        model_summary['difference_from_fixed_branches']={}
        for branch in BRANCHES:
            difference=[y-int(v['labels'][branch]['recovered']) for y,v in zip(ys,values)]
            model_summary['difference_from_fixed_branches'][branch]=dict(rate_difference=mean(difference),
                layout_bootstrap95=cluster_ci(difference,layouts))
        if len(valid)==4*n:model_summary['brier_layout_bootstrap95']=cluster_ci(briers,layouts)
        summary['models'][model]=model_summary
    a,b=MODELS;diff=[y-x for x,y in zip(selected[a],selected[b])]
    summary['differences']['spatial_minus_qwen_selection']=dict(rate_difference=mean(diff),
        layout_bootstrap95=cluster_ci(diff,layouts),
        only_spatial=sum(x==0 and y==1 for x,y in zip(selected[a],selected[b])),
        only_qwen=sum(x==1 and y==0 for x,y in zip(selected[a],selected[b])))
    if all(summary['models'][m]['valid_states']==n for m in MODELS):
        delta=[y-x for x,y in zip(summary['models'][a]['state_briers'],summary['models'][b]['state_briers'])]
        summary['differences']['spatial_minus_qwen_brier']=dict(difference=mean(delta),layout_bootstrap95=cluster_ci(delta,layouts))
    dump(ROOT/'SUMMARY.json',summary);dump(ROOT/'INPUT_AND_LABEL_AUDIT.json',audits)
    dump(ROOT/'MODEL_INPUT_PARITY.json',parity)
    lines=['# 遮挡恢复分支预测：初步实验结果','',
        f'冻结 V4：{n} 个合格状态、{len(set(layouts))} 个物理布局、{4*n} 条正式分支，另 {n} 条重复校验分支。',
        '这是零样本动作后果预测与一次候选选择实验；没有训练，也没有执行多轮自主导航。',
        '', '| 方法 | 语义可解析 | 所选分支恢复 | 有区分度状态恢复 | Brier ↓ | 重现时间 MAE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for m,s in summary['models'].items():
        brier=f'{s["brier"]:.4f}' if s['brier'] is not None else 'NA'
        mae=f'{s["time_mae_s"]:.3f}s ({s["time_prediction_count_on_true_recoveries"]}/{s["true_recovery_branches"]})' if s['time_mae_s'] is not None else 'NA'
        lines.append(f'| {m} | {s["valid_states"]}/{n} | {s["selected_recovered"]}/{n} ({100*s["selected_rate"]:.1f}%) | {s["discriminative_recovered"]}/{s["discriminative_total"]} | {brier} | {mae} |')
    for b in BRANCHES:
        s=summary['baselines'][b];lines.append(f'| 固定 {b} | — | {s["recovered"]}/{n} ({100*s["rate"]:.1f}%) | {s["discriminative_recovered"]}/{sum(discriminative)} | — | — |')
    lines.extend(['',f'四分支中至少一个能恢复的状态：{sum(oracle)}/{n}；四条都恢复：{summary["all_success_states"]}；四条都不恢复：{summary["all_failure_states"]}。',
        f'均匀随机选择的期望恢复率为 {100*summary["baselines"]["uniform_random"]["expected_rate"]:.1f}%。',
        '', '## 输出与分组诊断', ''])
    for model,s in summary['models'].items():
        ci=s['selected_rate_layout_bootstrap95'];base=s['one_base_state_per_layout']
        lines.extend([f'- {model}：严格 JSON {s["strict_json_states"]}/{n}，仅补末尾括号 {s["syntax_repaired_states"]}/{n}；'
            f'恢复率布局聚类 95% 区间 [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]；'
            f'仅保留每布局首次状态时为 {base["selected_recovered"]}/{base["states"]}。',
            f'  选择分支计数：{json.dumps(s["selected_branch_counts"])}；四候选给出相同概率的状态 {s["constant_probability_states"]}/{n}。'])
    d=summary['differences']['spatial_minus_qwen_selection'];ci=d['layout_bootstrap95']
    lines.extend(['',f'Spatial−Qwen 恢复率差 {100*d["rate_difference"]:.1f} 个百分点，布局聚类 95% 区间 [{100*ci[0]:.1f}, {100*ci[1]:.1f}]；只有 Spatial 恢复 {d["only_spatial"]} 个状态，只有 Qwen 恢复 {d["only_qwen"]} 个。',
        '', '| 地形族 | Qwen 所选恢复 | Spatial 所选恢复 |','|---|---:|---:|'])
    for family,s in summary['models'][MODELS[0]]['per_family'].items():
        t=summary['models'][MODELS[1]]['per_family'][family]
        lines.append(f'| {family} | {s["selected_recovered"]}/{s["states"]} | {t["selected_recovered"]}/{t["states"]} |')
    lines.extend([
        '', '## 解释边界', '',
        '- 表中恢复指重新看见目标，不表示已经到达目标，也不表示 VLM 正确触发 STOP。',
        '- 所有模型接收相同的三时刻 RGB-D、自身理想里程计、传感器计算的粗略目标表面点与四个候选。模型未接收目标真实坐标、地形图或分支标签。',
        '- 粗略目标点由几何变换得到；本实验不能单独证明模型学会了运动积分或跨帧记忆。',
        '- 仅有 22 个物理布局，7 个布局贡献两个相邻状态；95% 区间按布局整体自助抽样，不能当作 116 个独立实验。',
        '- 固定分支基线完整列出，不能只与最弱基线比较；oracle 是有限候选中的离线上界。',
        '- 概率对全部四条分支计算 Brier；时间误差只在真实恢复且模型给出时间的分支上计算，同时列出覆盖数量。未恢复分支按 6 秒观察窗删失，不填 0 秒。',
        '- 语义解析允许仅补齐最多两个末尾花括号，单列修复数量；不修改选择、概率或时间。原始输出、严格解析结果和语义解析审计均保留。',
        '- 选择与概率独立于时间字段评分：Spatial 在 S10_0 的三个时间为不合规的 0 秒，保留原值并剔除这些时间预测；明确给出的选择和概率仍计分。该回答完整 schema 不合规，不能算严格格式成功。',
        '- 29 次重复 advance 校验均在执行其他候选后进行；恢复完整物理状态及低层策略历史，避免仅恢复位置速度造成分支不公平。',
        '- 本批状态只用于诊断；若用于调整方法，应视为开发集，后续泛化评测必须使用新的完整几何布局。',
        '', '详细数值、置信区间与逐分支记录见 SUMMARY.json、forecast_scores.csv；可视化见 index.html。',''])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    # Offline trajectory visualization: evaluator target is drawn here only.
    cards=[]
    colors=dict(zip(BRANCHES,['#e69f00','#009e73','#0072b2','#cc79a7']))
    plotdir=ROOT/'plots';plotdir.mkdir(exist_ok=True)
    for key,v in by_state.items():
        folder=ROOT/'states'/key;checkpoint=np.load(folder/'checkpoint.npz');start=checkpoint['qpos'][:2]
        target=np.array(read(folder/'gate_evaluator_only.json')['current']['target_centre_xyz_evaluator'][:2])
        fig,ax=plt.subplots(figsize=(5.5,4))
        for name,l in v['labels'].items():
            points=np.array([start]+[row['own_qpos'][:2] for row in l['samples']])
            ax.plot(points[:,0],points[:,1],color=colors[name],label=name+(' [RECOVERED]' if l['recovered'] else ' [no recovery]'))
            ax.scatter(*points[-1],color=colors[name],s=20)
            if l['recovered']:
                row=min(l['samples'],key=lambda r:abs(r['time_from_start_s']-l['first_recovery']['time_s']))
                ax.scatter(*row['own_qpos'][:2],marker='*',s=90,color=colors[name],edgecolors='black',zorder=5)
        ax.scatter(*start,c='black',marker='s',label='lost-state start');ax.scatter(*target,c='red',marker='*',s=110,label='target (evaluator only)')
        ax.set(xlabel='world x (m)',ylabel='world y (m)',title=f'{key}: {v["state"]["layout"]}');ax.axis('equal');ax.grid(alpha=.2)
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(plotdir/(key+'.png'),dpi=120);plt.close(fig)
        rows_html=[]
        for model in MODELS:
            result=v['models'][model];pred=result['parsed'];chosen=v['mapping'][pred['choice']] if pred else 'INVALID'
            rows_html.append(f'<p><b>{model}</b>: {chosen}, '+('recovered' if pred and v['labels'][chosen]['recovered'] else 'no recovery / invalid')+'</p><details><summary>Raw response</summary><pre>'+html.escape(result['raw'])+'</pre></details>')
        rgb=''.join(f'<figure><img src="states/{key}/inputs/{role}/rgb.png"><figcaption>{role}</figcaption></figure>' for role in ['initial','last_visible','current'])
        cards.append(f'<article><h2>{key} / {v["state"]["layout"]}</h2><div class="inputs">{rgb}</div><img class="plot" src="plots/{key}.png">'+''.join(rows_html)+'</article>')
    rows=''.join('<tr><td>'+m+'</td><td>'+str(s['valid_states'])+'/'+str(n)+'</td><td>'+str(s['selected_recovered'])+'/'+str(n)+'</td><td>'+f'{s["brier"]:.3f}'+'</td><td>'+f'{s["time_mae_s"]:.2f}'+'</td></tr>' for m,s in summary['models'].items() if s['brier'] is not None and s['time_mae_s'] is not None)
    page='<!doctype html><meta charset="utf-8"><title>Occlusion branch pilot</title><style>body{font-family:system-ui;max-width:1200px;margin:auto;padding:24px;background:#fafafa}article{background:white;margin:20px 0;padding:20px;border:1px solid #ddd}.inputs{display:flex}.inputs figure{margin:5px;width:32%}.inputs img{width:100%}.plot{max-width:660px;width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}</style><h1>Occlusion branch pilot — offline evaluator</h1><p>29 eligible states / 22 layouts; four fixed 6-second branches per state. Recovery is visibility, not navigation arrival. Target positions in these plots are evaluator-only; no terrain map or oracle branch outcomes were supplied to the models. Stars on paths mark first stable recovery. Figures show paths, not obstacle maps.</p><table><tr><th>Model</th><th>Valid</th><th>Chosen recovery</th><th>Brier ↓</th><th>Time MAE (s) ↓</th></tr>'+rows+'</table>'+''.join(cards)
    (ROOT/'index.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
