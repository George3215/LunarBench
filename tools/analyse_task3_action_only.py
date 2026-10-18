"""Report action-only navigation without inventing omitted actor assessments."""
import argparse
import collections
import csv
import json
from pathlib import Path

from analyse_task3_candidates import analysis, paired_comparison

REPO = Path(__file__).resolve().parents[1]
KEYS = ('qwen3vl8b','spatial_interactor8b','spatialreasoner','spatialrgpt','robobrain25_8b_nv')
OLD = REPO/'paper/vlm_goal_navigation'


def episode(folder):
    result = analysis.read(folder/'result.json')
    prefix = result['controlled_approach_actions']
    streak = 0
    recovered = False
    for sample in analysis.lines(folder/'evaluator.jsonl'):
        if sample['step'] <= prefix:
            continue
        good = sample['target_visible_rgb_pixels'] >= 20 and sample['upright'] >= .25 and sample['in_bounds']
        streak = streak+1 if good else 0
        recovered |= streak >= 3
    actions = result['policy_action_sequence']
    return dict(scene=result['scene'], arrived=result['visible_near_arrival'],
        reobserved=bool(recovered),termination=result['termination'],
        autonomous_actions=result['autonomous_actions'],
        legal_vlm_stop=result['legal_vlm_stop'],rejected_stops=result['rejected_stop_count'],
        initial_plan_valid=result['initial_plan_valid'],
        policy_forward_only=bool(actions) and all(a in ('FORWARD','STOP') for a in actions),
        low_translation_actions=result['low_translation_actions'],
        turn_actions=sum(a in ('TURN_LEFT','TURN_RIGHT') for a in actions),
        alternating_turn_pairs=sum(a in ('TURN_LEFT','TURN_RIGHT') and b in ('TURN_LEFT','TURN_RIGHT') and a!=b for a,b in zip(actions,actions[1:])),
        path_m_including_prefix=result['path_m'])


def summary(rows):
    return dict(completed=len(rows),arrivals=sum(r['arrived'] for r in rows),
        reobserved=sum(r['reobserved'] for r in rows),
        successful_scenes=[r['scene'] for r in rows if r['arrived']],
        recovered_not_arrived_scenes=[r['scene'] for r in rows if r['reobserved'] and not r['arrived']],
        autonomous_actions=sum(r['autonomous_actions'] for r in rows),
        legal_vlm_stop=sum(r['legal_vlm_stop'] for r in rows),
        rejected_stops=sum(r['rejected_stops'] for r in rows),
        valid_initial_plans=sum(r['initial_plan_valid'] for r in rows),
        forward_only_arrivals=sum(r['arrived'] and r['policy_forward_only'] for r in rows),
        terminations=dict(collections.Counter(r['termination'] for r in rows)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    rows={}
    metrics={}
    audits={}
    smoke={}
    old={}
    for key in KEYS:
        run=root/'runs'/key
        rows[key]=[]
        for result in sorted(run.glob('rgbd/*/trials/*/result.json')):
            rows[key].append(episode(result.parent))
            analysis.preserve_case(root,key,result.parent)
        metrics[key]=summary(rows[key])
        metrics[key]['state']='complete' if len(rows[key])==22 else 'partial' if rows[key] else 'not_evaluated'
        smoke_rows=[]
        for result in sorted((root/'smoke'/key).glob('rgbd/*/trials/*/result.json')):
            smoke_rows.append(episode(result.parent))
            analysis.preserve_case(root/'smoke_evidence',key,result.parent)
        smoke[key]=summary(smoke_rows)
        smoke[key]['input_audit']=analysis.audit_run(root/'smoke'/key)
        old_run=OLD/('spatial_transfer_v2_20261018' if key in KEYS[:2] else 'spatial_candidates_20261018')/'runs'/key
        old[key]=[analysis.episode(p.parent) for p in sorted(old_run.glob('rgbd/*/trials/*/result.json'))]
        audits[key]=analysis.audit_run(run)
        audits[key]['paired_vs_original_initial_states']=analysis.matched_initial_inputs(old_run,run)
        audits[key]['initial_state_mismatches']=[p['scene'] for p in audits[key]['paired_vs_original_initial_states'] if not all(p[k] for k in ('same_initial_rgb','same_lost_rgb','same_loss_qpos'))]
        reference=root/'runs/qwen3vl8b/protocol.json'
        protocol=run/'protocol.json'
        if reference.exists() and protocol.exists():
            a,b=analysis.read(reference),analysis.read(protocol)
            fields=('system_prompt','plan_system_prompt','action_budget','max_new_tokens','plan_max_new_tokens','scenes','admission_manifest_sha256','stop_gate','arrival_endpoint','action_constraint')
            audits[key]['shared_protocol_mismatches']=[k for k in fields if a.get(k)!=b.get(k)]
        else:
            audits[key]['shared_protocol_mismatches']=None
        audits[key]['complete_input_pairing_pass']=bool(len(rows[key])==22 and len(audits[key]['paired_vs_original_initial_states'])==22 and not audits[key]['initial_state_mismatches'] and not audits[key]['forbidden_field_or_role_violations'] and audits[key]['shared_protocol_mismatches']==[])
    comparisons={key:paired_comparison(rows['qwen3vl8b'],rows[key]) for key in KEYS[1:]}
    for key,value in comparisons.items():
        value['input_audit_passed_for_comparison']=audits[key]['complete_input_pairing_pass'] and audits['qwen3vl8b']['complete_input_pairing_pass']
    changes={key:dict(original_schema=summary(old[key]),action_only=metrics[key],
        interpretation='protocol change: actor prompt, constrained answer space, token budget and optional plan updates differ; not isolated evidence of improved spatial capability') for key in KEYS}
    analysis.write(root/'metrics.json',dict(status='complete' if all(m['completed']==22 for m in metrics.values()) else 'in_progress',models=metrics,paired_vs_action_only_qwen=comparisons,within_model_protocol_changes=changes,smoke_not_formal_SR=smoke,missing_episodes_imputed_as_failures=False))
    analysis.write(root/'INPUT_AUDIT.json',audits)
    flattened=[dict(model=key,**row) for key,values in rows.items() for row in values]
    if flattened:
        with (root/'episodes.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(flattened[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(flattened)
    report=['# TASK3 仅动作输出接口对照','',
        '与原复杂 JSON 实验分开保存。所有模型共享新 actor prompt、合法动作约束、64动作预算和22冻结起点。',
        '初始计划仍由各自模型按原首帧计划 prompt 生成，解析失败保留 null。每步不请求或补造可见性、距离、证据引用、计划更新或重现距离预测。',
        '此组是接口迁移实验，不是新的空间学习方法；不能将协议差异解释为权重能力提升。','',
        '| 模型 | 完成 | 到达 | 恢复可见 | 原复杂JSON到达 | 当前终止类型 |',
        '| --- | --- | --- | --- | --- | --- |']
    for key,m in metrics.items():
        report.append(f"| {key} | {m['completed']}/22 | {m['arrivals'] if m['completed'] else '待测'} | {m['reobserved'] if m['completed'] else '待测'} | {sum(r['arrived'] for r in old[key])}/22 | {json.dumps(m['terminations'])} |")
    report.extend(['','## 成功场景与剩余失败',''])
    for key,m in metrics.items():
        report.append(f"- {key}: 成功 {', '.join(m['successful_scenes']) or '暂无'}；恢复可见但未到达 {len(m['recovered_not_arrived_scenes'])}；有效首帧计划 {m['valid_initial_plans']}/{m['completed']}；仅前进/STOP成功 {m['forward_only_arrivals']}。")
    report.extend(['','## 完整场景比较',''])
    for key,value in comparisons.items():
        if not value['ready'] or not value['input_audit_passed_for_comparison']:
            report.append(f'- {key}: 等待22场景完成及配对审计，不给最终排序。')
        else:
            lo,hi=value['terrain_family_cluster_bootstrap_95_ci']
            report.append(f"- {key}: 相对本组Qwen到达率差 {100*value['candidate_minus_qwen_arrival_rate']:+.1f}pp；地形族重采样95%区间 [{100*lo:.1f}, {100*hi:.1f}]pp；配对McNemar p={value['mcnemar_exact_two_sided_p']:.4g}。")
    report.extend(['','## 两场景短测试（不计正式成功率）',''])
    for key,m in smoke.items():
        report.append(f"- {key}: {m['completed']}/2，执行 {m['autonomous_actions']} 个自主动作；终止 {json.dumps(m['terminations'])}。")
    report.extend(['','原始RGB-D、序列化请求、token、输出、逐步动作及评估轨迹保留在本机；GitHub备份协议、脚本、回合结果、精简逐步证据和分析。',
        '目标仍为红球；理想模拟里程计/RGB-D，不代表真实传感器、开放词汇导航或真机结果。SpatialRGPT仍为通用多图配置，不代表其原生区域深度分支。',
        'STOP只在当前红像素>=20且有限0<红区域距离<1m时可选；自动连续3采样到达与模型STOP分开统计。',
        'V4准入只证明可重观测，不保证22场景都已认证完整到达路线。按地形族相关的小样本解释统计。'])
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    if args.plots and flattened:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        scenes=sorted({r['scene'] for r in flattened})
        fig,axes=plt.subplots((len(scenes)+3)//4,4,figsize=(17,3.8*((len(scenes)+3)//4)),squeeze=False)
        for ax,scene in zip(axes.flat,scenes):
            for key,color in zip(KEYS,('#1769aa','#ed8311','#009e73','#aa3377','#666666')):
                folder=root/'runs'/key/'rgbd/event4_odom_plan/trials'/scene
                if not (folder/'result.json').exists():continue
                states=analysis.lines(folder/'state.jsonl')
                origin=np.array(analysis.read(folder/'initialization.json')['qpos'][:2])
                xy=np.array([s['qpos'][:2] for s in states])-origin
                result=analysis.read(folder/'result.json')
                ax.plot(xy[:,0],xy[:,1],color=color,label=key,lw=1.2)
                ax.scatter(*xy[-1],color=color,marker='*' if result['visible_near_arrival'] else 'x',s=65)
                target=np.array(states[-1]['evaluation']['target_centre_xyz_evaluator'][:2])-origin
                ax.scatter(*target,color='red',marker='+',s=40)
            ax.set_title(scene,fontsize=10);ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
        for ax in list(axes.flat)[len(scenes):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9)
        fig.suptitle('Action-only: *=arrival, x=terminated, red+=target (evaluator only)',y=.98)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/'trajectories.svg');plt.close(fig)
        path=root/'trajectories.svg';path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    print(json.dumps({k:{f:m[f] for f in ('completed','arrivals','reobserved')} for k,m in metrics.items()}))


if __name__=='__main__':main()
