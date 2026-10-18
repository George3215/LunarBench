"""Cross-backbone results and input audits; never fill missing trials as failures."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('task3_analysis', REPO / 'paper/vlm_goal_navigation/analyse_spatial_transfer_v4.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
KEYS = ('spatialreasoner', 'spatialrgpt', 'robobrain25_8b_nv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    old = REPO / 'paper/vlm_goal_navigation/spatial_transfer_v2_20261018'
    rows = []
    metrics = {}
    smoke_metrics = {}
    smoke_audits = {}
    audits = {}
    locations = {}
    for key in ('qwen3vl8b', 'spatial_interactor8b') + KEYS:
        run = (old if key not in KEYS else root) / 'runs' / key
        locations[key] = run
        cases = []
        for result in sorted(run.glob('rgbd/*/trials/*/result.json')):
            row = analysis.episode(result.parent)
            row['model'] = key
            cases.append(row)
            if key in KEYS:
                analysis.preserve_case(root, key, result.parent)
        rows.extend(cases)
        metrics[key] = analysis.summary(cases)
        metrics[key]['state'] = 'complete' if len(cases) == 22 else 'partial' if cases else 'not_evaluated'
        if key in KEYS:
            smoke = root/'smoke'/key
            smoke_rows = []
            for result in sorted(smoke.glob('rgbd/*/trials/*/result.json')):
                smoke_rows.append(analysis.episode(result.parent))
                analysis.preserve_case(root/'smoke_evidence',key,result.parent)
            smoke_metrics[key] = analysis.summary(smoke_rows)
            smoke_audits[key] = analysis.audit_run(smoke)
            audits[key] = analysis.audit_run(run)
            protocol = run / 'protocol.json'
            ref = old / 'runs/qwen3vl8b/protocol.json'
            if protocol.exists() and ref.exists():
                a, b = analysis.read(protocol), analysis.read(ref)
                fields = ('system_prompt','plan_system_prompt','action_budget','max_new_tokens',
                    'plan_max_new_tokens','scenes','admission_manifest_sha256','stop_gate','arrival_endpoint')
                audits[key]['common_semantic_protocol_mismatches'] = [k for k in fields if a.get(k) != b.get(k)]
                audits[key]['token_or_native_preprocessing_equality_claimed'] = False
                audits[key]['paired_initial_states'] = analysis.matched_initial_inputs(old/'runs/qwen3vl8b',run)
                audits[key]['initial_rgb_loss_rgb_qpos_mismatches'] = [
                    p['scene'] for p in audits[key]['paired_initial_states'] if not all(p[k] for k in
                    ('same_initial_rgb','same_lost_rgb','same_loss_qpos'))]
    analysis.write(root / 'metrics.json', {'comparison_scope':'cross-backbone transfer, not a causal spatial-pretraining ablation',
        'models':metrics, 'smoke_diagnostics_not_formal_SR':smoke_metrics, 'missing_episodes_imputed_as_failures':False})
    analysis.write(root / 'INPUT_AUDIT.json', {'formal_runs':audits,'smoke_runs':smoke_audits})
    if rows:
        with (root/'episodes.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['model']+[k for k in rows[0] if k != 'model'], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    report = ['# TASK3 新增空间模型测试', '', '仅统计已完成回合；接口/资源失败与导航失败分别报告。', '',
        '| 模型 | 正式完成 | 确认到达 | 恢复可见 | 终止类型 |', '| --- | --- | --- | --- | --- |']
    for key, value in metrics.items():
        report.append(f"| {key} | {value['completed']}/22 | {value['arrivals'] if value['completed'] else '未测'} | {value['reobserved'] if value['completed'] else '未测'} | {json.dumps(value['terminations'])} |")
    report.extend(['', '## 两场景接口诊断（每场景最多2动作，不纳入正式成功率）', '',
        '| 模型 | 完成案例 | 有效动作总数 | 当前可见性判断正确/有效判断 | 终止 |', '| --- | --- | --- | --- | --- |'])
    for key,value in smoke_metrics.items():
        report.append(f"| {key} | {value['completed']}/2 | {value['valid_decisions']} | {value['current_visibility_correct_decisions']}/{value['valid_decisions']} | {json.dumps(value['terminations'])} |")
    report.extend(['', 'SpatialReasoner 的前两例没有执行自主动作：首次输出为不完整自然语言字段；纠正后仍缺 action，或 evidence_frames 类型错误。',
        '这属于当前复杂结构化接口的直接迁移失败，不能推出它没有空间能力；两个接口样例不能代替22场景正式评测。',
        'SpatialReasoner 已另排完整22场景诊断，保留相同 prompt 与解析规则，以免只根据两个样例作判断；执行状态见 REASONER_QUEUE_STATUS.json。',
        'SpatialRGPT、RoboBrain 通过接口诊断后接续正式场景。SpatialRGPT 的通用多图配置与原生区域深度输入不同。'])
    report.extend(['', 'Qwen/Spatial-Interactor 是前轮冻结协议记录；新增模型有原生 tokenizer/vision/chat template 差异。',
        'RoboBrain 使用作者链接的官方 ModelScope NV 权重，分片与 HF 哈希一致。SpatialRGPT 首轮为无额外region的通用多图配置，不能代表其原生区域深度分支完整能力。',
        '未完成全场景且未通过输入审计前，不给出最终模型排序。即使到达率不同，也不能直接归因于空间训练数据。',
        '当前目标是红球；结果不能单独证明开放词汇语义理解或跨目标类别泛化。',
        '原V4准入证明可重观测，不保证全部场景已经认证完整<1m到达路线。', '',
        '新方法及开源导航对照的取舍见 [RELATED_WORK.md](RELATED_WORK.md)。'])
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    if args.plots and rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        scenes = sorted({r['scene'] for r in rows})
        fig, axes = plt.subplots((len(scenes)+3)//4, 4, figsize=(17, 3.9*((len(scenes)+3)//4)), squeeze=False)
        colors = ('#1769aa','#ed8311','#009e73','#aa3377','#666666')
        for ax, scene in zip(axes.flat, scenes):
            target = None
            for (key, run), color in zip(locations.items(), colors):
                trial = run / 'rgbd/event4_odom_plan/trials' / scene
                if not (trial/'result.json').exists():
                    continue
                states = analysis.lines(trial/'state.jsonl')
                if not states:
                    continue
                result = analysis.read(trial/'result.json')
                origin = np.array(analysis.read(trial/'initialization.json')['qpos'][:2])
                xy = np.array([s['qpos'][:2] for s in states])-origin
                ax.plot(xy[:,0],xy[:,1],color=color,label=key,linewidth=1.2)
                ax.scatter(*xy[-1],color=color,marker='*' if result['visible_near_arrival'] else 'x',s=75)
                target = np.array(states[-1]['evaluation']['target_centre_xyz_evaluator'][:2])-origin
            if target is not None:
                ax.scatter(*target,color='red',marker='+',s=60)
            ax.set_title(scene,fontsize=10)
            ax.set_aspect('equal',adjustable='datalim')
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(scenes):]:
            ax.axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9)
        fig.suptitle('TASK3: *=confirmed arrival, x=terminated, red+=target (evaluator only)',y=.98,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(root/'trajectories.svg')
        path=root/'trajectories.svg'
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    print(json.dumps({k:{f:v[f] for f in ('completed','arrivals','state')} for k,v in metrics.items()}))


if __name__ == '__main__':
    main()
