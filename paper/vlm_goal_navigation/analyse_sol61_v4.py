"""Camera wire audit and paired six-condition summaries, without reading GT poses."""
import argparse
import collections
import csv
import json
import time
from pathlib import Path

MODES = ('rgb', 'rgbd')
HORIZONS = ('current', 'first_current', 'all_history')
BASELINE = Path(__file__).parent / 'runs/20261009_v4_qwen3vl_six_condition'


def write(root):
    current = [json.loads(p.read_text()) for p in root.glob('*/*/trials/*/result.json')]
    baseline = [json.loads(p.read_text()) for p in BASELINE.glob('*/*/trials/*/result.json')]
    protocol = json.loads((root / 'protocol.json').read_text())
    status = json.loads((root / 'RUN_STATUS.json').read_text())
    audit = {'requests_checked': 0, 'responses_checked': 0, 'violations': []}
    for p in root.glob('*/*/trials/*/steps/*/inference_*/network_input.json'):
        data = json.loads(p.read_text())
        audit['requests_checked'] += 1
        context = json.loads((p.parent / 'context.json').read_text())
        issues = []
        if data.get('tools') != [] or data.get('tool_choice') != 'none':
            issues.append('tools_not_disabled')
        if data.get('previous_response_id') or len(data['input']) != 1:
            issues.append('unexpected_session_history')
        content = data['input'][0]['content']
        text = '\n'.join(b['text'] for b in content if b['type'] == 'input_text')
        for forbidden in ('geom_xpos', 'target_centre_xyz', 'qpos', 'target_mask', 'terrain_design_v4'):
            if forbidden in text:
                issues.append(forbidden)
        expected_images = context['input_frame_count'] * (2 if context['mode'] == 'rgbd' else 1)
        if sum(b['type'] == 'input_image' for b in content) != expected_images:
            issues.append('image_count_mismatch')
        if context['horizon'] == 'current' and context['input_frame_count'] != 1:
            issues.append('current_has_history')
        if context['horizon'] == 'first_current' and context['input_frame_ids'][0] != 0:
            issues.append('first_frame_missing')
        if context['horizon'] == 'all_history' and context['input_frame_count'] != context['all_recorded_frame_count']:
            issues.append('history_frames_dropped')
        if issues:
            audit['violations'].append({'request': str(p.relative_to(root)), 'issues': issues})
        response = p.parent / 'response.json'
        if response.exists():
            value = json.loads(response.read_text())
            audit['responses_checked'] += 1
            if value.get('model') != 'gpt-6.1-sol':
                audit['violations'].append({'request': str(p.relative_to(root)), 'issues': ['model_identity']})
    (root / 'INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    rows = []
    for mode in MODES:
        for horizon in HORIZONS:
            selected = [r for r in current if r['mode'] == mode and r['horizon'] == horizon]
            qwen = [r for r in baseline if r['mode'] == mode and r['horizon'] == horizon]
            paired = {r['scene']: r for r in qwen}
            rows.append({'mode': mode, 'horizon': horizon, 'sol_completed': len(selected),
                         'sol_success': sum(r['success'] for r in selected),
                         'qwen_completed': len(qwen), 'qwen_success': sum(r['success'] for r in qwen),
                         'paired_sol_only_success': sum(r['success'] and not paired[r['scene']]['success'] for r in selected),
                         'paired_qwen_only_success': sum(not r['success'] and paired[r['scene']]['success'] for r in selected),
                         'sol_terminations': dict(collections.Counter(r['termination'] for r in selected))})
    with (root / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['# 第四版地形：GPT-6.1 Sol 相机观测测试', '',
             f"当前状态：`{status['status']}`；完成 {status['completed']}/{status['expected']}，成功 {status['success_count']}。", '',
             '| 输入 | 帧条件 | Sol 成功/已完成 | Qwen3-VL-8B 成功/已完成 |', '|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['horizon']} | {r['sol_success']}/{r['sol_completed']} | {r['qwen_success']}/{r['qwen_completed']} |")
    lines.extend(['', '未完成时，不能将两列成功数直接当作最终成功率比较。', '',
                  f"相机输入审计：检查 {audit['requests_checked']} 个实际请求、{audit['responses_checked']} 个模型回复，发现 {len(audit['violations'])} 个违规项。", '',
                  '仅发送相机 RGB；RGBD 增加对齐理想深度和 RGB 红色区域的深度测量。模型无工具、无目标坐标、无评分器信息、无先前回复或动作文本。', '',
                  '历史帧保留每个动作边界的相机观测，包含共同接近过程；不进行历史抽样。', '',
                  '渲染一致性限制：Sol 原始相机为 Mesa llvmpipe；恢复和后续本地模型锁定相同后端。现有 Qwen 对照使用不同渲染后端：barrier_a0 初始物理状态完全相同，但 RGB 像素不同，因此现有 Qwen 列只作参考，不能声称逐像素配对比较。', '',
                  f"比较限制：{protocol['comparison_limitations']}。每组每地形仅一次运行；结果属于描述性比较。", '',
                  '第一轮六路并发遇到 HTTP 429，未执行模型动作；中断记录保留在相邻的无 r2 后缀目录。正式重跑使用串行模型请求。'])
    # Runner owns REPORT.md; this independent audit owns COMPARISON.md.
    (root / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    while True:
        try:
            write(args.root)
            status = json.loads((args.root / 'RUN_STATUS.json').read_text())['status']
            recovery_path = args.root/'RECOVERY_STATUS.json'
            restoring = recovery_path.exists() and json.loads(recovery_path.read_text()).get('phase') == 'restoring'
            if not args.watch or (status != 'running' and not restoring):
                break
        except (FileNotFoundError, json.JSONDecodeError):
            if not args.watch:
                raise
        time.sleep(30)


if __name__ == '__main__':
    main()
