"""离线统计已录制轨迹；用法：python tools/analyze_rl_trace.py outputs/<run>。

不重放动作、不连接模拟器。旧日志只包含接触奖励，不能据此推断石头身份或抓取成功。
报告写到对应运行目录，保留连续接触片段供逐条核对。
"""
from collections import Counter
import json
import logging
from pathlib import Path
import sys
import numpy as np


def analyze(directory):
    episodes, endings = Counter(), Counter()
    active, segments, errors = {}, [], []
    samples, placed = 0, 0
    with (directory / 'train.jsonl').open() as trace:
        for line in trace:
            row = json.loads(line)
            samples += 1
            index = row['env_id']
            terms = row['reward_terms']
            errors.append(row['tcp_error_m'])
            placed = max(placed, terms['placed_count'])
            if terms['grasp'] > 0:
                active.setdefault(index, []).append(dict(step=row['step'], env_id=index,
                    episode=episodes[index], episode_step=row['step_id'], terms=terms,
                    error_m=row['tcp_error_m'], action=row['action'],
                    grasped_stones=row.get('grasped_stones')))
            elif index in active:
                segments.append(active.pop(index))
            if row['termination_reason']:
                endings[row['termination_reason']] += 1
                episodes[index] += 1
                if index in active:
                    segments.append(active.pop(index))
    segments.extend(active.values())
    return dict(samples=samples, episode_ends=dict(endings), max_placed=placed,
                positive_grasp_samples=sum(map(len, segments)),
                longest_contact_run_steps=max(map(len, segments), default=0),
                tcp_error_quantiles_m=np.quantile(errors, [.5, .9, .99, 1]).tolist(),
                fraction_error_over_5cm=float(np.mean(np.array(errors) > .05)), segments=segments)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    directory = Path(sys.argv[1])
    report = analyze(directory)
    output = directory / 'grasp_trajectory_analysis.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    logging.info('轨迹分析：%s', {k: v for k, v in report.items() if k != 'segments'})
    logging.info('完整接触片段：%s', output)
