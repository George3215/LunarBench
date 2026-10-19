"""Score only final answers, accepting standard native letter/text/boxed answers."""
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task3_native_diagnostics as diag


def normalize(text):
    return re.sub(r'[^a-z0-9]+','',text.lower())


def final_choice(raw, question):
    tagged=re.findall(r'<answer>(.*?)</answer>',raw,re.S|re.I)
    if tagged:
        answer=tagged[-1].strip()
    elif '<think>' in raw and '</think>' not in raw:
        return None,'incomplete_reasoning'
    elif '</think>' in raw:
        # Interactor emits a closed (often empty) think block then a bare letter.
        answer=raw.rsplit('</think>',1)[1].strip()
    else:
        answer=raw.strip()
    boxed=re.findall(r'\\boxed\{\s*(?:\\text\{\s*)?([A-D])(?:[\s.}]|$)',answer)
    if boxed:return boxed[-1],'boxed'
    # Exact option text is a valid native final answer; compare against ALL options.
    choices=dict(re.findall(r'^([A-D])\. (.+)$',question,re.M))
    exact=[k for k,v in choices.items() if normalize(answer)==normalize(v)]
    if len(exact)==1:return exact[0],'exact_option_text'
    last=re.split(r'(?:Final Answer|final answer)\s*:?',answer)[-1].strip()
    match=re.match(r'^\s*(?:\*\*)?\(?([A-D])(?:[\s\).:,]|$)',last)
    if match:return match.group(1),'letter_prefix'
    match=re.search(r'(?:answer is|answer:|correct (?:answer|option) is|correct option is)\s*(?:\*\*)?\(?([A-D])(?:[\s\).:]|$)',answer,re.I)
    if match:return match.group(1).upper(),'final_letter_statement'
    return None,'unparsed'


def main():
    # Preserve the original strict scorer output for audit before format normalization.
    diag.analyse()
    root=diag.ROOT
    strict=json.loads((root/'scores.json').read_text())
    diag.dump(root/'scores_strict_letter_parser.json',strict)
    items={x['id']:x for x in json.loads((root/'inputs.json').read_text())}
    labels=json.loads((root/'labels_evaluator_only.json').read_text())
    rows=[];unparsed=[]
    for score in strict:
        row=items[score['id']]
        result=json.loads((root/'runs'/score['model']/score['id']/'result.json').read_text())
        raw=result['raw']
        score['non_answer_artifact']='ComponentPlacement' in raw
        score['raw_final_output']=raw
        if labels[score['id']]['kind']=='mcq':
            pred,method=final_choice(raw,row['question'])
            score.update(prediction=pred,parseable=pred is not None,
                         correct=pred==labels[score['id']]['answer'],parse_method=method)
        if not score['parseable']:unparsed.append(dict(model=score['model'],id=score['id'],raw=raw))
        rows.append(score)
    metrics={}
    for model in diag.MODELS:
        rr=[x for x in rows if x['model']==model]
        metrics[model]={task:dict(completed=len(a),correct=sum(x['correct'] for x in a),
            parseable=sum(x['parseable'] for x in a),token_limits=sum(x['hit_token_limit'] for x in a),
            artifact_outputs=sum(x['non_answer_artifact'] for x in a))
            for task in sorted(set(x['task'] for x in rr)) if (a:=[x for x in rr if x['task']==task])}
    diag.dump(root/'scores.json',rows);diag.dump(root/'metrics.json',metrics);diag.dump(root/'unparsed_outputs.json',unparsed)
    with (root/'results.csv').open('w',newline='') as f:
        fields=['model','id','task','prediction','correct','parseable','hit_token_limit','non_answer_artifact','raw_final_output']
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    # Independent evidence counts, request pairing and stored assets.
    failures=[];calls=0
    for model in diag.MODELS:
        for row in items.values():
            if model not in row['models']:continue
            p=root/'runs'/model/row['id']
            if not (p/'result.json').exists():failures.append([model,row['id'],'missing_result']);continue
            calls+=1
            request=json.loads((p/'serialized_request.json').read_text())
            if any(k in request for k in ['answer','bbox','centroid','correct','target_world_position']):
                failures.append([model,row['id'],'label_key_in_request'])
            audit=json.loads((p/'input_audit.json').read_text())
            for asset in audit.get('images',audit.get('assets',[])):
                if asset['sha256']!=diag.sha(diag.REPO/asset['path']):failures.append([model,row['id'],'image_hash_mismatch'])
    diag.dump(root/'FINAL_AUDIT.json',dict(completed_calls=calls,expected_calls=sum(len(x['models']) for x in items.values()),
        failures=failures,passed=not failures,scoring='final answer only: letters, boxed letters, exact text of any option',
        excluded_from_inputs='labels_evaluator_only.json; simulator target geometry/pose and evaluation trajectory',
        limitations='Key/hash checks plus source review, not a general natural-language leakage detector.'))
    common=['visibility','bearing','temporal_visibility','temporal_image_motion']
    lines=['# 原生格式诊断结果（2026-10-19）','',
           '这是 TASK3 自定义小样本诊断；不是官方基准复现，也不是新增导航成功率。', '',
           '|模型|可见性（9）|水平位置（6）|两帧可见性变化（3）|两帧图像位移（3）|',
           '|---|---:|---:|---:|---:|']
    for m in diag.MODELS:
        cells=[]
        for task in common:
            t=metrics[m].get(task);cells.append(f"{t['correct']}/{t['completed']}" if t else '不测（原生单图）')
        lines.append('|'+m+'|'+'|'.join(cells)+'|')
    lines+=['','选择题只按最终答案评分：支持字母、boxed 字母或任一选项的精确文字。未解析和截断单列在 scores.json；不从推理段寻找正确答案。',
        '', '## 原生坐标和区域接口','']
    for m,task in [('robobrain25_8b_nv','pointing'),('robobrain25_8b_nv','grounding'),('spatialrgpt','region_left'),('spatialrgpt','region_depth_sensor_variant')]:
        t=metrics[m].get(task,{})
        lines.append(f"* {m} / {task}: {t.get('correct',0)}/{t.get('completed',0)}，可解析 {t.get('parseable',0)}。")
    points=[x for x in rows if x['task']=='pointing'];boxes=[x for x in rows if x['task']=='grounding']
    lines+=['', '点坐标命中定义为落入可见红球紧框，框定位正确定义为 IoU≥0.5。',
            '点中心误差（像素）：'+', '.join(f"{x['id']}={x.get('center_error_pixels',float('nan')):.2f}" for x in points)+ '。',
            '框 IoU：'+', '.join(f"{x['id']}={x.get('iou',0):.3f}" for x in boxes)+ '。',
            '', 'SpatialRGPT 深度题是原生分支的 RGB-D 传感器变体，未使用作者 DepthAnything；不可声称完整官方流程复现。', '',
            '## 输入、资源和输出质量','',f"完成 {calls} / {sum(len(x['models']) for x in items.values())} 次固定推理；最终审计异常 {len(failures)}。"]
    for m in diag.MODELS:
        a=[x for x in rows if x['model']==m]
        lines.append(f"* {m}: 未解析 {sum(not x['parseable'] for x in a)}，输出上限 {sum(x['hit_token_limit'] for x in a)}，含 ComponentPlacement 异常串 {sum(x['non_answer_artifact'] for x in a)}。")
    lines+=['','最初四个 Qwen 系模型因缺少 qwen-vl-utils 在加载和推理前退出，补齐隔离依赖后续跑；保留首次退出记录，没有把环境错误算成能力失败。', '',
            '实际输入处理尺寸、token、pixel 哈希见 runs/*/*/input_audit.json。每题均保留完整问题、原始输出和 token ID。', '',
            '## 结论边界','',
            '这些题只检查视觉定位、当前可见性、简单图像变化和接口行为。双帧位移不是带里程计的空间记忆更新；区域深度题不是遮挡背后几何推断。三场景红球结果不能推断开放词汇、家庭/野外泛化、遮挡重现预测或导航成功率。', '',
            '对之前只转圈结果的解释应限制在证据范围：原生坐标/区域接口可回答，说明直接强制六动作会漏掉模型原有输出能力；它不能单独证明原来的失败完全由 prompt 造成。还需使用明确的动作转换器，比较相同感知、预算、初始状态下的闭环行为。', '',
            '复现：先运行 tools/task3_native_diagnostics.py build（已有冻结集勿重建），再以合适 Python 环境运行 tools/run_task3_native_diagnostics.py，最后运行 tools/analyse_task3_native_diagnostics.py。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=calls,failures=failures,unparsed=len(unparsed)),indent=2))


if __name__=='__main__':main()
