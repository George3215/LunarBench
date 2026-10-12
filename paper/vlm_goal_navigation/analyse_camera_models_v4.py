"""Audit local-model wire inputs and summarize formal runs, excluding archived attempts."""
import argparse
import collections
import datetime
import json
import time
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def audit(run, name, protocol):
    value = {'requests_checked':0, 'responses_checked':0, 'pending_requests':0,
             'violations':[], 'explicit_context_limits':0}
    for path in run.glob('*/*/trials/*/steps/*/inference_*/network_input.json'):
        try:
            request=json.loads(path.read_text())
            context=json.loads((path.parent/'context.json').read_text())
            response=json.loads((path.parent/'response.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            value['pending_requests']+=1
            continue
        value['requests_checked']+=1
        value['responses_checked']+=1
        issues=[]
        if set(request)!={'images','labels','text','system','protocol'}:
            issues.append('unexpected_wire_fields')
        text=request['text']+'\n'+request['system']+'\n'+'\n'.join(request['labels'])+'\n'+response.get('prompt','')
        for forbidden in ('geom_xpos','target_centre_xyz','qpos','target_mask','terrain_design_v4','graph_sprels'):
            if forbidden in text:issues.append('forbidden_input_'+forbidden)
        ids=context['input_frame_ids'];mode=context['mode'];horizon=context['horizon']
        expected=len(ids)*(2 if mode=='rgbd' else 1)
        if len(request['images'])!=expected or len(request['labels'])!=expected:
            issues.append('selected_image_count_mismatch')
        if mode=='rgb' and (any('depth' in label.lower() for label in request['labels']) or 'Camera measurement:' in request['text']):
            issues.append('depth_in_rgb')
        if horizon=='current' and len(ids)!=1:issues.append('current_has_history')
        if horizon=='first_current' and (len(ids)!=2 or ids[0]!=0):issues.append('first_current_frame_selection')
        if horizon=='all_history' and ids!=list(range(context['all_recorded_frame_count'])):
            issues.append('history_frames_dropped')
        if response.get('model')!=name:issues.append('model_identity')
        if response.get('history_truncated') is not False or context.get('truncated') is not False:
            issues.append('input_truncation')
        if context.get('past_assistant_outputs_in_context') is not False or context.get('past_action_history_in_context') is not False:
            issues.append('past_outputs_or_actions')
        if protocol=='matrix':
            if response['actual_input_labels']!=request['labels'] or response['input_image_count']!=expected:
                issues.append('actual_model_images_mismatch')
        elif response['input_image_count']!=8:
            issues.append('native8_image_count')
        if response['prompt'].count('<image>')!=response['input_image_count']:
            issues.append('image_placeholder_count')
        over_limit=response['input_tokens']>response['context_limit']
        if response.get('termination')=='context_limit':
            value['explicit_context_limits']+=1
            if response['generated_token_count']!=0:issues.append('generation_at_context_limit')
        elif over_limit:issues.append('generation_with_overlength_input')
        if issues:value['violations'].append({'request':str(path.relative_to(run)),'issues':issues})
    save(run/'INPUT_AUDIT.json',value)
    return value


def write(root):
    plan=json.loads((root/'models.json').read_text())
    rows=[]
    for item in plan:
        for protocol in item.get('protocols',[]):
            run=root/item['id']/protocol
            if not (run/'protocol.json').exists():continue
            try:
                config=json.loads((run/'protocol.json').read_text())
                status=json.loads((run/'RUN_STATUS.json').read_text())
                result=[json.loads(p.read_text()) for p in run.glob('*/*/trials/*/result.json')]
                checks=audit(run,item['id'],protocol)
            except (FileNotFoundError,json.JSONDecodeError):continue
            terminations=dict(collections.Counter(r['termination'] for r in result))
            rows.append({'model':item['id'],'protocol':protocol,'status':status['status'],
                         'completed':len(result),'expected':config['trial_count'],
                         'success':sum(r['success'] for r in result),
                         'terminations':terminations,'requests_checked':checks['requests_checked'],
                         'audit_violations':len(checks['violations'])})
    save(root/'FORMAL_PROGRESS.json',{'checked_at':datetime.datetime.now().astimezone().isoformat(),'runs':rows})
    lines=['# 本地模型正式测评进度','','Sol 已按用户要求暂停；请求次数不设配额。归档的裁剪异常试验不进入以下统计。','',
           '| 模型 | 协议 | 完成/计划 | 成功 | 终止原因 | 输入审计违规 |','|---|---|---:|---:|---|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['protocol']} | {row['completed']}/{row['expected']} | {row['success']} | {row['terminations']} | {row['audit_violations']} |")
    lines+=['','未完成结果仅为部分描述性记录；context_limit 表示模型输入资源限制，不作为导航失败率的直接证据。',
            '主矩阵不丢弃历史帧；NaVILA native8 为明确标注的八帧采样补充实验。Taga 为相机骨干适配，不能代表原生拓扑方法。']
    (root/'FORMAL_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        write(args.root)
        status=json.loads((args.root/'STATUS.json').read_text())
        if not args.watch or status['phase']=='queue_finished' or status['phase'].startswith('stopped'):return
        time.sleep(30)


if __name__=='__main__':main()
