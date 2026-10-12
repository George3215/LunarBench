"""Evidence-backed failure examples and testable improvements after a full suite."""
import json,collections
from pathlib import Path

def build(root,metrics):
    if not metrics['all_complete']:raise RuntimeError('Quality conclusions require complete results')
    analysis=json.loads((root/'analysis.json').read_text());examples=[]
    ranked=sorted(analysis['trials'],key=lambda t:(t['stalled_forward_actions'],t['autonomous_actions']),reverse=True)
    selected=[]
    for label,predicate in [('repeated_forward_stall',lambda t:t['stalled_forward_actions']>=10),
        ('premature_stop',lambda t:t['termination']=='premature_stop'),('wrong_visibility_claim',lambda t:t['false_visible_claims']>0),
        ('resource_boundary',lambda t:t['resource_limit']),('successful_trial',lambda t:t['success'])]:
        cases=[s for s in ranked if predicate(s)][:2]
        for summary in cases:
            folder=root/summary['mode']/summary['horizon']/'trials'/summary['scene'];traces=[]
            with (folder/'trace.jsonl').open() as stream:
                for line in stream:
                    t=json.loads(line)
                    if t['phase']=='vlm_autonomous':traces.append(t)
            relevant=[t for t in traces if t['action']=='FORWARD' and t['actual_displacement_xy_m']<.05] if label=='repeated_forward_stall' else [t for t in traces if t['decision'] and t['decision']['target_visible'] and t['before_evaluator']['target_segment_pixels']==0] if label=='wrong_visibility_claim' else traces[-2:]
            chosen=relevant[-2:];evidence=[]
            for t in chosen:
                evidence.append({'step':t['step'],'decision':t['decision'],'actual_displacement_xy_m':t['actual_displacement_xy_m'],
                    'current_target_pixels':t['before_evaluator']['target_segment_pixels'],'current_visible_range_m':t['before_evaluator']['median_target_surface_range_m'],
                    'actual_context':t['context'],'full_transition':str((folder/'steps'/f'{t["step"]:04d}'/'transition.json').relative_to(root)),
                    'inference_records':[str((folder/p).relative_to(root)) for p in t['inference_records']]})
            examples.append({'type':label,'trial':f'{summary["mode"]}/{summary["horizon"]}/{summary["scene"]}','summary':summary,'evidence':evidence})
    observations=[];suggestions=[];terminations=metrics['overall']['terminations']
    if terminations.get('premature_stop',0):
        observations.append('存在未达到可见且1m内条件的模型STOP；评估器全部记为提前停止。')
        suggestions.append({'hypothesis':'终止策略未可靠遵守成功条件','test':'后续另做允许继续探索的终止策略消融；保持原始直接VLM结果，不能追溯改成功判定。','evidence_type':'动作与当前物理评估一致性，不能推断隐藏思维'})
    if metrics['overall']['stalled_forward_actions']:
        observations.append('存在FORWARD执行后每步净位移小于5cm的重复动作，结合接触记录可定位无进展情况。')
        suggestions.append({'hypothesis':'模型未可靠识别或纠正运动无进展','test':'后续比较显式动作/位移/碰撞历史与纯图像历史；本轮没有向模型输入这些状态，不能提前宣称已验证效果。','evidence_type':'真实位移、接触与策略输出'})
    if metrics['overall']['false_visible_claims']:
        observations.append('存在模型声称当前看见目标但对应相机分割像素为0的决策。')
        suggestions.append({'hypothesis':'当前观测与历史目标信息混淆，或视觉识别失误','test':'逐例核对实际输入帧、公开引用依据与RGB图像；不能仅凭模型文字归因。','evidence_type':'对应输入和实际目标像素'})
    if metrics['overall']['resource_limit_count']:
        observations.append('存在上下文/GPU资源终止，单独列出，不能解释为地形导航能力失败。')
    small=[s for s in json.loads((root/'scene_setup.json').read_text()) if s['initial']['target_visible_rgb_pixels']<200]
    if small:
        observations.append(f'{len(small)}个场景首帧目标小于200个原始相机像素；统一图像缩放可能影响目标识别，但本轮没有做分辨率因果消融。')
        suggestions.append({'hypothesis':'小目标可能受图像预处理影响','test':'另做固定场景、固定上下文、不同图像预处理面积的配对消融；不能把更高分辨率效果算作地形降低难度的效果。','evidence_type':'首帧像素计数与已记录图像grid/预处理hash'})
    value={'scope':'complete-suite descriptive evidence, no hidden reasoning or causal claim','observations':observations,'testable_improvements':suggestions,'representative_examples':examples}
    (root/'quality_analysis.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));lines=['# 完整任务质量与失败证据','']+['- '+o for o in observations]+['','建议均是后续可验证假设，不是本轮已经证实的改进效果。','']
    for s in suggestions:lines+=['**'+s['hypothesis']+'**：'+s['test'],'']
    lines+=['具体trial、动作、真实位移、当前可见像素、实际上下文与完整输出路径见 quality_analysis.json。']
    (root/'QUALITY_REPORT.md').write_text('\n'.join(lines));return value
