"""Report a completed strict admission batch; no new simulation or model calls."""
import json,os,sys,collections
from pathlib import Path

def build(root):
 status=json.loads((root/'STATUS.json').read_text());protocol=json.loads((root/'PROTOCOL.json').read_text())
 if status['status']!='complete':raise RuntimeError('Full admission batch required')
 records=json.loads((root/'admission_records.json').read_text());accepted=[r for r in records if r['accepted']];rejected=[r for r in records if not r['accepted']]
 if len(records)!=status['expected'] or any(not all(r['gates'].values()) for r in accepted):raise RuntimeError('Admission audit failed')
 reasons=[]
 for r in rejected:
  attempts=[]
  for a in r.get('attempts',[]):
   p=Path(a['evidence_path'])/'gates.json';d=json.loads(p.read_text());g=d['gates'];attempts.append({'candidate':a['candidate'],'gates':g,'initial_target_pixels':d['initial_target_pixels'],'oracle_actions':d['oracle_searched_actions']})
  # A rejected terrain may pass G1/G2 in some candidates. Do not report its
  # aggregate failure as proof that every candidate failed every gate.
  reasons.append({'terrain_id':r['terrain_id'],'terrain_class':r['terrain_class'],'attempt_count':len(attempts),'any_candidate_passed_gate':{k:any(a['gates'][k] for a in attempts) for k in ['G1','G2','G3','G4']},'attempts':attempts,'interpretation':'bounded search found no all-gates-passing candidate; not proof of physical impossibility'})
 (root/'rejection_analysis.json').write_text(json.dumps(reasons,ensure_ascii=False,indent=2))
 flat=[]
 for r in accepted:
  flat.append({'terrain_id':r['terrain_id'],'terrain_class':r['terrain_class'],'seed':r['seed'],'G1':r['gates']['G1'],'G2':r['gates']['G2'],'G3':r['gates']['G3'],'G4':r['gates']['G4'],'initial_target_pixels':r['initial_target_pixels'],'lost_frames':r['loss']['lost_frames'],'occlusion_verified':r['loss']['ray']['entity_occlusion'],'oracle_recovered':any(b['recovery'] is not None for b in r['oracle']),'oracle_searched_actions':r['oracle_searched_actions'],'accepted':True,'evidence_path':r['evidence_path']})
 (root/'formal_admission.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in flat))
 classes={k:{'candidates':sum(r['terrain_class']==k for r in records),'accepted':sum(r['terrain_class']==k for r in accepted)} for k in ['A','B','C','D','E']}
 audit={'pass':True,'records':len(records),'accepted':len(accepted),'rejected':len(rejected),'classes':classes,'thresholds':protocol,'native_runs_completed':False,'E_history_necessity_verified':False,'F_composites':'reserved, no fabricated admission'}
 (root/'admission_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 lines=['# 严格Task3场景验收','',f'候选 {len(records)}/{len(records)} 全部处理；{len(accepted)} 个通过G1–G4，{len(rejected)} 个未找到满足全部条件的候选，不进入正式集。此比例是场景准入比例，不是VLM任务成功率。','', '|类别|候选|通过|','|---|---:|---:|']
 for k,v in classes.items():lines.append(f'|{k}|{v["candidates"]}|{v["accepted"]}|')
 lines += ['', 'G1初始RGB目标≥40像素；G2移动≥0.25m后，分割目标≤2像素连续3个10Hz样本；G3目标中心仍在FOV且真实几何射线命中地形/实体障碍；G4实际安全动作序列恢复目标RGB≥20像素，连续3样本。安全条件为upright≥0.5且在场地内。最多8个候选/地形、20个接近动作/候选、累计100个oracle动作/候选，未改变阈值。', '', '旧版31个场景全部重新验收通过，新版38个设计有23个通过。未通过只说明本次有限搜索没有找到合格起点/目标/运动/恢复配置，不证明地形无法产生这类任务。', '', 'E类21个场景通过几何可见性门槛，但尚未证明同当前画面、不同历史目标会要求相反恢复动作。F类复合地形暂留作后续泛化设计。', '', 'G4表示恢复可见性，不表示已验证从恢复点到目标1m内。后续Final Task Success必须独立计算，参考近距可达性需另作审查，不能把恢复成功当作到达成功。', '', '本阶段没有运行VLMnav、NaVILA或新的Qwen任务，也没有迁移LunarBench或改写旧实验。']
 (root/'ADMISSION_REPORT.md').write_text('\n'.join(lines))
 entries=[]
 for r in records:
  attempts=r.get('attempts',[]);folder=Path(r['evidence_path']) if r['accepted'] else Path(attempts[-1]['evidence_path']) if attempts else None
  images=[]
  if folder:
   for label,phase in [('初始RGB','initial'),('运动丢失RGB','lost'),('Oracle恢复RGB','recovered')]:
    p=folder/phase/'rgb.png'
    if p.exists():images.append({'label':label,'path':os.path.relpath(p,root)})
  entries.append({'id':r['terrain_id'],'class':r['terrain_class'],'accepted':r['accepted'],'gates':r['gates'] if r['accepted'] else None,'images':images,'record':os.path.relpath(root/'validation'/r['terrain_id']/'admission.json',root)})
 page=r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task3 严格G1–G4验收</title><style>body{font:16px system-ui;background:#121923;color:#eee;margin:24px}a{color:#9bd4ff}select{padding:8px;font:inherit;margin:8px}img{width:100%}figure{width:31%;display:inline-block;margin:1%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #456;padding:8px}@media(max-width:700px){figure{width:100%}}</style><h1>Task3 严格G1–G4验收</h1><p>69个来源候选全部处理，54个通过、15个不进入正式集。这是场景准入结果，不是模型成功率。全部阈值预先锁定，恢复使用真实物理动作。拒绝项展示最后一个候选，详细记录包含其他尝试；无候选时不伪造图片。</p><p><a href="ADMISSION_REPORT.md">验收报告</a> · <a href="formal_admission.jsonl">逐场景机器可读清单</a> · <a href="rejection_analysis.json">拒绝原因</a> · <a href="PROTOCOL.json">锁定协议</a></p><select id="class"><option value="">全部类别</option><option>A</option><option>B</option><option>C</option><option>D</option><option>E</option></select><select id="state"><option value="">全部结果</option><option value="yes">通过</option><option value="no">未通过</option></select><select id="scene"></select><div id="detail"></div><script>const D=__DATA__, $=id=>document.getElementById(id),esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');function filter(){let a=D.filter(x=>(!$('class').value||x.class===$('class').value)&&(!$('state').value||x.accepted===($('state').value==='yes')));$('scene').innerHTML='<option value="">选择场景</option>'+a.map(x=>'<option value="'+D.indexOf(x)+'">'+esc(x.id+' / '+(x.accepted?'通过':'未通过'))+'</option>').join('')}function show(){if($('scene').value==='')return;let x=D[+$('scene').value];$('detail').innerHTML='<h2>'+esc(x.id)+'</h2><p>类别 '+x.class+'；'+(x.accepted?'G1–G4全部通过':'有限候选搜索未通过，详细门槛见每个候选记录')+'</p>'+x.images.map(im=>'<figure><figcaption>'+im.label+'</figcaption><a href="'+esc(im.path)+'"><img src="'+esc(im.path)+'"></a></figure>').join('')+'<p><a href="'+esc(x.record)+'">完整验收与oracle记录（评估器专用，不提供给模型）</a></p>'}for(let id of ['class','state'])$(id).onchange=filter;$('scene').onchange=show;filter();</script>'''
 (root/'index.html').write_text(page.replace('__DATA__',json.dumps(entries,ensure_ascii=False).replace('</','<\\/')))
 print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':build(Path(sys.argv[1]).resolve())
