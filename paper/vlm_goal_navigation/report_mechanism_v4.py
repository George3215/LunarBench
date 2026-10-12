"""Generate a live, non-hardcoded mechanism admission browser and report."""
import json, os, hashlib, sys
from pathlib import Path
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent/'recovery_mechanism_v4_20261009_r2'
def build():
 inventory=json.loads((ROOT/'candidate_inventory.json').read_text())
 records=json.loads((ROOT/'admission_records.json').read_text()) if (ROOT/'admission_records.json').exists() else []
 lookup={r['terrain_id']:r for r in records};entries=[];counts={}
 for item in inventory:
  c=item['config'];r=lookup.get(item['terrain_id']);family=c['kind'];counts.setdefault(family,{'designed':0,'processed':0,'accepted':0});counts[family]['designed']+=1
  state='待验收' if r is None else '通过' if r['accepted'] else '未通过有限搜索'
  if r:counts[family]['processed']+=1;counts[family]['accepted']+=int(r['accepted'])
  folder=Path(r['evidence_path']) if r and r['accepted'] else Path(r['attempts'][-1]['evidence_path']) if r and r.get('attempts') else None
  images=[]
  if folder:
   for phase,label in [('initial','初始RGB'),('lost','实体挡视线后RGB'),('recovered','恢复可见RGB')]:
    p=folder/phase/'rgb.png'
    if p.exists():images.append({'label':label,'path':os.path.relpath(p,ROOT)})
  entries.append({'id':item['terrain_id'],'family':family,'level':c['provisional_difficulty'],'state':state,'config':c,'images':images,'record':os.path.relpath(ROOT/'validation'/item['terrain_id']/'admission.json',ROOT) if r else None})
 status=json.loads((ROOT/'STATUS.json').read_text()) if (ROOT/'STATUS.json').exists() else {'status':'running','processed':0,'expected':36,'accepted':0}
 summary={'status':status,'families':counts,'model_trials_run':0,'model_success_rate':None,'task_pose_pairing_verified':False}
 (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 text=['# 机制纯净地形 v4 验收','',f"状态：{status['status']}；处理 {status['processed']}/36；通过 {status['accepted']}。此处是地形准入数，不是模型成功率。",'', '|族|设计数|已处理|通过|','|---|---:|---:|---:|']
 text += [f"|{k}|{v['designed']}|{v['processed']}|{v['accepted']}|" for k,v in counts.items()]
 text += ['', '配置与因素定义见DESIGN.md；所有高程任务不添加辅助方块。','初始位姿与目标位置的跨档配对尚未验收，暂不作为完整配对模型消融。','通过G1–G4只证明初始可见、真实地形/实体挡视线、实际动作恢复可见；未证明到目标1m内。']
 (ROOT/'ADMISSION_REPORT.md').write_text('\n'.join(text)+'\n')
 page=r'''<!doctype html><meta charset="utf-8"><title>Task3 v4机制地形</title><style>body{font:16px system-ui;background:#111827;color:#eee;margin:24px}select,button{padding:10px;font:inherit}a{color:#9bd4ff}.images{display:flex;flex-wrap:wrap;gap:12px}.images figure{flex:1;min-width:240px;margin:0}img{width:100%}pre{white-space:pre-wrap}</style><h1>Task3 v4：机制独立与递进消融</h1><p id="status"></p><p>12族，每族3档。高程任务无辅助方块；简单/中等/复杂是暂定设计标签。验收通过不等于VLM成功或到达1m内。</p><select id="family"><option value="">全部地形族</option></select> <select id="level"><option value="">全部档位</option><option value="simple">简单</option><option value="medium">中等</option><option value="complex">复杂</option></select><p><select id="scene"></select></p><div id="view"></div><p><a href="DESIGN.md">设计说明</a> · <a href="ADMISSION_REPORT.md">完整进度报告</a> · <a href="summary.json">机器记录</a></p><script>const D=__DATA__,S=__STATUS__, $=x=>document.getElementById(x);$('status').textContent='当前快照：'+S.status+'；已处理 '+S.processed+'/36；通过 '+S.accepted+'。';for(const f of [...new Set(D.map(r=>r.family))])$('family').add(new Option(f,f));function show(){const r=D.find(r=>r.id===$('scene').value);if(!r){$('view').textContent='无匹配场景';return}sessionStorage.setItem('v4scene',r.id);$('view').replaceChildren();const h=document.createElement('h2');h.textContent=r.id+' / '+r.state;$('view').append(h);const ims=document.createElement('div');ims.className='images';for(const im of r.images){const f=document.createElement('figure'),cap=document.createElement('figcaption'),img=document.createElement('img');cap.textContent=im.label;img.src=im.path;f.append(cap,img);ims.append(f)}$('view').append(ims);const nav=document.createElement('p');for(const [label,n] of [['上一个场景',-1],['下一个场景',1]]){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{$('scene').selectedIndex=Math.max(0,Math.min($('scene').options.length-1,$('scene').selectedIndex+n));show()};nav.append(b)}$('view').append(nav);const details=document.createElement('details');details.open=true;const title=document.createElement('summary');title.textContent='递进因素与配置';const pre=document.createElement('pre');pre.textContent=JSON.stringify(r.config,null,2);details.append(title,pre);$('view').append(details);if(r.record){const a=document.createElement('a');a.href=r.record;a.textContent='完整真实物理验收记录';$('view').append(a)}}function filter(){const prev=$('scene').value||sessionStorage.getItem('v4scene');$('scene').replaceChildren();for(const r of D.filter(r=>(!$('family').value||r.family===$('family').value)&&(!$('level').value||r.level===$('level').value)))$('scene').add(new Option(r.id+' / '+r.state,r.id));if([...$('scene').options].some(o=>o.value===prev))$('scene').value=prev;show()}$('family').onchange=$('level').onchange=filter;$('scene').onchange=show;filter();</script>'''
 (ROOT/'index.html').write_text(page.replace('__DATA__',json.dumps(entries,ensure_ascii=False)).replace('__STATUS__',json.dumps(status)))
 hashes=json.loads((ROOT/'source_hashes.json').read_text());assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':build()
