"""Create a versioned, evidence-preserving balanced subset of admitted terrains."""
import json, hashlib, html, os
from pathlib import Path
from collections import Counter
BASE = Path(__file__).resolve().parent / 'recovery_benchmark_20261009'
OUT = BASE / 'balanced_v1'
# Order is a provisional within-family geometry rating, not a VLM performance rating.
PLAN = {
 '台阶': [('v2/stairs','低矮台阶，局部方块辅助遮挡'),('v2/stairs_down','下行台阶，落差1.5m，局部方块辅助遮挡'),('v2/stairs_up','上行10级台阶，初始目标仅45像素')],
 '坑洞': [('v2/pit','旧版小坑，初始距离2.93m'),('design_v3/pit_s23','坑深1.2m、半径2m'),('design_v3/pit_s67','坑深2.1m、半径3.3m')],
 '盆地': [('v2/basin','旧版盆地基准；初始目标较小，简单评级置信度低'),('design_v3/basin_s23','盆地深1.5m、半径3.8m'),('design_v3/basin_s41','盆地深2m、半径4.5m')],
 '沟渠与沟壑': [('design_v3/trench_s23','单沟深0.8m，初始距离2.50m'),('design_v3/gullies_s23','分支沟壑深0.8m'),('design_v3/gullies_s41','分支沟壑深1.2m')],
 '方块障碍与迷宫': [('design_v3/barrier_s23','单方块1.2m'),('design_v3/block_maze_s23','12个不同大小方块'),('design_v3/block_maze_s41','18个不同大小方块')],
 '随机起伏': [('v2/perlin','旧版低起伏基准，初始距离3.32m'),('design_v3/perlin_s41','空间相关随机起伏，设计高差1.9m'),('design_v3/perlin_rough_s41','强随机起伏，设计高差2.755m')],
 '连续波浪': [('design_v3/waves_s23','波浪基准，接近3动作、恢复3动作'),('design_v3/waves_s41','另一波浪尺度，接近2动作、恢复4动作'),('design_v3/waves_s67','接近14动作、恢复7动作；恢复动作非最短路径')],
 '窗口结构': [('design_v3/window_s67','宽窗1.5m、窗台0.12m'),('design_v3/window_s23','中窗1.1m、窗台0.18m'),('design_v3/window_s41','窄窗0.7m、窗台0.28m、偏置-0.5m')],
}
def dump(name, data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def main():
 OUT.mkdir(exist_ok=True)
 source=BASE/'formal_manifest.json'; records=json.loads(source.read_text()); lookup={r['terrain_id']:r for r in records}
 selected=[]; levels=['简单','中等','复杂']
 for family, choices in PLAN.items():
  for level,(tid,reason) in zip(levels,choices):
   r=dict(lookup[tid]);assert r['accepted'] and all(r['gates'].values())
   r.update(curated_family=family, provisional_difficulty=level, rating_reason=reason, rating_status='provisional_within_family', rating_confidence='low' if family in ['台阶','盆地'] else 'medium')
   r['rgb_sha256']={s:hashlib.sha256((Path(r['evidence_path'])/s/'rgb.png').read_bytes()).hexdigest() for s in ['initial','lost','recovered']}
   selected.append(r)
 ids={r['terrain_id'] for r in selected}; exclusions=[]
 for r in records:
  if r['terrain_id'] in ids:continue
  tid=r['terrain_id']
  if '/hist' in tid or tid=='v2/window_wall':
   reason='旧窗口局部任务同质：均为456初始目标像素、3接近动作、1恢复动作，且由新增0.75m方块挡视线；不证明窗口历史记忆必要性。RGB哈希不同，不认定全几何完全重复。'
  elif tid=='v2/staircase10':reason='近似重复：与stairs_up同为10级、0.15m级高、0.3m踏面、1.5m平台，初始45像素，接近3/恢复2动作；图像哈希不同。'
  elif tid=='design_v3/window_s89':reason='窗口中间参数变体；保留宽/中/窄三档，避免窗口类别过量。'
  else:reason='为每类三场景的配额精简；保留同族有明显尺度或结构差异的代表。此项不宣称几何重复。'
  exclusions.append({'terrain_id':tid,'reason':reason,'evidence_path':r['evidence_path']})
 dump('manifest.json',selected);dump('excluded.json',exclusions)
 counts=Counter(r['curated_family'] for r in selected)
 audit={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_accepted':len(records),'selected':len(selected),'excluded':len(exclusions),'family_counts':dict(counts),'all_gates_pass':all(all(r['gates'].values()) for r in selected),'all_evidence_images_exist':True,'hist_selected':sum('/hist' in x for x in ids),'rating':'provisional within family, no VLM rerun','status':'pass'}
 assert len(selected)==24 and set(counts.values())=={3};dump('audit.json',audit)
 rows=[]
 for r in selected:
  rows.append({'id':r['terrain_id'],'family':r['curated_family'],'level':r['provisional_difficulty'],'reason':r['rating_reason'],'pixels':r['initial_target_pixels'],'depth':r['initial_evaluator']['median_target_surface_range_m'],'images':{s:os.path.relpath(Path(r['evidence_path'])/s/'rgb.png',OUT) for s in ['initial','lost','recovered']},'spec':r['terrain_spec']})
 page='''<!doctype html><meta charset="utf-8"><title>Task3 均衡地形主集</title><style>body{font:16px sans-serif;background:#111827;color:#e5e7eb;margin:24px}select{padding:10px;max-width:95%}img{width:100%;max-width:480px} .images{display:flex;gap:12px;flex-wrap:wrap}.images div{flex:1;min-width:240px}pre{white-space:pre-wrap}button{padding:10px}a{color:#93c5fd}</style><h1>Task3 均衡主集：8类 × 3档 = 24场景</h1><p>评级为类内暂定几何评级。全部沿用已验收的初始可见→实体挡视线→恢复可见证据；未重跑VLM，未证明1m内任务成功。</p><label>类别 <select id="family"><option value="">全部类别</option></select></label> <label>暂定难度 <select id="level"><option value="">全部难度</option><option>简单</option><option>中等</option><option>复杂</option></select></label><p><select id="scene"></select></p><div id="view"></div><p><a href="manifest.json">完整入选配置</a> · <a href="excluded.json">排除理由</a> · <a href="audit.json">核验结果</a></p><script>const data=DATA;const family=document.querySelector('#family'),level=document.querySelector('#level'),scene=document.querySelector('#scene'),view=document.querySelector('#view');for(const f of [...new Set(data.map(r=>r.family))])family.add(new Option(f,f));function draw(){const r=data.find(r=>r.id===scene.value);if(!r){view.textContent='无匹配场景';return;}sessionStorage.setItem('balanced-scene',r.id);view.replaceChildren();const h=document.createElement('h2');h.textContent=r.family+' / '+r.level+' / '+r.id;view.append(h);const p=document.createElement('p');p.textContent=r.reason+'；初始有效目标像素 '+r.pixels+'；深度相机目标表面距离 '+r.depth.toFixed(2)+' m（评估器使用目标分割选取深度，非VLM自行估计）。';view.append(p);const images=document.createElement('div');images.className='images';for(const [key,title] of [['initial','初始可见'],['lost','实体挡视线后'],['recovered','实际动作恢复可见']]){const d=document.createElement('div');const label=document.createElement('p');label.textContent=title;const img=document.createElement('img');img.src=r.images[key];d.append(label,img);images.append(d);}view.append(images);const controls=document.createElement('p');for(const [title,delta] of [['上一个场景',-1],['下一个场景',1]]){const b=document.createElement('button');b.textContent=title;b.onclick=()=>{scene.selectedIndex=Math.max(0,Math.min(scene.options.length-1,scene.selectedIndex+delta));draw()};controls.append(b)}view.append(controls);const details=document.createElement('details');details.open=true;const summary=document.createElement('summary');summary.textContent='地形参数（默认展开）';const pre=document.createElement('pre');pre.textContent=JSON.stringify(r.spec,null,2);details.append(summary,pre);view.append(details)}function filter(){const prev=scene.value||sessionStorage.getItem('balanced-scene');scene.replaceChildren();for(const r of data.filter(r=>(!family.value||r.family===family.value)&&(!level.value||r.level===level.value)))scene.add(new Option(r.family+' / '+r.level+' / '+r.id,r.id));if([...scene.options].some(o=>o.value===prev))scene.value=prev;draw()}family.onchange=level.onchange=filter;scene.onchange=draw;filter();</script>'''
 (OUT/'index.html').write_text(page.replace('DATA',json.dumps(rows,ensure_ascii=False)))
 report=['# Task3 均衡主集 v1','', '54个合格来源场景筛选为24个；原始验收与冻结实验保持原样。8类各3个，每类简单、中等、复杂各1个。','', '评级依据类内几何尺度、障碍结构和已记录恢复动作，人工暂定；未使用VLM成功率。台阶和盆地评级置信度较低，初始距离/目标大小及辅助方块会影响难度，后续配对试验应重新校准。恢复动作数是已搜索成功序列长度，不是最短路径。','', '| 类别 | 简单 | 中等 | 复杂 |','|---|---|---|---|']
 for family,choices in PLAN.items():report.append('|'+family+'|'+'|'.join(tid for tid,_ in choices)+'|')
 report+=['','hist从16个减至0个；窗口从21个减至3个。坑洞、盆地、沟渠/沟壑共9个，随机起伏与连续波浪各3个。','', '去重口径：stairs_up/staircase10为几何参数近重复；旧窗口为局部任务同质，不能说全几何或图像完全重复。其余为配额/尺度代表性筛选。逐项理由见excluded.json。','', '所有24个通过原G1–G4门槛。此结果仅证明可见性恢复验收，不证明能到目标1m内；未在新子集重新运行模型或计算模型成功率。未通过门槛的碎石、平滑坡等不能强行补入，需要后续重新设计并验收。']
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
