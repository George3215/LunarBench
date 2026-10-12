"""Select one scene to inspect first-visible/lost/near evidence and terrain changes."""
import json,sys,os
from pathlib import Path
root=Path(sys.argv[1]).resolve();setups=json.loads((root/'manifest.json').read_text());entries=[]
for s in setups:
 p=Path(s['evidence_path']);info={k:s[k] for k in ('added_occluder','approach_actions','reference_termination')}
 info.update({'original_target':s['source_scene']['target_xy'],'adjusted_target':s['adjusted_scene']['target_xy'],
 'target_height_m':s['adjusted_scene']['target_height_m'],'initial_camera_depth_range_m':s['initial_depth']['red_region_median_surface_range_m'],
 'blocking_geom':s['loss']['blocking_geom'],'start':s['adjusted_scene']['lost_xy'],
 'original_terrain_geometry_unchanged':s['original_terrain_geometry_unchanged'],'terrain_changes':s['terrain_spec'].get('navigation_design_changes',[])})
 entries.append({'name':s['scene_name'],'info':info,'images':[{'label':label,'path':os.path.relpath(p/phase/'rgb.png',root)} for label,phase in [('初始可见','initial'),('运动后不可见','lost'),('参考控制近距终点','reference_final')]]})
page='''<!doctype html><meta charset="utf-8"><title>Task3 场景准入</title><style>body{font:16px system-ui;margin:24px;background:#151b24;color:#eee}select,button{font:inherit;padding:8px;margin:8px}img{max-width:100%}figure{display:inline-block;width:31%;margin:1%}pre{white-space:pre-wrap;background:#24303b;padding:12px}</style><h1>Task3 场景准入</h1><p>默认不展示全部场景。选择一个场景查看初始可见、运动后不可见和参考近距终点。参考控制器仅验证可行性，不是 VLM 成功。</p><p id="count"></p><select id="scene"><option value="">选择场景</option></select><button onclick="show()">查看所选场景</button><div id="detail"></div><script>const D=__DATA__;const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');document.getElementById('count').textContent='已通过 '+D.length+' 个场景';D.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.name+' / '+(s.info.added_occluder?s.info.added_occluder.kind||'旧版薄墙':'地形起伏');document.getElementById('scene').appendChild(o)});function show(){let v=document.getElementById('scene').value;if(v==='')return;let s=D[+v];document.getElementById('detail').innerHTML='<h2>'+esc(s.name)+'</h2>'+s.images.map(im=>'<figure><figcaption>'+esc(im.label)+'</figcaption><img src="'+esc(im.path)+'"></figure>').join('')+'<pre>'+esc(JSON.stringify(s.info,null,2))+'</pre>'}</script>'''
(root/'scene_review.html').write_text(page.replace('__DATA__',json.dumps(entries,ensure_ascii=False).replace('</','<\\/')));print(len(entries),'selectable review scenes')
