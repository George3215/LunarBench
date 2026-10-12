import json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
b=Path(sys.argv[1]);p=json.loads((b/'PROTOCOL.json').read_text());rows=json.loads((b/'formal_manifest.json').read_text());checks=[];reject=[]
for r in rows:
 f=Path(r['evidence_path']);item={'terrain_id':r['terrain_id'],'phases':{}}
 assert all(r['gates'].values())
 for phase in ['initial','lost','recovered']:
  rgb=np.asarray(Image.open(f/phase/'rgb.png'));mask=np.asarray(Image.open(f/phase/'target_mask.png'))>0
  red,green,blue=[rgb[...,i].astype(np.int16) for i in range(3)];n=int((mask&(red>50)&(red>1.5*green)&(red>1.5*blue)).sum());m=json.loads((f/phase/'measurement.json').read_text());assert n==m['target_visible_rgb_pixels']
  depth=np.load(f/phase/'depth.npy');assert depth.shape==mask.shape
  item['phases'][phase]={'rgb_target_pixels_recounted':n,'depth_shape':list(depth.shape)}
 assert item['phases']['initial']['rgb_target_pixels_recounted']>=p['initial_rgb_min_pixels']
 assert item['phases']['recovered']['rgb_target_pixels_recounted']>=p['recovery_rgb_min_pixels']
 assert r['loss']['ray']['entity_occlusion'] and r['loss']['ray']['target_centre_in_fov']
 assert r['loss']['lost_frames']>=3 and r['loss']['displacement_m']>=.25
 assert any(o.get('recovery') and o['recovery']['consecutive_frames']>=3 for o in r['oracle'])
 assert all((f/name).stat().st_size>0 for name in ['state.jsonl','evaluator.jsonl'])
 assert not r['terrain_spec'].get('navigation_added_occluder')
 checks.append(item)
for r in json.loads((b/'admission_records.json').read_text()):
 if r['accepted']:continue
 attempts=[json.loads((Path(a['evidence_path'])/'gates.json').read_text()) for a in r.get('attempts',[])]
 reject.append({'terrain_id':r['terrain_id'],'candidate_attempts':len(attempts),'candidate_gate_pass_counts':{g:sum(a['gates'][g] for a in attempts) for g in ['G1','G2','G3','G4']},'reason':'no geometric candidate' if not attempts else 'finite physical candidate search exhausted'})
hashes=json.loads((b/'source_hashes.json').read_text());assert all(hashlib.sha256((b/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
result={'status':'pass','accepted_evidence_checked':len(checks),'source_hashes_unchanged':True,'images_depth_state_checked':True,'details':checks}
(b/'evidence_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(b/'rejection_analysis.json').write_text(json.dumps(reject,ensure_ascii=False,indent=2));print('EVIDENCE_AUDIT_PASS',len(checks),'REJECTED',len(reject))
