"""Verify saved model inputs and first executed steps without rerunning policies."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'paper/vlm_goal_navigation/runs/20261004_qwen35'
old=json.loads((ROOT/'paper/audit_20261004/strict_camera/camera_rollouts.json').read_text())
records={r['scene']:r for r in old['records']};checks=[];hash_count=0;source_count=0
for folder in ['closed_loop_v2','closed_loop_9b']:
 for f in (RUN/folder).glob('episodes/*/*/*/result.json'):
  r=json.loads(f.read_text())
  for dp in f.parent.glob('decision_*.json'):
   d=json.loads(dp.read_text())
   for name,h in zip(d['image_files'],d['image_sha256'],strict=True):
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    hash_count+=1
   for name in d.get('source_image_files',d['image_files']):
    assert Path(name).exists(),name
    source_count+=1
  if not r['steps'] or r['steps'][0]['action']=='stop':continue
  step=r['steps'][0];branch=next(b for b in records[r['scene']]['branches'] if b['action']==step['action'])
  a=step['curve'];b=branch['curve']
  delta=max((max(abs(np.array(x['xy'])-np.array(y['xyz'][:2]))) for x,y in zip(a,b)),default=None)
  checks.append({'folder':folder,'scene':r['scene'],'mode':r['mode'],'action':step['action'],
   'camera_success_matches':step['reacquired_so_far']==branch['success'],'sample_count_matches':len(a)==len(b),
   'max_xy_difference_vs_rounded_legacy_m':delta})
  assert step['reacquired_so_far']==branch['success'],str(f)
  assert len(a)==len(b),str(f)
  assert delta is not None and delta<0.00006,str(f)
(RUN/'first_step_physics_check.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
result={'online_input_hashes_verified':hash_count,'source_image_paths_verified':source_count,'executed_first_steps_verified_against_old_branches':len(checks),
 'maximum_xy_difference_m':max((x['max_xy_difference_vs_rounded_legacy_m'] for x in checks),default=None),
 'outcome':'pass','note':'Only matching first steps are compared. Later actions have continuous controller history and are new online trajectories.'}
(RUN/'artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
