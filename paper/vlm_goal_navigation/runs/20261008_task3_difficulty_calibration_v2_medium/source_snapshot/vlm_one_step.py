"""Exact existing one-step benchmark: VLM selects among already physically executed six branches.
No future outcome enters a model request. Camera prefix and executed branch hashes are pinned.
"""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
from local_vlm import LocalVLM,MODES
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--model',default='qwen3.5:4b');p.add_argument('--modes',nargs='+',default=list(MODES),choices=MODES);p.add_argument('--tokens',type=int,default=512);a=p.parse_args()
root=ROOT/'paper/vlm_goal_navigation/runs/20261004_qwen35';out=root/'original_six_action';out.mkdir(exist_ok=True)
raw_path=ROOT/'paper/audit_20261004/strict_camera/camera_rollouts.json';raw=json.loads(raw_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
vlm=LocalVLM(model=a.model,max_tokens=a.tokens,history_layout='sheet',task_mode='reacquire')
(out/'source_snapshot').mkdir(exist_ok=True)
for f in [Path(__file__),Path(__file__).with_name('local_vlm.py')]:shutil.copyfile(f,out/'source_snapshot'/(sha(f)[:12]+'_'+f.name))
protocol={'model':a.model,'task':'original strict-camera single 3-second action; six original actions; no STOP',
 'evaluation':'VLM selects action; score from pinned physically executed branch table. This is offline one-step selection, not online search. Separate closed_loop_v2 records real online multi-step execution.',
 'source_branch_table':str(raw_path),'branch_table_sha256':sha(raw_path),'candidate_worlds':24,'eligible_worlds':23,
 'modes':a.modes,'history':'up to 3 uniform chronological prefix frames + current frame, explicitly labeled contact sheet',
 'thinking_budget':a.tokens,'finalization_budget':180,'temperature':0,'seed':7,'new_training':False,
 'privileged_inputs':False,'source_sha256':{f.name:sha(f) for f in [Path(__file__),Path(__file__).with_name('local_vlm.py')]}}
(out/('protocol_'+a.model.replace(':','_')+'.json')).write_text(json.dumps(protocol,indent=2),encoding='utf8')
for rec in raw['records']:
 if not rec['eligible']:continue
 prefix=root/'prefixes'/rec['scene'];meta=json.loads((prefix/'metadata.json').read_text())
 frames=[Path(x['path']) for x in meta['prefix_frames'] if x['time']<meta['loss_t']]
 if len(frames)>3:frames=[frames[i] for i in np.linspace(0,len(frames)-1,3,dtype=int)]
 for mode in a.modes:
  d=out/a.model.replace(':','_')/mode/rec['scene'];d.mkdir(parents=True,exist_ok=True);result=d/'result.json'
  if result.exists():continue
  current=d/'current.png';shutil.copyfile(prefix/'loss.png',current)
  parsed,log=vlm.decide(current,frames,[meta['approach_command']],mode,0)
  (d/'decision.json').write_text(json.dumps(log,indent=2),encoding='utf8')
  branch=next((b for b in rec['branches'] if parsed and b['action']==parsed['action']),None)
  r={'model':a.model,'mode':mode,'scene':rec['scene'],'group':rec['group'],'action':None if not parsed else parsed['action'],
   'success':bool(branch and branch['success']),'valid_output':branch is not None,'model_s':log['elapsed_s'],
   'source_branch_sha256':None if branch is None else hashlib.sha256(json.dumps(branch,sort_keys=True).encode()).hexdigest(),
   'evaluator_motion_cost_m':None if branch is None else branch['cost_m'],'evaluation':'existing physically executed branch; not an online closed-loop episode'}
  result.write_text(json.dumps(r,indent=2),encoding='utf8')
  print('SIX_ACTION',a.model,mode,rec['scene'],r['action'],r['success'],round(r['model_s'],1),flush=True)
