"""Perception diagnostic, separate from navigation and never given to its policy."""
import argparse,base64,json,time
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--model',default='qwen3.5:4b');a=p.parse_args()
root=Path('paper/vlm_goal_navigation/runs/20261004_qwen35');out=root/('perception_'+a.model.replace(':','_')+'.json')
old=json.loads(Path('paper/audit_20261004/strict_camera/camera_rollouts.json').read_text());rows=[]
for r in old['records']:
 if not r['eligible']:continue
 for phase in ['seen','lost']:
  f=Path('paper/audit_20261004/strict_camera/frames')/(r['scene']+'_'+phase+'.png')
  payload={'model':a.model,'messages':[{'role':'user','content':'Inspect ONLY this image. Is any red target sphere or small red target dot visible? Do not infer a hidden object. Return JSON with visible (boolean), position (left, center, right, or absent).','images':[base64.b64encode(f.read_bytes()).decode()]}],
   'think':False,'stream':False,'format':{'type':'object','properties':{'visible':{'type':'boolean'},'position':{'type':'string','enum':['left','center','right','absent']}},'required':['visible','position']},'options':{'temperature':0,'seed':7,'num_ctx':8192,'num_predict':100},'keep_alive':'30m'}
  start=time.time();resp=requests.post('http://127.0.0.1:11435/api/chat',json=payload,timeout=180);resp.raise_for_status();d=resp.json()
  try: answer=json.loads(d['message']['content'])
  except Exception:answer=None
  rows.append({'scene':r['scene'],'phase':phase,'image':str(f),'expected_marker_visible':phase=='seen','answer':answer,'response':d,'wall_s':time.time()-start})
  out.write_text(json.dumps(rows,indent=2),encoding='utf8')
 print('PERCEPTION',a.model,r['scene'],[(x['phase'],x['answer']) for x in rows[-2:]],flush=True)
