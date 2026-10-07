"""Run a real camera-image request and record runtime/model/GPU evidence."""
from pathlib import Path
import argparse,json,sys,subprocess,time,requests
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tasks/task3_reobservability'))
from local_vlm import LocalVLM
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
r=ROOT/'paper/vlm_goal_navigation/runs/20261004_qwen35'
client=LocalVLM(model=a.model,max_tokens=512,history_layout='sheet',task_mode='reacquire')
evidence={'model':a.model,'started':time.strftime('%Y-%m-%dT%H:%M:%S'),'api_tags':requests.get(client.endpoint+'/api/tags',timeout=30).json(),
 'api_show':requests.post(client.endpoint+'/api/show',json={'model':a.model},timeout=30).json()}
try:
 parsed,log=client.decide(r/'prefixes/stairs_up/loss.png',[],[],'current_direct',0)
 evidence.update({'inference_completed':True,'parsed':parsed,'log':log,'api_ps':requests.get(client.endpoint+'/api/ps',timeout=30).json(),
  'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()})
except Exception as exc:
 evidence.update({'inference_completed':False,'error':repr(exc)})
(r/('runtime_'+a.model.replace(':','_')+'.json')).write_text(json.dumps(evidence,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in evidence.items() if k not in ['api_tags','api_show','log']},indent=2),flush=True)
if not evidence['inference_completed']:raise SystemExit(1)
