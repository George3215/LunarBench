from pathlib import Path
import json,base64,argparse,requests,hashlib
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
r=Path('paper/vlm_goal_navigation/runs/20261004_qwen35');out=[]
for first,second in [('left','none'),('none','right'),('left','right')]:
 image=r/f'sheet_check_{first}_{second}.png'
 payload={'model':a.model,'messages':[{'role':'user','content':'Inspect the two labeled panels separately. For each panel report the red dot location, left, right, or none. Return JSON with first for the first panel and second for the second panel.','images':[base64.b64encode(image.read_bytes()).decode()]}],
 'think':False,'stream':False,'format':{'type':'object','properties':{k:{'type':'string','enum':['left','right','none']} for k in ['first','second']},'required':['first','second']},'options':{'temperature':0,'seed':7,'num_ctx':8192,'num_predict':100},'keep_alive':'30m'}
 resp=requests.post('http://127.0.0.1:11435/api/chat',json=payload,timeout=180);resp.raise_for_status();data=resp.json();q=json.loads(data['message']['content'])
 out.append({'expected':[first,second],'parsed':q,'correct':[q.get('first'),q.get('second')]==[first,second],'response':data,'image':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'prompt':payload['messages'][0]['content']})
(r/('labeled_sheet_'+a.model.replace(':','_')+'.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print('Labeled image identity:',sum(x['correct'] for x in out),'/',len(out),flush=True)
