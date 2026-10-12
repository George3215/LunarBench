"""Local VLM client. Only supplied image bytes and explicit text reach the model."""
from __future__ import annotations
import base64, hashlib, json, time
from pathlib import Path
import requests

ACTIONS = ('forward', 'back', 'left', 'right', 'turn_left', 'turn_right', 'stop')
MODES = ('current_direct', 'current_reasoned', 'actions_reasoned', 'images_reasoned', 'images_actions_reasoned')
SYSTEM = '''You control a simulated quadruped using its forward-facing body camera.
Goal: find the red sphere that was seen earlier, recover a useful view, then walk close to it and stop.
Only the supplied camera images and, when provided, executed-command history describe the world.
No target world coordinates or map are available. Images are ordered oldest to newest; the final image is CURRENT.
Select exactly one action. forward/back translate along the body heading; left/right strafe;
turn_left/turn_right rotate the body (about 45 degrees per full step, approximate).
Each motion lasts 3 seconds. Forward is about 1 m/s commanded, backward 0.8 m/s, strafe 0.6 m/s;
actual displacement depends on terrain. Stop only when close to the correct visible sphere.
Do not assume the object is in a fixed direction. Camera perspective changes with robot motion.
Return a JSON object with action (one of forward, back, left, right, turn_left, turn_right, stop),
target_visible (boolean), and reason (one short sentence).'''
SCHEMA = {'type':'object','properties':{'action':{'type':'string','enum':list(ACTIONS)},
 'target_visible':{'type':'boolean'},'reason':{'type':'string'}},
 'required':['action','target_visible','reason'],'additionalProperties':False}

class LocalVLM:
 def __init__(self, model='qwen3.5:4b', endpoint='http://127.0.0.1:11435', max_tokens=1024, context=8192, history_layout='native', task_mode='approach'):
  self.model,self.endpoint,self.max_tokens,self.context=model,endpoint,max_tokens,context
  self.history_layout=history_layout
  self.task_mode=task_mode
 def decide(self, current:Path, past_images:list[Path], commands:list[dict], mode:str, step:int):
  assert mode in MODES
  history=mode in ('images_reasoned','images_actions_reasoned')
  actions=mode in ('actions_reasoned','images_actions_reasoned')
  images=(past_images if history else [])+[current]
  prompt='Choose the next action toward the goal. '
  if mode=='current_direct': prompt+='Respond directly without extended deliberation. '
  else: prompt+='Consider camera evidence, obstacles, movement effects and uncertainty before selecting. '
  if history: prompt+=f'There are {len(images)-1} historical images followed by the CURRENT image. '
  else: prompt+='Only the CURRENT image is provided. '
  if actions: prompt+='Executed commands, oldest first (commands are not measured displacements): '+json.dumps(commands,ensure_ascii=True)
  source_images=list(images)
  if history and self.history_layout=='sheet':
   from PIL import Image,ImageDraw,ImageFont
   cols=2;rows=(len(images)+1)//2;w,h=320,240;label_h=28
   sheet=Image.new('RGB',(cols*w,rows*(h+label_h)),(225,225,225));draw=ImageDraw.Draw(sheet)
   try: font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
   except OSError: font=ImageFont.load_default()
   for i,p in enumerate(images):
    x=(i%cols)*w;y=(i//cols)*(h+label_h)
    label='CURRENT (use for immediate action)' if i==len(images)-1 else f'HISTORY {i+1} (older observation)'
    draw.rectangle((x,y,x+w-1,y+label_h-1),fill=(180,220,250) if i==len(images)-1 else (240,240,240))
    draw.text((x+5,y+4),label,fill=(0,0,0),font=font)
    sheet.paste(Image.open(p).convert('RGB'),(x,y+label_h))
   sheet_path=current.with_name('input_sheet_'+current.stem+'.png');sheet.save(sheet_path)
   images=[sheet_path]
   prompt+=' The supplied single contact sheet has labeled HISTORY panels and one CURRENT panel. Only the CURRENT panel depicts what is visible now. A target in a HISTORY panel is not currently visible. '
  blobs=[p.read_bytes() for p in images]
  system=SYSTEM
  schema=SCHEMA
  if self.task_mode=='reacquire':
   system=SYSTEM.replace('Goal: find the red sphere that was seen earlier, recover a useful view, then walk close to it and stop.',
    'Goal: reacquire a camera view of the previously seen red target sphere within the next 3-second action. Choose one motion even when uncertain.')
   system=system.replace('Stop only when close to the correct visible sphere.','There is no STOP action in this single-step task.')
   system=system.replace('turn_left, turn_right, stop)','turn_left, turn_right)')
   schema=json.loads(json.dumps(SCHEMA));schema['properties']['action']['enum']=list(ACTIONS[:-1])
  payload={'model':self.model,'messages':[{'role':'system','content':system},
   {'role':'user','content':prompt,'images':[base64.b64encode(x).decode() for x in blobs]}],
   'stream':False,'think':mode!='current_direct','format':schema,'keep_alive':'30m',
   'options':{'temperature':0,'seed':7,'num_ctx':self.context,'num_predict':self.max_tokens if mode!='current_direct' else 180}}
  start=time.perf_counter()
  r=requests.post(self.endpoint+'/api/chat',json=payload,timeout=(10,240));r.raise_for_status(); raw=r.json()
  content=raw.get('message',{}).get('content','')
  deliberation=None
  # Bounded deliberation is a separate phase. Always request a final structured
  # action with thinking disabled, instead of treating a thinking-token cutoff
  # as robot task failure. This rule is identical in all four reasoning modes.
  if mode!='current_direct':
   deliberation=raw
   reasoning=raw.get('message',{}).get('thinking','') or content
   final_payload=dict(payload)
   final_payload['think']=False
   final_payload['options']=dict(payload['options'],num_predict=180)
   final_payload['messages']=payload['messages']+[
    {'role':'assistant','content':'My preliminary analysis: '+reasoning},
    {'role':'user','content':'Now select one action using that analysis and the supplied images. Return only the requested JSON object.'}]
   r=requests.post(self.endpoint+'/api/chat',json=final_payload,timeout=(10,240));r.raise_for_status();raw=r.json()
   content=raw.get('message',{}).get('content','')
  parsed=None; error=None
  try:
   parsed=json.loads(content)
   if parsed.get('action') not in (ACTIONS[:-1] if self.task_mode=='reacquire' else ACTIONS): raise ValueError('invalid action')
  except (ValueError,TypeError) as e: error=str(e)
  log={'model':self.model,'mode':mode,'step':step,'task_mode':self.task_mode,'system_prompt':system,'user_prompt':prompt,
       'history_layout':self.history_layout,'source_image_files':[str(p) for p in source_images],'image_files':[str(p) for p in images],'image_sha256':[hashlib.sha256(b).hexdigest() for b in blobs],
       'settings':{k:v for k,v in payload.items() if k!='messages'},'response':raw,'deliberation_response':deliberation,'finalization_rule':'reasoning modes use a bounded thinking call followed by a non-thinking JSON call',
       'elapsed_s':time.perf_counter()-start,'parsed':parsed,'parse_error':error}
  return parsed,log
