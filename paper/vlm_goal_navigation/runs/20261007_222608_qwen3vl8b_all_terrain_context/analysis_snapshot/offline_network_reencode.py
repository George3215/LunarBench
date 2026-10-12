import json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import AutoProcessor
root=Path(sys.argv[1]);processor=AutoProcessor.from_pretrained('/home/lry/MoonUnrealEnv/MoonSim/assets/model/Qwen3-VL-8B-Instruct',local_files_only=True);processor.tokenizer.padding_side='left';processor.image_processor.size={'shortest_edge':65536,'longest_edge':65536}
results=[]
for folder in sorted(root.glob('*/*/trials/barrier')):
 p=sorted(folder.glob('steps/*/inference_1/network_input.json'))[0];net=json.loads(p.read_text());saved=json.loads((p.parent/'messages.json').read_text())['messages'];msg=[]
 for m in saved:
  if isinstance(m['content'],str):msg.append(m);continue
  blocks=[]
  for b in m['content']:
   blocks.append({'type':'image','image':Image.open(b['image_path']).convert('RGB')} if b['type']=='image' else b)
  msg.append({'role':m['role'],'content':blocks})
 prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
 images=[b['image'] for b in msg[1]['content'] if b['type']=='image']
 batch=processor(text=[prompt],images=images,padding=True,return_tensors='pt')
 ids=[v for v,m in zip(net['input_ids'],net['attention_mask']) if m];assert ids==batch['input_ids'][0].tolist()
 assert prompt==(p.parent/'rendered_prompt.txt').read_text()
 pixels=batch['pixel_values'].float().numpy();offset=0
 for index,im in enumerate(net['images']):
  grid=batch['image_grid_thw'][index].tolist();assert grid==im['image_grid_thw'];count=int(np.prod(grid));chunk=pixels[offset:offset+count];offset+=count
  assert hashlib.sha256(chunk.tobytes()).hexdigest()==im['pixel_values_float32_sha256']
 results.append({'trial':str(folder.relative_to(root)),'token_ids_match':True,'all_preprocessed_pixel_hashes_match':True,'images':len(images),'active_input_tokens':len(ids)})
 print(results[-1],flush=True)
assert len(results)==6
(root/'offline_network_reencoding_audit.json').write_text(json.dumps({'pass':True,'conditions':results},indent=2))
