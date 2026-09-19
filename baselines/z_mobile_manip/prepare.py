"""One-time model preparation. Run with .local/zmobile-env/bin/python; no CLI flags."""
from pathlib import Path
import hashlib
import json
import os
workspace=Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME",str(workspace/".local/huggingface"))
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
import torch
from ultralytics import YOLOE

models=Path(__file__).resolve().parents[3]/'.local/models/z_mobile_manip'
models.mkdir(parents=True,exist_ok=True);os.chdir(models)
model=YOLOE('yoloe-11s-seg.pt')
names=['rock','stone'];embedding=model.get_text_pe(names)
torch.save({'names':names,'embeddings':embedding.cpu()},'rock_text_embeddings.pt')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in models.iterdir() if p.suffix in ('.pt','.ts')}
(models/'manifest.json').write_text(json.dumps({'detector':'ultralytics YOLOE-11s-seg','ultralytics':'8.4.104','prompts':names,'files':files},indent=2)+'\n')
from huggingface_hub import snapshot_download
snapshot_download('yonigozlan/EdgeTAM-hf',local_dir=models/'edgetam',allow_patterns=['*.json','*.safetensors','README.md'])
snapshot_download('timm/repvit_m1.dist_in1k',allow_patterns=['*.json','*.safetensors'])
print('Models prepared:',models)
