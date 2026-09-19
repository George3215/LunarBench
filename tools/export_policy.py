"""One-time TorchScript -> NumPy weights; no retraining or runtime torch needed."""
from pathlib import Path
import json
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]/'assets/robots/go2_demo/policy'
torch.set_num_threads(1)
p=torch.jit.load(str(root/'policy.pt'),map_location='cpu').eval()
assert p.normalizer.original_name=='Identity'
assert [m.original_name for m in p.actor.children()]==['Linear','ELU','Linear','ELU','Linear','ELU','Linear']
weights={k:v.detach().numpy() for k,v in p.state_dict().items()}
x=np.random.default_rng(42).normal(size=(128,45)).astype(np.float32)
y=x.copy()
for i in (0,2,4,6):
    y=y@weights[f'actor.{i}.weight'].T+weights[f'actor.{i}.bias']
    if i!=6:
        y=np.where(y>0,y,np.expm1(np.minimum(y,0)))
with torch.no_grad():
    expected=p(torch.from_numpy(x)).numpy()
error=float(np.max(np.abs(y-expected)))
assert error<1e-4,error
np.savez(root/'policy.npz',**weights)
(root/'equivalence.json').write_text(json.dumps({'max_abs_error':error,'samples':128,'seed':42,'architecture':[45,512,256,128,12],'activation':'ELU','normalizer':'Identity'},indent=2))
print('POLICY_EXPORT_PASS',error)
