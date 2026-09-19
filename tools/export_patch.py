"""Only the physical 10 m patch and its seven rocks; no full-world UE dependency."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'mujoco'
meta=json.loads((ROOT/'generated/scene.json').read_text())
origin=np.array(meta['origin_ue_m'])
raw=np.frombuffer((ROOT/'generated/height.bin').read_bytes()[8:],dtype='<f4').reshape(21,21)
lo,hi=meta['terrain_z_range_local_m'];z=raw*(hi-lo)+lo
points=[];indices=[]
dzdr,dzdc=np.gradient(z,.5)
normals=np.stack([-dzdc,dzdr,np.ones_like(z)],axis=-1)
normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
for r in range(21):
    for c in range(21):points.append([(-5+c*.5)*100,(5-r*.5)*100,float(z[r,c])*100])
for r in range(20):
    for c in range(20):
        a=r*21+c
        # Match MuJoCo hfield diagonal: verify triangles via off-grid rays in tests.
        indices.extend([a,a+1,a+22,a,a+22,a+21])
rows=list(csv.DictReader((ROOT.parent/'ue/import_data/v2/rock_positions_ue.csv').open()))
rocks=[]
for i in meta['rocks']:
    r=rows[i]
    rocks.append({'id':i,'mesh':int(r['mesh']),'position':[float(r[k]) for k in ('x_cm','y_cm','z_cm')],
                  'quaternion':[float(r[k]) for k in ('qx','qy','qz','qw')],'scale':[float(r[k]) for k in ('sx','sy','sz')]})
(ROOT/'generated/patch.json').write_text(json.dumps({'origin_cm':(origin*100).tolist(),'points':points,'normals':normals.reshape(-1,3).tolist(),'indices':indices,'rocks':rocks},separators=(',',':')))
print('PATCH_EXPORT',len(points),'vertices',len(indices)//3,'triangles',len(rocks),'rocks')
