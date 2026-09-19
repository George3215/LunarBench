import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"mujoco"))
"""Export MuJoCo compiled meshes and initial geom poses for exact UE display."""
import json
import numpy as np
from run import Simulation,ROOT

s=Simulation();m=s.model
meshes=[]
for mid in sorted(set(int(m.geom_dataid[i]) for i in s.visual)):
    a,n=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid])
    f,nf=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid])
    vertices=m.mesh_vert[a:a+n]*[100,-100,100]
    # UE reverse culling convention matches existing successful rock importer.
    faces=m.mesh_face[f:f+nf]
    meshes.append({'id':mid,'points':vertices.tolist(),'indices':faces.ravel().tolist()})
geoms=[]
for i in s.visual:
    color=m.mat_rgba[m.geom_matid[i]].tolist() if m.geom_matid[i]>=0 else m.geom_rgba[i].tolist()
    geoms.append({'id':i,'mesh':int(m.geom_dataid[i]),'rgba':color})
out={'meshes':meshes,'geoms':geoms,'initial':s.packet(),'origin_ue_m':s.meta['origin_ue_m']}
(ROOT/'generated/visual.json').write_text(json.dumps(out,separators=(',',':')))
print('VISUAL_EXPORT',len(meshes),len(geoms))
