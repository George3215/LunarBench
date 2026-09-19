"""Audit original USD/R16 samples and all original rock faces, without resampling."""
from pathlib import Path
import json
import sys
import csv
import numpy as np
from pxr import Usd, UsdGeom
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mujoco'))
from go2 import Simulation

root=Path(__file__).resolve().parents[1]
source=root/'assets/environments/lunar/terrain/landscape_cropped/Props/Landscape_1.usd'
stage=Usd.Stage.Open(str(source),load=Usd.Stage.LoadNone)
mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Root/Landscape_1'))
points=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float32)
x=np.unique(points[:,0]);y=np.unique(points[:,1])
raw=np.empty((len(y),len(x)),dtype=np.float32)
raw[np.searchsorted(y,points[:,1]),np.searchsorted(x,points[:,0])]=points[:,2]
r16=np.fromfile(root/'ue/import_data/Landscape_1_2795x2795.r16',dtype='<u2').reshape(len(y),len(x))
decoded=(r16.astype(np.float32)-32768)/128
height_error=float(np.abs(decoded-raw[::-1]).max())
assert height_error==0, height_error
assert points.shape==(2795*2795,3)
scale=json.loads((root/'ue/import_data/v2/scene.json').read_text())['scale']
sim=Simulation()
rows=list(csv.DictReader((root/'ue/import_data/v2/rock_positions_ue.csv').open()))
rocks=[]
for ident in sim.meta['rocks']:
 kind=int(rows[ident]['mesh'])
 original=json.loads((root/f'ue/import_data/v2/rock{kind}.json').read_text())
 mid=sim.model.mesh(f'rock_{ident:06d}').id
 record={'rock_id':ident,'source_vertices':len(original['points']),
         'source_triangles':len(original['indices'])//3,
         'compiled_vertices':int(sim.model.mesh_vertnum[mid]),
         'compiled_triangles':int(sim.model.mesh_facenum[mid])}
 assert record['source_triangles']==record['compiled_triangles'],record
 rocks.append(record)
report={'terrain_source_grid':[len(y),len(x)],'source_points':len(points),
        'source_spacing_m':float(x[1]-x[0])*UsdGeom.GetStageMetersPerUnit(stage),
        'scene_xy_scale':scale[:2],'scene_spacing_m':float(x[1]-x[0])*.01*scale[0],
        'source_to_r16_max_height_error_cm':height_error,'terrain_resampled':False,
        'physics_patch_grid':[int(sim.model.hfield_nrow[0]),int(sim.model.hfield_ncol[0])],
        'rocks':rocks,'rock_render_faces_preserved':True,
        'collision_semantics':'MuJoCo native convex mesh collision; original render faces retained',
        'robot_meshes':'unchanged original OBJ files; group 3 collision primitives hidden in viewer'}
(root/'mujoco/generated/precision_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
