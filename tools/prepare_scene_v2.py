"""Audit USD and export full-resolution rock geometry and per-instance transforms."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ue/import_data/v2'
OUT.mkdir(parents=True, exist_ok=True)
src = ROOT / 'assets/environments/lunar/terrain/landscape_cropped'
# Open root layer with payloads deferred, mute empty obsolete grass references
# in an anonymous copy, leaving source files intact.
layer = Sdf.Layer.CreateAnonymous()
layer.TransferContent(Sdf.Layer.FindOrOpen(str(src / 'landscape_cropped.usd')))
for spec in list(layer.GetPrimAtPath('/Root/Landscape1').nameChildren.values()):
    if spec.name.startswith('GrassInstanced'):
        del layer.GetPrimAtPath('/Root/Landscape1').nameChildren[spec.name]
layer.UpdateExternalReference('./Props/Landscape_1.usd', str(src / 'Props/Landscape_1.usd'))
stage = Usd.Stage.Open(layer)
parent = UsdGeom.Xformable(stage.GetPrimAtPath('/Root/Landscape1')).ComputeLocalToWorldTransform(0)
print('scene transform', parent)
terrain = Usd.Stage.Open(str(src / 'Props/Landscape_1.usd'))
shader = terrain.GetPrimAtPath('/Root/Looks/MI_Landscape0/MI_Landscape0')
def params(p):
    result = {}
    for a in p.GetAttributes():
        if a.GetName().startswith('inputs:'):
            v = a.Get()
            result[a.GetName()[7:]] = v.path if isinstance(v, Sdf.AssetPath) else list(v) if hasattr(v, '__len__') and not isinstance(v, str) else v
    return result
parameters = params(shader)
(OUT/'material_parameters.json').write_text(json.dumps(parameters, indent=2))
for k in (1, 2):
    rs = Usd.Stage.Open(str(src/f'Props/SM_Rock_0{k}.usd'))
    mesh = next(UsdGeom.Mesh(p) for p in rs.Traverse() if p.IsA(UsdGeom.Mesh))
    points = np.array(mesh.GetPointsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    assert np.all(counts == 3)
    normals = np.array(mesh.GetNormalsAttr().Get())
    uv = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').ComputeFlattened())
    points[:,1] *= -1
    normals[:,1] *= -1
    uv[:,1] = 1-uv[:,1]
    # Reflection changes handedness; UE's clockwise front face restores USD winding.
    data = dict(points=points.tolist(), indices=indices.tolist(), normals=normals.tolist(), uv=uv.tolist())
    (OUT/f'rock{k}.json').write_text(json.dumps(data, separators=(',', ':')))
    print('mesh', k, len(points), len(indices)//3)
fol = Usd.Stage.Open(str(src/'Props/Landscape_1_Foliage.usd'))
inst = next(UsdGeom.PointInstancer(p) for p in fol.Traverse() if p.IsA(UsdGeom.PointInstancer))
targets = inst.GetPrototypesRel().GetTargets()
types = [1 if 'SM_Rock_01' in str(fol.GetPrimAtPath(t).GetMetadata('references')) else 2 for t in targets]
transforms = inst.ComputeInstanceTransformsAtTime(0, 0, UsdGeom.PointInstancer.IncludeProtoXform, UsdGeom.PointInstancer.IgnoreMask)
proto = inst.GetProtoIndicesAttr().Get()
mask = list(inst.ComputeMaskAtTime(0))
records = []
flip = Gf.Matrix4d(1); flip[1,1] = -1
max_shear_error = 0
for idx, local in enumerate(transforms):
    if mask and not mask[idx]:
        continue
    world = flip * local * parent * flip
    tr = Gf.Transform(world)
    rot = tr.GetRotation().GetQuat()
    pos, scale = tr.GetTranslation(), tr.GetScale()
    # Gf decomposes shear into scale orientation; record matrix for audit.
    rebuilt = Gf.Matrix4d().SetScale(scale) * Gf.Matrix4d().SetRotate(rot) * Gf.Matrix4d().SetTranslate(pos)
    err = float(np.max(np.abs(np.array(world)-np.array(rebuilt))))
    max_shear_error = max(max_shear_error, err)
    records.append(dict(id=idx, mesh=types[proto[idx]], position=list(pos), quaternion=[*rot.GetImaginary(),rot.GetReal()], scale=list(scale), matrix=np.array(world).tolist()))
(OUT/'rocks.json').write_text(json.dumps(records, separators=(',', ':')))
origin = parent.Transform(Gf.Vec3d(1270,-2794,0))
scene = dict(location=[origin[0],-origin[1],origin[2]], scale=[50,50,25], rocks=len(records), max_matrix_decomposition_error=max_shear_error)
(OUT/'scene.json').write_text(json.dumps(scene, indent=2))
print(scene)
