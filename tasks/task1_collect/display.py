"""Task-owned UE display descriptor. Dynamic stones use body poses, never mesh COM poses."""
import json
import zipfile
from pathlib import Path
import mujoco
import numpy as np
try:
    from .scene import _prototype
except ImportError:
    from scene import _prototype

ROOT = Path(__file__).resolve().parents[2]

# Anonymous meshes take their name from the file stem, so these identify Go2 meshes in any model
# that embeds the Go2 body. They must occupy mesh ids 0-15: UE addresses them as SM_Go2_<id>.
# tools/prepare_go2_piper.py asserts the order for go2_piper.xml.
GO2_MESH_NAMES=frozenset({'base_0','base_1','base_2','base_3','base_4','hip_0','hip_1','thigh_0','thigh_1',
    'thigh_mirror_0','thigh_mirror_1','calf_0','calf_1','calf_mirror_0','calf_mirror_1','foot'})

def _texture_file(out_dir,kind,texture,extracted):
    """石头贴图在磁盘上的可读路径。

    lunar 原型的贴图封在 usdz 里，先解出来放 generated/；apollo 的本来就是普通 jpg，直接用。
    100 块原型一共 ~190 MB 贴图，而一次场景只用其中几十块，所以这里按需取、取过就缓存，
    不把贴图复制进 assets/。返回 None 表示这块原型没有配准过的贴图。
    """
    if not texture:return None
    source=ROOT/texture['path']
    member=texture.get('member')
    if member is None:return str(source)
    if kind in extracted:return extracted[kind]
    target=out_dir/f'rock_texture_{kind}{Path(member).suffix}'
    if not target.is_file():
        with zipfile.ZipFile(source) as archive:target.write_bytes(archive.read(member))
    extracted[kind]=str(target.resolve())
    return extracted[kind]


def export(task,out_dir):
    m=task.model;geoms=[];out_dir=Path(out_dir)
    def mesh_file(name,vertices,faces,uvs=None):
        path=out_dir/(name+'.json')
        payload={'points':(vertices*[100,-100,100]).tolist(),'indices':np.asarray(faces).ravel().tolist()}
        # UV 与 indices 平行（每个面角一个），UE 侧直接按顶点实例写进去。
        if uvs is not None:payload['uvs']=np.asarray(uvs,dtype=float).reshape(-1,2).tolist()
        path.write_text(json.dumps(payload,separators=(',',':')))
        return str(path.resolve())
    # 池子有 100+ 块原型，一次只用到 count 块：只导出真正上场的那几块。
    pool=[int(k) for k in task.config['rock']['meshes']]
    rocks={};extracted={}
    for kind in sorted({pool[i%len(pool)] for i in range(len(task.meta['rocks']))}):
        data=_prototype(kind)
        rocks[kind]=(mesh_file(f'solid_rock_{kind}',np.array(data['points'],dtype=float),data['indices'],data.get('uvs')),
                     _texture_file(out_dir,kind,data.get('texture'),extracted))
    primitives={int(mujoco.mjtGeom.mjGEOM_BOX):'Cube',int(mujoco.mjtGeom.mjGEOM_SPHERE):'Sphere',int(mujoco.mjtGeom.mjGEOM_CYLINDER):'Cylinder'}
    for i in task.visual:
        name=m.geom(i).name;typ=int(m.geom_type[i]);size=m.geom_size[i];file=None;texture=None;role='robot'
        if name.startswith('sample_'):
            index=int(name.split('_')[1]);kind=pool[index%len(pool)]
            asset=None;file,texture=rocks[kind];role='rock'
            # 逐块缩放：UE 画的是未缩放的原型网格 + actor scale，而 MJCF 里已经把
            # rock_scales[index] 乘进顶点。两边必须用同一个值，否则 UE 显示和 MuJoCo
            # 物理里的石头尺寸对不上。
            scale=[float(task.meta['rock_scales'][index])]*3
        elif typ==mujoco.mjtGeom.mjGEOM_MESH and (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,int(m.geom_dataid[i])) or '') in GO2_MESH_NAMES:
            asset=f'/Game/MoonGo2/Meshes/SM_Go2_{int(m.geom_dataid[i])}';scale=[1,1,1]
        elif typ==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(m.geom_dataid[i]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);b=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid])
            asset=None;scale=[1,1,1]
            file=mesh_file(f'robot_mesh_{mid}',m.mesh_vert[a:a+n],m.mesh_face[b:b+nf])
        elif typ in primitives:
            primitive=primitives[typ];asset=f'/Engine/BasicShapes/{primitive}.{primitive}'
            scale=(size*2).tolist() if primitive=='Cube' else [2*size[0]]*3 if primitive=='Sphere' else [2*size[0],2*size[0],2*size[1]]
        else:
            raise ValueError(f'UE display asset mapping required for {name}; this external MJCF remains runnable in MuJoCo')
        rgba=m.mat_rgba[m.geom_matid[i]] if m.geom_matid[i]>=0 else m.geom_rgba[i]
        geoms.append({'id':i,'asset':asset,'scale':scale,'rgba':rgba.tolist(),'mesh_file':file,
                      'texture':texture,'role':role,'sensor_housing':name=='d435_visual'})
    result={'task':'task1_collect','origin_ue_m':task.field.origin.tolist(),
            'landscape':task.meta['landscape'],'geoms':geoms,'initial':task.packet()}
    path=Path(out_dir)/'visual.json';path.write_text(json.dumps(result,indent=2));return path
