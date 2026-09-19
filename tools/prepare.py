"""Build a small collision island aligned with the saved UE scene, no USD dependency."""
from pathlib import Path
import csv,json,struct
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]/'mujoco'
DATA=ROOT.parent/'ue/import_data/v2'
OUT=ROOT/'generated'

def build():
    OUT.mkdir(exist_ok=True)
    h=np.fromfile(ROOT.parent/'ue/import_data/Landscape_1_2795x2795.r16',dtype='<u2').reshape(2795,2795)
    heights=(h.astype(float)-32768)/128*25/100
    scene=json.loads((DATA/'scene.json').read_text())
    base=np.array(scene['location'])/100
    rows=list(csv.DictReader((DATA/'rock_positions_ue.csv').open()))
    positions=np.array([[float(r[k])/100 for k in ('x_cm','y_cm','z_cm')] for r in rows])
    # Pick a low-slope, clear spawn near the UE inspection area, never flatten terrain.
    best=None
    for iy in range(1300,1361,2):
        for ix in range(2180,2261,2):
            patch=heights[iy-2:iy+3,ix-2:ix+3]
            xy=base[:2]+np.array([ix,iy])*.5
            dist=np.min(np.linalg.norm(positions[:,:2]-xy,axis=1))
            if dist<1.4: continue
            score=np.ptp(patch)+.0001*np.linalg.norm(np.array([ix-2228,iy-1329]))
            if best is None or score<best[0]:best=(score,ix,iy)
    assert best is not None
    _,ix,iy=best
    origin=base+np.array([ix*.5,iy*.5,heights[iy,ix]])
    crop=heights[iy-10:iy+11,ix-10:ix+11][::-1].copy()+base[2]-origin[2]
    zmin=float(crop.min()); span=max(float(np.ptp(crop)),.001)
    with (OUT/'height.bin').open('wb') as f:
        f.write(struct.pack('<ii',21,21));f.write(((crop-zmin)/span).astype('<f4').tobytes())
    tree=ET.parse(ROOT.parent/'assets/robots/go2_demo/go2/go2.xml');xml=tree.getroot()
    xml.find('compiler').set('meshdir','../../assets/robots/go2_demo/go2/assets')
    xml.find('option').attrib.update(timestep='0.002',gravity='0 0 -9.81',iterations='30')
    for key in list(xml.findall('keyframe')):xml.remove(key)
    world=xml.find('worldbody');asset=xml.find('asset')
    ET.SubElement(asset,'hfield',name='local_terrain',file='height.bin',size=f'5 5 {span} 1')
    ET.SubElement(world,'geom',name='terrain',type='hfield',hfield='local_terrain',pos=f'0 0 {zmin}',rgba='.42 .42 .42 1',friction='.8 .02 .001',condim='3')
    ET.SubElement(world,'light',pos='0 0 10',dir='-.3 -.5 -1',diffuse='.8 .8 .8')
    prototypes={k:np.array(json.loads((DATA/f'rock{k}.json').read_text())['points'])/100 for k in (1,2)}
    faces={k:json.loads((DATA/f'rock{k}.json').read_text())['indices'] for k in (1,2)}
    selected=[]
    # Include intersecting rock geometry, not merely centres within the square.
    for i in np.flatnonzero(np.max(np.abs(positions[:,:2]-origin[:2]),axis=1)<8):
        r=rows[i]; q=np.array([float(r[k]) for k in ('qw','qx','qy','qz')]);mat=np.empty(9)
        mujoco.mju_quat2Mat(mat,q)
        scale=np.array([float(r[k]) for k in ('sx','sy','sz')])
        v=(prototypes[int(r['mesh'])]*scale)@mat.reshape(3,3).T+positions[i]-origin
        if np.any(v[:,:2].min(axis=0)>5) or np.any(v[:,:2].max(axis=0)<-5):continue
        v[:,1]*=-1  # UE left-handed -> MuJoCo right-handed.
        # MuJoCo's convex mesh collision is explicit; vertices are not decimated.
        name=f'rock_{i:06d}'
        ET.SubElement(asset,'mesh',name=name,vertex=' '.join(f'{x:.9g}' for x in v.ravel()),face=' '.join(map(str,faces[int(r['mesh'])])))
        ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,rgba='.3 .3 .3 1',friction='.8 .02 .001',group='0')
        selected.append(int(i))
    ET.indent(xml)
    tree.write(OUT/'scene.xml',encoding='unicode')
    meta={'origin_ue_m':origin.tolist(),'grid_index':[ix,iy],'width_m':10,'shape':[21,21],'spacing_m':.5,'rocks':selected,'gravity':-9.81,'source':'saved UE V2 terrain and rock transforms','rock_collision':'convex hull of full transformed prototype vertices','terrain_z_range_local_m':[zmin,zmin+span]}
    (OUT/'scene.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    return meta

if __name__=='__main__':build()
