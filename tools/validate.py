import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"mujoco"))
"""Headless geometric and collision checks; deliberately separate from walking tests."""
import json,struct,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from run import Simulation,ROOT
from model import scene_tree

s=Simulation();m,d=s.model,s.data
blob=(ROOT/'generated/height.bin').read_bytes()
heights=np.frombuffer(blob[8:],dtype='<f4').reshape(21,21)*m.hfield_size[0,2]+m.geom_pos[m.geom('terrain').id,2]
group=np.array([1,0,0,0,0,0],dtype=np.uint8)
# Terrain-only mask for rays, then restore every rock collision group.
rockids=[m.geom(f'rock_{i:06d}').id for i in s.meta['rocks']]
for i in rockids:m.geom_group[i]=1
errors=[]
for r in range(21):
    for c in range(21):
        gid=np.array([-1],dtype=np.int32)
        dist=mujoco.mj_ray(m,d,np.array([-5+c*.5,-5+r*.5,10.]),np.array([0.,0.,-1.]),group,True,-1,gid)
        assert gid[0]==m.geom('terrain').id,(r,c,dist,gid)
        errors.append(abs(10-dist-heights[r,c]))
assert max(errors)<1e-6,max(errors)
for i in rockids:m.geom_group[i]=0
collision_results=[]
for gid in rockids:
    # Drop a physical sphere on the top of each actual convex rock geom.
    mid=m.geom_dataid[gid];a=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid]
    v=m.mesh_vert[a:a+n]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
    top=v[np.argmax(v[:,2])]
    xml=scene_tree(ROOT/'generated/scene.xml').getroot()
    for tag in ('actuator','sensor','contact','keyframe'):
        for e in list(xml.findall(tag)):xml.remove(e)
    world=xml.find('worldbody');world.remove(world.find('body'))
    body=ET.SubElement(world,'body',name='probe',pos=' '.join(map(str,top+[0,0,.3])))
    ET.SubElement(body,'freejoint');ET.SubElement(body,'geom',name='probe',type='sphere',size='.05',mass='.1',friction='.8 .02 .001')
    probe=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));pd=mujoco.MjData(probe)
    rockname=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gid)
    expected=probe.geom(rockname).id;ball=probe.geom('probe').id;hit=False
    for step in range(400):
        mujoco.mj_step(probe,pd)
        hit |= any(set(c.geom)=={expected,ball} for c in pd.contact)
    assert hit,rockname
    collision_results.append({'rock':rockname,'sphere_contact':hit})
out={'terrain_grid_rays':441,'max_height_error_m':max(errors),'rock_drop_tests':collision_results,'all_pass':True}
(ROOT/'generated/collision_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
