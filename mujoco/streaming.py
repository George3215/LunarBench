"""Asynchronous 10 m collision windows; immutable world coordinates and robot state."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv,json,time,copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from model import scene_tree

ROOT=Path(__file__).resolve().parent

class CollisionWindows:
    def __init__(self,meta,scene_path=None):
        source=ROOT.parent/'ue/import_data'
        self.origin=np.array(meta['origin_ue_m'])
        self.base=np.array(json.loads((source/'v2/scene.json').read_text())['location'])/100
        self.heights=(np.fromfile(source/'Landscape_1_2795x2795.r16',dtype='<u2').reshape(2795,2795).astype(float)-32768)/128*.25
        self.rows=list(csv.DictReader((source/'v2/rock_positions_ue.csv').open()))
        self.positions=np.array([[float(r[k])/100 for k in ('x_cm','y_cm','z_cm')] for r in self.rows])
        self.prototypes={k:np.array(json.loads((source/f'v2/rock{k}.json').read_text())['points'])/100 for k in (1,2)}
        self.faces={k:json.loads((source/f'v2/rock{k}.json').read_text())['indices'] for k in (1,2)}
        # Conservative bounding sphere includes every intersecting rotated/scaled rock.
        self.radii=np.array([np.linalg.norm(self.prototypes[int(r['mesh'])]*[float(r[k]) for k in ('sx','sy','sz')],axis=1).max() for r in self.rows])
        self.template=scene_tree(scene_path or ROOT/'generated/scene.xml').getroot()
        for parent in (self.template.find('worldbody'),self.template.find('asset')):
            for child in list(parent):
                if child.get('name','').startswith('rock_'):parent.remove(child)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='collision-window')
        self.pending=None;self.center=np.zeros(2);self.swaps=0;self.last_build_ms=0.
        lo=self.base[:2]-self.origin[:2];hi=lo+1397
        self.bounds=np.array([[lo[0]+2,-hi[1]+2],[hi[0]-2,-lo[1]-2]])

    def build(self,xy):
        started=time.perf_counter()
        ij=np.rint((self.origin[:2]+np.array(xy)*[1,-1]-self.base[:2])/.5).astype(int)
        ij=np.clip(ij,10,2784);ix,iy=ij
        center=(self.base[:2]+ij*.5-self.origin[:2])*[1,-1]
        crop=self.heights[iy-10:iy+11,ix-10:ix+11][::-1]+self.base[2]-self.origin[2]
        zmin=float(crop.min());span=max(float(np.ptp(crop)),.001)
        xml=copy.deepcopy(self.template)
        hf=xml.find('asset/hfield');hf.attrib.pop('file',None)
        hf.attrib.update(nrow='21',ncol='21',size=f'5 5 {span} 1')
        xml.find("worldbody/geom[@name='terrain']").set('pos',f'{center[0]} {center[1]} {zmin}')
        world_center=self.origin[:2]+center*[1,-1]
        candidates=np.flatnonzero(np.max(np.abs(self.positions[:,:2]-world_center),axis=1)<=5+self.radii)
        selected=[]
        for i in candidates:
            r=self.rows[i];mat=np.empty(9)
            mujoco.mju_quat2Mat(mat,np.array([float(r[k]) for k in ('qw','qx','qy','qz')]))
            v=(self.prototypes[int(r['mesh'])]*[float(r[k]) for k in ('sx','sy','sz')])@mat.reshape(3,3).T+self.positions[i]-self.origin
            v[:,1]*=-1
            if np.any(v[:,:2].min(0)>center+5) or np.any(v[:,:2].max(0)<center-5):continue
            name=f'rock_{i:06d}'
            ET.SubElement(xml.find('asset'),'mesh',name=name,vertex=' '.join(f'{x:.9g}' for x in v.ravel()),face=' '.join(map(str,self.faces[int(r['mesh'])])))
            ET.SubElement(xml.find('worldbody'),'geom',name=name,type='mesh',mesh=name,friction='.8 .02 .001',group='0')
            selected.append(int(i))
        model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
        model.hfield_data[:]=((crop-zmin)/span).ravel()
        return model,center,selected,(time.perf_counter()-started)*1000

    def update(self,sim):
        if self.pending is not None and self.pending.done():
            model,center,rocks,ms=self.pending.result();self.pending=None
            old,data=sim.model,sim.data
            spec=mujoco.mjtState.mjSTATE_INTEGRATION
            assert mujoco.mj_stateSize(old,spec)==mujoco.mj_stateSize(model,spec)
            state=np.empty(mujoco.mj_stateSize(old,spec));mujoco.mj_getState(old,data,state,spec)
            model.opt.gravity[:]=old.opt.gravity
            new=mujoco.MjData(model);mujoco.mj_setState(model,new,state,spec);mujoco.mj_forward(model,new)
            sim.model,sim.data=model,new
            sim.visual=[i for i in range(model.ngeom) if model.geom_group[i]==2 and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
            sim.meta['rocks']=rocks;self.center=center;self.swaps+=1;self.last_build_ms=ms
            print(f'COLLISION_WINDOW swap={self.swaps} center={center.tolist()} rocks={len(rocks)} build_ms={ms:.1f}',flush=True)
        delta=sim.data.qpos[:2]-self.center
        if self.pending is None and np.max(np.abs(delta))>=1.5:
            target=np.round(sim.data.qpos[:2]/.5)*.5
            ij=np.clip(np.rint((self.origin[:2]+target*[1,-1]-self.base[:2])/.5),10,2784)
            target=(self.base[:2]+ij*.5-self.origin[:2])*[1,-1]
            if not np.array_equal(target,self.center):self.pending=self.pool.submit(self.build,target)
        # Freeze simulation time if preparation is late, never step off collision coverage.
        return np.max(np.abs(delta))<3.5

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
