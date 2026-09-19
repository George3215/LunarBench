"""Native-resolution terrain. One source R16 sample is one centimetre at scale 1.

All task coordinates are metres, Z up. UE reflects Y and converts metres to cm.
The Landscape actor is placed at terrain_origin_m and uses terrain_scale, so the
physics crop and the rendered full terrain share exactly the same transform.
"""
from pathlib import Path
import struct
from functools import lru_cache
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATCH_CENTERS = {'center': [13.97, 13.97], 'southwest': [10., 10.], 'northeast': [17.9, 17.9]}

HEIGHTMAP = ROOT / 'ue/import_data/Landscape_1_2795x2795.r16'

@lru_cache(maxsize=1)
def _source():
    return (np.fromfile(HEIGHTMAP, dtype='<u2').reshape(2795, 2795).astype(np.float64)-32768)/128/100

class Field:
    def __init__(self, spec, out_dir):
        self.scale=np.asarray(spec['terrain_scale'],dtype=float)
        self.base=np.asarray(spec['terrain_origin_m'],dtype=float)
        self.spacing=.01*self.scale[0]
        self.size_m=float(spec['size_m']);self.half=self.size_m/2
        intervals=round(self.size_m/self.spacing)
        if intervals%2 or abs(intervals*self.spacing-self.size_m)>1e-8:
            raise ValueError('field.size_m must contain an even number of source intervals')
        self.samples=intervals+1;self.edge=intervals//2
        heights=_source()*self.scale[2]
        center=spec.get('center_m')
        patch=spec.get('patch', 'center')
        if patch not in PATCH_CENTERS:raise ValueError(f'Unknown terrain patch: {patch}')
        if center is None:center=PATCH_CENTERS[patch]
        self.grid=tuple(np.rint((np.asarray(center)-self.base[:2])/self.spacing).astype(int))
        ix,iy=self.grid
        if not all(self.edge<=v<=2794-self.edge for v in self.grid):
            raise ValueError('Task field extends beyond the original terrain; check center_m and size_m')
        self.center_height=float(heights[iy,ix])
        self.origin=np.array([self.base[0]+ix*self.spacing,self.base[1]+iy*self.spacing,self.base[2]+self.center_height])
        self.crop=heights[iy-self.edge:iy+self.edge+1,ix-self.edge:ix+self.edge+1][::-1]-self.center_height
        self.zmin=float(self.crop.min());self.span=max(float(np.ptp(self.crop)),.001)
        self.normalized=((self.crop-self.zmin)/self.span).astype('<f4')
        out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
        with (out/'height.bin').open('wb') as f:
            f.write(struct.pack('<ii',self.samples,self.samples));f.write(self.normalized.tobytes())

    @property
    def relief_m(self):return float(np.ptp(self.crop))

    def height_local(self,x,y):
        # Match MuJoCo's two planar triangles per cell, not bilinear interpolation.
        row=np.clip(self.edge+np.asarray(y)/self.spacing,0,self.samples-1)
        col=np.clip(self.edge+np.asarray(x)/self.spacing,0,self.samples-1)
        r=np.minimum(np.floor(row).astype(int),self.samples-2)
        c=np.minimum(np.floor(col).astype(int),self.samples-2)
        fy=row-r;fx=col-c
        a=self.crop[r,c];b=self.crop[r,c+1];d=self.crop[r+1,c];e=self.crop[r+1,c+1]
        return np.where(fy>=fx,a+(d-a)*fy+(e-d)*fx,a+(b-a)*fx+(e-b)*fy)

    def to_world(self,x,y,z=0.):
        return np.array([self.origin[0]+np.asarray(x),self.origin[1]-np.asarray(y),self.origin[2]+np.asarray(z)])
