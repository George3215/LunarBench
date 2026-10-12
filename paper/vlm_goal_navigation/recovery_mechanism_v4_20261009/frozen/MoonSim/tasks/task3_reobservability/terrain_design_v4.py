"""Mechanism-pure paired geometry ablations; no helper occluders on relief."""
import numpy as np
from slabs import Slab, SlabTerrain, grid
from terrain_design_v3 import build_terrain_v3 as build_base, scene_for_config

FAMILIES=('barrier','trench','stairs_up','stairs_down','slope_up','slope_down','pit','basin','waves','perlin','perlin_rough','window')

def designs():
    result=[]
    for kind in FAMILIES:
        for level in range(3):
            # medium changes one primary factor; complex adds one secondary factor.
            c=dict(design_version=4,name=f'{kind}_a{level}',kind=kind,seed=23,
                   provisional_difficulty=['simple','medium','complex'][level],
                   ablation_group=kind,ablation_level=level,
                   primary_strength=1. if level==0 else 1.5,
                   secondary_relief_m=.16 if level==2 and kind not in ('barrier','window') else 0.,
                   helper_occluder_allowed=False)
            s=c['primary_strength']
            if kind in ('perlin','perlin_rough'):c.update(feature_m=4.2,relief_m=(1.4 if kind=='perlin' else 2.03)*s,octaves=3,lateral_blend=1.)
            elif kind in ('pit','basin'):c.update(radius_m=2.7 if kind=='pit' else 4.5,depth_m=(1.1 if kind=='pit' else 1.4)*s)
            elif kind=='trench':c.update(width_m=1.8,depth_m=.8*s)
            elif kind in ('stairs_up','stairs_down','slope_up','slope_down'):c.update(rise_m=.10*s,tread_m=.45,steps=9)
            elif kind=='waves':c.update(amplitude_m=.45*s,wavelength_m=4.8)
            elif kind=='barrier':c.update(edge_m=1.0*s,secondary_cube=level==2)
            elif kind=='window':c.update(opening_width_m=1.5 if level==0 else 1.,sill_m=.18,opening_height_m=1.05,center_y_m=.5 if level==2 else 0.,frame_width_m=3.8,wall_thickness_m=.35)
            c['factor_delta']='baseline' if level==0 else 'primary strength only' if level==1 else 'add secondary relief / second cube / window offset only'
            result.append(c)
    return result

def build_terrain_v3(c):
    kind=c['kind'];name=c['name']
    if kind in ('stairs_up','stairs_down','slope_up','slope_down'):
        n=c['steps'];rise=c['rise_m'];tread=c['tread_m'];start=-2.;end=start+n*tread;height=n*rise
        down=kind.endswith('down')
        def z(x):
            q=np.clip((np.asarray(x)-start)/(end-start),0,1)
            if kind.startswith('stairs'):q=np.ceil(q*n)/n
            return height*(1-q if down else q)
        edges=np.unique(np.r_[[-8.,start],start+np.arange(n+1)*tread,[8.]])
        slabs=[]
        for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            za=zb=float(z((a+b)/2)) if kind.startswith('stairs') else None
            if za is None:za,zb=float(z(a)),float(z(b))
            slabs.append(Slab(float(a),float(b),-6,6,za,zb,2.,name+f'_surface_{i}'))
        t=SlabTerrain(name,slabs,{'profile':name,'size_m':[16,12],'design':c,'navigation_extra_geom_xml':''})
    else:t=build_base(c)
    if c['secondary_relief_m']:
        original=t;amplitude=c['secondary_relief_m']
        def profile(x,y):
            xx,yy=np.broadcast_arrays(x,y)
            base=np.array([original.height(float(a),float(b)) for a,b in zip(xx.ravel(),yy.ravel())]).reshape(xx.shape)
            return base+amplitude*np.sin(2*np.pi*xx/3.2)
        t=grid(name,profile,16,12,cell_m=.25,quantum=.025,max_rise_m=.25)
        t.spec['navigation_extra_geom_xml']=''
    if kind=='barrier' and c['secondary_cube']:
        from terrain_design_v3 import box
        t.spec['navigation_extra_geom_xml']+=box(name+'_second_cube',2.,1.3,.5,1.,1.,1.)
    t.spec.update(design_version=4,design=c,mechanism_pure=kind not in ('barrier','window'),helper_occluder_added=False,
                  navigation_qualification='pending strict G1-G4; not VLM success')
    return t
