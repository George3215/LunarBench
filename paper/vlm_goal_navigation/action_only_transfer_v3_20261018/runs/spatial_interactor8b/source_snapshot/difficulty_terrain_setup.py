"""Versioned difficulty recipes: native relief first, seeded cubic blocks second.
Recipes are materialised only after a complete-suite difficulty decision.
"""
from dataclasses import replace
import copy,hashlib,math
import numpy as np
from all_terrain_setup import originals,place
from scenes import Scene
from slabs import Slab,SlabTerrain


def catalogue(config):
    source=originals();level=config['level'];out={}
    for name,(scene,terrain) in source.items():
        if isinstance(terrain,SlabTerrain):terrain.spec=copy.deepcopy(terrain.spec)
        else:terrain=replace(terrain,spec=copy.deepcopy(terrain.spec))
        terrain.spec.pop('navigation_extra_geom_xml',None);terrain.spec.pop('navigation_added_occluder',None)
        changes=[]
        if name=='flat':
            # Only the originally flat scene receives a mild valley/crest profile;
            # other terrain families retain their native terrain rather than sharing it.
            start_height={'intro':.24,'easy':.28,'medium':.30,'hard':.34,'very_hard':.38}[level]
            valley={'intro':-.32,'easy':-.34,'medium':-.38,'hard':-.42,'very_hard':-.45}[level]
            crest={'intro':.28,'easy':.30,'medium':.34,'hard':.38,'very_hard':.42}[level]
            xs=[-6.,-3.4,-2.8,-1.6,-.7,.5,.9,1.3,6.]
            zs=[start_height,start_height,start_height,valley,valley,crest,crest,0.,0.]
            # Fine treads retain the low-relief profile while matching the
            # existing locomotion controller's supported step representation.
            edges=np.linspace(-6.,6.,81);tops=np.round(np.interp((edges[:-1]+edges[1:])/2,xs,zs)/.04)*.04
            slabs=[Slab(float(edges[i]),float(edges[i+1]),-3.,3.,float(tops[i]),float(tops[i]),1.5,f'flat_relief_{i}') for i in range(len(tops))]
            spec=copy.deepcopy(terrain.spec);spec.update({'navigation_flat_relief':{'x_m':xs,'z_m':zs,'max_absolute_height_m':max(map(abs,zs))},'navigation_original_geometry_unchanged':False})
            terrain=SlabTerrain('flat',slabs,spec)
            changes.append({'kind':'flat_to_low_relief','original_height_m':0.,'profile_x_m':xs,'profile_z_m':zs,'only_this_family_modified':True,
                'collider':'0.15m treads, 0.04m height quantisation','max_adjacent_rise_m':float(np.max(np.abs(np.diff(tops))))})
        elif name=='barrier':
            # Replace the original thin finite wall with an exposed cube.
            old=[s for s in terrain.slabs if s.name=='barrier_wall'];keep=[s for s in terrain.slabs if s.name!='barrier_wall']
            for s in old:keep.append(Slab(s.x0,s.x1,s.y0,s.y1,0.,0.,1.5,'barrier_removed_wall_floor'))
            spec=copy.deepcopy(terrain.spec);spec['navigation_original_geometry_unchanged']=False
            terrain=SlabTerrain('barrier',keep,spec)
            changes.append({'kind':'remove_original_nonwindow_wall','removed_slabs':[vars(s) for s in old],'replacement':'seeded cube candidate'})
        elif level in ('hard','very_hard') and scene.terrain_profile!='window_wall':
            factor=1.15 if level=='hard' else 1.3
            if isinstance(terrain,SlabTerrain):
                spec=copy.deepcopy(terrain.spec)
                terrain=SlabTerrain(terrain.name,[replace(s,z0=s.z0*factor,z1=s.z1*factor) for s in terrain.slabs],spec)
            else:
                spec=copy.deepcopy(terrain.spec)
                if 'step_rise_m' in spec:spec['step_rise_m']*=factor
                terrain=replace(terrain,heights=terrain.heights*factor,spec=spec)
            terrain.spec['navigation_original_geometry_unchanged']=False
            changes.append({'kind':'scale_original_relief','height_factor':factor,'original_family_retained':True})
        terrain.spec['navigation_design_changes']=changes
        terrain.spec['navigation_difficulty_level']=level
        terrain.spec['navigation_recipe_seed']=config.get('seed',23)
        terrain.spec.setdefault('navigation_original_geometry_unchanged',True)
        out[name]=(scene,terrain)
    return out


def geometric_visible(terrain,start,target,height):
    # Geometric prefilter only. Full RGB/seg/depth/Go2 physics remains mandatory.
    sx,sy=start;tx,ty=target
    def sampled(x,y):
        ix=np.clip(np.searchsorted(terrain.x,x),0,len(terrain.x)-1)
        iy=np.clip(np.searchsorted(terrain.y,y),0,len(terrain.y)-1)
        return terrain.heights[iy,ix]
    eye=np.array([sx+.31,sy,float(sampled(sx,sy))+.52])
    centre=np.array([tx,ty,float(sampled(tx,ty))+height-.15])
    dx=centre[0]-eye[0];dy=centre[1]-eye[1]
    if dx<=.3 or abs(dy/dx)>1.:return False
    points=[centre,centre+[0,0,.15],centre+[0,0,-.15],centre+[0,.15,0],centre+[0,-.15,0]]
    visible=[]
    for point in points:
        ray=eye+(point-eye)*np.linspace(.02,.97,40)[:,None]
        ground=np.asarray(sampled(ray[:,0],ray[:,1]))
        visible.append(not bool(np.any(ground>=ray[:,2])))
    return sum(visible)


def candidates(name,source,terrain,config):
    level=config['level'];max_native=config.get('max_native_candidates',24)
    if name=='flat':
        # Actual motion descends into the shallow valley; the low crest conceals
        # the target while it remains geometrically inside camera FOV.
        for sx in (-3.2,-3.0,-2.8):
            for tx in (2.3,2.6,2.0):
                yield ((sx,0.),(tx,0.),.30,0.,'FORWARD')
    elif source.terrain_profile!='window_wall' and name!='barrier':
        ranked=[]
        near={'intro':1.8,'easy':2.4,'medium':3.3,'hard':4.2,'very_hard':4.8}[level]
        heights=(.30,.38,.45,.60,.75) if level in ('intro','easy') else (.30,.4,.55,.65,.75) if level=='medium' else (.25,.35,.45,.60,.75)
        for y in (0.,-.8,.8):
            for sx in np.arange(-4.5,3.2,.5):
                # Some original stair crests only yield a native seen/lost
                # transition over a longer approach. Keep these explicit harder
                # candidates before adding blocks rather than avoiding the terrain.
                for distance in (near,near+.8,near+1.6,7.,8.5):
                    tx=sx+distance
                    if tx>float(terrain.x[-1])-.7:continue
                    for height in heights:
                        start=(float(sx),y);target=(float(tx),y)
                        if geometric_visible(terrain,start,target,height)<3:continue
                        loss=None
                        for offset in np.arange(.4,min(distance-1.,3.)+.01,.2):
                            if geometric_visible(terrain,(sx+offset,y),target,height)==0:loss=float(offset);break
                        if loss is not None:ranked.append((distance,loss,start,target,height))
        ranked.sort(key=lambda r:(r[0],r[1]))
        for distance,loss,start,target,height in ranked[:max_native]:yield (start,target,height,0.,'FORWARD')
    # Native relief candidates are always tried before a block. Blocks are
    # equilateral exposed cubes, never narrow randomly placed wall segments.
    digest=int(hashlib.sha256((name+str(config.get('seed',23))).encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(digest)
    edge={'intro':.60,'easy':.65,'medium':.75,'hard':.95,'very_hard':1.05}[level]
    distance={'intro':2.2,'easy':2.6,'medium':3.5,'hard':4.5,'very_hard':5.2}[level]
    if source.terrain_profile=='window_wall':
        wx=terrain.spec['wall_x_m'];locations=[(wx-distance-1.1,wx-1.1)]
    else:
        locations=[(-3.8,-3.8+distance),(-4.7,-4.7+distance),(-1.6,-1.6+distance),(0.,distance)]
    for sx,tx in locations:
        if sx<float(terrain.x[0])+.5 or tx>float(terrain.x[-1])-.6:continue
        for height in (.55,.7):
            wx=sx+.32*(tx-sx)+float(rng.uniform(-.10,.10))
            yield ((sx,-1.35),(tx,0.),height,0.,'STRAFE_LEFT',{'kind':'cube','x':wx,'edge':edge,'seed':digest})


def adjusted(source,terrain,candidate,config):
    start,target,height,yaw,action=candidate[:5];extra=candidate[5] if len(candidate)>5 else None
    terrain.spec.pop('navigation_extra_geom_xml',None);terrain.spec.pop('navigation_added_occluder',None)
    if extra:
        wx=extra['x'];edge=extra['edge'];half=edge/2
        base=max(float(terrain.height(x,y)) for x in (wx,wx+edge) for y in (-half,0.,half))-.02
        top=base+edge
        block=Slab(wx,wx+edge,-half,half,top,top,edge,'navigation_added_occluder')
        terrain.spec['navigation_extra_geom_xml']=block.geom_xml()
        terrain.spec['navigation_added_occluder']={'kind':'cube','x':wx,'width':edge,'half_y':half,'top_z':top,'bottom_z':base,
            'exposed_dimensions_m':[edge,edge,edge],'seed':extra['seed'],'ablation_id':f'cube_{extra["seed"]}',
            'generation':'one seeded cube, finite fixed candidate list; no walls outside window structures'}
    changes=terrain.spec['navigation_design_changes']
    original_notes=(source.notes,) if isinstance(source.notes,str) else tuple(source.notes)
    scene=replace(source,target_xy=tuple(target),target_height_m=height,target_z_absolute=float(terrain.height(*target))+height,
        approach_xy=tuple(start),lost_xy=tuple(start),yaw_rad=yaw,history_xy=None,history_saw_target=False,
        notes=original_notes+(f'Calibrated {config["level"]}; native relief preferred; cubic fallback; versioned terrain changes.',))
    return scene,action
