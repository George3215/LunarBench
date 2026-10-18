"""Versioned, diverse terrain designs. Old experimental worlds stay frozen.

New designs are inspected/physics-checked here; they are not automatically
admitted into the VLM navigation suite without its visibility/reachability gate.
"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from slabs import Slab, SlabTerrain, grid, fbm2d
from scenes import Scene

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/vlm_goal_navigation/terrain_design_v3'


def designs():
    configs = []
    def add(kind, seed, **kw):
        configs.append(dict(design_version=3, name=f'{kind}_s{seed}', kind=kind, seed=seed, **kw))
    for seed, scale, relief in [(23,4.2,1.6),(41,3.2,1.9),(67,5.0,2.2)]:
        add('perlin',seed,feature_m=scale,relief_m=relief,octaves=3,lateral_blend=1.)
        add('perlin_rough',seed,feature_m=scale*.65,relief_m=relief*1.45,octaves=4,lateral_blend=1.)
    for seed, radius, depth in [(23,2.,1.2),(41,2.7,1.7),(67,3.3,2.1)]:
        add('pit',seed,radius_m=radius,depth_m=depth)
        add('basin',seed,radius_m=radius+1.8,depth_m=depth+0.3)
    for seed, width, sill, cy in [(23,1.1,.18,0.),(41,.7,.28,-.5),(67,1.5,.12,.6),(89,.9,.32,0.)]:
        add('window',seed,opening_width_m=width,sill_m=sill,opening_height_m=1.05,center_y_m=cy,frame_width_m=3.8,wall_thickness_m=.35)
    for seed, count, clearance in [(23,12,1.6),(41,18,1.3),(67,24,1.1)]:
        add('block_maze',seed,block_count=count,corridor_width_m=clearance,edge_range_m=[.6,1.7])
    for seed, count, friction in [(23,48,.65),(41,72,.9),(67,96,1.1)]:
        add('movable_rubble',seed,rock_count=count,radius_range_m=[.045,.15],density_kg_m3=1800.,friction=[friction,.01,.002])
    for seed, amplitude, wavelength in [(23,.45,7.),(41,.7,8.),(67,.95,10.)]:
        add('smooth_slope',seed,amplitude_m=amplitude,wavelength_m=wavelength)
    for seed, depth, width in [(23,.8,1.3),(41,1.2,1.8),(67,1.6,2.4)]:
        add('gullies',seed,depth_m=depth,width_m=width,branches=3)
        add('trench',seed,depth_m=depth,width_m=width)
    for seed, edge in [(23,1.2),(41,1.7)]:add('barrier',seed,edge_m=edge)
    for seed, rise in [(23,.12),(41,.17)]:add('stairs_up',seed,rise_m=rise,tread_m=.45,steps=9)
    for seed, amp, wave in [(23,.45,4.),(41,.65,4.8),(67,.8,5.6)]:add('waves',seed,amplitude_m=amp,wavelength_m=wave)
    return configs


def flat(name):
    return SlabTerrain(name,[Slab(-8,8,-6,6,0,0,1.,f'{name}_floor')],{'profile':name,'size_m':[16.,12.]})


def box(name,x,y,z,dx,dy,dz,color='.42 .43 .46 1'):
    return f'<geom name="{name}" type="box" pos="{x} {y} {z}" size="{dx/2} {dy/2} {dz/2}" friction=".9 .01 .001" rgba="{color}"/>\n'


def build_terrain_v3(c):
    name,kind,seed=c['name'],c['kind'],c['seed'];rng=np.random.default_rng(seed)
    extras='';assets='';bodies='';objects=[]
    if kind in ('perlin','perlin_rough'):
        x=np.linspace(-8,8,81);y=np.linspace(-6,6,61);xx,yy=np.meshgrid(x,y)
        raw=fbm2d(xx/c['feature_m'],yy/c['feature_m'],c['octaves'],seed=seed)
        lo,hi=float(raw.min()),float(raw.max())
        def profile(x,y):
            z=fbm2d(x/c['feature_m'],y/c['feature_m'],c['octaves'],seed=seed)
            # Full 2D correlation; macro relief retained rather than flattening
            # every seed towards the same y=0 trace. Record realised relief.
            return (z-(lo+hi)/2)/(hi-lo)*c['relief_m']
        t=grid(name,profile,16,12,cell_m=.25,quantum=.025,max_rise_m=.20 if kind=='perlin' else .25)
    elif kind in ('pit','basin'):
        # Closed bowl, rather than an x-only trench. Multiple shapes and rims.
        radius,depth=c['radius_m'],c['depth_m'];cy=float(rng.uniform(-.45,.45));ellipticity=float(rng.uniform(.8,1.15))
        def profile(x,y):
            r=np.sqrt((x-.5)**2+((y-cy)*ellipticity)**2);q=np.clip(r/radius,0,1)
            bowl=-depth*.5*(1+np.cos(np.pi*q))
            rim=.18*np.exp(-((r-radius)/.35)**2) if kind=='pit' else 0.
            if kind=='pit' and seed==67:
                bowl-=.35*np.exp(-((x+3.8)**2+(y-2.8)**2)/1.2)
            return bowl+rim
        t=grid(name,profile,16,12,cell_m=.25,quantum=.025,max_rise_m=.25)
        c={**c,'center_xy_m':[.5,cy],'ellipticity':ellipticity,'secondary_pit':kind=='pit' and seed==67}
    elif kind in ('smooth_slope','waves'):
        # Connected piecewise-linear ramps, no quantised stair risers.
        edges=np.linspace(-8,8,161);amp=c['amplitude_m'];w=c['wavelength_m']
        z=amp*np.sin(2*np.pi*edges/w+seed*.1)
        slabs=[Slab(float(a),float(b),-6,6,float(z[i]),float(z[i+1]),2.5,f'{name}_ramp_{i}') for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))]
        t=SlabTerrain(name,slabs,{'profile':name,'size_m':[16,12],'ramp_segments':160,'max_slope_degrees':float(np.degrees(np.arctan(np.max(np.abs(np.diff(z)/np.diff(edges))))))})
    elif kind in ('trench','gullies'):
        def profile(x,y):
            if kind=='trench':r=np.abs(x-.5)
            else:
                a=np.abs(y-.8*np.sin(x*.55))
                b=np.abs((y-.6*x-1.7)/math.sqrt(1.36))
                d=np.abs((y+.65*x+1.7)/math.sqrt(1.4225))
                r=np.minimum(np.minimum(a,b),d)
            q=np.clip(r/c['width_m'],0,1)
            return -c['depth_m']*.5*(1+np.cos(np.pi*q))
        t=grid(name,profile,16,12,cell_m=.25,quantum=.025,max_rise_m=.25)
    elif kind=='stairs_up':
        rise,tread,n=c['rise_m'],c['tread_m'],c['steps'];slabs=[Slab(-8,-2,-6,6,0,0,2.,name+'_apron')]
        for i in range(n):slabs.append(Slab(-2+i*tread,-2+(i+1)*tread,-6,6,(i+1)*rise,(i+1)*rise,2.,name+f'_step_{i}'))
        slabs.append(Slab(-2+n*tread,8,-6,6,n*rise,n*rise,2.,name+'_landing'))
        t=SlabTerrain(name,slabs,{'profile':name,'size_m':[16,12]})
    else:
        t=flat(name)
        if kind=='window':
            cy=c['center_y_m'];fw=c['frame_width_m'];ow=c['opening_width_m'];sill=c['sill_m'];oh=c['opening_height_m'];th=c['wall_thickness_m'];top=sill+oh+.35
            for side in [-1,1]:
                width=(fw-ow)/2;y=cy+side*(ow/2+width/2)
                extras+=box(name+f'_jamb_{side}',0,y,top/2,th,width,top)
            extras+=box(name+'_sill',0,cy,sill/2,th,ow,sill)
            extras+=box(name+'_lintel',0,cy,sill+oh+.175,th,ow,.35)
            t.spec['side_bypass_width_m']=(12-fw)/2-abs(cy)
        elif kind=='barrier':
            e=c['edge_m'];extras+=box(name+'_cube',0,0,e/2,e,e,e);objects.append({'xyz':[0,0,e/2],'dimensions_m':[e]*3})
        elif kind=='block_maze':
            route=np.array([[-6,0],[-3.5,-2],[-1,-2],[1,2],[3.5,2],[6,0]],float)
            def clearance(p):
                a=route[:-1];d=route[1:]-a;u=np.clip(np.sum((p-a)*d,axis=1)/np.sum(d*d,axis=1),0,1)
                return np.min(np.linalg.norm(p-(a+u[:,None]*d),axis=1))
            for _ in range(2000):
                if len(objects)>=c['block_count']:break
                edge=float(rng.uniform(*c['edge_range_m']));p=rng.uniform([-4.8,-4.6],[4.8,4.6])
                if clearance(p)<c['corridor_width_m']/2+edge/math.sqrt(2):continue
                if any(np.linalg.norm(p-np.array(o['xyz'][:2]))<(edge+o['dimensions_m'][0])/math.sqrt(2)+.12 for o in objects):continue
                i=len(objects);extras+=box(name+f'_block_{i}',*p,edge/2,edge,edge,edge)
                objects.append({'xyz':[*p,edge/2],'dimensions_m':[edge]*3})
            if len(objects)!=c['block_count']:raise RuntimeError('finite maze placement search exhausted')
            t.spec['geometric_clearance_route_xy_m']=route.tolist();t.spec['route_status']='geometric clearance only, not locomotion qualification'
        elif kind=='movable_rubble':
            from scipy.spatial import ConvexHull
            for i in range(c['rock_count']):
                for _ in range(500):
                    r=rng.uniform(*c['radius_range_m'],size=3);p=rng.uniform([-4.8,-4.8],[5.2,4.8])
                    if all(np.linalg.norm(p-np.array(o['xyz'][:2]))>max(r[:2])+max(o['radii_m'][:2])+.08 for o in objects):break
                else:raise RuntimeError('finite rubble placement search exhausted')
                verts=np.array([[a,b,d] for a in [-1,1] for b in [-1,1] for d in [-1,1]],float)*r
                verts*=rng.uniform(.8,1.15,verts.shape);verts-=verts.mean(axis=0)
                volume=float(ConvexHull(verts).volume);mass=volume*c['density_kg_m3'];z=float(-verts[:,2].min()+.025);mesh=name+f'_mesh_{i}';body=name+f'_rock_{i}'
                assets+=f'<mesh name="{mesh}" vertex="'+ ' '.join(f'{v:.7f}' for v in verts.ravel())+'"/>\n'
                fr=' '.join(map(str,c['friction']));shade=float(rng.uniform(.28,.55))
                bodies+=f'<body name="{body}" pos="{p[0]} {p[1]} {z}"><freejoint name="{body}_free"/><geom name="{body}_geom" type="mesh" mesh="{mesh}" mass="{mass}" friction="{fr}" contype="1" conaffinity="1" priority="1" condim="6" rgba="{shade} {shade*.95} {shade*.88} 1"/></body>\n'
                objects.append({'body':body,'xyz':[*p,z],'radii_m':r.tolist(),'mass_kg':mass,'volume_m3':volume,'friction':c['friction'],'collision':'convex irregular mesh','freejoint':True,'contact_priority':1})
    t.spec.update({'design_version':3,'design':c,'seed':seed,'objects':objects,'navigation_extra_geom_xml':extras,'navigation_extra_asset_xml':assets,'navigation_dynamic_body_xml':bodies,'navigation_qualification':'pending initial-visible / moving-loss / near-visible reference gate'})
    return t


def scene_for_config(c):
    y=c.get('center_y_m',0.)
    return Scene(c['name'],c['kind'],(6.,y),.55,(-6.5,y),(-6.5,y),terrain_params=c,
                 mechanism=c['kind'],notes=('Version 3 terrain design; not a completed VLM trial.',))


def inspect_all(out):
    import mujoco
    import torch
    from api import MuJoCoEnvironment
    from terrain_viewer import support_height
    from controllers import Primitive
    from benchmark import rollout_with_visibility
    from go2_piper_camera import CAMERA_NAME
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True);records=[];cards=[]
    for c in designs():
        name=c['name'];folder=out/name;folder.mkdir(exist_ok=True);t=build_terrain_v3(c);scene=scene_for_config(c)
        env=MuJoCoEnvironment(scene,terrain=t);renderer=None
        try:
            env.reset(replay_approach=False);env.data.qpos[2]=support_height(t,*scene.lost_xy,0)+.3;mujoco.mj_forward(env.model,env.data)
            env.model.vis.global_.offwidth=800;env.model.vis.global_.offheight=600
            renderer=mujoco.Renderer(env.model,height=600,width=800)
            cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,0];cam.distance=23;cam.azimuth=125;cam.elevation=-50
            def capture(filename):
                renderer.update_scene(env.data,camera=cam);Image.fromarray(renderer.render()).save(folder/filename)
            capture('overview_initial.png')
            # Physical settle and locomotion inspection, not task success.
            stand=rollout_with_visibility(env.model,env.data,t,env.controller,Primitive('stand',duration_s=2.),env.target)
            renderer.update_scene(env.data,camera=CAMERA_NAME);Image.fromarray(renderer.render()).save(folder/'initial_rgb.png')
            before=env.data.xpos.copy();dynamic=t.spec['objects'] if c['kind']=='movable_rubble' else []
            # Apply a documented finite force to one stone to verify it really
            # moves under physics; not part of any autonomous trial.
            pushed=None
            if dynamic:
                bid=env.model.body(dynamic[0]['body']).id;start=env.data.xpos[bid].copy();env.data.xfrc_applied[bid,:3]=[dynamic[0]['mass_kg']*30.,0,0]
                rollout_with_visibility(env.model,env.data,t,env.controller,Primitive('stand',duration_s=.25),env.target)
                env.data.xfrc_applied[:]=0
                rollout_with_visibility(env.model,env.data,t,env.controller,Primitive('stand',duration_s=.75),env.target)
                pushed={'body':dynamic[0]['body'],'force_N':dynamic[0]['mass_kg']*30.,'duration_s':.25,'displacement_m':float(np.linalg.norm(env.data.xpos[bid]-start))}
            rock_contact_samples=[]
            def record_contacts(model,data):
                hits=[(int(data.contact[i].geom1),int(data.contact[i].geom2)) for i in range(data.ncon)
                      if any('rock_' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '')
                             for g in (data.contact[i].geom1,data.contact[i].geom2))]
                if hits:rock_contact_samples.append({'sim_time_s':float(data.time),'pairs':hits})
                return None
            walk=rollout_with_visibility(env.model,env.data,t,env.controller,Primitive('forward',vx=.5,duration_s=8. if dynamic else 2.),env.target,sample_observer=record_contacts if dynamic else None)
            (folder/'rock_contacts.json').write_text(json.dumps(rock_contact_samples,indent=2))
            capture('overview_after.png')
            state={'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist(),'body_positions':env.data.xpos.tolist(),'contacts':[{'geom1':int(env.data.contact[i].geom1),'geom2':int(env.data.contact[i].geom2)} for i in range(env.data.ncon)]}
            (folder/'physical_state.json').write_text(json.dumps(state,indent=2))
            t.save(folder)
            rec={'name':name,'kind':c['kind'],'design':c,'height_range_m':[float(t.heights.min()),float(t.heights.max())],'height_std_m':float(t.heights.std()),'stand':stand,'walk':walk,'finite_state':bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()),'dynamic_body_count':len(dynamic),'stone_force_check':pushed,'rock_contact_sample_count':len(rock_contact_samples),'walk_duration_s':8 if dynamic else 2,'nq':env.model.nq,'nv':env.model.nv,'nu':env.model.nu,'terrain_spec':t.spec,'task_gate':'not run; physical design inspection only'}
            (folder/'inspection.json').write_text(json.dumps(rec,indent=2));records.append(rec)
            cards.append(f'<article id="{name}"><h2>{name}</h2><p>高度范围 {rec["height_range_m"][0]:.2f} 至 {rec["height_range_m"][1]:.2f} m；{8 if dynamic else 2}秒前进位移 {walk["forward_m"]:.2f} m；可移动石头 {len(dynamic)} 个</p><div><a href="{name}/overview_initial.png"><img loading="lazy" src="{name}/overview_initial.png"></a><a href="{name}/initial_rgb.png"><img loading="lazy" src="{name}/initial_rgb.png"></a><a href="{name}/overview_after.png"><img loading="lazy" src="{name}/overview_after.png"></a></div><a href="{name}/inspection.json">配置及物理记录</a></article>')
            (out/'inspection_summary.json').write_text(json.dumps({'version':3,'completed':len(records),'expected':len(designs()),'records':records},indent=2))
            print('INSPECTED',name,'range',rec['height_range_m'],'walk',round(walk['forward_m'],3),'stone',pushed,flush=True)
        finally:
            if renderer:renderer.close()
            env.close()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);cols=4;rows=math.ceil(len(records)/cols);sheet=Image.new('RGB',(1600,rows*330),'#121923');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        x=i%cols*400;y=i//cols*330;draw.text((x+10,y+4),r['name'],font=font,fill='white')
        with Image.open(out/r['name']/'overview_initial.png') as im:sheet.paste(im.resize((400,300)),(x,y+30))
    sheet.save(out/'all_terrains.png')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task3 新版地形设计 v3</title><style>body{font:16px system-ui;background:#121923;color:#eee;margin:24px}a{color:#9bd4ff}article{border:1px solid #456;padding:14px;margin:18px 0}article div{display:flex;gap:12px}img{width:100%}article div a{width:33%}@media(max-width:800px){article div{display:block}article div a{display:block;width:100%}}</style><h1>Task3 新版地形设计 v3</h1><p>__COUNT__个设计，全部展开。每行依次为：初始地形全景、实际站立后相机RGB、短程物理检查后的全景。碎石检查额外施加一次已记录外力验证可移动性。此页为地形设计与物理检查，不是VLM导航成功率；正式测试仍需初始可见、运动丢失与近距可达准入。</p><a href="all_terrains.png">全部设计总览原图</a> · <a href="inspection_summary.json">完整物理检查记录</a>__CARDS__</html>'
    (out/'index.html').write_text(page.replace('__COUNT__',str(len(records))).replace('__CARDS__',''.join(cards)))
    (out/'designs.json').write_text(json.dumps(designs(),indent=2))
    (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},indent=2))
    print('DESIGN_COMPLETE',len(records),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--list',action='store_true');p.add_argument('--view');a=p.parse_args()
    if a.list:
        for c in designs():print(c['name'],json.dumps(c))
    elif a.view:
        import terrain_viewer as viewer
        byname={c['name']:c for c in designs()}
        if a.view not in byname:raise SystemExit('Unknown scene; use --list')
        viewer.scene_for=lambda n:scene_for_config(byname[n]);viewer.view(a.view,'go2rl','auto',None,'approach',None,None,False)
    else:inspect_all(a.out)

if __name__=='__main__':main()
