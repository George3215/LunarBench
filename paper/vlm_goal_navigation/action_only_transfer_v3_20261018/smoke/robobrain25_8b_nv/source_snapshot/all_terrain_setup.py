"""Preserve every existing Task3 terrain; adjust only navigation anchors.
15 terrain types + 16 existing parameterised paired worlds = 31 scenes.
"""
from dataclasses import replace
import math
import api
from benchmark import build_scene_terrain
from terrain_viewer import ALL_TERRAINS, support_height
from scenes import scene_for
from slabs import Slab


def originals():
    scenes={name:scene_for(name) for name in ALL_TERRAINS}
    scenes.update(api.scene_catalogue())
    return {name:(scene,build_scene_terrain(scene)) for name,scene in scenes.items()}


def candidates(name,source,terrain):
    native={
      'stairs_up':[((-4.6,0),(5,0),.5,0,'FORWARD'),((-3.7,0),(5,0),.6,0,'FORWARD')],
      'staircase10':[((-4.6,0),(5,0),.5,0,'FORWARD')],
      'trench':[((-5,0),(4.6,0),.35,0,'FORWARD')],
      'pit':[((-4.8,-.5),(-2.1,1.25),.45,0,'FORWARD')],
      'waves':[((-4,0),(5.2,0),.5,0,'FORWARD')],
      'basin':[((-3,-.3),(2,1.5),.6,0,'FORWARD')],
      'perlin':[((2,0),(5,0),h,0,'FORWARD') for h in (.35,.45,.6)],
      'perlin_rough':[((2,0),(5,0),h,0,'FORWARD') for h in (.3,.35,.45)],
      'stairs_down':[((-1.5,-.3),(3,1.5),.6,0,'FORWARD')],
      'barrier':[((-1,-2),(5,0),.6,0,'STRAFE_LEFT'),((-1,2),(5,0),.6,0,'STRAFE_RIGHT')],
    }
    for candidate in native.get(name,[]):yield candidate
    # Added finite occluders retain all original terrain geometry.
    # Lateral approach reduces target range while crossing into the shadow.
    if source.terrain_profile=='window_wall':
        wx=terrain.spec['wall_x_m']
        for height in (.6,.85):
            yield (wx-4.2,-1.8),(wx-1.2,0.),height,0.,'STRAFE_LEFT',{'x':wx-2.2,'half_y':.35}
    else:
        for sx,tx in [(-3.,2.),(-4.8,-1.8),(-4.8,-2.5),(-1.8,2.5),(0.,4.)]:
            if not terrain.x[0]+.5<sx<terrain.x[-1]-.5 or not terrain.x[0]+.5<tx<terrain.x[-1]-.5:continue
            for height in (.6,.85,1.):
                yield (sx,-1.8),(tx,0.),height,0.,'STRAFE_LEFT',{'x':(sx+tx)/2,'half_y':.4}


def adjusted(source,terrain,candidate):
    start,target,height,yaw,action=candidate[:5]
    extra=candidate[5] if len(candidate)>5 else None
    terrain.spec.pop('navigation_extra_geom_xml',None)
    terrain.spec.pop('navigation_added_occluder',None)
    if extra:
        wx=extra['x'];half=extra['half_y']
        base=max(float(terrain.height(wx,y)) for y in (-half,0,half))
        top=max(base+1.0,float(terrain.height(*target))+height+.4)
        # Raster holes use a -1e9 sentinel; they are not physical ground.
        bottom=min(float(terrain.height(wx,y)) for y in (-half,0,half))-.5
        slab=Slab(wx,wx+.16,-half,half,top,top,top-bottom,name='navigation_added_occluder')
        terrain.spec['navigation_extra_geom_xml']=slab.geom_xml()
        terrain.spec['navigation_added_occluder']={'x':wx,'width':.16,'half_y':half,'top_z':top,'bottom_z':top-slab.thickness_m}
    return replace(source,target_xy=tuple(target),target_height_m=height,
        target_z_absolute=float(terrain.height(*target))+height,
        approach_xy=tuple(start),lost_xy=tuple(start),yaw_rad=yaw,
        history_xy=None,history_saw_target=False,
        notes=((source.notes,) if isinstance(source.notes,str) else tuple(source.notes))+('Navigation anchors adjusted; original terrain geometry retained.',)),action


def place(env):
    env.reset(replay_approach=False)
    scene=env.scene_spec
    env.data.qpos[2]=support_height(env.terrain,*scene.lost_xy,scene.yaw_rad)+.3
    env._mujoco.mj_forward(env.model,env.data)
