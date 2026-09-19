"""Real MuJoCo TASK1 verification; reward teleports are test fixtures, not policy success."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from MoonSim.tasks.task1_collect import make
from MoonSim.tasks.task1_collect import scene,config

OUT=Path(__file__).resolve().parents[1]/'tasks/task1_collect/generated'

def terrain(env):
    f=env.field;m=env.model;d=env.data
    from MoonSim.tasks.task1_collect.field import _source
    ix,iy=f.grid;e=f.edge
    original=_source()[iy-e:iy+e+1,ix-e:ix+e+1][::-1]*f.scale[2]-f.center_height
    assert np.array_equal(original,f.crop)
    height_error=0.;ids=[]
    for tile in env.meta['terrain_tiles']:
        geom=m.geom(tile['name']).id;ids.append(geom);h=int(m.geom_dataid[geom]);a=int(m.hfield_adr[h])
        shape=(tile['rows'],tile['cols']);n=shape[0]*shape[1]
        decoded=m.hfield_data[a:a+n].reshape(shape)*m.hfield_size[h,2]+m.geom_pos[geom,2]
        r,c=tile['row'],tile['col'];ref=original[r:r+shape[0],c:c+shape[1]]
        height_error=max(height_error,float(np.abs(decoded-ref).max()))
    assert height_error<2e-7,height_error
    groups=m.geom_group.copy();m.geom_group[:]=1;m.geom_group[ids]=0
    rng=np.random.default_rng(71);worst=0.
    try:
        for xy in rng.uniform(-f.half+.01,f.half-.01,(512,2)):
            gid=np.array([-1],np.int32)
            distance=mujoco.mj_ray(m,d,np.r_[xy,10.],np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],np.uint8),True,-1,gid)
            assert gid[0] in ids
            worst=max(worst,abs(10-distance-f.height_local(*xy)))
    finally:m.geom_group[:]=groups
    assert worst<2e-7,worst
    return {'native_points_compared':original.size,'source_stride':1,'spacing_m':f.spacing,
            'max_encoded_error_m':height_error,'off_grid_ray_checks':512,'max_ray_error_m':worst}

def layout(env):
    xy=np.array([p[:2] for p,q in env.rock_home])
    assert len(xy)==env.config['rock']['count'],(len(xy),env.config['rock']['count'])
    assert not any(env._in_zone(*p) for p in xy)
    distances=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)+np.eye(len(xy))*1e6
    assert distances.min()>=env.config['rock']['min_spacing_m']
    counts=[]
    for i in range(len(xy)):
        gid=env.model.geom(f'sample_{i}').id
        assert env.model.geom_contype[gid] and env.model.geom_conaffinity[gid]
        assert env.model.body_mass[env.model.body(f'sample_{i}').id]>0
        assert env.model.jnt_type[env.model.joint(f'sample_{i}_joint').id]==mujoco.mjtJoint.mjJNT_FREE
        mid=int(env.model.geom_dataid[gid])
        kind=env.config['rock']['meshes'][i%len(env.config['rock']['meshes'])]
        data=scene._prototype(kind)
        assert int(env.model.mesh_facenum[mid])==len(data['indices'])//3
        counts.append(int(env.model.mesh_facenum[mid]))
    # 只查这一场真正上场的原型：池子有 100+ 块，全查一遍要几十秒，而没上场的块
    # 出问题也不影响这一集。不开孔是转换时就保证的（边折叠不撕洞 + 补掉源扫描件自带的洞），
    # 也是显示上唯一看得出来的毛病——有洞就能从石头背面看穿。
    solid=[];pinched=0
    pool=env.config['rock']['meshes']
    for kind in sorted({pool[i%len(pool)] for i in range(len(xy))}):
        data=scene._prototype(kind);vertices=np.array(data['points']);faces=np.array(data['indices']).reshape(-1,3)
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        _,edge_counts=np.unique(edges,axis=0,return_counts=True)
        assert not np.any(edge_counts==1),f'rock{kind} 有边界边（破洞）'
        # 被三个以上面共用的边不判失败：它来自摄影测量源件的奇点（整池 2 百万条边里 4 条），
        # 去掉只能删面、删了就成洞，渲染与 MuJoCo 取凸包都不受影响。记下来即可。
        pinched+=int((edge_counts>2).sum())
        volume=abs(float(np.einsum('ij,ij->i',vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6))
        extent=np.ptp(vertices,axis=0)*env.config['rock']['scale']
        # 石头得是立体的，不能是一片板。这里按**形状**判、不按绝对高度判：绝对高度会随
        # rock.scale 变，那是任务参数，不该出现在原型校验里。整池最扁的是 rock14（Apollo
        # 扫描件）高/宽 0.30，其余都 ≥ 0.53（lunar 中位 0.94、Apollo 0.70），所以取 0.25。
        assert extent[2]>.25*max(extent[0],extent[1]) and volume>0,f'rock{kind} 太扁或体积非正'
        solid.append({'kind':kind,'extent_m':extent.tolist(),'closed_mesh':True,'has_uv':data.get('uvs') is not None,
                      'textured':data.get('texture') is not None,
                      'nonmanifold_edges':int((edge_counts>2).sum()),
                      'volume_m3':volume*env.config['rock']['scale']**3})
    return {'solid_meshes':solid,'rocks':len(xy),'minimum_spacing_m':float(distances.min()),
            'full_mesh_triangles':counts,'nonmanifold_edges':pinched}

def api(env):
    obs,info=env.reset();spec=env.action_spec
    assert env.observation_spec['rocks']['pose']['shape']==[env.config['rock']['count'],7]
    # Returned arrays must not alias the simulator.
    obs['robot']['position'][:]=1e6
    assert np.linalg.norm(env.base_position)<100
    for bad in ([float('nan')]*spec['shape'][0],np.zeros(spec['shape'][0]+1),np.array(spec['high'])+1):
        try:env.step(bad)
        except ValueError:pass
        else:raise AssertionError('invalid action accepted')
    zero=np.zeros(spec['shape']);total=0.
    for _ in range(50):
        _,r,terminated,truncated,_=env.step(zero);total+=r
        assert not terminated and not truncated
    start=env.base_position.copy();state=env.get_state()
    action=np.array([.15,0,0] if env.robot.name=='go2' else [1.,1.])
    first=[]
    for _ in range(15):first.append(env.step(action)[1:4])
    expected=env.get_state()['physics'].copy()
    env.set_state(state)
    second=[env.step(action)[1:4] for _ in range(15)]
    error=float(np.max(np.abs(env.get_state()['physics']-expected)))
    assert error<1e-9,(env.robot.name,error)
    assert first==second
    displacement=float(np.linalg.norm(env.base_position[:2]-start[:2]))
    assert displacement>.001,(env.robot.name,displacement)
    assert np.isfinite(env.data.qpos).all() and not env.fallen
    return {'action_spec':spec,'state_replay_max_error':error,'controlled_displacement_m':displacement,
            'fallen':env.fallen,'contacts':env.data.ncon,
            'rocks_in_contact':len({int(env.model.geom_bodyid[g]) for c in env.data.contact for g in (c.geom1,c.geom2) if int(env.model.geom_bodyid[g]) in env.rock_bodies}),'min_contact_distance_m':min((c.dist for c in env.data.contact),default=0.)}

def reward(env):
    # Explicit scoring fixtures; these teleports do not demonstrate a collecting policy.
    env.reset();a=env.rock_adr[0]
    env.data.qpos[a:a+2]=env.zone_center
    env.data.qpos[a+2]=env.field.height_local(*env.zone_center)+.10
    mujoco.mj_forward(env.model,env.data)
    for _ in range(1000):env.physics_step()
    assert env.score==0, 'ungrasped rocks must not score'
    env.scoring.lifted[0]=True  # inject grasp history only for this scoring fixture
    for _ in range(1000):env.physics_step()
    r=env.config['reward']['per_rock']
    assert env.score==r, (env.score, env.scoring.dwell[0])
    assert env.physics_step()==0
    env.data.qpos[a:a+2]=env.zone_center+env.zone_size;env.physics_step()
    env.data.qpos[a:a+2]=env.zone_center;assert env.physics_step()==0
    env.reset();assert env.score==0 and not env.collected and not env.scoring.lifted.any()
    for a,(p,q) in zip(env.rock_adr,env.rock_home):assert np.array_equal(env.data.qpos[a:a+7],np.r_[p,q])
    env.robot.fallen=True
    assert env.physics_step()==-env.config['reward']['fall_penalty']
    assert env.physics_step()==0
    env.reset();env.config['episode']['time_limit_s']=.04
    env.step(np.zeros(env.action_spec['shape']))
    assert env.step(np.zeros(env.action_spec['shape']))[3]
    return {'per_rock':r,'ungrasped_reward':0,'reentry_reward':0,
            'fall_penalty':env.config['reward']['fall_penalty'],
            'reset_restores_all_rocks':True,'time_limit_truncation':True}

def main():
    result={}
    with make(robot='go2') as env:
        result['terrain']=terrain(env);result['layout']=layout(env)
        result['go2']=api(env);result['reward']=reward(env)
        poses=[p.copy() for p,q in env.rock_home]
        env.reset(seed=19);changed=[p.copy() for p,q in env.rock_home]
        assert not np.array_equal(poses,changed)
        env.reset(seed=0);assert np.array_equal(poses,[p for p,q in env.rock_home])
        result['seed_reset_reproducible']=True
        names=[env.model.geom(i).name for i in range(env.model.ngeom)]
        lines=[i for i,n in enumerate(names) if n.startswith('viz_')]
        assert all(env.model.geom_contype[i]==env.model.geom_conaffinity[i]==0 for i in lines)
        # 收集区的箱子是另一回事：墙必须**有**碰撞，否则石头放进去会滚出来，
        # 而"白色箱子"就退化成了地板上的白线。
        walls=[i for i,n in enumerate(names) if n.startswith('zone_wall_')]
        assert walls, '收集区箱子没有生成墙体'
        assert all(env.model.geom_contype[i] and env.model.geom_conaffinity[i] for i in walls)
        assert all(env.model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX for i in walls)
        result['non_colliding_boundary_segments']=len(lines)
        result['solid_zone_walls']=len(walls)
    with make(robot='reference_rover') as env:result['reference_rover']=api(env)
    result['all_pass']=True
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'task1_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
