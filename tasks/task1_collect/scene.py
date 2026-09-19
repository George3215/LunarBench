"""TASK1: native-resolution terrain tiles, chosen robot and closed 3D sample bodies."""
from pathlib import Path
import json
import struct
import xml.etree.ElementTree as ET
import numpy as np

try:
    from .field import Field
except ImportError:
    from field import Field

ROOT = Path(__file__).resolve().parents[2]
PROTOTYPES = ROOT / 'assets/objects/task1_rocks'
GENERATED = Path(__file__).resolve().parent / 'generated'
GROUND_CLEARANCE = 0.02  # rock.scale=0.14 时的石头底面离地高度上限，避免初始互穿


def _prototype(mesh_id):
    """原型 JSON：points / indices / uvs / texture / audit。由 tools/prepare_task_rocks.py 生成。"""
    return json.loads((PROTOTYPES / f'rock{mesh_id}.json').read_text())


def _robot(config):
    try:
        from .robot import model_tree
    except ImportError:
        from robot import model_tree
    return model_tree(config['robot'])


def _samples(config, field):
    """按 rock.seed 在 rock.region_m 内撒石头，保证间距且避开机器人出生点。

    返回 (points, yaws, volume)：volume 是每块石头的体积倍率（基准体积的倍数）。
    """
    spec = config['rock']
    spawn = np.array(config['robot']['spawn_m'], dtype=float)
    rng = np.random.default_rng(spec['seed'])
    x0, y0, x1, y1 = spec['region_m']
    points = []
    for _ in range(20000):
        if len(points) >= spec['count']:
            break
        candidate = rng.uniform([x0, y0], [x1, y1])
        if np.linalg.norm(candidate - spawn) < spec['spawn_clearance_m']:
            continue
        zone=config['collection_zone']
        if np.all(np.abs(candidate-np.array(zone['center_m']))<=np.array(zone['size_m'])/2+.4):
            continue
        if points and min(float(np.linalg.norm(candidate - p)) for p in points) < spec['min_spacing_m']:
            continue
        # Random placement on locally supportable terrain; do not modify heights.
        # 坡度看的是石头自己那块地，不是固定的 0.25 m：采样半径跟着 rock.scale 走，
        # 否则缩到 3.5 cm 之后，0.25 m 半径内的平均坡度会把大片本来放得下的点判掉。
        radius=max(2*field.spacing,2*float(spec['scale']))
        level=float(field.height_local(*candidate))
        heights=[float(field.height_local(*(candidate+offset))) for offset in
                 ((radius,0),(-radius,0),(0,radius),(0,-radius))]
        if max(abs(z-level) for z in heights)/radius > np.tan(np.deg2rad(spec.get('max_spawn_slope_deg',25))):
            continue
        points.append(candidate)
    if len(points) < spec['count']:
        raise ValueError(
            f'rock.region_m 里放不下 {spec["count"]} 块石头（间距 {spec["min_spacing_m"]} m）；'
            '请放大区域或减小数量/间距')
    yaws = rng.uniform(-np.pi, np.pi, size=len(points))
    # 每块石头的体积倍率。刻意抽在撒点与朝向之后：前两者的随机序列因此逐位不变，
    # 加抖动不会让石头位置或朝向挪动，只是每块石头大小不同。
    volume = rng.uniform(*spec['volume_jitter'], size=len(points))
    return np.array(points), yaws, volume


def _footprint_height(field, x, y, radius):
    """石头占地范围内地形最高点，避免斜面上一侧埋进地里。"""
    offsets = [(0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)]
    return max(float(field.height_local(x + dx, y + dy)) for dx, dy in offsets)


def build(config, out_dir=GENERATED):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    field = Field(config['field'], out_dir)

    root = _robot(config)

    # 休眠：静止的刚体树跳过自身的窄相碰撞，醒着的物体仍会与它求交并把它唤醒。
    # 这片场景的开销几乎全在石头上——每块石头都贴着 1 cm 高度场做窄相，关掉石头碰撞后
    # 单步从 9.33 ms 掉到 0.18 ms。休眠把"已经静止"的那部分省掉，而**不动机器人的地形与
    # 精度**：地形仍是原始 1 cm 网格，`sleep_tolerance` 仍是 MuJoCo 默认的 0.001。
    # 实测每步 9.33 → 0.079 ms（RTF 0.21 → 25）；机器人 8 s 位移 2.82 m vs 不休眠 2.84 m；
    # 被压到的休眠石头位移 15.9 mm vs 15.7 mm。见 tasks/task1_collect/README.md。
    option = root.find('option')
    flags = option.find('flag')
    if flags is None:
        flags = ET.SubElement(option, 'flag')
    flags.set('sleep', 'enable')

    world = root.find('worldbody')
    asset = root.find('asset')

    # Partition, never decimate: every 1 cm cell appears in exactly one tile.
    # Small bounds allow renderer frustum culling and physics broadphase pruning.
    stride=max(2,round(2.0/field.spacing));tiles=[]
    for row in range(0,field.samples-1,stride):
        for col in range(0,field.samples-1,stride):
            r1=min(row+stride,field.samples-1);c1=min(col+stride,field.samples-1)
            crop=field.crop[row:r1+1,col:c1+1]
            tile_min=float(crop.min());tile_span=max(float(np.ptp(crop)),.001)
            values=((crop-tile_min)/tile_span).astype('<f4')
            name='terrain' if row==col==0 else f'terrain_{row}_{col}'
            filename=f'{name}.bin'
            with (out_dir/filename).open('wb') as handle:
                handle.write(struct.pack('<ii',*values.shape));handle.write(values.tobytes())
            sx=(c1-col)*field.spacing/2;sy=(r1-row)*field.spacing/2
            x=-field.half+col*field.spacing+sx;y=-field.half+row*field.spacing+sy
            ET.SubElement(asset,'hfield',name=name,file=filename,size=f'{sx:.9g} {sy:.9g} {tile_span:.12g} 1')
            ET.SubElement(world,'geom',name=name,type='hfield',hfield=name,
                          pos=f'{x:.9g} {y:.9g} {tile_min:.12g}',rgba='.42 .42 .42 1',
                          friction='.8 .02 .001',condim='3',group='0')
            tiles.append({'name':name,'row':row,'col':col,'rows':values.shape[0],'cols':values.shape[1]})
    ET.SubElement(world, 'light', pos='0 0 10', dir='-.3 -.5 -1', diffuse='.8 .8 .8')

    points, yaws, volume = _samples(config, field)
    spec = config['rock']
    # 池子可能有 100+ 块原型，但一次只用到 count 块：只加载真正上场的那几块。
    pool = [int(m) for m in spec['meshes']]
    used = {pool[index % len(pool)] for index in range(len(points))}
    meshes = {mesh_id: _prototype(mesh_id) for mesh_id in used}
    placed = []
    scales = []
    for index, (xy, yaw) in enumerate(zip(points, yaws)):
        mesh_id = pool[index % len(pool)]
        vertices = np.array(meshes[mesh_id]['points'], dtype=float)
        faces = meshes[mesh_id]['indices']
        # volume[index] 是这块石头的体积倍率；边长倍率是它的立方根。质量用同一个倍率放大，
        # 所以密度与基准石头完全一致——变大不等于变轻。
        scale = float(spec['scale']) * float(volume[index]) ** (1.0 / 3.0)
        mass = float(spec['mass_kg']) * float(volume[index])
        scaled = vertices * scale
        # 离地高度按石头尺寸等比给。2 cm 的落差对 35 cm 的石头是"贴地放下"，对 3.5 cm 的
        # 石头却是从半身高扔下去——会弹起来、在坡上滚走。0.14*scale 在原来的 rock.scale=0.14
        # 上正好是 0.0196 m，与原来的 0.02 一致；上限仍是 GROUND_CLEARANCE。
        clearance = min(GROUND_CLEARANCE, 0.14 * scale)
        # 只绕 Z 转，竖直角点不变，所以底面高度可以直接用局部包围盒。
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        z = (_footprint_height(field, xy[0], xy[1], float(np.abs(scaled[:, :2]).max()))
             - float(scaled[:, 2].min()) + clearance)
        name = f'sample_{index}'
        ET.SubElement(asset, 'mesh', name=name,
                      vertex=' '.join(f'{v:.9g}' for v in scaled.ravel()),
                      face=' '.join(map(str, faces)))
        body = ET.SubElement(world, 'body', name=name,
                             pos=f'{xy[0]:.9g} {xy[1]:.9g} {z:.9g}',
                             quat=' '.join(f'{v:.9g}' for v in quat))
        ET.SubElement(body, 'freejoint', name=f'{name}_joint')
        ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name,
                      mass=f'{mass:.6g}', friction='.8 .02 .001', solref='.01 1',
                      rgba='.62 .56 .46 1', group='0')  # 比地形亮，远处也能看见样本
        placed.append([float(xy[0]), float(xy[1]), float(z)])
        scales.append(scale)

    (out_dir/'ue_overlay.json').write_text(json.dumps({'segments':[],'walls':[]}))
    viz_count = {'boundary_segments':0,'zone_walls':0}
    zone = config['collection_zone']
    # Box collision is task physics; hiding boundary decoration must never remove walls.
    if __package__:
        from . import viz
    else:
        import viz
    viz.decorate(root, field, config['visualization'], zone)
    viz_count = viz.write_ue_overlay(out_dir / 'ue_overlay.json', field,
                                     config['visualization'], zone)

    ET.indent(root)
    ET.ElementTree(root).write(out_dir / 'scene.xml', encoding='unicode')

    meta = {
        'origin_ue_m': field.origin.tolist(),
        'grid_index': list(map(int,field.grid)),
        'field_size_m': field.size_m,
        'field_center_ue_m': field.origin[:2].tolist(),
        'relief_m': field.relief_m,
        'samples': field.samples,
        'spacing_m': field.spacing,
        'terrain_resampled': False,
        'build_version': 8,
        'terrain_tiles': tiles,
        'landscape': {'location_m': field.base.tolist(), 'scale': field.scale.tolist()},
        'rocks': list(range(len(placed))),
        'rock_positions_local': placed,
        # 每块石头的实际边长比例（基准 scale × 该块体积倍率的立方根）。UE 显示按原型网格
        # 加 actor 缩放来画，必须用这个逐块的值，不能用全局 rock.scale。
        'rock_scales': scales,
        'viz_segments': viz_count['boundary_segments'],
        'zone_walls': viz_count['zone_walls'],
        'visualization': bool(viz_count['boundary_segments'] or viz_count['zone_walls']),
        'gravity': -9.81,
        'source': 'TASK1 rock collection; terrain from demo R16 and rock transforms',
        'config': config,
    }
    (out_dir / 'scene.json').write_text(json.dumps(meta, indent=2))
    return meta, field


def ensure(config, out_dir=GENERATED, force=False):
    """Regenerate task-owned descriptions so edits to models/assets cannot leave stale caches.

    No USD conversion or source resampling occurs. force is retained for CLI compatibility.
    """
    return build(config,out_dir)
