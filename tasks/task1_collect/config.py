"""TASK1 参数：只做 YAML 载入、默认值合并和范围检查，不引入配置框架。

YAML 只放实验条件；实现常数留在 Python。这里只保证缺省的键有合理默认值，
并且几何参数彼此不矛盾（石头区域、收集区、机器人出生点都要落在场地内）。
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
PROTOTYPES = ROOT.parents[1] / 'assets/objects/task1_rocks'

DEFAULTS = {
    'field': {
        'size_m': 20.0,
        'patch': 'center',
        'center_m': None,
        'terrain_scale': [1.0, 1.0, 1.0],
        'terrain_origin_m': [0.0, 0.0, 0.0],
    },
    'rock': {
        'count': 150,
        'region_m': [-8.0, -8.0, 8.0, 2.0],
        'scale': 0.014,                   # 基准边长比例，约 3.5 cm；Piper 夹爪净开口 6.996 cm
        'volume_jitter': [1.5, 2.0],      # 每块石头的体积在基准的 1.5–2 倍之间随机取。
                                          # 边长倍率是它的立方根（×1.145–1.260），与质量一起
                                          # 按同一个倍率放大，所以密度不变。
        'mass_kg': 0.002,                 # 基准质量，与 scale 等比：长度 1/10 → 体积 1/1000
        'min_spacing_m': 0.5,
        'spawn_clearance_m': 2.0,
        'seed': 0,
        'max_spawn_slope_deg': 6,         # 3.5 cm 的石头在 25° 坡上会滚走，见 task1.yaml
        'meshes': 'all',                  # 'all' 或整数 id 列表；见 tools/prepare_task_rocks.py
    },
    # 收集区是一个有围栏的白色箱子：四面墙沿地形切开、有碰撞、无底板。
    # 0.3 m 边长 + 墙高 0.3 m，所以是个立方体框。
    'collection_zone': {'center_m': [6.0, 6.0], 'size_m': [0.3, 0.3],
                        'wall_height_m': 0.3, 'wall_thickness_m': 0.05, 'wall_embed_m': 0.05},
    'reward': {'per_rock': 10.0, 'fall_penalty': 20.0},
    'episode': {'control_dt': 0.02, 'time_limit_s': 120.0},
    'robot': {'name': 'go2_piper', 'spawn_m': [0.2, 2.4], 'yaw_deg': 0.0},
    # 任务级组合策略（policy/）。默认 none：不跑策略时任务行为与从前完全一致。
    'policy': {'name': 'z_mobile_manip', 'seed': 0},
    # UE 地图资产路径，可选。缺省 None = 用 launch.py 的默认 demo 地图。
    # 三个地形裁片各有一张换过皮的图（tools/build_task1_maps.py），
    # 想换背景月面贴图时在这里填，例如
    # /Game/MoonMacro/Maps/MoonTerrain_Task1_Center
    'ue_map': None,
    'visualization': {
        'enabled': True,
        'boundary_color': [1.0, 0.0, 0.0, 1.0],
        'zone_color': [1.0, 1.0, 1.0, 1.0],
        'line_width_m': 0.06,
        'segment_m': 0.5,
    },
}


def _merge(base, override):
    merged = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _rect(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f'{name} 必须是 [xmin, ymin, xmax, ymax]，当前为 {value!r}')
    x0, y0, x1, y1 = (float(v) for v in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f'{name} 的 max 必须大于 min，当前为 {value!r}')
    return [x0, y0, x1, y1]


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{name} 必须是 [x, y]，当前为 {value!r}')
    return [float(v) for v in value]


def _check(config):
    half = config['field']['size_m'] / 2
    if config['field']['size_m'] <= 0:
        raise ValueError('field.size_m 必须为正')

    # 石头区域、收集区、出生点都用相对场地中心的局部坐标，必须落在场地内。
    zone_center = _point(config['collection_zone']['center_m'], 'collection_zone.center_m')
    zone_size = _point(config['collection_zone']['size_m'], 'collection_zone.size_m')
    spawn = _point(config['robot']['spawn_m'], 'robot.spawn_m')
    region = _rect(config['rock']['region_m'], 'rock.region_m')

    inside = {
        'rock.region_m': all(-half <= v <= half for v in region),
        'collection_zone': all(-half <= c - s / 2 and c + s / 2 <= half
                               for c, s in zip(zone_center, zone_size)),
        'robot.spawn_m': all(-half <= v <= half for v in spawn),
    }
    for name, ok in inside.items():
        if not ok:
            raise ValueError(f'{name} 超出 {config["field"]["size_m"]} m 场地范围，请调整 task1.yaml')

    import math
    def finite(value):
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return math.isfinite(value) if isinstance(value,(int,float)) else True
    if not finite(config):raise ValueError('配置数值不能是 NaN 或 Inf')
    if config['robot']['name']=='mjcf' and not config['robot'].get('model_path'):
        raise ValueError('robot.name=mjcf 必须提供 model_path')
    if config['rock']['count'] < 0:
        raise ValueError('rock.count 不能为负')
    _check_pool(config)
    if config['rock']['min_spacing_m']<0 or config['rock']['spawn_clearance_m']<0:
        raise ValueError('石头间距与出生点间距不能为负')
    if not 0<config['rock']['max_spawn_slope_deg']<90:
        raise ValueError('max_spawn_slope_deg 必须在 0 到 90 度之间')
    if config['visualization']['segment_m'] <= 0:
        raise ValueError('visualization.segment_m 必须为正')
    scale=config['field']['terrain_scale']
    if len(scale)!=3 or any(float(v)<=0 for v in scale) or scale[0]!=scale[1]:
        raise ValueError('terrain_scale requires positive isotropic XY scale')
    if config['robot']['name'] not in ('go2','go2_piper','reference_rover','piper_arm','rover_piper','mjcf'):
        raise ValueError('robot.name must be go2, go2_piper, reference_rover, piper_arm, rover_piper or mjcf')
    if config['episode']['control_dt']<=0 or config['episode']['time_limit_s']<0:
        raise ValueError('Episode timing must be positive')
    if config['rock']['count']!=int(config['rock']['count']) or not config['rock']['meshes']:
        raise ValueError('rock.count must be integer; rock.meshes must be nonempty')
    if min(config['rock']['scale'],config['rock']['mass_kg'],*zone_size)<=0:
        raise ValueError('Rock scale/mass and zone size must be positive')
    jitter=config['rock']['volume_jitter']
    if len(jitter)!=2 or float(jitter[0])<=0 or float(jitter[0])>float(jitter[1]):
        raise ValueError('rock.volume_jitter 必须是 [min, max]，且 0 < min <= max')
    wall=config['collection_zone']
    if float(wall['wall_height_m'])<0 or float(wall['wall_thickness_m'])<=0 or float(wall['wall_embed_m'])<0:
        raise ValueError('collection_zone 的墙高/埋深必须非负，墙厚必须为正')
    if min(config['reward']['per_rock'], config['reward']['fall_penalty']) < 0:
        raise ValueError('Reward magnitudes must be nonnegative')
    if min(zone_size) <= wall['wall_thickness_m']:
        raise ValueError('Collection box must have positive interior clearance')
    if wall['wall_height_m'] <= 0:
        raise ValueError('Collection requires a physical box with positive wall height')
    _check_policy(config)
    return config


def _check_pool(config):
    """把 rock.meshes 解析成实际的 id 列表，并对着磁盘上的原型校验。

    写成 'all' 是为了不必在换石头时同步一份 100 多个数字的清单——清单会漏、会过时，
    而漏掉的那块石头不会报错，只会静默不出现在场上。
    """
    available = sorted(int(p.stem[4:]) for p in PROTOTYPES.glob('rock*.json') if p.stem[4:].isdigit())
    if not available:
        raise ValueError(f'{PROTOTYPES} 里没有石头原型，先跑 tools/prepare_task_rocks.py')
    spec = config['rock']['meshes']
    if isinstance(spec, str):
        if spec != 'all':
            raise ValueError(f"rock.meshes 只能是 'all' 或整数 id 列表，当前为 {spec!r}")
        pool = available
    elif isinstance(spec, (list, tuple)) and spec:
        pool = [int(m) for m in spec]
        missing = sorted(set(pool) - set(available))
        if missing:
            raise ValueError(f'rock.meshes 里的 {missing} 不存在；{PROTOTYPES} 现有 {len(available)} 块原型')
    else:
        raise ValueError(f"rock.meshes 必须是非空列表或 'all'，当前为 {spec!r}")
    config['rock']['meshes'] = pool
    return config


def _check_policy(config):
    policy=config['policy']
    if set(policy)-{'name','seed'}:raise ValueError('Policy implementation constants belong in Python')
    if policy['name'] not in ('none','z_mobile_manip'):raise ValueError('policy.name must be none or z_mobile_manip')


def load(path=None):
    """读取 YAML（缺省 task1.yaml），补默认值并校验后返回。"""
    path = Path(path or ROOT / 'task1.yaml')
    config = _merge(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    return _check(config)
