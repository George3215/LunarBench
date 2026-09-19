"""TASK1 在 UE 里的可视化叠加层：红色场地边界 + 白色收集区箱子。

独立模块，可以整个删掉：demo_receiver.py 里的 import 包在 try/except 里，
文件不在时 UE 照常显示原 demo，只是看不到场地线条。

线条本身是静态的，所以不上 UDP、不改包格式、不碰 demo 的 33 个 Go2Visual 断言：
descriptor 由 MuJoCo 侧的 tasks/task1_collect/viz.py 写进 ue_overlay.json，
这里只在启动时读一次并 spawn 成 Cube。

渲染的是场景里的 actor，不是 UMG，因此跟随编辑器相机，任何视角都能看到。
"""
import json
import math
from pathlib import Path

TAG = 'TaskOverlay'          # 用于每次重建时清掉上一轮
FOLDER = 'Task1_Overlay'
CUBE = '/Engine/BasicShapes/Cube.Cube'      # 边长 100 cm、以原点为中心
MATERIALS = '/Game/MoonGo2/Materials'       # 与 MoonGo2.inl 造材质放同一个目录


def descriptor(root):
    """ue_overlay.json 的路径；root 是 ue/demo_receiver.py 里的 MoonSim/mujoco。"""
    root=Path(root)
    return root/'ue_overlay.json' if (root/'ue_overlay.json').is_file() else root.parent/'tasks/task1_collect/generated/ue_overlay.json'


def _color_key(rgba):
    return 'M_TaskOverlay_%02X%02X%02X' % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgba[:3])


def _material(unreal, name, rgba, unlit=True):
    """按 MoonGo2.inl:96-103 的同一套 API 造一个纯色材质，已存在就直接复用。"""
    path = f'{MATERIALS}/{name}'
    existing = unreal.EditorAssetLibrary.load_asset(path)
    if existing:
        return existing
    factory = unreal.MaterialFactoryNew()
    asset = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, MATERIALS, unreal.Material, factory)
    if asset is None:
        return None
    color = unreal.MaterialEditingLibrary.create_material_expression(
        asset, unreal.MaterialExpressionConstant3Vector, 0, 0)
    color.set_editor_property('constant', unreal.LinearColor(*[float(c) for c in rgba[:3]], 1.0))
    unreal.MaterialEditingLibrary.connect_material_property(
        color, '', unreal.MaterialProperty.MP_BASE_COLOR)
    try:
        if not unlit:
            unreal.MaterialEditingLibrary.recompile_material(asset)
            unreal.EditorAssetLibrary.save_loaded_asset(asset)
            return asset
        # 场景是暗的月面，自发光才能让线条在任何光照下都读得出来。
        unreal.MaterialEditingLibrary.connect_material_property(
            color, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        asset.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_UNLIT)
    except Exception as error:                # 属性名随版本变，退化成普通材质即可
        unreal.log_warning(f'TASK1 overlay: 材质退化为受光材质 ({error})')
    unreal.MaterialEditingLibrary.recompile_material(asset)
    unreal.EditorAssetLibrary.save_loaded_asset(asset)
    return asset


def clear(unreal, sub):
    """删掉上一轮 spawn 的线条，重复 install 不会累积。"""
    removed = 0
    for actor in sub.get_all_level_actors():
        if actor.actor_has_tag(TAG):
            sub.destroy_actor(actor)
            removed += 1
    return removed


def install(unreal, sub, root):
    """读 ue_overlay.json 并画线。返回 (画了几段, 删了几段)；没有 descriptor 就什么都不做。"""
    path = descriptor(root)
    if not path.is_file():
        unreal.log('TASK1 overlay: 没有 ue_overlay.json，跳过（先跑一次 run.py 生成场景）')
        return 0, 0

    # 变量名不能叫 descriptor：那会和本模块的 descriptor() 函数重名，
    # 使 install 整个作用域里 descriptor 变成本地名，L76 的调用直接 UnboundLocalError。
    data = json.loads(path.read_text())
    segments = data.get('segments') or []
    walls = data.get('walls') or []
    removed = clear(unreal, sub)
    if not segments and not walls:
        unreal.log('TASK1 overlay: descriptor 里没有线段')
        return 0, 0

    mesh = unreal.EditorAssetLibrary.load_asset(CUBE)
    colors = {}          # 颜色 -> 材质，同一颜色的所有线段共用一个材质
    for segment in segments:
        rgba = segment['rgba']
        key = _color_key(rgba)
        if key not in colors:
            colors[key] = _material(unreal, key, rgba)

    drawn = 0
    for index, segment in enumerate(segments):
        start = unreal.Vector(*[float(v) for v in segment['a']])
        end = unreal.Vector(*[float(v) for v in segment['b']])
        length = math.dist(segment['a'],segment['b'])
        if length <= 0.0:
            continue
        rotation = unreal.MathLibrary.find_look_at_rotation(start, end)
        # EditorActorSubsystem.spawn_actor_from_class 的第 4 个参数是 transient(bool)，
        # 不是 ActorSpawnParameters（那是 C++ 里 World->SpawnActor 的用法）。
        actor = sub.spawn_actor_from_class(
            unreal.StaticMeshActor, (start + end) / 2.0, rotation, True)
        if actor is None:
            continue
        name = f'{TAG}_{index:03d}'
        actor.set_actor_label(name)
        actor.set_folder_path(FOLDER)
        actor.tags = [unreal.Name(TAG)]
        actor.set_actor_enable_collision(False)
        component = actor.static_mesh_component
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
        component.set_static_mesh(mesh)
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        # Cube 边长 100 cm、原点居中：X 拉到线段长度，Y 是线宽，Z 压成薄片。
        component.set_world_scale3d(unreal.Vector(
            length / 100.0, float(segment['width_cm']) / 100.0, float(segment['width_cm']) / 100.0))
        material = colors.get(_color_key(segment['rgba']))
        if material:
            component.set_material(0, material)
        drawn += 1

    # 收集区的白箱子：同一批几何在 MuJoCo 侧是有碰撞的实体墙，这里只画出来。
    # 和线段不同，Cube 的 Z 不是压扁的线宽，而是墙的实际高度。
    wall_count = 0
    for index, wall in enumerate(walls):
        start = unreal.Vector(*[float(v) for v in wall['a']])
        end = unreal.Vector(*[float(v) for v in wall['b']])
        length = math.dist(wall['a'], wall['b'])
        if length <= 0.0:
            continue
        rotation = unreal.MathLibrary.find_look_at_rotation(start, end)
        actor = sub.spawn_actor_from_class(
            unreal.StaticMeshActor, (start + end) / 2.0, rotation, True)
        if actor is None:
            continue
        name = f'{TAG}_wall_{index:03d}'
        actor.set_actor_label(name)
        actor.set_folder_path(FOLDER)
        actor.tags = [unreal.Name(TAG)]
        actor.set_actor_enable_collision(False)
        component = actor.static_mesh_component
        component.set_mobility(unreal.ComponentMobility.MOVABLE)
        component.set_static_mesh(mesh)
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        component.set_world_scale3d(unreal.Vector(
            length / 100.0, float(wall['width_cm']) / 100.0, float(wall['height_cm']) / 100.0))
        # colors 原本只从 segments 收集，而墙用的是 zone_color（白）、边界用 boundary_color（红），
        # 两个 key 不同，于是这里的 colors.get() 对每一面墙都返回 None，白箱子会用引擎 Cube 的
        # 默认灰格材质渲染。和线段一样按需建材质。
        key = _color_key(wall['rgba'])
        if key not in colors:
            colors[key] = _material(unreal, key, wall['rgba'])
        material = colors[key]
        if material:
            component.set_material(0, material)
        wall_count += 1

    unreal.log(f'TASK1_OVERLAY_PASS segments={drawn} walls={wall_count} '
               f'removed={removed} colors={len(colors)}')
    return drawn + wall_count, removed
