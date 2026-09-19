"""Install a task display without changing saved demo assets or map.

Task actors are transient. Existing actor visibility and Landscape transforms
are restored on cleanup; caller must never save the temporary task view as demo.
"""
from pathlib import Path
import json
import re
import task_overlay

TEXTURES = '/Game/Task1_Textures'            # 石头贴图导入到这里的临时资产


def _textured_material(unreal, source, cache):
    """把磁盘上的 jpg/png 导进 UE 并做成贴图材质；同一个文件只导一次。

    石头原型由 tools/prepare_task_rocks.py 从月面扫描件里带出基色贴图。贴图本来封在
    usdz 里或散在 assets 下，MuJoCo 侧的 display.py 已经取好并给出可读路径，这里只管导入。
    导入失败就返回 None，调用方退回纯色材质——贴图缺失不该让整个场景起不来。
    """
    if source in cache:
        return cache[source]
    cache[source] = None
    try:
        stem = re.sub(r'[^0-9A-Za-z_]', '_', Path(source).stem)
        name = f'T_Rock_{stem}'
        unreal.EditorAssetLibrary.make_directory(TEXTURES)
        asset = unreal.EditorAssetLibrary.load_asset(f'{TEXTURES}/{name}')
        if asset is None:
            task = unreal.AssetImportTask()
            task.set_editor_property('filename', source)
            task.set_editor_property('destination_path', TEXTURES)
            task.set_editor_property('destination_name', name)
            task.set_editor_property('automated', True)
            task.set_editor_property('replace_existing', True)
            task.set_editor_property('save', True)
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
            paths = task.get_editor_property('imported_object_paths')
            asset = unreal.load_asset(paths[0]) if paths else None
        if asset is None:
            unreal.log_warning(f'TASK1 scene: 石头贴图导入失败，退回纯色 ({source})')
            return None
        material_name = f'M_Rock_{stem}'
        material = unreal.EditorAssetLibrary.load_asset(f'{task_overlay.MATERIALS}/{material_name}')
        if material is None:
            factory = unreal.MaterialFactoryNew()
            material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                material_name, task_overlay.MATERIALS, unreal.Material, factory)
            if material is None:
                return None
            sample = unreal.MaterialEditingLibrary.create_material_expression(
                material, unreal.MaterialExpressionTextureSample, 0, 0)
            sample.set_editor_property('texture', asset)
            sample.set_editor_property('sampler_type', unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
            unreal.MaterialEditingLibrary.connect_material_property(
                sample, 'RGB', unreal.MaterialProperty.MP_BASE_COLOR)
            unreal.MaterialEditingLibrary.recompile_material(material)
            unreal.EditorAssetLibrary.save_loaded_asset(material)
        cache[source] = material
        return material
    except Exception as error:               # 属性名/API 随版本变，退回纯色即可
        unreal.log_warning(f'TASK1 scene: 贴图材质建不起来，退回纯色 ({source}: {error})')
        return None

def install(unreal,sub,directory,visual):
    previous=[];created=[];actors={}
    old=getattr(unreal,'_moon_task_cleanup',None)
    if old:old()
    def cleanup():
        for a in created:
            if unreal.SystemLibrary.is_valid(a):sub.destroy_actor(a)
        for a,hidden,transform in previous:
            if not unreal.SystemLibrary.is_valid(a):continue
            a.set_is_temporarily_hidden_in_editor(hidden)
            if transform is not None:a.set_actor_transform(transform,False,False)
        unreal._moon_task_cleanup=None
    unreal._moon_task_cleanup=cleanup
    try:
        for a in sub.get_all_level_actors():
            landscape=isinstance(a,unreal.LandscapeProxy)
            hide=a.actor_has_tag('Go2Visual') or a.actor_has_tag('MoonRock') or a.actor_has_tag('TaskOverlay')
            if not (landscape or hide):continue
            previous.append((a,a.is_temporarily_hidden_in_editor(),a.get_actor_transform() if landscape else None))
            if landscape:
                spec=visual['landscape']
                a.set_actor_scale3d(unreal.Vector(*spec['scale']))
                a.set_actor_location(unreal.Vector(*[v*100 for v in spec['location_m']]),False,False)
            else:a.set_is_temporarily_hidden_in_editor(True)
        materials={};meshes={}
        for g in visual['geoms']:
            a=sub.spawn_actor_from_class(unreal.StaticMeshActor,unreal.Vector(),unreal.Rotator(),True)
            created.append(a);a.set_actor_label(f'TaskGeom_{g["id"]}');a.tags=[unreal.Name('TaskVisual')]
            if g.get('sensor_housing'):a.tags=list(a.tags)+[unreal.Name('D435Housing')]
            a.set_folder_path('TASK1');a.set_actor_enable_collision(False)
            c=a.static_mesh_component;c.set_mobility(unreal.ComponentMobility.MOVABLE)
            if g.get('mesh_file'):
                path=g['mesh_file']
                if path not in meshes:meshes[path]=unreal.MoonViewportLibrary.create_task_mesh(path)
                mesh=meshes[path]
            else:mesh=unreal.load_asset(g['asset'])
            if mesh is None:raise RuntimeError(f'Missing task display mesh: {g["asset"]}')
            c.set_static_mesh(mesh);c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
            a.set_actor_scale3d(unreal.Vector(*g['scale']))
            material=None
            texture=g.get('texture')
            if texture:
                if texture not in materials:materials[texture]=_textured_material(unreal,texture,materials)
                material=materials[texture]
            if material is None:                 # 没有贴图（或贴图导不进来）：退回纯色
                key=task_overlay._color_key(g['rgba'])+'_Lit'
                if key not in materials:materials[key]=task_overlay._material(unreal,key,g['rgba'],unlit=False)
                material=materials[key]
            if material:c.set_material(0,material)
            actors[f'Go2Geom_{g["id"]:03d}']=a
        task_overlay.install(unreal,sub,Path(directory))
        created.extend(a for a in sub.get_all_level_actors() if a.actor_has_tag('TaskOverlay') and all(a!=p[0] for p in previous))
        unreal.log(f'TASK1_SCENE_READY parts={len(actors)} landscape_scale={visual["landscape"]["scale"]}')
        return actors
    except Exception:
        cleanup();raise
