"""Render recorded Task3 scene geometry and publish all scene pictures.

Overview cameras are review-only; recorded initial/lost images are original
experiment evidence. This script never runs a policy or changes trial data.
"""
import argparse
import copy
import html
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build(root):
    setups = json.loads((root / 'scene_setup.json').read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tasks/task3_reobservability'))
    import mujoco
    from api import MuJoCoEnvironment
    from scenes import Scene
    from all_terrain_setup import originals, place
    from difficulty_terrain_setup import catalogue

    config_path = root / 'terrain_config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    if config is None:
        protocol = json.loads((root / 'protocol.json').read_text())
        candidate = protocol.get('terrain_config')
        if isinstance(candidate, dict):
            config = candidate
        elif candidate and Path(candidate).exists():
            config = json.loads(Path(candidate).read_text())
    if root.name.endswith('_v2_medium') and config is None:
        raise RuntimeError('Cannot render revised geometry without its terrain config')
    terrains = catalogue(config) if config else originals()
    dest = root / 'terrain_images'
    dest.mkdir(exist_ok=True)
    cards = []
    for setup in setups:
        name = setup['scene_name']
        terrain = terrains[name][1]
        terrain.spec.clear()
        terrain.spec.update(copy.deepcopy(setup['terrain_spec']))
        scene = Scene(**setup['adjusted_scene'])
        overview = dest / (name + '_overview.png')
        if not overview.exists():
            env = MuJoCoEnvironment(scene, terrain=terrain)
            try:
                place(env)
                initial = json.loads((root / 'rgb/current/trials' / name / 'initialization.json').read_text())
                env.data.qpos[:] = initial['qpos']
                env.data.qvel[:] = initial['qvel']
                mujoco.mj_forward(env.model, env.data)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = [float((terrain.x[0] + terrain.x[-1]) / 2),
                                 float((terrain.y[0] + terrain.y[-1]) / 2), 0.]
                cam.distance = max(float(terrain.x[-1] - terrain.x[0]),
                                   float(terrain.y[-1] - terrain.y[0])) * 1.45
                cam.azimuth = 125
                cam.elevation = -48
                env.model.vis.global_.offwidth = 800
                env.model.vis.global_.offheight = 600
                renderer = mujoco.Renderer(env.model, height=600, width=800)
                try:
                    renderer.update_scene(env.data, camera=cam)
                    Image.fromarray(renderer.render()).save(overview)
                finally:
                    renderer.close()
            finally:
                env.close()
        evidence = Path(setup['evidence_path'])
        images = [(label, os.path.relpath(path, root)) for label, path in (
            ('地形全景（额外渲染，仅供查看）', overview),
            ('初始相机 RGB（实验原始记录）', evidence / 'initial/rgb.png'),
            ('运动后目标不可见（实验原始记录）', evidence / 'lost/rgb.png'))]
        mechanism = '新增方块' if setup['added_occluder'] else '未新增方块'
        figures = ''.join(f'<figure><figcaption>{label}</figcaption><a href="{html.escape(path)}" target="_blank"><img loading="lazy" src="{html.escape(path)}" alt="{html.escape(name)} {label}"></a></figure>' for label, path in images)
        cards.append(f'<article id="{html.escape(name)}"><h2>{html.escape(name)}</h2><p>{mechanism}；初始深度相机测距 {setup["initial_depth"]["red_region_median_surface_range_m"]:.3f} m</p><div class="images">{figures}</div></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task3 全部地形图片</title><style>body{font:16px system-ui;background:#121923;color:#e5edf5;margin:24px}a{color:#9bd4ff}article{border:1px solid #455;padding:16px;margin:18px 0;border-radius:8px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{margin-bottom:8px}img{width:100%;height:auto}nav{line-height:2.2}nav a{margin-right:16px}@media(max-width:800px){.images{grid-template-columns:1fr}}</style><h1>当前全部地形与场景图片</h1><p>31个场景：15类地形及16个窗口参数变体。全部内容默认展开，点击图片可看原图。地形全景使用已记录配置和初始机器人状态额外渲染，并非模型输入；相机图来自正式场景准入记录。</p><p><a href="index.html">返回 Trial 浏览页</a> · <a href="terrain_images/terrain_families.png">15类地形全景总览</a> · <a href="terrain_images/window_variants.png">16个窗口变体全景总览</a></p><nav>__NAV__</nav>__CARDS__</html>'''
    nav = ''.join(f'<a href="#{html.escape(s["scene_name"])}">{html.escape(s["scene_name"])}</a>' for s in setups)
    (root / 'terrain_gallery.html').write_text(page.replace('__NAV__', nav).replace('__CARDS__', ''.join(cards)))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    for filename, group in [('terrain_families.png', [s for s in setups if not s['scene_name'].startswith('hist')]),
                            ('window_variants.png', [s for s in setups if s['scene_name'].startswith('hist')])]:
        cols = 3 if len(group) == 15 else 4
        rows = (len(group) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * 400, rows * 330), '#121923')
        draw = ImageDraw.Draw(sheet)
        for i, setup in enumerate(group):
            name = setup['scene_name']; x = i % cols * 400; y = i // cols * 330
            draw.text((x + 12, y + 6), name, font=font, fill='white')
            with Image.open(dest / (name + '_overview.png')) as im:
                sheet.paste(im.resize((400, 300)), (x, y + 30))
        sheet.save(dest / filename)
    print('GALLERY_COMPLETE', len(setups), str(root / 'terrain_gallery.html'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root.resolve())
