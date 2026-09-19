"""TASK1 可视化：红色场地边界 + 白色的收集区箱子。

同时定义收集框的物理墙体，不能作为可选装饰删除。边界线是纯几何：MuJoCo 侧是一串 contype=0 的细长 box，
UE 侧是同一批线段的世界坐标，由 ue/task_overlay.py 生成静态 mesh actor。

收集区是一个**有围栏的白色箱子**：四面墙沿地形切成小段，每段从地形最低点下方埋进去
一点、往上高出 wall_height_m。和边界线不同，**墙是实体**（contype/conaffinity 保持默认），
所以石头被夹爪放进箱子之后会留在里面；箱子没有底板，地形本身就是箱底。
"""

from pathlib import Path
import json
import math
import numpy as np

DEGENERATE = 1e-9


def _quat_from_x(direction):
    """把局部 +X 轴转到 direction 的四元数 (w, x, y, z)。"""
    axis = np.cross([1.0, 0.0, 0.0], direction)
    norm = float(np.linalg.norm(axis))
    if norm < DEGENERATE:
        # 已经共线：同向为单位四元数，反向绕 Z 转 180 度。
        return [1.0, 0.0, 0.0, 0.0] if direction[0] > 0 else [0.0, 0.0, 0.0, 1.0]
    axis = axis / norm
    angle = math.acos(float(np.clip(direction[0], -1.0, 1.0)))
    half = angle / 2
    s = math.sin(half)
    return [math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s]


def _outline(x0, y0, x1, y1, step):
    """矩形轮廓上按 step 采样的点，首尾不重复。"""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    points = []
    for i in range(4):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % 4]
        length = math.hypot(bx - ax, by - ay)
        count = max(1, int(math.ceil(length / step)))
        for k in range(count):
            t = k / count
            points.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return points


def _loop(points, height_local, offset):
    """把闭合折线转成 (p0, p1) 线段，端点贴地形并抬高半个线宽。"""
    segments = []
    for i, (ax, ay) in enumerate(points):
        bx, by = points[(i + 1) % len(points)]
        a = np.array([ax, ay, float(height_local(ax, ay)) + offset])
        b = np.array([bx, by, float(height_local(bx, by)) + offset])
        if np.linalg.norm(b - a) > DEGENERATE:
            segments.append((a, b))
    return segments


def boundary(field, spec):
    """场地外框：贴地的红色折线，返回 [(a, b, rgba)]。"""
    if not spec['enabled']:
        return []
    offset = spec['line_width_m'] / 2
    half = field.half
    return [(a, b, spec['boundary_color'])
            for a, b in _loop(_outline(-half, -half, half, half, spec['segment_m']),
                              field.height_local, offset)]


def walls(field, spec, zone):
    """收集区箱子的四面墙，切成贴地形的小段。

    每段给出中心线两端点 ``a``/``b``（局部米，z 是该段的墙中高）、厚度与高度。同一段内
    顶面是平的，段与段之间随地形递变，所以墙顶读起来仍然贴着地走，而墙脚不会在坡上悬空。
    ``wall_height_m`` 为 0 时不建箱子，退回原来只有地面轮廓线的画法。
    """
    height = float(zone.get('wall_height_m', 0.0))
    if height <= 0.0:
        return []
    thickness = float(zone.get('wall_thickness_m', spec['line_width_m']))
    embed = float(zone.get('wall_embed_m', 0.0))
    half_thickness = thickness / 2
    cx, cy = zone['center_m']
    sx, sy = zone['size_m']
    corners = [(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2),
               (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)]
    boxes = []
    for i in range(4):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % 4]
        length = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / length, (by - ay) / length
        count = max(1, int(math.ceil(length / spec['segment_m'])))
        for k in range(count):
            t0, t1 = k / count, (k + 1) / count
            p0 = (ax + (bx - ax) * t0, ay + (by - ay) * t0)
            p1 = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
            base = min(float(field.height_local(*p0)), float(field.height_local(*p1))) - embed
            top = max(float(field.height_local(*p0)), float(field.height_local(*p1))) + height
            # 沿墙方向两端各外扩半个厚度：拐角处相邻两面墙自然搭接，不留缝。
            boxes.append({'a': (p0[0] - ux * half_thickness, p0[1] - uy * half_thickness, (base + top) / 2),
                          'b': (p1[0] + ux * half_thickness, p1[1] + uy * half_thickness, (base + top) / 2),
                          'rgba': list(spec['zone_color']),
                          'thickness_m': thickness, 'height_m': top - base})
    return boxes


def decorate(root, field, spec, zone):
    """把边界线和箱子墙写进 MJCF。

    边界线 contype/conaffinity=0，不产生任何物理接触；箱子墙是实体几何，机器人进不去、
    放进去的石头出不来。
    """
    import xml.etree.ElementTree as ET
    width = spec['line_width_m'] / 2
    world = root.find('worldbody')
    for index, (a, b, color) in enumerate(boundary(field, spec)):
        delta = b - a
        length = float(np.linalg.norm(delta))
        center = (a + b) / 2
        ET.SubElement(world, 'geom',
                      name=f'viz_{index:04d}', type='box',
                      pos=' '.join(f'{v:.6g}' for v in center),
                      quat=' '.join(f'{v:.9g}' for v in _quat_from_x(delta / length)),
                      size=f'{length / 2 + width:.6g} {width:.6g} {width:.6g}',
                      rgba=' '.join(f'{v:.4g}' for v in color),
                      contype='0', conaffinity='0', group='0')
    boxes = walls(field, spec, zone)
    for index, box in enumerate(boxes):
        a = np.asarray(box['a'], dtype=float)
        b = np.asarray(box['b'], dtype=float)
        delta = b - a
        length = float(np.linalg.norm(delta))
        center = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, a[2]])
        ET.SubElement(world, 'geom',
                      name=f'zone_wall_{index:04d}', type='box',
                      pos=' '.join(f'{v:.6g}' for v in center),
                      quat=' '.join(f'{v:.9g}' for v in _quat_from_x(delta / length)),
                      size=f'{length / 2:.6g} {box["thickness_m"] / 2:.6g} {box["height_m"] / 2:.6g}',
                      rgba=' '.join(f'{v:.4g}' for v in box['rgba']),
                      friction='.8 .02 .001', condim='3', group='0')
    return {'boundary_segments': len(boundary(field, spec)), 'zone_walls': len(boxes)}


def write_ue_overlay(path, field, spec, zone):
    """导出 UE 侧叠加层描述（世界坐标 cm），静态内容，不上 UDP。

    ``segments`` 是贴地的边界线（Cube 沿线段拉长、压扁），``walls`` 是箱子的墙段
    （Cube 按长/厚/高三个方向缩放）。两边引用同一批几何，MuJoCo 与 UE 看到的箱子是同一个。
    """
    lines = [{'a': (field.to_world(*a) * 100).tolist(),
              'b': (field.to_world(*b) * 100).tolist(),
              'rgba': list(color),
              'width_cm': spec['line_width_m'] * 100}
             for a, b, color in boundary(field, spec)]
    boxes = [{'a': (field.to_world(*box['a']) * 100).tolist(),
              'b': (field.to_world(*box['b']) * 100).tolist(),
              'rgba': list(box['rgba']),
              'width_cm': box['thickness_m'] * 100,
              'height_cm': box['height_m'] * 100}
             for box in walls(field, spec, zone)]
    Path(path).write_text(json.dumps({'segments': lines, 'walls': boxes}, separators=(',', ':')))
    return {'boundary_segments': len(lines), 'zone_walls': len(boxes)}
