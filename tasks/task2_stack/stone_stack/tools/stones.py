"""石头生成/导入工具，统一返回 FlatStone。

load_stones 用于场景装配；import_stones 也可被其他脚本直接调用。
导入网格转换为居中的凸包，保留源坐标轴，显式换算为米；不修改源文件。
此工具只准备初始资产，不注册为 trial 内可调用的造石/重置工具。
"""
import json
import logging
from pathlib import Path

import numpy as np
import trimesh

from ..rock_wall_stones import make_rock_wall_stones
from ..rocks import FlatStone
from ..robots.scene import scale_stones

logger = logging.getLogger(__name__)


def import_stones(paths, unit_scale=1.0, density=2200.0):
    """导入 OBJ/STL/PLY/GLB/GLTF，unit_scale 表示源单位换算到米的倍率。

    仅使用凸包几何与统一颜色，不保留源纹理或凹面碰撞。路径按输入顺序对应石头编号。
    """
    stones = []
    for index, filename in enumerate(paths):
        path = Path(filename)
        mesh = trimesh.load(path, force="mesh").convex_hull
        mesh.apply_scale(unit_scale)
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        length, width, thickness = mesh.extents
        stone = FlatStone(
            name=f"imported_{index + 1:02d}",
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.tolist(),
            rgba=(0.55, 0.52, 0.45, 1.0),
            mass=float(mesh.volume * density),
            length=float(length), width=float(width), thickness=float(thickness),
        )
        stones.append(stone)
        logger.info("导入石头 %s ← %s", stone.name, path.resolve())
    return stones


def load_stones(config, scale=1.0, count=None):
    """按 YAML 选择程序生成或仓库资产；统一执行最终几何/质量缩放。"""
    robot = config["robot"]
    options = config["stones"]
    if count is None:
        count = robot["stone_pool"]
        if count <= 0:
            count = max(robot["stones"], int(np.ceil(sum(robot["courses"]) * 1.6)))
    source = options["source"]
    if source == "procedural":
        stones = make_rock_wall_stones(
            seed=robot["stone_seed"], count=count, style=robot["rock_style"],
            irregularity=robot["rock_irregularity"], subdivisions=robot["rock_subdivisions"],
        )
    elif source == "moonsim":
        from ..moonsim_rocks import make_moonsim_wall_rocks
        stones = make_moonsim_wall_rocks(seed=robot["stone_seed"], count=count, rock_dir=options["directory"])
    elif source == "mesh":
        paths = options["paths"]
        if options["directory"]:
            paths = sorted(Path(options["directory"]).glob(options["pattern"]))
        if len(paths) < count:
            raise ValueError(f"需要 {count} 个石头资产，实际提供 {len(paths)} 个")
        stones = import_stones(paths[:count], options["unit_scale"], options["density"])
    else:
        raise ValueError(f"未知石头来源：{source}")
    logger.info("石头来源=%s，数量=%d，缩放=%g", source, len(stones), scale)
    return scale_stones(stones, scale)


def load_report_stones(path):
    """恢复上游报告选中的十块石头；entry 只用于恢复初始 yaw。"""
    report = json.loads(Path(path).read_text())
    params = report["parameters"]
    generated = make_rock_wall_stones(
        seed=params["stone_seed"], count=params["stones"],
        irregularity=params["rock_irregularity"], subdivisions=params["rock_subdivisions"],
        style=params.get("rock_style", "paper"),
    )
    by_name = {}
    for stone in generated:
        by_name[stone.name] = stone
    entries = report["wall"][:10]
    stones = []
    for entry in entries:
        stones.append(by_name[entry["name"]])
    return stones, entries
