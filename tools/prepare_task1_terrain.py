"""把 TASK1 的三个地形裁片导出成可烘焙的资产目录。

TASK1 的物理地形取自 ue/import_data/Landscape_1_2795x2795.r16，每个实验条件
在它上面取一个 20 m 的滑动窗口（tasks/task1_collect/field.py 的 PATCH_CENTERS）。
这里把那三个窗口各裁一份出来，配上 terrain.yaml，交给
assets/environments/lunar/terrain/terrain_baking/terrain_bake.py 烘 PBR 贴图。

裁片算术与 Field.__init__ 完全一致（grid / crop / 翻转），这样烘出来的贴图
和跑起来的场景是同一块地，不是"大致同一区域"。

源高程只读。运行：

    python tools/prepare_task1_terrain.py
    python tools/prepare_task1_terrain.py --patch center      # 只做一个
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.task1_collect import config as task_config          # noqa: E402
from tasks.task1_collect.field import PATCH_CENTERS, _source   # noqa: E402

TERRAIN_ROOT = ROOT / 'assets/environments/lunar/terrain'
MATERIAL_ROOT = '../../materials'      # 相对每个资产目录

# 与 terrain_baking/terrain.yaml 同口径；裁片是 1 cm/px 的真实月面，坑与岩的
# 推导尺度取厘米级，跟 lunalab_lidar01 一个量级。
BAKING = {
    'resolution': 2048,
    'normal_strength': 1.0,
    'detail_amount': 0.0,
    'crater_radius_m': 0.2,
    'rock_sigma_m': 0.06,
    'normal_from_height': False,
    'preview': True,
}

HEADER = """\
# TASK1 地形裁片 {patch}，由 tools/prepare_task1_terrain.py 生成，请勿手改。
#
# 高程来自 ue/import_data/Landscape_1_2795x2795.r16 上以 {centre} 为中心的
# {size:g} m 窗口，与 field.py 的 PATCH_CENTERS['{patch}'] 是同一块地。
# 烘焙：python terrain_bake.py --config ../task1_{patch}/terrain.yaml
"""


def patch_extent(spec, patch):
    """复算 Field.__init__ 的窗口算术，返回 (grid_index, samples, edge)。"""
    scale = np.asarray(spec['terrain_scale'], dtype=float)
    base = np.asarray(spec['terrain_origin_m'], dtype=float)
    spacing = 0.01 * scale[0]
    size_m = float(spec['size_m'])
    intervals = round(size_m / spacing)
    if intervals % 2 or abs(intervals * spacing - size_m) > 1e-8:
        raise ValueError('field.size_m must contain an even number of source intervals')
    edge = intervals // 2
    grid = tuple(np.rint((np.asarray(PATCH_CENTERS[patch]) - base[:2]) / spacing).astype(int))
    return grid, intervals + 1, edge


def crop(patch, spec):
    grid, samples, edge = patch_extent(spec, patch)
    source = _source() * np.asarray(spec['terrain_scale'], dtype=float)[2]
    iy, ix = grid[1], grid[0]
    if not all(edge <= v <= source.shape[0] - 1 - edge for v in (iy, ix)):
        raise ValueError(f'{patch}: 窗口超出源地形范围；检查 PATCH_CENTERS 与 size_m')
    # [::-1] 与 Field.__init__ 的 crop 一致：MuJoCo 的 Y 向下，渲染侧反射回来。
    # 对贴图本身无所谓（世界位置投影，本来就不逐像素对齐），但保持字面一致，
    # 免得以后有人拿它和 height.bin 比对时对不上。
    return source[iy - edge:iy + edge + 1, ix - edge:ix + edge + 1][::-1], grid, samples


def write_yaml(path, patch, centre, size_m):
    def scalar(value):
        return 'null' if value is None else str(value).lower() if isinstance(value, bool) else value

    text = HEADER.format(patch=patch, centre=list(centre), size=size_m)
    text += f"""
terrain:
  name: task1_{patch}
  heightfield: dem.npy
  resolution_m: 0.01
  z_scale: 1.0
  crater_mask: null
  rock_mask: null

material:
  root: {MATERIAL_ROOT}
  base: LunarRegolith8k
  detail: null
  # 20 m 的裁片，1000 cm 会铺 2x2 次。取 2000 让贴图在任务区里 1:1，
  # 与 UE 侧换皮时传的 -TileCm 一致。
  tile_cm: 2000

baking:
"""
    for key, value in BAKING.items():
        text += f'  {key}: {scalar(value)}\n'
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch', action='append', choices=sorted(PATCH_CENTERS),
                        help='只导出指定裁片；缺省三个都做')
    parser.add_argument('--out-root', type=Path, default=TERRAIN_ROOT)
    args = parser.parse_args()

    patches = args.patch or sorted(PATCH_CENTERS)
    spec = dict(task_config.DEFAULTS['field'])
    size_m = float(spec['size_m'])

    for patch in patches:
        heights, grid, samples = crop(patch, spec)
        out = args.out_root / f'task1_{patch}'
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'dem.npy', heights.astype(np.float32))
        write_yaml(out / 'terrain.yaml', patch, PATCH_CENTERS[patch], size_m)
        relief = float(np.ptp(heights))
        print(f'{patch:<10} {out.relative_to(ROOT)}  '
              f'{samples}x{samples} @ {size_m / (samples - 1) * 100:.2f} cm/px  '
              f'grid={grid}  起伏 {relief:.3f} m')
    return 0


if __name__ == '__main__':
    sys.exit(main())
