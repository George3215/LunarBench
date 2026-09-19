#!/usr/bin/env python3
"""校验 terrain_bake.py 的产物。

跟着仓库里 build_* / validate_* 成对的惯例。只读产物，不改任何东西。

    python validate_terrain_bake.py --config ../lunalab_lidar01/terrain.yaml

退出码 0 表示全部通过，1 表示有检查项没过。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SUFFIXES = ("diff_4k", "nor_gl_4k", "rough_4k")
MASKS = ("height", "slope", "crater_mask", "rock_mask")


def load_yaml(path):
    import yaml
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


class Report:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def check(self, ok, message):
        if ok:
            self.passes += 1
            print(f"  ok    {message}")
        else:
            self.failures.append(message)
            print(f"  FAIL  {message}")
        return ok


def read(path):
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def source_shape(path):
    """只要 DEM 的形状，不把整个矩阵读进来。"""
    if not path.is_file():
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r").shape[:2]
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    return None if image is None else tuple(s * 8 for s in image.shape[:2])


def tile_repeats(doc, asset_dir, out_hw):
    """按 terrain_bake.py 的口径复算平铺次数，用于对齐源材质。"""
    terrain = doc.get("terrain") or {}
    if "heightfield" not in terrain or "resolution_m" not in terrain:
        return None
    src = source_shape(asset_dir / str(terrain["heightfield"]))
    if src is None:
        return None
    res = terrain["resolution_m"]
    dx, dy = (float(res), float(res)) if np.isscalar(res) else \
        (abs(float(res[0])), abs(float(res[1])))
    extent_x, extent_y = (src[1] - 1) * dx, (src[0] - 1) * dy
    tile_m = float((doc.get("material") or {}).get("tile_cm", 1000.0)) / 100.0
    return max(extent_x / tile_m, 0.25), max(extent_y / tile_m, 0.25)


def convention_correlation(doc, asset_dir, baked_normal, repeats, side=512):
    """把源材质法线按同样的平铺参数采样一遍，和烘出来的法线做相关。

    绿通道翻转在单张图上没有统计特征可查（源材质 G 均值本来就贴着 0.5），
    只有和源逐点比才能确定没被翻过。顺带也验证了平铺参数没算错。
    """
    material = doc.get("material") or {}
    root = (asset_dir / str(material.get("root", "../../materials"))).resolve()
    name = str(material.get("base", "LunarRegolith8k"))
    candidates = [root / name / f"{name}_nor_gl.png", root / name / f"{name}_nor_gl_4k.png"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None

    # 降到小图再比，精度足够且不用碰 8K 原图
    src = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if src is None:
        return None
    # 按平铺次数降到够用的精度：平铺越密，源材质每个循环占的像素越少。
    # 不降的话采样步长远超源图一个纹素，相关性会被走样淹没。
    want = max(8, int(round(side / max(repeats) * 2.0)))
    if max(src.shape[:2]) > want:
        src = cv2.resize(src, (want, want), interpolation=cv2.INTER_AREA)

    small = cv2.resize(baked_normal, (side, side), interpolation=cv2.INTER_AREA)
    u = (np.arange(side, dtype=np.float32) + 0.5) / side
    map_x = np.tile((u * repeats[0])[None, :], (side, 1)) * src.shape[1] - 0.5
    map_y = np.tile((u * repeats[1])[:, None], (1, side)) * src.shape[0] - 0.5
    sampled = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    baked_g = small[..., 1].astype(np.float32) - 128.0
    source_g = sampled[..., 1].astype(np.float32) - 128.0
    if baked_g.std() < 1e-3 or source_g.std() < 1e-3:
        return None
    return float(np.corrcoef(baked_g.ravel(), source_g.ravel())[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        print(f"找不到配置：{cfg_path}", file=sys.stderr)
        return 1
    doc = load_yaml(cfg_path)
    asset_dir = cfg_path.parent
    name = (doc.get("terrain") or {}).get("name") or asset_dir.name

    print(f"[validate] {name}  {asset_dir}")
    report = Report()

    # ---- 配置还在产物目录里 ----
    report.check((asset_dir / "terrain.yaml").is_file(), "资产目录里有 terrain.yaml")

    # ---- 三件套贴图 ----
    texture_dir = asset_dir / "textures"
    textures = {}
    for suffix in SUFFIXES:
        path = texture_dir / f"{name}_{suffix}.png"
        image = read(path)
        if report.check(image is not None, f"贴图存在且可读：textures/{path.name}"):
            textures[suffix] = image

    if len(textures) == len(SUFFIXES):
        shapes = {image.shape[:2] for image in textures.values()}
        report.check(len(shapes) == 1, f"三张贴图尺寸一致 {shapes.pop()}")

        diff = textures["diff_4k"]
        report.check(diff.ndim == 3 and diff.shape[2] == 3, "albedo 是三通道")
        report.check(float(diff.std()) > 1.0, f"albedo 不是常数（std={diff.std():.2f}）")

        rough = textures["rough_4k"]
        report.check(rough.ndim == 2, "roughness 是单通道")
        lo, hi = float(rough.min()) / 255.0, float(rough.max()) / 255.0
        report.check(lo >= 0.34 and hi <= 1.001,
                     f"roughness 落在月壤区间 [0.35, 1.0]（实测 {lo:.2f}..{hi:.2f}）")

        # 法线：解码成向量，验单位长度与 GL 约定
        nrm = textures["nor_gl_4k"].astype(np.float32) / 255.0 * 2.0 - 1.0
        nrm = nrm[..., ::-1]                      # BGR -> RGB
        length = np.linalg.norm(nrm, axis=2)
        report.check(abs(float(length.mean()) - 1.0) < 0.02,
                     f"法线是单位向量（平均模长 {length.mean():.4f}）")
        # Z 必须为正，否则法线朝背面
        report.check(float(nrm[..., 2].min()) > 0.0,
                     f"法线 Z 分量为正（最小 {nrm[..., 2].min():.3f}）")
        report.check(np.isfinite(nrm).all(), "法线没有 NaN / Inf")
        report.check(float(nrm[..., :2].std()) > 0.005,
                     f"法线 XY 有信号（std={nrm[..., :2].std():.4f}）")

        # 绿通道有没有被翻，只能跟源材质逐点比
        repeats = tile_repeats(doc, asset_dir, diff.shape[:2])
        if repeats is None:
            print("  skip  法线朝向：读不到源 DEM，无法复算平铺参数")
        else:
            r = convention_correlation(doc, asset_dir, textures["nor_gl_4k"], repeats)
            if r is None:
                print("  skip  法线朝向：读不到源材质法线")
            else:
                report.check(r > 0.5,
                             f"法线与源材质同向、绿通道未翻转（G 通道相关 r={r:+.3f}）")

    # ---- 掩码 ----
    mask_dir = asset_dir / "masks"
    mask_images = {}
    for mask in MASKS:
        path = mask_dir / f"{mask}.png"
        image = read(path)
        if report.check(image is not None, f"掩码存在且可读：masks/{path.name}"):
            mask_images[mask] = image

    if len(mask_images) == len(MASKS):
        shapes = {image.shape[:2] for image in mask_images.values()}
        report.check(len(shapes) == 1, f"四个掩码尺寸一致 {shapes.pop()}")
        if textures:
            report.check(mask_images["height"].shape[:2] == textures["diff_4k"].shape[:2],
                         "掩码与贴图尺寸一致")
        height = mask_images["height"]
        report.check(height.dtype == np.uint16, f"高程是 16 位（实际 {height.dtype}）")
        report.check(float(height.std()) > 1.0, f"高程不是常数（std={height.std():.1f}）")
        for mask in ("crater_mask", "rock_mask"):
            image = mask_images[mask]
            frac = float((image > 127).mean())
            report.check(frac < 0.95,
                         f"{mask} 没有全命中（占比 {frac:.1%}）")

    # ---- 预览 ----
    report.check((texture_dir / "preview.png").is_file(), "预览图存在")

    print(f"\n{report.passes} 项通过，{len(report.failures)} 项失败")
    for message in report.failures:
        print(f"  - {message}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
