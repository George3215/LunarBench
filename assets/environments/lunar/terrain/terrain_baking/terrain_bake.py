#!/usr/bin/env python3
"""月面地形烘焙：OmniLRS DEM → UE 可直接用的 PBR 贴图集。

输入
    heightfield（.npy / .png / .tif），可选 crater_mask、rock_mask
输出
    textures/<name>_diff_4k.png    albedo
    textures/<name>_nor_gl_4k.png  normal，OpenGL 约定（绿通道不翻转）
    textures/<name>_rough_4k.png   roughness，线性
    textures/preview.png           2x3 预览
    masks/height.png  slope.png  crater_mask.png  rock_mask.png

用法
    python terrain_bake.py --config <terrain_dir>/terrain.yaml

约定
    · 参数全部来自 YAML，命令行只给 --config / --out
    · 路径相对 YAML 文件所在目录解析，产物默认写到同一目录
    · 不产出 displacement：本仓库地形物理是原始高度场，UE 渲染同一份几何，
      位移贴图会破坏这条不变量（见 assets/README.md）
    · 法线贴图只搬运材质自身的细节，不烘焙 DEM 梯度——DEM 的起伏已经由
      真实几何产生了，再烘一次就是重复计算
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# 源贴图先降到这个量级再重采样，避免 8K 大图无谓占用内存。
# 实际取 min(SRC_MAX, 烘焙长边 x 平铺次数 x 2)，即按 Nyquist 取够用的精度。
SRC_MAX = 4096

# 材质目录里按后缀识别贴图，按顺序取第一个命中的。优先 png/jpg，其次 exr：
# 部分 OpenCV 构建不带 EXR 支持。
DIFFUSE_HINTS = ("_diff_4k", "_diff", "_basecolor", "_base_color", "_albedo", "_col")
NORMAL_HINTS = ("_nor_gl_4k", "_nor_gl", "_normal", "_nor_dx_4k", "_nor_dx", "_nor")
ROUGH_HINTS = ("_rough_4k", "_rough", "_roughness")
ORM_HINTS = ("_orm", "_arm", "_rma")
AO_HINTS = ("_ao", "_ambientocclusion")
PREFER_EXT = (".png", ".jpg", ".jpeg", ".tif", ".exr")


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def sstep(x, a, b):
    """平滑阶跃：x<=a 为 0，x>=b 为 1，中间三次平滑。"""
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def robust_norm(x, lo=1.0, hi=99.0):
    """按百分位拉伸到 [0,1]，避免个别极值吃掉动态范围。"""
    p_lo, p_hi = np.percentile(x, [lo, hi])
    return np.clip((x - p_lo) / (p_hi - p_lo + 1e-9), 0.0, 1.0)


def blur(x, sigma_px):
    """高斯模糊；sigma 以像素为单位。"""
    if sigma_px <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigma_px, borderType=cv2.BORDER_REFLECT)


def resize_to(a, shape):
    """把 a 重采样到 shape=(h,w)。下采样用 INTER_AREA 抗锯齿，上采样用 CUBIC。"""
    h, w = shape
    if a.shape[:2] == (h, w):
        return a
    shrink = h < a.shape[0] or w < a.shape[1]
    interp = cv2.INTER_AREA if shrink else cv2.INTER_CUBIC
    return cv2.resize(a, (w, h), interpolation=interp)


def target_shape(src_hw, dx, dy, long_side):
    """按物理长宽比（不是像素长宽比）算输出尺寸：DEM 像素通常不是正方形。"""
    h, w = src_hw
    ex, ey = (w - 1) * dx, (h - 1) * dy
    if ex >= ey:
        wb, hb = long_side, max(4, int(round(long_side * ey / ex / 4.0)) * 4)
    else:
        hb, wb = long_side, max(4, int(round(long_side * ex / ey / 4.0)) * 4)
    return hb, wb


def meters_per_px(value):
    """resolution_m 允许写标量或 [x, y]。"""
    if np.isscalar(value):
        return float(value), float(value)
    pair = list(value)
    if len(pair) != 2:
        raise SystemExit(f"resolution_m 应该是标量或 [x, y]，实际是 {value!r}")
    return abs(float(pair[0])), abs(float(pair[1]))


# --------------------------------------------------------------------------
# 读入
# --------------------------------------------------------------------------
def resolve(base_dir, value):
    return (base_dir / str(value)).resolve()


def load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def load_heightfield(path, z_scale):
    """读高程，返回 float32 米制矩阵。"""
    if not path.is_file():
        raise SystemExit(f"heightfield 不存在：{path}")
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise SystemExit(f"读不出高程文件：{path}")
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise SystemExit(f"高程应该是二维，实际是 {raw.shape}")
    return np.ascontiguousarray(raw, dtype=np.float32) * float(z_scale)


def load_scalar_field(path, invert=False):
    """读单通道掩码，归一到 [0,1] float32。

    uint8/uint16 按满量程归一；浮点掩码取值超过 1 时也按满量程归一，
    否则当作已归一的权重。归一化之后的高值一律当作「命中」。

    掩码极性没有统一约定（有的 1 表示坑心、有的 1 表示有效地面），
    用之前先看一眼数据：命中率接近 100% 说明极性和预期相反，
    在 YAML 里把它翻过来（invert: true）。
    """
    if not path.is_file():
        raise SystemExit(f"掩码文件不存在：{path}")
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise SystemExit(f"读不出掩码文件：{path}")
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[..., 0]

    if raw.dtype == np.uint8:
        field = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        field = raw.astype(np.float32) / 65535.0
    else:
        field = raw.astype(np.float32)
        peak = float(field.max())
        if peak > 1.0:
            field = field / peak
    field = np.clip(field, 0.0, 1.0)
    return np.ascontiguousarray(1.0 - field if invert else field)


def mask_spec(section, default_invert=False):
    """把 crater_mask / rock_mask 的写法统一成 (path, invert) 或 None。

    支持三种写法：
        rock_mask: null
        rock_mask: ../path/mask.npy
        rock_mask: {path: ../path/mask.npy, invert: true}
    """
    if not section:
        return None
    if isinstance(section, str):
        return section, default_invert
    if isinstance(section, dict):
        if "path" not in section:
            raise SystemExit(f"掩码配置缺少 path：{section}")
        return section["path"], bool(section.get("invert", default_invert))
    raise SystemExit(f"无法识别的掩码配置：{section!r}")


def find_texture(directory, hints):
    """在目录里按后缀找一张贴图；优先 png/jpg，其次 exr。"""
    if not directory.is_dir():
        return None
    files = [p for p in directory.iterdir() if p.is_file()]
    for hint in hints:
        for ext in PREFER_EXT:
            for path in files:
                if path.stem.lower().endswith(hint) and path.suffix.lower() == ext:
                    return path
    return None


def load_texture(path, channel=None, max_side=SRC_MAX):
    """读贴图 → float32 RGB [0,1]；channel 非空时返回单通道二维。

    超大贴图读完立刻降采样，避免把 8K 原图一路带到底。
    """
    if path is None:
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"  ! 读不出贴图，跳过：{path}")
        return None
    if raw.dtype == np.uint16:
        raw = raw.astype(np.float32) / 65535.0
    elif raw.dtype == np.uint8:
        raw = raw.astype(np.float32) / 255.0
    else:
        raw = raw.astype(np.float32)
    if raw.ndim == 3 and raw.shape[2] >= 4:
        raw = raw[..., :3]
    # 取通道要放在缩放之前：cv2.resize 对 HxWx1 会退化成二维
    if channel is not None and raw.ndim == 3:
        raw = raw[..., min(channel, raw.shape[2] - 1)]
    if max(raw.shape[:2]) > max_side:
        # 等比缩放，别把非方形贴图压扁
        scale = max_side / max(raw.shape[:2])
        raw = cv2.resize(raw, (max(1, round(raw.shape[1] * scale)),
                               max(1, round(raw.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    if channel is not None:
        return np.ascontiguousarray(raw)
    if raw.ndim == 2:
        raw = np.repeat(raw[..., None], 3, axis=2)
    return np.ascontiguousarray(raw[..., ::-1])       # OpenCV 读进来是 BGR


def to_gray(a):
    """标量场贴图（粗糙度、AO）时常存成三通道灰度，取平均压成一维。"""
    if a is None:
        return None
    return np.ascontiguousarray(a.mean(axis=2) if a.ndim == 3 else a)


def load_material(root, name, max_side=SRC_MAX):
    """按名字装一套 PBR 贴图。缺哪张，对应项就是 None。"""
    material_dir = (root / name).resolve()
    # 贴图可能直接放在材质目录里，也可能在该目录的同名子目录里
    search_dirs = [material_dir]
    if (material_dir / name).is_dir():
        search_dirs.insert(0, material_dir / name)

    def pick(hints):
        for directory in search_dirs:
            hit = find_texture(directory, hints)
            if hit:
                return hit
        return None

    diffuse_p = pick(DIFFUSE_HINTS)
    normal_p = pick(NORMAL_HINTS)
    rough_p = pick(ROUGH_HINTS)
    ao_p = pick(AO_HINTS)

    # ORM 打包：R=AO G=Roughness B=Metallic，缺单张时从这里取通道
    rough_channel = None
    if rough_p is None and (orm_p := pick(ORM_HINTS)) is not None:
        rough_p, rough_channel = orm_p, 1
    if ao_p is None:
        ao_p = pick(ORM_HINTS)

    material = {
        "name": name,
        "dir": material_dir,
        "diffuse": load_texture(diffuse_p, max_side=max_side),
        "normal": load_texture(normal_p, max_side=max_side),
        "roughness": to_gray(load_texture(rough_p, channel=rough_channel, max_side=max_side)),
        "ao": to_gray(load_texture(ao_p, channel=0, max_side=max_side)),
    }
    material["paths"] = {
        key: (path.relative_to(root) if path.is_relative_to(root) else path)
        for key, path in (("diffuse", diffuse_p), ("normal", normal_p),
                          ("roughness", rough_p), ("ao", ao_p)) if path
    }
    return material


EMPTY_MATERIAL = {"name": None, "diffuse": None, "normal": None,
                  "roughness": None, "ao": None, "paths": {}}


# --------------------------------------------------------------------------
# 平铺采样
# --------------------------------------------------------------------------
def mip_for_source(tex, repeats, bake_long):
    """按输出需要的精度降采样源贴图，省内存也避免无谓的重采样开销。

    平铺次数越多，每个循环占的输出像素越少，源贴图就该越小：
    每个循环在输出里占 bake_long / repeats 像素，按 Nyquist 取两倍即可。
    """
    if tex is None:
        return None
    want = int(min(SRC_MAX, max(64, bake_long / max(repeats) * 2.0)))
    long_src = max(tex.shape[:2])
    if long_src <= want:
        return tex
    scale = want / long_src
    return resize_to(tex, (max(1, int(tex.shape[0] * scale)),
                           max(1, int(tex.shape[1] * scale))))


def tile_sample(tex, shape, repeats):
    """把 tex 按 repeats=(rx,ry) 次平铺重采样到 shape。世界空间平铺，无缝。"""
    if tex is None:
        return None
    h, w = shape
    rx, ry = repeats
    u = (np.arange(w, dtype=np.float32) + 0.5) / w * rx
    v = (np.arange(h, dtype=np.float32) + 0.5) / h * ry
    # texel 中心对齐（-0.5），否则平铺接缝处会错开半像素
    map_x = np.tile(u[None, :], (h, 1)) * tex.shape[1] - 0.5
    map_y = np.tile(v[:, None], (1, w)) * tex.shape[0] - 0.5
    out = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return out if out.ndim == 3 else out[..., None]


def tile_material(material, shape, repeats):
    """就地把一套材质的所有通道铺到目标尺寸，后续烘焙直接取用。"""
    for key in ("diffuse", "normal", "roughness", "ao"):
        value = material.get(key)
        if value is None:
            continue
        small = mip_for_source(value, repeats, max(shape))
        material[key] = tile_sample(small, shape, repeats)
        if value.ndim == 2:                       # 单通道贴图保持二维
            material[key] = material[key][..., 0]


# --------------------------------------------------------------------------
# 派生图层
# --------------------------------------------------------------------------
def compute_slope(height, dx, dy):
    """坡度，单位度。dx/dy 是米/像素。"""
    gy, gx = np.gradient(height, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def derive_crater(height, radius_m, dx, dy, zone=None):
    """陨石坑影响场，返回 (坑底, 坑缘)，各自 [0,1]。

    判据是「局部凹陷」：高程低于周围邻域均值的地方就是坑体。
    取原图与高斯平滑版的差，负的是坑底、正的是坑缘。
    zone 非空时（外部给了 crater_mask）用它做门控，抑制地形噪声造成的误检。
    """
    sigma = max(1.0, radius_m / max((dx + dy) * 0.5, 1e-6) * 0.5)
    local = blur(height, sigma)
    # 再低通一次：高程里的散斑会原封不动地留在差值里，不做抑制的话
    # 阈值一放低就满屏噪点
    smooth = max(1.0, sigma * 0.5)
    floor = sstep(robust_norm(blur(np.clip(local - height, 0.0, None), smooth)), 0.40, 0.80)
    rim = sstep(robust_norm(blur(np.clip(height - local, 0.0, None), smooth)), 0.50, 0.92)

    if zone is not None:
        # OmniLRS 的 mask 往往只在坑心标一个点，直接高斯模糊会把它抹平。
        # 改用距离场：坑心权重 1，到坑半径处线性的衰减到 0。
        binary = (zone > 0.5).astype(np.uint8)
        if binary.any():
            # distanceTransform 量的是「到最近的 0 的距离」，所以要传取反后的图，
            # 这样坑心是 0、向外递增，才有从坑心衰减的权重
            dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
            gate = blur(np.clip(1.0 - dist / (sigma * 2.0), 0.0, 1.0), 2.0)
            floor, rim = floor * gate, rim * gate
    return floor.astype(np.float32), rim.astype(np.float32)


def derive_rock(height, sigma_m, dx, dy):
    """岩石 / 碎块影响场，[0,1]。取高频凸起部分，不取凹处。"""
    sigma = max(1.0, sigma_m / max((dx + dy) * 0.5, 1e-6))
    residual = np.clip(height - blur(height, sigma * 2.0), 0.0, None)
    return sstep(robust_norm(residual), 0.25, 0.80).astype(np.float32)


def height_to_normal(height, dx, dy, flip_y=False):
    """从 DEM 梯度算切空间法线。仅用于 normal_from_height 与预览渲染。"""
    gy, gx = np.gradient(height, dy, dx)
    n = np.dstack([-gx, gy if flip_y else -gy, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    return n.astype(np.float32)


# --------------------------------------------------------------------------
# 烘焙
# --------------------------------------------------------------------------
def macro_weight(layer, cfg):
    """新鲜 / 岩质区域的位置权重，用来在基础材质与第二套材质之间过渡。"""
    return np.clip(0.65 * layer["rock"] + 0.55 * layer["crater_rim"], 0, 1) \
        * float(cfg["detail_amount"])


def bake_albedo(base, detail, layer, cfg):
    """反照率 = 平铺的基础材质，再按坑底/坑缘/岩石/坡度做乘性调制。

    物理依据：坑底是成熟月壤（更暗），坑缘是新鲜溅射物（更亮），
    岩石是未风化的新鲜面（更亮），陡坡挂不住浮尘（更暗）。
    """
    alb = base["diffuse"].copy() if base["diffuse"] is not None else \
        np.full((*layer["slope"].shape, 3), 0.5, np.float32)

    weight = macro_weight(layer, cfg)
    if detail["diffuse"] is not None and weight.any():
        alb = alb * (1.0 - weight[..., None]) + detail["diffuse"] * weight[..., None]

    gain = (1.0 - 0.30 * layer["crater"]
            + 0.35 * layer["crater_rim"]
            + 0.25 * layer["rock"]
            - 0.15 * layer["slope_n"])
    alb = alb * gain[..., None]

    if base["ao"] is not None:
        alb = alb * (1.0 - 0.5 * (1.0 - base["ao"]))[..., None]
    return np.clip(alb, 0.0, 1.0)


def bake_roughness(base, detail, layer, cfg):
    """粗糙度：岩石更光（新鲜面），坑底更糙（细粒月壤）。"""
    rgh = base["roughness"].copy() if base["roughness"] is not None else \
        np.full_like(layer["crater"], 0.85)

    weight = macro_weight(layer, cfg)
    if detail["roughness"] is not None and weight.any():
        rgh = rgh * (1.0 - weight) + detail["roughness"] * weight

    rgh = rgh * (1.0 - 0.25 * layer["rock"]) * (1.0 + 0.12 * layer["crater"])
    # 月壤整体很粗糙，不要出现镜面高光
    return np.clip(rgh, 0.35, 1.0)


def bake_normal(base, detail, layer, cfg):
    """法线 = 平铺的材质法线（GL 约定），按 normal_strength 缩放 xy。

    可选叠加 DEM 梯度，默认关闭：真实几何已经提供了起伏，
    再烘进法线就是重复计算，还会破坏「UE 渲染与物理同一份几何」。
    """
    if base["normal"] is not None:
        nrm = base["normal"].copy() * 2.0 - 1.0
    else:
        nrm = np.zeros((*layer["crater"].shape, 3), np.float32)
        nrm[..., 2] = 1.0

    weight = macro_weight(layer, cfg)
    if detail["normal"] is not None and weight.any():
        nrm = nrm * (1.0 - weight[..., None]) \
            + (detail["normal"] * 2.0 - 1.0) * weight[..., None]

    if cfg["normal_from_height"]:
        nrm[..., :2] += height_to_normal(layer["height"], cfg["dx"], cfg["dy"])[..., :2] * 2.0

    nrm[..., :2] *= float(cfg["normal_strength"])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-9
    return np.clip(nrm * 0.5 + 0.5, 0.0, 1.0)


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------
def write_png(path, array, bits=8):
    """array 是 float [0,1] 的 HxW 或 HxWx3，或已归一的整数矩阵。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype in (np.uint8, np.uint16):
        data = array
    elif bits == 16:
        data = (np.clip(array, 0, 1).astype(np.float64) * 65535.0 + 0.5).astype(np.uint16)
    else:
        data = (np.clip(array, 0, 1).astype(np.float64) * 255.0 + 0.5).astype(np.uint8)
    if data.ndim == 3:
        data = data[..., ::-1]              # RGB → BGR，OpenCV 的约定
    if not cv2.imwrite(str(path), data):
        raise SystemExit(f"写不出 PNG：{path}")
    return path


def shade_preview(albedo, height, dx, dy):
    """用固定低角度太阳做一次 Lambert 着色，只为预览好看。"""
    n = height_to_normal(height, dx, dy, flip_y=True)
    elev, azim = np.radians(15.0), np.radians(135.0)
    light = np.array([np.cos(elev) * np.cos(azim),
                      np.cos(elev) * np.sin(azim),
                      np.sin(elev)], np.float32)
    lam = np.clip((n * light).sum(axis=2), 0.0, 1.0)
    return np.clip(albedo * (0.12 + 0.88 * lam[..., None]), 0, 1)


def make_preview(path, panels, thumb=420):
    """2x3 面板预览。panels 是 (标题, HxWx3 float [0,1]) 列表。"""
    tiles = []
    for title, image in panels:
        h, w = image.shape[:2]
        scale = thumb / max(h, w)
        small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        canvas = np.zeros((thumb, thumb, 3), np.float32)
        y0, x0 = (thumb - small.shape[0]) // 2, (thumb - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        strip = np.full((26, thumb, 3), 0.08, np.float32)
        cv2.putText(strip, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0.95, 0.95, 0.95), 1, cv2.LINE_AA)
        tiles.append(np.vstack([strip, canvas]))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    return write_png(path, np.vstack([np.hstack(tiles[i:i + 3])
                                      for i in range(0, len(tiles), 3)]))


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build(cfg_yaml, out_dir=None):
    cfg_path = Path(cfg_yaml).resolve()
    base_dir = cfg_path.parent
    doc = load_yaml(cfg_path)
    out_dir = base_dir if out_dir is None else Path(out_dir).resolve()

    terrain = doc.get("terrain") or {}
    material_cfg = doc.get("material") or {}
    baking = doc.get("baking") or {}

    if "heightfield" not in terrain:
        raise SystemExit(f"{cfg_path} 里缺少 terrain.heightfield")
    if "resolution_m" not in terrain:
        raise SystemExit(f"{cfg_path} 里缺少 terrain.resolution_m（米/像素）")

    name = terrain.get("name") or out_dir.name
    print(f"[terrain] {name}")

    # ---- 高程 ----
    height = load_heightfield(resolve(base_dir, terrain["heightfield"]),
                              terrain.get("z_scale", 1.0))
    dx, dy = meters_per_px(terrain["resolution_m"])

    # 物理跨度以源数据为准；重采样之后每像素代表的米数随之变化
    extent_x, extent_y = (height.shape[1] - 1) * dx, (height.shape[0] - 1) * dy
    src_hw = height.shape

    long_side = int(baking.get("resolution", 2048))
    out_hw = target_shape(src_hw, dx, dy, long_side)
    height = resize_to(height, out_hw).astype(np.float32)
    dx, dy = extent_x / (out_hw[1] - 1), extent_y / (out_hw[0] - 1)

    print(f"  高程      {src_hw[1]}x{src_hw[0]} px 源 -> {height.shape[1]}x{height.shape[0]} px 烘焙"
          f"  {dx:g} m/px  范围 {height.min():.3f}..{height.max():.3f} m")
    print(f"  世界范围  {extent_x:.2f} x {extent_y:.2f} m")

    # ---- 掩码：给了就用，没给就从高程推导 ----
    crater_spec = mask_spec(terrain.get("crater_mask"))
    rock_spec = mask_spec(terrain.get("rock_mask"))

    crater_zone = None
    if crater_spec:
        crater_zone = resize_to(load_scalar_field(resolve(base_dir, crater_spec[0]),
                                                  crater_spec[1]), out_hw)
        hit = float(crater_zone.mean())
        print(f"  crater_mask  {crater_spec[0]}  invert={crater_spec[1]}  "
              f"坑区占比 {hit:.2%}")
        if hit > 0.9:
            print("  ! 掩码几乎全命中，极性和预期多半相反——检查后改用 invert: true")
    else:
        print("  crater_mask  未提供，从高程推导")

    crater, crater_rim = derive_crater(height, float(baking.get("crater_radius_m", 1.2)),
                                       dx, dy, zone=crater_zone)

    if rock_spec:
        rock = resize_to(load_scalar_field(resolve(base_dir, rock_spec[0]),
                                           rock_spec[1]), out_hw)
        print(f"  rock_mask    {rock_spec[0]}  岩区占比 {float(rock.mean()):.2%}")
    else:
        rock = derive_rock(height, float(baking.get("rock_sigma_m", 0.06)), dx, dy)
        # 推导出来的高频凸起里混着坑缘。坑缘已经有自己的一层，
        # 这里减掉，免得同一处起伏被算两遍。外部给的掩码不做这个处理。
        rock = np.clip(rock - crater_rim, 0.0, 1.0).astype(np.float32)
        print(f"  rock_mask    未提供，从高程推导  覆盖 {float((rock > 0.5).mean()):.1%}")

    slope = compute_slope(height, dx, dy)
    layers = {
        "height": height, "slope": slope,
        "slope_n": np.clip(slope / 60.0, 0.0, 1.0),
        "crater": crater, "crater_rim": crater_rim, "rock": rock,
    }

    # ---- 材质 ----
    tile_m = float(material_cfg.get("tile_cm", 1000.0)) / 100.0
    repeats = (max(extent_x / tile_m, 0.25), max(extent_y / tile_m, 0.25))
    # 读源贴图时就直接降到需要的精度，平铺次数多的时候能省下大量内存
    max_side = int(min(SRC_MAX, max(256, long_side / max(repeats) * 2.0)))

    tex_root = resolve(base_dir, material_cfg.get("root", "../../materials"))
    base_mat = load_material(tex_root, material_cfg.get("base", "LunarRegolith8k"), max_side)
    detail_name = material_cfg.get("detail")
    detail_mat = load_material(tex_root, detail_name, max_side) if detail_name else \
        dict(EMPTY_MATERIAL)

    print(f"  材质      base={base_mat['name']}  diff={base_mat['paths'].get('diffuse', '-')}")
    if detail_name:
        print(f"            detail={detail_name}  diff={detail_mat['paths'].get('diffuse', '-')}")
    print(f"  平铺      tile={tile_m:g} m  ->  {repeats[0]:.2f} x {repeats[1]:.2f} 次")

    bake_cfg = {
        "detail_amount": float(baking.get("detail_amount", 0.6)),
        "normal_strength": float(baking.get("normal_strength", 1.0)),
        "normal_from_height": bool(baking.get("normal_from_height", False)),
        "dx": dx, "dy": dy,
    }
    if bake_cfg["normal_from_height"]:
        print("  ! normal_from_height=true：DEM 梯度会烘进法线，与真实几何"
              "重复计算，只适合非 UE 用途")

    tile_material(base_mat, out_hw, repeats)
    tile_material(detail_mat, out_hw, repeats)

    albedo = bake_albedo(base_mat, detail_mat, layers, bake_cfg)
    roughness = bake_roughness(base_mat, detail_mat, layers, bake_cfg)
    normal = bake_normal(base_mat, detail_mat, layers, bake_cfg)

    # ---- 导出 ----
    tex_dir, mask_dir = out_dir / "textures", out_dir / "masks"
    written = [
        write_png(tex_dir / f"{name}_diff_4k.png", albedo),
        write_png(tex_dir / f"{name}_nor_gl_4k.png", normal),
        write_png(tex_dir / f"{name}_rough_4k.png", roughness),
        write_png(mask_dir / "height.png", robust_norm(height), bits=16),
        write_png(mask_dir / "slope.png", np.clip(slope / 90.0, 0, 1)),
        write_png(mask_dir / "crater_mask.png", crater),
        write_png(mask_dir / "rock_mask.png", rock),
    ]

    if baking.get("preview", True):
        written.append(make_preview(tex_dir / "preview.png", [
            ("preview (albedo x Lambert)", shade_preview(albedo, height, dx, dy)),
            ("albedo", albedo),
            ("normal (GL)", normal * 0.5 + 0.5),
            ("roughness", np.repeat(roughness[..., None], 3, 2)),
            ("crater mask", np.repeat(crater[..., None], 3, 2)),
            ("rock mask", np.repeat(rock[..., None], 3, 2)),
        ]))

    # 配置随产物落到资产目录，资产自解释、可复现
    if out_dir != base_dir:
        (out_dir / "terrain.yaml").write_text(cfg_path.read_text())

    print(f"\n输出目录 {out_dir}")
    for path in written:
        print(f"  {path.relative_to(out_dir)}  {path.stat().st_size / 1e6:.2f} MB")
    print(f"\nUE 换皮：tools/build_moon_macro_map.py 加 "
          f"-TextureRoot={tex_dir} -TileCm={tile_m * 100:g}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="从 OmniLRS DEM 烘焙月面 PBR 贴图集")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "terrain.yaml",
                        help="terrain.yaml 路径；默认取脚本同目录的 terrain.yaml")
    parser.add_argument("--out", type=Path, default=None,
                        help="产物目录；默认写到 YAML 同级目录")
    args = parser.parse_args()
    build(args.config, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
