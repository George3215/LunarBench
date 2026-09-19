# Terrain Baking Skill

把 OmniLRS 等方法生成的月面高程（DEM / heightfield）离线烘成 UE 能直接用的
PBR 贴图集：反照率、粗糙度、法线，外加坡度与坑 / 岩掩码。

只做离线烘焙。不含 runtime shader、神经纹理、Diffusion。

## 输入 YAML

一个资产一个 `terrain.yaml`，放在资产目录里。参数全在 YAML，命令行只给路径。
路径一律相对 YAML 所在目录解析，产物也默认写到同一目录——**YAML 所在的目录
就是资产目录**。

```yaml
terrain:
  name: lunalab_lidar01
  heightfield: ../Lunalab/Terrain_lidar01/dem.npy   # .npy/.png/.tif
  resolution_m: 0.01        # 米/像素，标量或 [x, y]
  z_scale: 1.0              # 高程已是米制就留 1.0
  crater_mask:              # 可选，缺省从高程推
    path: ../Lunalab/Terrain_lidar01/mask.npy
    invert: false           # 归一化后高值算「命中」
  rock_mask: null           # 可选，缺省从高程的高频凸起推

material:
  root: ../../materials     # 相对本文件
  base: LunarRegolith8k     # 目录名，贴图按后缀自动识别
  detail: null              # 第二套材质，可选，在岩质/新鲜区混合
  tile_cm: 1000             # 世界空间平铺边长，cm

baking:
  resolution: 2048          # 输出长边像素数
  normal_strength: 1.0      # 法线强度，1.0 是恒等变换
  detail_amount: 0.0        # detail 材质的混合上限
  crater_radius_m: 0.15     # 推导坑体的邻域尺度，取大致等于坑半径
  rock_sigma_m: 0.04        # 推导岩石的高频尺度
  normal_from_height: false # 见下「为什么默认关闭」
  preview: true
```

可复制的模板见 [`terrain.yaml`](terrain.yaml)；跑通的实例见
[`../lunalab_lidar01/`](../lunalab_lidar01/)、[`../southpole_site20/`](../southpole_site20/)，
以及 TASK1 的三个 20 m 地形裁片 [`../task1_center/`](../task1_center/)、
[`../task1_southwest/`](../task1_southwest/)、[`../task1_northeast/`](../task1_northeast/)
（那三份带 `crater_mask: null`，坑与岩都从高程自己推，是「只有 DEM 也能用」的例子）。

## 运行命令

```bash
cd assets/environments/lunar/terrain/terrain_baking
python terrain_bake.py --config ../lunalab_lidar01/terrain.yaml
```

`--config` 缺省取脚本同目录的 `terrain.yaml`；`--out` 可以把产物写到别处
（默认写回 YAML 所在目录）。

依赖只有 `numpy` / `opencv-python` / `pyyaml`。

## 输出内容

```text
lunalab_lidar01/
├── terrain.yaml                        # 配置（产物目录即资产目录，自解释可复现）
├── textures/
│   ├── lunalab_lidar01_diff_4k.png     反照率，8 位 sRGB
│   ├── lunalab_lidar01_nor_gl_4k.png   法线，8 位线性，OpenGL 约定
│   ├── lunalab_lidar01_rough_4k.png    粗糙度，8 位线性，单通道
│   └── preview.png                     2x3 预览（着色/albedo/法线/粗糙度/坑/岩）
└── masks/
    ├── height.png    16 位高程，便于回查
    ├── slope.png     坡度
    ├── crater_mask.png
    └── rock_mask.png
```

贴图文件名沿用仓库既有约定（`_diff` / `_nor_gl` / `_rough`），法线是 OpenGL
约定、绿通道不翻转，UE 侧靠 `UTexture::bFlipGreenChannel` 处理，不生成
`_nor_dx` 派生文件。

各图层怎么来的：

| 输出 | 依据 |
| --- | --- |
| 坑底（变暗） | 高程低于局部邻域均值；给了 `crater_mask` 就用它做距离场门控 |
| 坑缘（变亮） | 高程高于局部邻域均值，取新鲜溅射物 |
| 岩石（变亮、变光） | 高程的高频凸起；自动扣掉坑缘，避免同一处起伏算两遍 |
| 坡度（变暗） | 陡坡挂不住浮尘 |
| 粗糙度 | 岩石更光、坑底更糙，下限夹在 0.35，月壤不该出现镜面高光 |

**不产出位移贴图。** 本仓库的地形物理就是原始高度场、UE 必须渲染同一份几何，
位移贴图会破坏这条不变量（见 `assets/README.md`）。同理，
`normal_from_height` 默认关闭：DEM 的起伏已经由真实几何产生，再烘进法线
就是重复计算。它只对非 UE 用途（比如离线渲染一张带起伏的法线图）才开着有意义。

坑 / 岩 / 坡度这几层是按高程逐像素算的，本来就与地形 **1:1** 对应，不参与平铺。
只有 `material.base` 的贴图会平铺，`tile_cm` 调的只是它。

`tile_cm` 取多大是视觉取舍：取值远小于地形跨度时，基础贴图会重复很多次，
一眼能看出规律（SouthPole 16 km 用 400 m 一次，就是 40x40 次）。可取地形
跨度本身或其整数分之一。运行时打印的 `平铺 x.xx x y.yy 次` 就是这个比值。

平铺用的是 `BORDER_WRAP`，源贴图自身是否无缝决定了接缝明不明显——
`LunarRegolith8k` 是扫描件，边界未必对得上，密铺时值得看一眼 preview。

## 如何从 OmniLRS 接入

OmniLRS 只是数据来源，**这个 Skill 不依赖它**，跑烘焙时不需要装 OmniLRS，
也没有改过它任何代码。接入方式是手工把产物文件拷过来。

| OmniLRS | 位置 | 目标 YAML 字段 |
| --- | --- | --- |
| `TerrainManager` 的 heightfield（MoonYard 路径） | `terrain_conf.resolution` 决定米/像素 | `heightfield` + `resolution_m` |
| LargeScale 的真实 DEM（LOLA 5 mpp） | `SouthPole/*_5mpp_surf/dem.npy` | 同上 |
| 陨石坑掩码 `mask.npy` | 与 DEM 同目录 | `crater_mask` |
| `LunarRegolith8k` 材质 | 已入库在 `assets/environments/lunar/materials/` | `material.base` |

OmniLRS 的 `LunarRegolith8k` 贴图本仓库已经有一份，直接用 `material.base` 指过去
即可，不需要从 OmniLRS 拷贝材质。

掩码极性没有统一约定，用之前先看日志里的「坑区占比」：接近 100% 说明反了，
把 `invert` 打开。工具会在超过 90% 时主动提醒。

## 接到 UE

产物命名沿用了仓库既有的后缀（`_diff_4k` / `_nor_gl_4k` / `_rough_4k`），
但换皮 commandlet 认死的是另一组文件名，需要转一次格式：

| 本工具产出 | commandlet 要的文件名 |
| --- | --- |
| `<name>_diff_4k.png` | `moon_macro_01_diff_4k.jpg` |
| `<name>_nor_gl_4k.png` | `moon_macro_01_nor_gl_4k.exr` |
| `<name>_rough_4k.png` | `moon_macro_01_rough_4k.exr` |

图本身不用改，只是换名字与容器。`tools/build_task1_maps.py` 是这条路的完整
例子：它把产物摆成上面那三个名字（jpg 用 OpenCV 转，EXR 用 ImageMagick
的 `convert -depth 16`——OpenCV 这个构建写了 EXR，`isFormatSupported(CV_8U)`
会直接断言失败），再逐个裁片调一次 commandlet。

源保持 GL 约定不要预翻转：UE 侧会设 `bFlipGreenChannel=true` 自己翻。

拿到 `-TileCm` 的取值参考：它是世界空间平铺边长，取地形跨度本身或其整数分之一。
运行时最后一行会打印可用的命令，形如：

```bash
python tools/build_moon_macro_map.py \
    -TextureRoot=<摆好那三个文件名的目录> \
    -TileCm=<tile_cm>
```

## 校验

```bash
python validate_terrain_bake.py --config ../lunalab_lidar01/terrain.yaml
```

检查产物齐全、尺寸一致、法线是单位向量且为 GL 约定、粗糙度落在月壤合理区间、
高程与掩码对得上。退出码非 0 表示没过。
