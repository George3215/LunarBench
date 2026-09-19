# 完整 USD 场景迁移记录

## 入口

- UE 工程：`/home/lry/文档/Unreal Projects/Moon/Moon.uproject`
- 新地图：`/Game/MoonTerrainV2/Maps/MoonTerrain_Full`
- 旧地图保留：`/Game/MoonTerrain/Maps/MoonTerrain`
- 地表材质：`/Game/MoonTerrainV2/Materials/M_MoonLandscape_Full`
- 岩石材质：`/Game/MoonTerrainV2/Materials/M_MoonRock_Full`

## 修复内容

1. 从完整根 USD 读取父层变换，而不只读取 Landscape 原型。源场景本身为 X/Y 50、Z 25 的非均匀缩放；迁移保留此变换，不额外扭曲源场景比例。最终地形宽 1397 米，2795×2795 个高度采样，水平间隔 0.5 米；未减小高度图尺寸。旧版本 27.94 米是未应用父层变换的原型尺寸。
2. 从真正有效的 `Props/Landscape_1_Foliage.usd` 提取 54,753 个实例，不依据旁边的空 PointInstancer 文件推断岩石数量。两个岩石原型保留完整三角形，分别为 10,903 和 8,648 三角形。
3. 岩石为独立 StaticMeshActor（不是只有一个不可逐个选取的合并网格），放在 Outliner 的 `MoonRocks` 文件夹，名称 `MoonRock_000000` 至 `MoonRock_054752`，共享两个网格资产以控制存储。
4. 按 MDL 有效参数重建三种土壤的混合、近远景采样、细节法线、两层颜色变化和岩石材质。源 Soil_N 的 RG 为法线、B 为 specular、A 为混合高度；使用线性 RGBA BC7 保存这些通道，不再用会丢 B/A 的 BC5 普通法线压缩。颜色贴图使用 sRGB；所有贴图保留源分辨率上限，不设置额外 LODBias，禁用流送降级。BC7 本身仍为有损压缩，不能称为像素无损。

## 位置与精度证据

2026-09-15 16:52:28 编辑器实际资源回读通过：7 张 8192×8192、2 张 4096×4096，与源 PNG 尺寸逐张一致。16:53 的实际 Lit 视口已确认地表纹理细节和独立岩石可见，截图 `ue_viewport_verified.png`。这证明材质实际参与渲染，不代表与 Isaac Sim 像素级一致。

- `rock_positions_ue.csv`：从实际 UE Actor 导出的编号、网格类型、世界位置（厘米）、四元数和缩放。米制坐标将 x_cm/y_cm/z_cm 除以 100。这是 UE 坐标，Y 轴已按 USD→UE 转换。
- `rocks.json`：源 USD 换算后的完整目标 4×4 矩阵与稳定实例 ID。
- `ue_reload_validation.json`：重新打开已保存地图后，54,753 个 Actor 的位置逐一核对，最大位置误差 0 cm；484 个 LandscapeComponent，scale=(50,50,25)。只有 `texture_resolution_verified=true` 才代表异步编译后的纹理尺寸全部验证完毕；初始 32px 是编译占位，不能据此宣称完成。
- 高度图沿用此前逐点回读通过的 2795×2795 R16，本次仅更改 Landscape 的变换和材质，未重采样高度数据。

## 已知边界

- 普通 UE Actor 的平移/旋转/缩放不能表示源父层非均匀缩放与岩石旋转组合产生的剪切。位置准确，但当前岩石形状存在近似：逐原型顶点对比，单块岩石最大偏差的中位数 1.861 cm、95 分位 5.408 cm、全场最坏 22.059 cm（实例 2826）。详见 `rock_transform_audit.json`；不要将当前结果表述为所有顶点变换完全一致。
- 材质为 MDL 活跃逻辑的 UE 重建，不是 Isaac Sim 渲染器移植；不同照明、色调映射、切线空间实现会影响最终外观，未作同相机同灯光像素级一致性验收。
- 54,753 个独立 Actor 首次打开和加载较慢；没有为了提高帧率而合并实例或删减岩石。地形仍使用 UE 正常的屏幕距离 LOD，保留完整源数据不代表远处每帧强制绘制最高 LOD。

## 可复现入口

- `prepare_scene_v2.py`：从 USD 导出网格、材质参数和实例矩阵。
- 插件命令：`-run=MoonTerrainImport -UpgradeScene`（会重建 V2 资产，不用于日常打开）。
