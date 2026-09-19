# 既有 UE 导入实现

这里保留原有 `MoonTerrainImporter` C++ 源码，负责固定月面 Landscape、材质、岩石和 Go2 显示导入，以及已有 demo 的视口控制。它是 UE 引擎侧的实现，不是通用插件框架。

源地形：`assets/environments/lunar/terrain/landscape_cropped/`。导入数据：`ue/import_data/`。离线生成工具位于 `tools/prepare_unreal_landscape.py`、`tools/prepare_scene_v2.py`。

运行路径由启动器传入 `LUNARBENCH_WORKSPACE`（MoonSim 目录）和 `LUNARBENCH_PYTHON`。部署到已有 UE 工程、备份和构建方式见 MoonSim 根 README；普通 demo 启动不需要重新导入地图。

导入命令模板（将路径替换为实际值）：

```bash
UnrealEditor-Cmd Project.uproject -run=MoonTerrainImport -unattended -nullrhi -SourceRoot=/absolute/path/to/landscape_cropped -Heightmap=/absolute/path/to/MoonSim/ue/import_data/Landscape_1_2795x2795.r16
```

### commandlet 开关

| 开关 | 作用 |
| --- | --- |
| `-UpgradeScene` | `MoonSceneV2::Run()`：建 V2 贴图/材质/岩石，产出 `/Game/MoonTerrainV2/Maps/MoonTerrain_Full` |
| `-ImportGo2 -FullGo2Render` | `MoonGo2::Import()`：在上面那张图基础上加 Go2 显示部件，另存为 `MoonTerrain_Go2_FullRender` |
| `-MoonMacroMap` | `MoonMacro::Run()`：给地形换皮，另存为新地图。源地图与源材质只读 |
| `-ValidateMacroMap` | 校验换皮地图（几何、transform、高度场逐点、材质、贴图、Go2 部件数） |
| `-ValidateOnly -Heightmap=` | 校验初代地图 |
| 无开关 + `-SourceRoot= -Heightmap=` | 初代地形导入 |

`-MoonMacroMap` 的可选参数：`-SrcMap=`、`-DstMap=`、`-TextureRoot=`、`-TextureDest=`、`-MaterialAsset=`、`-InstanceAsset=`、`-TileCm=`、`-Specular=`、`-NormalScale=`，全部有默认值，通常直接用 `tools/build_moon_macro_map.py` 即可。

**换皮不要加 `-nullrhi`。** 纹理的平台数据编译需要 RHI，无 RHI 时导入会停在 32px 的占位尺寸上（`MIGRATION_REPORT.md` 记过这个坑）。初代导入模板里的 `-nullrhi` 沿用旧例，与换皮无关。

本次整理只更新运行路径和编译插件，没有重建地图或再次验收材质导入。
