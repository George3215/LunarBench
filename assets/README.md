# Asset — 月面仿真资产库

本目录存放月面机器人仿真所需的全部三维资产：月面地形、岩石、材质、天空、场景、机器人、物体与传感器模型。

资产按**类别**组织，四个顶层分类：

| 分类 | 内容 | 体积 | 文件数 |
| --- | --- | ---: | ---: |
| `environments/` | 月面地形、岩石、材质、天空、场景 | 23 G | 2880 |
| `robots/` | 机器人本体（移动、机械臂、着陆器、航天器） | 703 M | 300 |
| `objects/` | 可交互物体（岩石样品、工具、构件、载荷） | 263 M | 188 |
| `sensors/` | 独立传感器模型（相机、激光雷达） | 4.6 M | 4 |
| **合计** | | **24 G** | **3373** |

上表由 `du -sh` 与 `find <分类> -type f | wc -l` 实测（2026-09-21 复核，含本轮新增的
`robots/robotiq_2f85/`）；下面各小节表格里的细分数来自更早的一次统计，未随本次一并复核。

---

## 目录结构

```text
Asset/
├── environments/
│   └── lunar/
│       ├── terrain/          月面地形
│       │   ├── landscape_cropped/   大面积月面地形（含材质与附属物体）
│       │   ├── Lunalab/             室内月面试验场地形（激光扫描）
│       │   ├── Lunaryard/           月面试验场地形（占位，待填充）
│       │   ├── SouthPole/           月面南极 DEM 高程瓦片
│       │   ├── terrain_baking/      DEM 烘 PBR 贴图的工具（见其 SKILL.md）
│       │   ├── lunalab_lidar01/     Terrain_lidar01 烘出的贴图资产
│       │   ├── southpole_site20/    Site20 烘出的贴图资产
│       │   ├── task1_center/        TASK1 三个 20 m 裁片烘出的贴图资产
│       │   ├── task1_southwest/     （各含 dem.npy / terrain.yaml /
│       │   ├── task1_northeast/     textures / masks）
│       │   └── crater_spline_profiles.pkl   陨石坑剖面曲线参数
│       ├── rocks/            岩石
│       │   ├── apollo_rocks/        按样品编号归档的岩石模型
│       │   ├── lunalab_rocks/       试验场岩石（rock_1 ~ rock_4）
│       │   ├── lunar_rocks/         按尺寸/分辨率分级生成的大中型岩石
│       │   └── small_rocks/         小粒径碎石（rock_0 ~ rock_99）
│       ├── materials/        月面材质与灯光
│       │   ├── BlackBrick/ BlackIron/ BlackPaint/ ClothBlack/
│       │   ├── Earth/ Glass.mdl/ GravelStones/ LunarRegolith8k/
│       │   ├── moon_macro_01_4k/ Sand/ rock_boulder_dry_2k/
│       │   ├── seaside_rock_2k/
│       │   └── lights/              投影灯 IES 光域网文件
│       ├── skydomes/         天空盒
│       │   ├── high_res/            高分辨率 EXR/JPG
│       │   └── low_res/             低分辨率 EXR/JPG
│       └── scenery/          组合场景
│           ├── environments/        场景装配入口（Lunalab.usd）
│           ├── lunalab/             试验场构件（墙体、格栅、帘幕、细节）
│           ├── lunalab.usdc
│           └── oberpfaffenhofen_test_site.usdc
│
├── robots/
│   ├── mobile/               移动机器人
│   │   ├── rovers/                  轮式巡视器（含 common/ 共用传感器）
│   │   ├── legged/                  足式机器人
│   │   ├── aerial/                  无人机
│   │   └── go2/                     四足机器人（占位，待填充）
│   ├── manipulators/
│   │   ├── manipulator/             机械臂
│   │   └── gripper/                 夹爪 / 灵巧手
│   ├── ur10/ ur10e/ ur12e/   固定基座六轴机械臂（自包含：MuJoCo + UE）
│   ├── franka/               Franka Emika Panda 七轴机械臂 + 夹爪（原样取自 MuJoCo Menagerie，只有 MuJoCo）
│   ├── robotiq_2f85/         Robotiq 2F-85 自适应两指夹爪（同上）
│   ├── so101/                LeRobot SO-ARM101 桌面机械臂（同上）
│   ├── piper/                Piper 机械臂与参考移动操作机器人（同上）
│   ├── mobile_manipulators/  移动操作机器人
│   ├── landers/              着陆器
│   └── spacecraft/           航天器
│
├── objects/
│   ├── rocks/                岩石样品（可收集目标物）
│   ├── tools/                工具（铲斗、螺丝刀等）
│   ├── construction/         构件（螺栓螺母、基座、销孔机构）
│   ├── payloads/             载荷与设备
│   └── debris/               碎片（占位，待填充）
│
└── sensors/
    ├── cameras/              相机
    ├── lidar/                激光雷达
    └── imu/                  惯性测量单元（占位，待填充）
```

---

## 分类说明

### environments/ — 月面环境

| 目录 | 体积 | 说明 |
| --- | ---: | --- |
| `lunar/terrain/landscape_cropped` | 2.4 G | 大面积月面地形，含 `Materials/`、`Props/` 附属资源 |
| `lunar/terrain/SouthPole` | 7.5 G | 27 个 5 m/px 高程瓦片（`*_final_adj_5mpp_surf`、`ldem_87s_5mpp`），另含 `dem.npy` 高程矩阵 |
| `lunar/terrain/Lunalab` | 16 M | 试验场激光扫描地形 `Terrain_lidar01/02` |
| `lunar/terrain/terrain_baking` | 76 K | Terrain Baking Skill：把 DEM 离线烘成 albedo / roughness / normal。用法见其 `SKILL.md`，入口 `terrain_bake.py --config <资产目录>/terrain.yaml`，配套 `validate_terrain_bake.py` |
| `lunar/terrain/lunalab_lidar01` `southpole_site20` | 40 M | 上面这个工具烘出来的贴图资产，每个目录自带 `terrain.yaml`、`textures/`、`masks/` |
| `lunar/terrain/task1_center` `task1_southwest` `task1_northeast` | 53 M | TASK1 三个 20 m 地形裁片（`ue/import_data/Landscape_1_2795x2795.r16` 的滑动窗口）烘出的贴图资产。裁片导出见 `tools/prepare_task1_terrain.py`，换到 UE 地图见 `tools/build_task1_maps.py` |
| `lunar/rocks/lunar_rocks` | 1.7 G | 按尺寸等级 × 分辨率命名（`rocks_s<尺寸>_r<分辨率>`），适合批量散布 |
| `lunar/rocks/apollo_rocks` | 241 M | 23 组按样品编号归档的岩石 |
| `lunar/rocks/lunalab_rocks` | 165 M | 4 组试验场岩石 |
| `lunar/rocks/small_rocks` | 11 M | 100 个小粒径碎石，适合作为地表散落物 |
| `lunar/materials` | 1.7 G | 月壤、砂、砾石、岩石、黑色涂层等 PBR 材质，含贴图与 `.mdl` 材质定义 |
| `lunar/skydomes` | 207 M | 天空盒，`high_res/` 与 `low_res/` 两档同内容不同精度 |
| `lunar/scenery` | 43 M | 试验场组合场景与构件 |

`materials/` 中每个材质目录（如 `LunarRegolith8k/`）内含贴图，同名的 `.mdl` 文件为材质定义；`lights/` 存放投影灯的 `.ies` 光域网文件。

`moon_macro_01_4k/` 是给 UE 地形换皮用的一套 4K PBR 贴图（源：Blender 资产 `moon_macro_01_4k.blend`）。**只入库了三张**——`_diff_4k.jpg`、`_nor_gl_4k.exr`、`_rough_4k.exr`；源目录里的 `_disp_4k.png`（sha256 `2ddc17fe9fea4f3e124ea35662ecf589a86ba39b80453208bf6de627b62b5b73`）**有意未入库**：本仓库的地形物理是 1 cm 原始高度场、UE 必须渲染同一份几何，位移贴图会破坏这条不变量。目录里也没有 `_nor_dx`，UE 侧改用 `UTexture::bFlipGreenChannel` 翻转 OpenGL 约定的绿通道，不生成派生文件。

### robots/ — 机器人

| 目录 | 体积 | 内容 |
| --- | ---: | --- |
| `mobile/rovers` | 111 M | 巡视器：`perseverance`、`mars_rover`、`osr`、`leo_rover`、`construction_rover`、`pragyan`/`pragyaan`、`ros2_husky_*`、`ros2_jackal_*` 等；`common/` 为共用传感器 |
| `mobile/legged` | 2.7 M | `cassie` |
| `mobile/aerial` | 1.5 M | `ingenuity` |
| `manipulators/` | 108 M | 机械臂 `manipulator/`（96 M）：`franka_arm`、`kinova_gen3n7`、`kinova_j2n6s/j2n7s`、`canadarm3_large`、`so_arm100_5dof`；夹爪 `gripper/`（13 M）：`franka_hand`、`robotiq_hand_e`、`kinova300`、`so_arm100_5dof_gripper` |
| `mobile_manipulators` | 2.3 M | `jackal_ur3_423` |
| `landers` | 82 M | `apollo`、`peregrine`、`resilience`、`vikram` |
| `spacecraft` | 193 M | `gateway`、`iss`、`starship`、`super_heavy`、`satellite_mockup`、`venus_express` |

`ros2_husky_*`、`ros2_jackal_*` 等文件已内置相机、激光雷达与 IMU，可直接使用，无需再外挂 `sensors/` 下的模型。

**自包含机械臂目录**（`ur10/`、`ur10e/`、`ur12e/`、`so101/`、`piper/`）结构与上面按类别分组的目录不同：每个目录是一个完整机械臂，含 `arm.xml`（MuJoCo 模型）、`urdf/` 与 `meshes/`、给 UE 的 `<名字>.usdz`、`source.json`（上游仓库与 commit、逐文件 SHA256、本次改动清单、未声明事项）、`UPSTREAM_LICENSE` 与 `README.md`。搬走整个目录即可用，不需要额外的路径配置。

| 目录 | 体积 | 来源 | 说明 |
| --- | ---: | --- | --- |
| `ur10/` | 21 M | `ros-industrial/universal_robot` | UR10，6 关节 |
| `ur10e/` | 23 M | 同上 | UR10e，6 关节 |
| `ur12e/` | 23 M | 同上 | UR12e。上游这份 commit 里 UR12e 的描述文件与 UR10e 逐字节相同，网格目录也不存在，所以两个模型在把机器人名统一后完全一致——不是复制错误，详见其 `README.md` |
| `franka/` | 42 M | `google-deepmind/mujoco_menagerie` | Franka Emika Panda，7 关节 + 夹爪。**与上面四个不同：这是原样搬运**，`panda.xml` 与 67 个网格逐字节等于上游 commit `71f066a`，本目录只加了 `source.json`/`README.md`/`fetch_report.json` 三个说明文件，且每个文件都对着上游 git blob SHA1 校验过。只有 MuJoCo 模型，没有 `.usdz`；上游 `panda.xml` 也没有 `tool_tip` site |
| `robotiq_2f85/` | 3.1 M | 同上，同一个 commit `71f066a` | Robotiq 2F-85 自适应两指夹爪，8 关节 + 1 执行器（`ctrlrange` 0–255），指间距全开 0.0984 m / 闭合 0.0150 m，总质量 1.0526 kg。同样是**原样搬运**，8 个 STL 与 `2f85.xml` 逐字节等于上游。只有 MuJoCo 模型，没有 `.usdz`。TASK2 空白环境用它 |
| `so101/` | 27 M | `TheRobotStudio/SO-ARM100` | LeRobot SO-ARM101，6 关节含夹爪。保留上游已调好的 MJCF 伺服增益与 geom class，只改 `meshdir`、补 `option`、`tool_tip` 与 `home` 关键帧 |
| `piper/` | 8.9 M | 本机 ATEC_UE_sim | 见 `piper/README.md` |

`ur10`/`ur10e`/`ur12e`/`so101` 由 `tools/prepare_arms.py` 从上游 xacro/URDF/MJCF 生成，`tools/validate_arms.py` 验收；验收覆盖 MuJoCo 载入与步进、home 位姿自接触与重力下垂、末端随动、以及 `.usdz` 的连杆坐标系与网格摆位是否逐一对上 MuJoCo 自身运动学。证据写在 `.local/validation/arms.json`。

`franka/` 与 `robotiq_2f85/` 不走这条链：它们是原样搬运的 Menagerie 模型，没有 `.usdz` 可验收，`prepare_arms.py` 与 `validate_arms.py` 都不覆盖。它们的来源凭证是 `fetch_report.json` 里对着 GitHub API 的 blob SHA1 校验（`franka/` 69 个文件、`robotiq_2f85/` 13 个文件，全部一致）。`robotiq_2f85/` 的装配由 TASK2 空白环境端到端跑通，见该目录的 `README.md`；`franka/` 目前没有接入任何任务，没有超过"能载入"这一条的验收。

生成的 `.obj`、`.stl`、`.dae`、`.usd`、`.usdz` 与上游缓存按 `.gitignore` 不入库，入库存放的是生成与验收脚本、`source.json`、许可与说明。第三方资产的公开再分发权尚未确认，`.usdz` 与网格仅在本机使用。

### objects/ — 可交互物体

| 目录 | 体积 | 内容 |
| --- | ---: | --- |
| `rocks` | 57 M | 33 个岩石样品：`apollo_sample1~22`、`lunalab_boulder1~4`、`spaceport_moon_rock1~7` |
| `tools` | 54 M | 铲斗 `scoop`（6 种形状）、`electric_screwdrivers`、`screwdriver_hex_m3/m5` |
| `construction` | 45 M | `bolt_and_nut`、`industrial_pedestal_25/50/100cm`、`peg_in_hole`、`beneficiation_unit` |
| `payloads` | 15 M | `cargo_bay`、`solar_panel`、`sample_tube`、`juggling_ball` |

### sensors/ — 传感器

| 目录 | 内容 |
| --- | --- |
| `cameras/` | `rsd455.usd` — 双目深度相机 |
| `lidar/` | `vlp16.usd`、`velodyne-vlp16.usd` — 16 线激光雷达 |
| `imu/` | 占位，待填充 |

---

## 文件格式

| 格式 | 数量 | 说明 |
| --- | ---: | --- |
| `.usdz` | 785 | 打包的 USD，单文件自包含，可直接拖入引擎 |
| `.usd` | 402 | 文本/二进制 USD，多为分层组装，含外部引用 |
| `.usdc` | 11 | 二进制 USD 层 |
| `.png` `.jpg` `.exr` `.tif` | 107 | 贴图与天空盒；`.exr` 为高动态范围 |
| `.npy` | 31 | 高程 / 地形数值矩阵 |
| `.yaml` | 27 | 配套参数文件 |
| `.mdl` | 15 | 材质定义 |
| `.ies` | 4 | 灯光光域网 |

---

## 使用说明

- **引擎分工**：`environments/`、`robots/`、`objects/`、`sensors/` 中的 USD 资产同时可供 UE（渲染、光照、材质）与 MuJoCo（碰撞、动力学）使用；视觉网格与碰撞网格按需各自简化。
- **分层引用**：`.usd` 资产之间存在相对路径引用（例如 `scenery/` 引用 `materials/` 与 `skydomes/`，`rovers/` 引用 `rovers/common/`）。移动或重命名文件时必须同步修改引用，否则资产会加载失败。`.usdz` 为自包含格式，可自由移动。
- **高低精度双份**：`skydomes/high_res` 与 `low_res`、`lunar_rocks` 的不同分辨率等级，用于在渲染质量与加载开销之间取舍。
- **占位目录**：`lunar/terrain/Lunaryard`、`robots/mobile/go2`、`objects/debris`、`sensors/imu` 目前为空占位，保留目录结构以便后续填充（以 `.keepme` / `.gitkeep` 标记）。
- **可收集目标物**：`objects/rocks` 中的岩石样品为独立可交互资产（可移动、可拾取），与 `environments/lunar/rocks` 中作为地形组成部分的固定岩石用途不同。
