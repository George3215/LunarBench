# TASK1 当前入口

当前默认使用 Z-Mobile-manip 的传感器与接近控制 baseline，关闭得分并持续运行。

```bash
bash /home/lry/MoonUnrealEnv/task1.sh
```

传感器页面：<http://127.0.0.1:19530>。完整接口、上游版本、数据来源和使用方式见 [baseline 文档](../../baselines/z_mobile_manip/README.md)。

- UE 输出机器人挂载 RGB-D；MuJoCo 输出 3D LiDAR、关节/IMU 自身反馈。
- 独立策略只通过 bridge 接收测量、发送动作，不读取场景石头真值。
- RGB 画面拖框选择目标；无目标时保持。UE W/S/A/D 接管，B 恢复 baseline，R 重置。
- 实验条件在 task1.yaml；普通实现变量直接改 Python，不使用命令行参数。
- 场景保留 20 m、150 块石头和固定种子，center/southwest/northeast 三个原始地形裁片。
- 原评分实现保留但默认禁用，后续抓取阶段可以重新启用。旧真值推击实现已移入工作区 .local/backups/task1-before-z-mobile-20260919。

评分与性能的上一阶段记录见 [STATUS.md](STATUS.md)，属于加入传感器之前的历史记录，不能当成当前 RGB-D/LiDAR 的性能结论。

## 三个地形裁片与它们的 UE 贴图

三个裁片各有一张换过皮的 UE 地图，地形贴图是从**这块地自己的高程**烘出来的
（陨石坑、坑缘、坡度都在反照率和粗糙度里），不是通用的月面噪声。红色边界线与
白色石头收集框由 `viz.py` / `task_scene.py` 在运行时叠加，与地图无关。

| 裁片 | 实验条件 | UE 地图 |
| --- | --- | --- |
| center | `task1_center.yaml` | `/Game/MoonMacro/Maps/MoonTerrain_Task1Center` |
| southwest | `task1_southwest.yaml` | `/Game/MoonMacro/Maps/MoonTerrain_Task1Southwest` |
| northeast | `task1_northeast.yaml` | `/Game/MoonMacro/Maps/MoonTerrain_Task1Northeast` |

三个 `task1_<patch>.yaml` 与 `task1.yaml` 内容相同，只多了 `ue_map`。地图路径
写在 YAML 里，`run.main()` 会读出来传给 UE，**不需要手工设 `LUNARBENCH_UE_MAP`**。

跑某一个裁片：

```bash
cd /home/lry/MoonUnrealEnv && PYTHONPATH= /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python -c "
from MoonSim.tasks.task1_collect import run
run.main(config='MoonSim/tasks/task1_collect/task1_southwest.yaml',
         output='MoonSim/tasks/task1_collect/generated/terrain_southwest')
"
```

三个裁片要**串行**跑：UE、bridge、状态与相机端口都是固定常量，同时起两个会互相抢。
`task1.yaml` 与 `task1.sh` 保持原样，不填 `ue_map` 时用 demo 地图，行为与从前一致。

贴图与地图怎么来的（改了贴图必须重跑换皮）：

```bash
python tools/prepare_task1_terrain.py                    # R16 -> 三个裁片 dem.npy + terrain.yaml
cd assets/environments/lunar/terrain/terrain_baking
for p in center southwest northeast; do
    python terrain_bake.py          --config ../task1_$p/terrain.yaml
    python validate_terrain_bake.py --config ../task1_$p/terrain.yaml
done
cd /home/lry/MoonUnrealEnv/MoonSim
python tools/build_task1_maps.py                         # 三张贴图 -> 三张 UE 地图
python tools/validate_task1_maps.py
```

换皮走的是仓库既有的 `-MoonMacroMap` commandlet，`MoonMacro.inl` 一行没改。
跑之前 UE 工程必须已关闭。
