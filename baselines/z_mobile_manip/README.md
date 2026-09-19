# TASK1 Z-Mobile-manip baseline

启动：`bash /home/lry/MoonUnrealEnv/task1.sh`。打开 http://127.0.0.1:19530 查看腕部 D435 RGB/Depth 和 Mid-360 点云。R 重置，B恢复baseline，WASD人工接管。此阶段关闭得分，无时间上限。

运行结构保持三个Python进程加UE：

```
Go2-Piper实体D435安装点 → UE RGB-D ─┐
Go2-Piper实体Mid-360 → MuJoCo射线 ─┼→ bridge → YOLOE / RGB-D / antipodal / IK / RRT → bridge动作 → MuJoCo
模拟编码器 / IMU ─────────────────┘
```

Policy只读bridge的camera、lidar、robot话题，不导入MuJoCo/UE，也不读取石头位置、场景ID或得分。独立机器人静态MJCF仅用于构建运动学链。实际相机光学点随腕部关节移动，激光雷达保留自身遮挡；安装来源与仿真精度见 `assets/robots/sensors/README.md`。

## 实际接入

- YOLOE-11s-seg 开放词汇检测/分割，提示词rock、stone，阈值.15；640输入，CUDA推理，拒绝超过画面12%的大片地形误检。一次准备模型可运行 `.local/zmobile-env/bin/python -m MoonSim.baselines.z_mobile_manip.prepare`。模型与缓存文本嵌入在 `.local/models/z_mobile_manip/`，无需每次联网。
- EdgeTAM使用上游服务里的原始模型后端类，剥离HTTP/ROS服务；权重与RepViT配置缓存本地，运行时强制离线。
- 使用上游RGB-D反投影；目标mask截取测量点云，并转换到机器人base坐标。
- 上游AntipodalGraspSource几何抓取后端，最大夹口70 mm；无需商业AnyGrasp许可。
- 上游RobustIKSolver + KinematicChain，静态机器人描述解析适配当前Go2-Piper；不是Pinocchio版本。
- 上游JointSpaceRRTConnect + retime_path；对测量场景点云作简化臂段碰撞约束，再输出有限速度关节轨迹。
- 搜索使用缓慢腕部扫描；接近使用上游VisualServoController及Mid-360前方障碍门限；近处尝试抓取规划，关节反馈到达后闭合夹爪。

这里是精简的仿真集成，不等于上游全部实机栈。没有移植ROS2、Isaac、CAN、NUC、Docker、外部导航、VLM、FFS、AnyGrasp、Pinocchio或CasADi。RGB-D取UE理想深度，YOLOE检测后由EdgeTAM保持目标mask；网页可以拖框提供图像目标，再使用上游EdgeTamBackend的EdgeTAM视频分割跟踪（这不是模型自动检测成功）；未观测区域及完整自碰撞尚无完整规划保证，抓取后没有成功判定、搬运或计分。

## 参数与依赖

日常变量直接编辑 `run.py` / `pipeline.py`；关键场景seed、机器人与policy选择使用TASK1现有YAML。没有新增CLI参数。

策略环境 `.local/zmobile-env` 以本机 `/home/lry/miniconda3/envs/ATEC` 的PyTorch2.7/CUDA12.8为基础，独立venv覆盖 ultralytics8.4.104、filelock4.0.1、pycollada0.9.3；不导入ATEC/Isaac包，也不修改原环境。MuJoCo仍使用原来的moonunreal-mujoco环境。pycollada仅生成D435 OBJ时使用。

上游选取文件与SHA256见 `third_party/z_mobile_manip/UPSTREAM.json`，commit `db76d242cf9210fae6068d63ce6e6e50a7ab0069`，保留PolyForm Noncommercial许可。没有复制整个上游仓库。

当前运行状态见 `tasks/task1_collect/generated/bridge/baseline.json`，包含epoch、相机帧号、模型推理耗时、检测数、抓取候选、IK、规划与执行累计数。只有这些计数实际增加，才能声称对应阶段在本轮运行发生。错误/无检测/不可达会明确记录，不能当作抓取成功。
