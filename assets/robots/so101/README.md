# SO101 机械臂

桌面级六自由度机械臂（LeRobot 的 SO-ARM101）。`arm.xml` 是 MuJoCo 模型，`so101.usdz` 是给 UE 的显示资产。

来源为 `TheRobotStudio/SO-ARM100`，commit `eecbe3e0a9ebb23e25ad7b2759b03884c6660903`，取 `Simulation/SO101` 下的 URDF、MJCF 与 STL 网格。`source.json` 记录上游仓库、commit、URDF、各网格与 `arm.xml`、`.usdz` 的 SHA256，以及本次改动清单；`UPSTREAM_LICENSE` 保留上游许可文本，不代表单独核实了全部网格的公开再分发授权。

与 UR 那几个臂不同，**上游本来就提供了一份可直接用的 MJCF**，所以这里没有走「编译 URDF 再补伺服」的路线，而是直接以上游 MJCF 为 MuJoCo 模型。网格全部是上游 STL 原文件，没有格式转换。

## 相对上游改动的内容

`source.json` 的 `modifications` 逐条列出，共四处：

- URDF 与网格路径改成相对资产目录，不再依赖 `package://` 或 `assets/`。
- MJCF 的 `meshdir` 由 `assets/` 改为 `meshes/`，并固定本工程的 timestep 与重力，避免模型行为随 MuJoCo 默认值漂移。
- 在上游 `gripper_frame_link` 的位姿处加 `tool_tip` site，作为末端参考点。
- 加一个全零的 `home` 关键帧（6 关节，`qpos` 与 `ctrl` 都是 0）。
- 生成显示用 USD（见下）。

**伺服增益是上游的，本次没改。** 上游 MJCF 自带的 `kp`/`kv` 直接沿用。关节力矩上限（`effort` 10）与限位也来自上游 URDF。

视觉网格进 `group 2`、碰撞网格进 `group 3`。隐藏组 3 只影响显示，不修改碰撞计算。

## 精度与验证

`tools/validate_arms.py` 逐条检查并写出 `.local/validation/arms.json`：

- home 位姿无自接触；去掉重力后 1 秒内偏离关键帧 **0**，即残差全部来自重力。
- home 位姿零控制误差下重力下垂 0.00054 rad。SO101 在 home 是竖直收拢姿态，力臂短，下垂比 UR 水平伸出时小两个数量级。
- 第 2 关节指令 +0.3 rad，末端移动 0.1021 m，全程 `qpos`/`qacc` 有限；同一条指令持续 16 秒后完全静止（末速 1.5e-11 rad/s），稳态最大出力占额定力矩的 19%。
- `.usdz` 用 usd-core 打开，根 prim、Z 轴朝上、单位米都正确；每个连杆坐标系与每个网格的摆放位置逐一对上 MuJoCo 自身的运动学，误差 5.6e-17。

一个如实记录的行为：把所有关节一起指到 +1.0 rad 并保持，肘部会停在中途不跟上。原因是该姿态下肩部与夹爪自接触，加上 10 的力矩上限，属于一台 hobby 舵机桌面臂的真实表现，不是模型错误。

`.usdz` 是**显示资产**，不是仿真：它带几何和零位下的连杆层级，不带自由度配置、不带 PhysX articulation。UE 只负责显示，物理一律在 MuJoCo 里算。

## 未声明

没有做硬件标定、没有接入 LeRobot 的策略或数据集，没有在 UE 侧实现物理。UE 侧只保证资产能导入，不改 C++ 插件、不改地图。

重新生成用 `tools/prepare_arms.py`（需要 `usd-core`），验收用 `tools/validate_arms.py`。两个脚本都不写本文件。
