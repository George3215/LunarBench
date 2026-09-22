# UR10 机械臂

固定基座六轴机械臂。`arm.xml` 是 MuJoCo 模型，`ur10.usdz` 是给 UE 的显示资产。

来源为 `ros-industrial/universal_robot` 的 `ur_description`，commit `39ad110d8f2e8f66856a201cca88aa7a7025e3eb`。`source.json` 记录上游仓库、commit、展开后的 URDF、各网格与 `arm.xml`、`.usdz` 的 SHA256，以及本次改动清单；`UPSTREAM_LICENSE` 保留上游 BSD-3-Clause 文本，不代表单独核实了全部网格的公开再分发授权。

网格保留上游原始文件：视觉网格是 Collada `.dae`，碰撞网格是 `.stl`。MuJoCo 读不了 DAE，所以每个视觉网格另存了一份 OBJ（Blender 无头导出，保持 Z 轴朝上）；DAE 原件一并保留以便对照。碰撞用的是上游 STL 原文件，MuJoCo 的网格碰撞是原生凸包语义，不是凹形三角网格。

## 相对上游新增的内容

上游只给 URDF/xacro，没有可直接仿真的模型。本次新增：

- 用 xacro 展开成独立 URDF，`package://` 网格路径改写为相对资产目录的路径。
- 用 MuJoCo 自己编译该 URDF（运动学、惯量、关节范围、力矩上限都来自上游），再补上上游无法表达的部分：6 路位置伺服、末端 `tool_tip` site（放在上游 `tool0` 位姿）、`home` 关键帧 `[0, -π/2, π/2, -π/2, -π/2, 0]`。
- 伺服增益按关节单独推导，不用全局常数。上游力矩上限跨了 6 倍（肩 330 N·m、腕 54 N·m），统一 kp 会让腕关节在 0.0135 rad 误差时就打满力矩，随后一直贴着限幅抖动、停不下来。做法是 kp 取该关节额定力矩对应 0.1 rad 误差的值，kv 按 qpos0 处质量矩阵对角元素取临界阻尼。这只是仿真接入参数，不是实机标定。
- 把编译后被并进 `worldbody` 的根连杆重新挂成一个具名的 `base_link` body，并排除 `base_link` 与 `shoulder_link` 的碰撞对。上游这两个碰撞网格本身就重叠 0.3 mm，不排除的话肩部回转关节在 home 位姿会常驻 4 个接触力，零控制误差下 1 秒内也会自己漂 0.04 rad。这与 Piper 排除底座与第一关节重叠轴承外壳是同一处理。
- 生成显示用 USD（见下）。

视觉网格进 `group 2`、碰撞网格进 `group 3`。隐藏组 3 只影响显示，不修改碰撞计算。

## 精度与验证

`tools/validate_arms.py` 逐条检查并写出 `.local/validation/arms.json`：

- home 位姿无自接触；去掉重力后 1 秒内偏离关键帧 **0**，即残差全部来自重力。
- home 位姿零控制误差下重力下垂 0.0275 rad（UR10 在 home 是水平伸出姿态，下垂最大）。这是有限伺服刚度的物理结果，不是求解器或接触假象。
- 第 2 关节指令 +0.3 rad，末端移动 0.2851 m，全程 `qpos`/`qacc` 有限；同一条指令持续 16 秒后完全静止（末速 5e-14 rad/s），稳态最大出力只占额定力矩的 26%。
- `.usdz` 用 usd-core 打开，根 prim、Z 轴朝上、单位米都正确；每个连杆坐标系与每个网格的摆放位置逐一对上 MuJoCo 自身的运动学，误差 0。

`.usdz` 是**显示资产**，不是仿真：它带几何和零位下的连杆层级，不带自由度配置、不带 PhysX articulation。UE 只负责显示，物理一律在 MuJoCo 里算。

## 未声明

没有做硬件标定，没有训练或接入任何策略，没有在 UE 侧实现物理。UE 侧只保证资产能导入，不改 C++ 插件、不改地图。

重新生成用 `tools/prepare_arms.py`（需要 `usd-core` 与 Blender 做 DAE 转 OBJ），验收用 `tools/validate_arms.py`。两个脚本都不写本文件。
