# UR12e 机械臂

固定基座六轴机械臂。`arm.xml` 是 MuJoCo 模型，`ur12e.usdz` 是给 UE 的显示资产。

来源为 `ros-industrial/universal_robot` 的 `ur_description`，commit `39ad110d8f2e8f66856a201cca88aa7a7025e3eb`。`source.json` 记录上游仓库、commit、展开后的 URDF、各网格与 `arm.xml`、`.usdz` 的 SHA256，以及本次改动清单；`UPSTREAM_LICENSE` 保留上游 BSD-3-Clause 文本，不代表单独核实了全部网格的公开再分发授权。

## 这个模型与 UR10e 完全相同

**上游仓库里 UR12e 没有独立描述。** 它的 `ur12e.xacro` 与 `ur12e` 网格目录实际内容与 UR10e 逐字节一致，也没有 `ur_description/meshes/ur12e` 这个目录——上游根本没有为它单独出网格。本目录的 URDF 是把上游 `ur12e.xacro` 展开后、把机器人名从 `ur12e` 换成 `ur10e` 的唯一差异；网格是从上游 `meshes/ur10e` 复制过来的。

因此两边的关节限位、力矩上限、连杆惯量、网格文件哈希全部相同（可比对两个 `source.json`），验证数值也相同：下垂 0.0283 rad、末端位移 0.2821 m。**这不是这里做错了，是上游就是这样发的。** 换成真实 UR12e 的手册参数（额定负载 12 kg、臂展 1300 mm）需要上游先给出对应描述，不能靠改名字造出来。

## 相对上游新增的内容

上游只给 URDF/xacro，没有可直接仿真的模型。跟 UR10/UR10e 一样，本次：

- 用 xacro 展开成独立 URDF，`package://` 网格路径改写为相对资产目录的路径，网格按上面说的从 `ur10e` 归一到 `meshes/ur12e/`。
- 用 MuJoCo 编译该 URDF（运动学、惯量、关节范围、力矩上限都来自上游），再补上上游无法表达的部分：6 路位置伺服、末端 `tool_tip` site（放在上游 `tool0` 位姿）、`home` 关键帧 `[0, -π/2, π/2, -π/2, -π/2, 0]`。
- 伺服增益按关节单独推导：kp 取该关节额定力矩对应 0.1 rad 误差的值，kv 按 qpos0 处质量矩阵对角元素取临界阻尼。上游力矩上限跨了 6 倍（肩 330 N·m、腕 54 N·m），统一 kp 会让腕关节在 0.0135 rad 误差时就打满力矩，随后贴着限幅抖动停不下来。这只是仿真接入参数，不是实机标定。
- 把编译后被并进 `worldbody` 的根连杆重新挂成具名的 `base_link` body，并排除 `base_link` 与 `shoulder_link` 的碰撞对。上游这两个碰撞网格本身就重叠 0.3 mm，不排除的话肩部回转关节在 home 位姿会常驻 4 个接触力，零控制误差下 1 秒内也会自己漂 0.04 rad。
- 生成显示用 USD（见下）。

视觉网格进 `group 2`、碰撞网格进 `group 3`。隐藏组 3 只影响显示，不修改碰撞计算。

## 精度与验证

`tools/validate_arms.py` 逐条检查并写出 `.local/validation/arms.json`：

- home 位姿无自接触；去掉重力后 1 秒内偏离关键帧 **0**，即残差全部来自重力。
- home 位姿零控制误差下重力下垂 0.0283 rad。这是有限伺服刚度的物理结果，不是求解器或接触假象。
- 第 2 关节指令 +0.3 rad，末端移动 0.2821 m，全程 `qpos`/`qacc` 有限；同一条指令持续 16 秒后完全静止（末速 5e-14 rad/s），稳态最大出力只占额定力矩的 26%。
- `.usdz` 用 usd-core 打开，根 prim、Z 轴朝上、单位米都正确；每个连杆坐标系与每个网格的摆放位置逐一对上 MuJoCo 自身的运动学，误差 0。

`.usdz` 是**显示资产**，不是仿真：它带几何和零位下的连杆层级，不带自由度配置、不带 PhysX articulation。UE 只负责显示，物理一律在 MuJoCo 里算。

## 未声明

没有做硬件标定，没有训练或接入任何策略，没有在 UE 侧实现物理。UE 侧只保证资产能导入，不改 C++ 插件、不改地图。

重新生成用 `tools/prepare_arms.py`（需要 `usd-core` 与 Blender 做 DAE 转 OBJ），验收用 `tools/validate_arms.py`。两个脚本都不写本文件。
