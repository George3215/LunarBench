# GPT-as-Policy 工具迁移到 TASK2 `stone_stack/robots`

记录时间：2026-09-22。回答"`stone_stack/robots` 是否已经迁移了 GPT-as-Policy 的工具"，
并记录本次迁移的内容与**未迁移项的原因**。

## 迁移前结论

**没有迁移。** `stone_stack/robots`（`control.py` / `profile.py` / `scene.py`）是 TASK2
自己独立实现的机器人层：MuJoCo 雅可比阻尼最小二乘 IK + 位置伺服 + 夹爪控制 + 抓取朝向，
与 GPT-as-Policy 的机器人工具只有**部分功能重叠**。

## 本次迁移的内容

新增 `stone_stack/robots/kinematics.py`（纯几何/契约，不导入 mujoco），并在
`robots/__init__.py` 导出；`RobotController` 增加 `tcp_pose_for(q)`。

| GPT-as-Policy 原工具 | 原文件 | 迁移结果 |
| --- | --- | --- |
| `transform()` / `pose()` | `action_edit_kinematics.py` | ✅ `transform` / `pose` / `pose_from_arrays` |
| 四元数↔矩阵 | 同上 | ✅ `quaternion_to_matrix` / `matrix_to_quaternion_wxyz`（带单位化校验） |
| `edited_targets()` | 同上 | ✅ 同契约：`alpha=(i+1)/steps`、5 cm / 0.35 rad 上限、`gripper∈{keep,open,closed}` |
| `ArmFK.bounded_target()` 的任务空间限幅 | `kinematics.py` | ✅ `bounded_task_error` / `bounded_eef_target`（默认 2 cm / 0.1 rad） |
| `validate_response()` 几何部分 | `validation.py` | ✅ `validate_execution_contract`：`student/edit/eef/stop`、`request_id` 防陈旧、步数上限 15/5/5 |
| `validate_eef_target` 的 5 cm / 0.35 rad 判定 | 同上 | ✅ `validate_eef_target`：单位四元数、布尔夹爪、相对当前 EEF 的界 |
| `validate_public_language()` | 同上 | ✅ `validate_public_language`（公开文本禁 CJK） |
| `DualKinematics.preview()` | `kinematics.py` | ✅ `preview_trajectory`（单臂版，用 `RobotController.tcp_pose_for`） |

测试：`tools/test_kinematics_tools.py`（36/36，纯离线，无需 GPU）。

## 迁移时发现的一个既有隐患（未修改）

`RobotController.__init__`（`control.py:37`）写的是：

```python
self.ik_data = data if ik_data is None else ik_data
```

而 task2 的三处构造（`execution.py:123`、`run.py:165`、`run.py:231`）**都没有传 `ik_data`**，
所以 `self.ik_data is self.data`——`solve_ik` / `servo_translate` 里那句
"IK 在独立的 MjData 上跑：不污染正在仿真的那份状态" 与注释不符：它们实际上在
**同一个 `MjData`** 上改 `qpos` 并 `mj_forward`。`solve_ik_best` 与调用方通常会把
关节角存回/恢复，所以现状能跑，但这是一个真实的状态耦合。

新增的 `tcp_pose_for` **不依赖** `ik_data`：它懒加载一份私有 `self._fk_data`
（真·独立 `MjData`），实测 `tcp_pose_for` 前后 `tcp_pose()` 逐位相等。

是否把 `ik_data` 默认改成独立 `MjData` 属于**行为变更**（会影响 `solve_ik` /
`servo_translate` 的中间状态），本次未改，留给你决定。


## 未迁移项与原因

| 未迁移 | 原因 |
| --- | --- |
| `ArmFK` 基于 URDF 的解析 FK | TASK2 已加载 MuJoCo 模型，`tcp_pose()` / `tcp_pose_for()` 直接给世界系 TCP 位姿；再维护一套 URDF FK 会引入"两套 FK 不一致"的风险。 |
| `DualKinematics` / 左右双臂 / `root()` 相对基座变换 | TASK2 是**单臂**、基座固定，TCP 已在世界系；双臂框架没有对应物。 |
| `ArmFK.bounded_target` 的数值差分雅可比 + ±0.05 rad 关节步长 | TASK2 的 `solve_ik` 用 MuJoCo 的解析雅可比（`mj_jacSite`）+ 关节限位裁剪，质量更高；只迁移了它缺的**任务空间限幅**。 |
| `gate_assessment.validate_assessment` / gate prompt | 属于**策略层**的 outcome/intent 评估，不是机器人层工具；TASK2 当前策略（`vlm` / `qwen-agent`）还没有 assessment 字段，先不做。`validate_execution_contract` 已预留 `gate_policy="failure-or-intent"` 挂钩。 |
| `decision_log.py` / `debug_recorder.py` / `video_panel*.py` / `scene_recording.py` | 属于记录与可视化，TASK2 已有自己的报告链路（`execution.write_report`、`--save-frames`）。 |
| `client.py` / `server.py` / `session.py` / `protocol.py` / `rpc.py` | RoboDojo/RoboLab 的 RPC 服务框架；TASK2 的策略在**同进程**同步调用，不需要 RPC 层。 |
| `proposal_diagnostics.py` / `pi05` 相关 | 依赖 π₀.₅ 提案，TASK2 没有该模型。 |

## 使用示例

```python
from stone_stack.robots import (
    edited_targets, bounded_eef_target, validate_execution_contract, preview_trajectory,
)

# 1) 有界平滑修正：在标称轨迹上叠加 ≤5 cm / ≤0.35 rad
corrected = edited_targets(nominal_trajectory, steps=3, edit={
    "delta_position": [0.01, 0.0, -0.005],
    "delta_rotation_vector": [0.0, 0.0, 0.05],
    "gripper": "closed",
})

# 2) 把任意绝对目标限成"离当前一步之内"再交给 IK
pos, rot, diagnostics = bounded_eef_target(tcp_pos, tcp_rot, wanted_pos, wanted_rot)

# 3) 执行契约校验（request_id / mode / steps / 5 cm / 0.35 rad）
validate_execution_contract(response, request)
```
