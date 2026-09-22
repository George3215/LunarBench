# 上游机械臂、可达料区与 Direct trial

## 源码对应

- stone-stacking-mujoco：`3fe9775f63563b924a62b3a87b3b8aa19331651b`。
- GPT-as-Policy：`8f3d362b077d8efb77e2a7274d5b2c20e2243846`。
- 机器可读溯源：[upstream_provenance.json](upstream_provenance.json)。

本轮对比远端源码，现有 `scripts/run_official_ur5e_robotiq_wall_stack.py` 的
`initial_supply_pose`、`build_wall_stack_scene`、`settle_initial_scene`、`report_stones`
四个函数的 AST 与上游一致。资产路径复用本地 robosuite；不依赖作者机器上的绝对路径。

`stone_stack/upstream_scene.py` 直接调用上游装配函数，保留机器人与石头参数，
按用户要求把石头移到可达料区，保留原生 eye_in_hand，并新增 top/front 相机：

- UR5e 基座：`[-0.30, -0.35, 0.0]` 米。
- 初始关节角：`[0.74, -1.30, 1.50, -1.76, -1.57, -0.83]` 弧度。
- 原始 Robotiq 2F-140；不替换成 MoonSim 的合成平行夹爪。
- 原始碰撞、摩擦、质量、重力及 0.0015 s 步长。
- 初始关节与开爪控制后，按上游预静置 0.65 s，再提供第一次观测。
- 不缩放石头，不扩增候选池，不根据可达性缩减层数。

当前十个初始 XY（米），直接配置在 `task2.yaml.scene.supply_positions_xy`：

| 序号 | x | y |
| --- | --- | --- |
| 1 | -0.70 | -0.10 |
| 2 | -0.70 | 0.07 |
| 3 | -0.74 | 0.24 |
| 4 | -0.52 | 0.24 |
| 5 | -0.30 | 0.24 |
| 6 | -0.08 | 0.24 |
| 7 | 0.14 | 0.24 |
| 8 | -0.52 | 0.41 |
| 9 | -0.30 | 0.41 |
| 10 | -0.08 | 0.41 |

两块在墙区左侧，其余八块在后侧两排，保留 `x±0.45, y±0.16` 的墙区。
这些料位在开发时已经验证过桌面边界、墙区、至少 2 cm 的包围盒间距、
0.82 m 水平半径及抓取/接近参考点 IK。验证代码已删除，启动时直接按 YAML 布置，
不重新运行相机、间距或 IK 自动检测。

本轮十块全部通过，最大水平半径约 0.7912 m，20 个参考 IK 位姿的位置误差均小于 0.1 mm。
这证明料区的端点运动学可达；不等于任意运动路径无碰撞，也不证明石头一定夹得住。
`scene.supply_layout: upstream` 可切回原始坐标作对照；正式配置默认用 `reachable`。

Z 由上游公式 `-min(vertices.z) + 0.004` 计算，之后遵循原始物理沉降。
上游 Git 仓库不包含 README 中生成的 planner report。正式配置默认通过 `stone_stack.tools.stones` 用上游生成器、seed 17 生成
十块未缩放的 paper 石头，初始 yaw=0；这不是上游演示“从 24 块中筛出的同十块及其 yaw”的复刻。
若已生成该报告，可在 `scene.upstream_report` 填绝对路径，恢复其选石与初始 yaw（XY 仍按 reachable 料区重新布置）；
报告中的最终放置位姿不会提供给 Qwen，也不会被执行器拿来完成任务。

上游执行器会在每次抓取前调用供料重定位。这里在 trial 开始前将所有石头一次布置在已经检查过的可达料区，trial 内不再重定位。
后续石头只能被机械臂通过物理接触移动，不能把上游有重定位辅助的结果当成新任务的成功证据。

## Qwen 工具

1. `plan_stone`：记录模型选择的石头、抓取 EE pose、放置 EE pose。没有自动运动。
2. `move_tcp_delta`：模型指定世界系局部位移，保持实测姿态，复用相同门控和伺服。
3. `move_tcp`：校验 world 目标后执行 1–5 个时长单位的伺服与夹爪指令。超过 0.05 m / 0.35 rad 则拒绝，不截短。
4. `record_failure`：记录阶段、石头、观测编号、可见证据、原因假设和下一次调整。
5. `finish`：请求 host 检查，不结束未完成的 trial。

新增工具仍共用同一份 schema。没有新增文本 parse、脚本抓取/放置规划器或自动补动作。
模型只能看到 RGB、本体状态、静态机械臂信息、自己的计划与失败记录。
完整计划/失败记录随观测传回并存盘。

提示词独立放在 [direct_stack.md](../prompts/direct_stack.md)，方便直接编辑。
它规定全部十块、从下到上 4+3+2+1、上层跨接、TCP/指垫转换、试抬检查、净空、
释放后检查、失败证据、具体调整和预算管理。修改提示词不需要修改 Python。

## trial 与经验回流

借鉴 Direct astra 的 trial 内规则：同一段持久对话、动作后新观测、工具执行留痕、
trial 内无 reset/rewind/隐藏规划器。用户要求的跨 trial 学习由 TASK2 额外实现：

```text
保存初始积分状态
  -> trial 1：Qwen 计划/动作/失败记录
  -> host 检查 + Qwen 复盘
  -> 恢复同一初始状态
  -> trial 2：加载上一轮复盘，再规划/动作
  -> ... 达到成功判据或 trial 预算
```

复盘字段为 `summary`、`failures[{evidence,cause_hypothesis,next_change}]`、`next_trial_plan`。
下一轮的 `context.json.previous_trial_lessons` 是上一轮完整 `review.json`。
这是经验与提示词回流，不更新模型权重，也不宣称上游 Direct astra 原本带有这种跨 trial 学习。

每个控制观测边界，host 检查全部十块是否位于墙区、支撑层数是否为 4/3/2/1、上层是否跨接至少
两块相邻下层石头、是否仍有机械臂接触及明显速度，要求合格的连续边界观测覆盖至少 1.2 s 仿真时间；动作间不会自动补等待。
这是离散观测检查，不证明两次观测之间从未失稳。
`task2_contact_wall_v1` 是本地几何/接触判据，不是上游原生 benchmark 计分，也不是抗扰稳定性证明。
物体真值只用于离线记录与 环境检查；检查结果在 trial 结束后供模型复盘。

预算耗尽标记 `step_limit`，`finish` 不改变终止状态；只有环境稳定成功或预算耗尽才正常结束。
API/执行异常写入 `error.json` 后抛出，不自动重放不确定动作或悄悄开启下一 trial。

## 运行与产物

正式配置：

```bash
bash /home/lry/MoonUnrealEnv/task2.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/upstream_qwen_trials.yaml --view
```

默认 3 个 trial，每 trial 最多 400 次工具调用，全部使用十块石头的可达料区场景。
这些是预算，不是已验证足够完成堆叠的步数。无窗口时去掉 `--view`。

两次 trial 的短链路验证：

```bash
bash /home/lry/MoonUnrealEnv/task2.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/upstream_trials_smoke.yaml
```

`outputs/direct_<时间戳>/` 保存：

```text
scene.xml                         # 本轮装配场景
initial_integration_state.npy     # trial 边界恢复的完整积分状态
run_config.json                   # 指令与策略配置，不含 API key
summary.json / lessons.json       # 各 trial 结果、最近经验
trial_001/
  context.json                    # 含上一 trial 复盘
  0000_state.json / *_sim.npz      # 本体观测、仅用于存证的完整仿真状态
  0000_eye_in_hand.jpg / top.jpg / ...   # 工具执行前后图像，不是连续帧率视频
  trajectory.jsonl                # 请求及实际结果
  stone_plans.json / failures.json
  result.json / review.json / NOTES.md
  final_eye_in_hand.jpg / ...           # 稳定观察后的最后图像
trial_002/ ...
```

只保存每次工具前后的图像和状态；本轮没有新增子步连续视频录制。
真实 Qwen 短 trial 证明了失败记录和跨 trial 调整链路，尚未完成整面墙。

## 本轮证据

[reachable_trials_validation.json](../reports/reachable_trials_validation.json) 汇总了：
12 项直接策略测试、6 项上游/料区/trial 测试；真实 Qwen 两次短 trial；
两个 trial 的初始 time/qpos/qvel/ctrl 完全一致；上一轮 review 与下一轮 context 中的 lessons 完全一致。
真实运行没有完成抓取，局部几何检查为十块仍在地面，不能把模型的失败解释当成确定的物理原因。

## 严格动作协议与可修改位置

`policy.direct.max_delta_m: 0.05`、`max_rotation_rad: 0.35` 是目标硬上限，可以收紧，不能增大。
`move_tcp.frame` 只接受 `world`；`steps` 为 1–5，每单位是 `servo_seconds` 秒（默认 0.30），不是上游 X5 的 25 Hz。
`max_steps` 仍是工具调用预算，含拒绝和记录调用。所有参数先校验，再执行；校验拒绝不推进仿真。
当前只接受单臂 TCP，left/right 等额外字段拒绝。目标限幅不保证物理轨迹无碰撞或实测位移完全相同。

主提示词在 `prompts/direct_stack.md`，具身契约与任务背景在 `prompts/context/`，三份均自动加载。
动作只能从 Python 白名单进入；模型无任意文件读取、shell、代码修改或额外仿真连接。
`plan_stone` 和 `record_failure` 保存工作记忆；trial 复盘由 host 写入 NOTES.md，并反馈下一轮。
Codex 的连接验证与能力边界见 [QWEN_CODEX.md](QWEN_CODEX.md)。

## 2026-09-22 不移动与 HTTP 400 修正

原问题有两个独立原因：模型目标被距离/姿态门控拒绝；第七次请求输入至少 15485 tokens + 输出预留 900 超过 16384。
保留原 5 cm / 0.35 rad 上限，增加显式局部位移工具，绝对工具允许 null 姿态表示保持实测姿态。
Python 不截短越界目标，不替模型决定方向。日志同时输出调用结果，不把“提交了目标”说成“执行了运动”。
默认两轮历史；精简三个提示文件、移除重复 schema、压缩旧观测，最新失败只携带最近三条，完整记录仍存盘。
HTTP 错误保留响应正文，不自动重试。viewer.close 后等待 MuJoCo UI 线程退出，再释放相机。

## 当前相机

上游正式配置使用 `[eye_in_hand, top, front]`。原生 `eye_in_hand` 保留，遮挡严重的额外 `wrist` 已删除。
`front` 的 `scene.front_camera_offset` 默认 `[0, -1.35, 0.35]` 米，位于工作区 -Y 侧，光轴水平沿 +Y，垂直墙面 XZ。
画面右为 +X、上为 +Z。它与斜向 `overview` 不再重复；正前视角的机器人遮挡需要 top/eye_in_hand 补充。
`--mode camera` 导出模型中实际存在的所有相机，不调用 Qwen。

## 观测与保持控制修正

Qwen 输入图像的 width/height/K 按真实发送 JPEG 尺寸生成；原始图片尺寸另存 source_width/source_height。
rotation_camera_to_world 为 MuJoCo 相机到世界旋转，局部 -Z 向前、+Y 向上；像素 v 向下。
具体射线公式见 prompts/context/eef_control.md。只发送 RGB，不采集无用途的深度。
trajectory.jsonl 的 call.policy_observation 保存该次模型收到的文本观测，便于和原始状态区分。

夹爪参数 0 表示命令闭合，max_grip_width_m 表示全开；实测开口与指令分开返回，接触不能仅凭夹爪指令判断。
直接控制保留已有位置执行器目标作为保持起点，避免每轮把重力静差重新写入关节目标。
到位阈值集中在 direct_control.py 的 TCP_POSITION_TOLERANCE_M / TCP_ROTATION_TOLERANCE_RAD（2 mm / 0.02 rad）。
这些是到位精度，不替代动作 5 cm / 0.35 rad 门限。超出实际跟踪精度时 reached=false，不能冒称到位。

工具返回实测运动、夹爪开口变化、重复目标次数和连续无运动次数。它们只反馈事实，不拒绝有意保持，也不替模型选动作。
保持/夹爪的独立物理对照结果见 reports/hold_control_validation.json；不等于完整抓取或堆叠成功。
