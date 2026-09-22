# TASK2 石块堆叠（MoonSim 接入版）

月面石块干砌堆叠任务，已经接进 MoonSim：**机械臂可切换、策略可切换、三路相机给观测**。
上游的 ICRA2017 复现代码（`scripts/`、`stone_stack/` 的其余部分）原样保留、仍可单独运行，
上游 README 移到 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

```bash
bash /home/lry/MoonUnrealEnv/task2.sh list                 # 有哪些臂
bash /home/lry/MoonUnrealEnv/task2.sh policies             # 有哪些策略
bash /home/lry/MoonUnrealEnv/task2.sh check piper          # 装配 + 可达性 + 相机自检
bash /home/lry/MoonUnrealEnv/task2.sh stack panda --courses 3,2,1 --stones 6 \
     --report reports/panda_321.json
bash /home/lry/MoonUnrealEnv/task2.sh stack ur5e --policy vlm --courses 3,2,1   # 需要 vLLM
```

## 这一层解决的四件事

| 需求 | 实现 | 位置 |
| --- | --- | --- |
| 接入 MoonSim | 统一入口 + YAML 配置 + 与 TASK1 相同的目录约定 | `run.py`、`task2.yaml` |
| 换机械臂 | `ArmProfile` 注册表：装配 / 控制 / 夹爪几何 / 工作区全部数据化 | `stone_stack/robots/profile.py` |
| Policy 接口 | 高层（选哪块放哪）与低层（末端下一步去哪）两个协议 + 注册表 | `stone_stack/policy/` |
| 三路相机 | 腕部（跟着夹爪动）+ 俯视图 + 前景，离屏 RGB-D | `stone_stack/cameras.py` |

## 结构

```text
tasks/task2_stack/
├── run.py                  # 统一入口（--arm/--policy/--check/--view/--camera-check）
├── task2.yaml              # 这次跑什么：臂、层数、石块、策略、执行、输出
├── stone_stack/            # 上游石块生成与几何（原样保留）
│   ├── robots/             # 本层新增：机械臂抽象
│   │   ├── profile.py      #   ArmProfile / GripperSpec / CameraMount 注册表
│   │   ├── scene.py        #   按 profile 装配 MJCF + 工作区几何
│   │   └── control.py      #   关节寻址 / 阻尼最小二乘 IK / 闭环伺服 / 夹爪
│   ├── cameras.py          #   腕部/俯视/前景 RGB-D 采集
│   ├── policy/             #   接口与实现
│   │   ├── base.py         #     Observation / Command / Decision / Goal / 协议
│   │   ├── scripted.py     #     脚本高层（铺层选石）+ 脚本低层（路径点）
│   │   └── vlm.py          #     Qwen3-VL 高层 + 低层（vLLM，OpenAI 兼容）
│   ├── execution.py        #   执行器：IK/伺服/接触/判定/报告（策略碰不到引擎）
│   └── task_config.py      #   YAML 默认值与校验
├── tools/                  # 任务自带工具
│   ├── probe_arm_geometry.py   # 逐臂实测几何（关节/TCP/开口曲线/可达范围）
│   ├── grasp_smoke.py          # 逐块石头真抓一次，量抓取成功率
│   ├── vlm_stub_server.py      # OpenAI 兼容桩服务（无 GPU 也能测 VLM 链路）
│   └── test_vlm_policy.py      # VLM 策略端到端测试（96 项）
├── docs/ARM_GEOMETRY.md    # 三套臂的实测几何与"文档与实测不一致"清单
├── reports/                # 自检报告、抓取报告、VLM 测试报告
└── scripts/                # 上游 ICRA2017 复现脚本（未改动，仍可单独跑）
```

## 支持的机械臂

| arm | 组成 | 关节 | 夹爪**净开口**（实测） | 石块缩放 | 基座 | 抓取倾角 |
| --- | --- | --- | --- | --- | --- | --- |
| `ur5e` | robosuite UR5e + Robotiq 2F-140 | 6 | 95.0 mm（指垫中心距 128.4 − 指垫厚 33.4） | 1.00 | (-0.30, -0.35, 0) | 0° |
| `panda` | MuJoCo Menagerie Franka Panda | 7 | 80.0 mm | 0.60 | (-0.30, -0.35, 0) | 0° |
| `piper` | MoonSim `assets/robots/piper` | 6 | 69.3 mm | 0.55 | (-0.24, -0.28, 0) | 15° |

夹爪的"能夹多宽"必须用**净开口**（指垫内表面之间的间距），不是指垫中心距：
UR5e 中心距 128.4 mm 看着能夹 110 mm 的石头，但指垫每片厚 16.7 mm，实际塞进去的
只有 95 mm。按中心距判"夹得住"会让指垫在下爪时直接撞上石头（实测把石头撞飞 26 cm）。
`GripperSpec` 因此把 `width_intercept/slope`（指垫中心距 ↔ ctrl 标定，用于控制）与
`pad_thickness_m` / `max_opening_m`（净开口，用于可夹性判定）分开。

`piper` 的 15° 倾角不是审美选择：实测它在"接近轴笔直向下"时关节 5 卡在限位，顶层槽位
差 18 mm 解不出来；倾 15° 后 20 个槽位 + 10 个料区口袋全部解到 0.1 mm 以内。细节与复现
命令见 [docs/ARM_GEOMETRY.md](docs/ARM_GEOMETRY.md)。

**加一套新臂**：在 `profile.py` 里加一个 `ArmProfile`（臂 MJCF、基座 body、关节名、home、
TCP site 或要现建的 site、夹爪执行器与开合命令、实测的开口方向/接近方向/指垫偏置、
石块缩放和工作区缩放），然后注册进 `ARM_PROFILES`。装配、IK、伺服、相机、策略都不用改；
`tools/probe_arm_geometry.py --arm <name>` 负责先把这些数字量出来（它目前只认
ur5e/panda/piper，加臂时要一并加进它的分支）。

## Policy 接口

策略分两层，各自独立可换（`--policy scripted|vlm|vlm-high|vlm-low`）：

```python
class HighLevelPolicy(Protocol):          # 每块石头问一次：下一块放哪
    def decide(self, obs: Observation, candidates: Sequence[Candidate]) -> Decision: ...

class LowLevelPolicy(Protocol):           # 每个控制周期问一次：末端下一步去哪
    def step(self, obs: Observation, goal: Goal) -> Command: ...
```

- `Observation`：时间、阶段、关节、TCP 位姿、夹爪开口、**多路相机 RGB-D**、石块真值、
  语言指令。`Observation.image_only()` 会剥掉真值——VLM 策略默认只拿图像和候选文字。
- `Decision`：石头 + 槽位 + 目标位姿，或者 `stop`。
- `Command`：**TCP 绝对目标位姿** + 可选夹爪开口宽度。执行器负责 IK、物理步进、接触、
  判定；策略拿不到 `MjModel`，所以"策略偷偷读引擎状态"在结构上不可能发生。
- `Goal.pick_pos/place_pos` 也是 TCP 位姿：指垫中心到 TCP 的固定偏置由执行器换算
  （搞混这两个空间会让末端稳定地停在偏 3 cm 的位置，实测踩过）。

两条实现：

| 实现 | 高层 | 低层 |
| --- | --- | --- |
| `scripted` | 先铺满低层、大石头优先放低层、按观测算支撑高度 | Goal 展开成路径点（接近/下降/闭合/抬起/转运/下降/张开/退开），到位或卡住就推进 |
| `vlm` | Qwen3-VL 在候选表里选一块石头 + 一个槽位，输出严格 JSON | Qwen3-VL 对目标位姿给增量修正（限幅），其余 Goal 交给脚本兜底 |

VLM 实现带完整的失败隔离：超时、HTTP 错误、JSON 解析失败、越界索引都会退到脚本策略，
并把计数写进报告。测试不需要 GPU：`tools/vlm_stub_server.py` 是 OpenAI 兼容的桩服务。

```bash
PYTHONPATH= /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python tools/test_vlm_policy.py
# -> 96/96 passed, reports/vlm_policy_test.json
```

接真模型（`bash assets/model/qwen3VL.sh` 起 vLLM，默认 `127.0.0.1:3001`）：

```bash
bash task2.sh stack ur5e --policy vlm --courses 3,2,1
```

## 三路相机

| 相机 | 装在哪 | 用途 | 默认 fovy |
| --- | --- | --- | --- |
| `wrist` | TCP 所在 body 上（跟着夹爪动） | 近距离对准抓取点 | 75–85° |
| `top` | 工作区正上方，垂直向下 | 全局俯视，看墙与料区 | 58° |
| `front` | 墙的右前方 45° | 前景视角，看整条臂与墙 | 45° |

- 全部离屏渲染，默认 `MUJOCO_GL=egl`（本机 osmesa 的 PyOpenGL 路径导入即失败、glfw 需要
  真实 GLX，只有 egl 可用）。本机软件渲染实测 640×480 约 9.5 ms/帧。
- 腕部相机是**实测安装**的：装配时先编译一遍模型量出 TCP 所在 body 的位姿，再把"相对
  TCP"的相机位姿换算成相对父 body 的静态位姿写进 MJCF。换臂不需要改数字。
- `CameraFrame` 带 RGB、米制深度、内外参；`jpeg_bytes()` 给 VLM，`depth_stats()` 给自检。
- 自检比的是**外参**不是图像差异：关节一动，世界相机同样会拍到臂在动。
  `run.py --check` 验证"腕部相机随臂移动、世界相机外参不动、三路画面都非空"。

## 配置

`task2.yaml` 覆盖全部可调项（命令行优先），未知键直接报错而不是静默忽略。要点：

```yaml
robot:
  arm: ur5e              # ur5e | panda | piper
  stones: 10             # 要放上墙的块数
  stone_pool: 0          # 候选池（0 = 自动：比槽位多 60%），策略从池子里挑夹得住的
  courses: [4, 3, 2, 1]  # 每层块数；够不到时装配阶段会自动削减并记录在报告里
  rock_style: moonsim    # moonsim（MoonSim 真实月岩）| paper | rough | natural
policy:
  name: scripted
  vlm: {base_url: http://127.0.0.1:3001/v1, model: Qwen3-VL-8B, max_delta_m: 0.05}
execution:
  max_placements: 0      # 0 = 放满所有槽位
  servo_seconds: 0.30    # 一个策略步伺服多久
```

## 工作区是怎么算出来的

墙固定在世界的原点附近（所有臂共用同一面墙，世界相机取景一致、结果可比），机械臂按
profile 摆在墙前，料区在"基座→墙"方位角两侧各 48° 的圆弧上、半径 0.72×臂展。

墙的距离不能随便定：太远最外侧槽位解不出来，太近最内侧槽位落到基座附近、肘部折不过来。
`build_workcell()` 用 `[min_reach + 0.651·span/2, 0.88·reach − 0.651·span/2]` 算可行区间，
区间为空就自动减掉最上层一块，并**把调整写进报告**（不静默改配置）。

抓取前只让**一块**石头上料区口袋，其余未上墙的石头回初始摆放位。上游做法是每块石头
各用一个口袋，但口袋间距只有 0.11 m 而石头长 0.19 m——没抓成功的石头会和下一块叠在
一起，物理直接炸开（实测抬升量出现 -4e8 mm 这种数）。

石块的世界位姿不能按生成器的顶点摆：编译后的几何用的是 `body ∘ geom_pos ∘ geom_quat`
（**不含** `mesh_quat`），按生成器顶点摆会整块陷进台面约 5 cm。`measure_ground_offsets()`
在编译后实测每块石头的贴面高度再摆。

## 当前状态

**已验证**（本机实测，命令与数字见 [STATUS.md](STATUS.md)）：

- 三套臂都能装配、编译、加载；每套 26 个位姿（10 个槽位 + 10 个口袋 + 边界）IK 全部收敛到 5 mm 内；
- 三路相机都能出图并带米制深度；腕部相机随臂移动 0.12–0.20 m，世界相机外参不动；
- Policy 接口、脚本策略、VLM 策略（含失败回退）通过 96 项端到端测试；
- 抓取/码放状态机能走完全程（接近 → 下降 → 闭合 → 抬起 → 转运 → 下降 → 张开 → 退开）。

**还没做到的**：石头夹不住。夹爪能合到石头两侧（指垫开口停在 88–118 mm，说明确实贴上了
石头），但抬起时石头不跟着走：末端抬了 130–190 mm，石头只动了 0–18 mm，抬起到位时接触
数掉到 1–3（只剩石头与台面）。硬化合拢（命令开口降到 0.6×石头宽度以下）会让接触求解炸开
（抬升量出现 -1e4～-1e5 mm 量级）。这是当前唯一的硬缺口，排查记录与已排除的原因见
[STATUS.md](STATUS.md)。

## 与旧代码的关系

- `scripts/`（ICRA2017 复现、稳定性规划器、官方 UR5e 执行器）**一个字节都没改**，
  仍按各自文档单独运行；它们与 `run.py` 是两条独立路径。
- `stone_stack/` 里上游的石块生成、`rocks.py`、`paper_scene.py` 等模块保持不变；
  本层新增的是 `robots/`、`cameras.py`、`policy/`、`execution.py`、`task_config.py`。
- 复用了上游的场景工具（`stone_body`、石块视觉网格与纹理、robosuite 资产的 mesh
  绝对化与显示清理），避免两套实现漂移。
