# LunarBench / MoonSim

月面机器人仿真与任务研发。GitHub 仓库名为 **LunarBench**，当前代码目录与实现名为 **MoonSim**。

![系统架构](docs/images/architecture.png)

架构图展示整体设计方向；当前实现状态以各任务文档为准。图中的 RL/VLA/WAM 等模块不代表所有方法都已接入，MJX GPU 物理尚未实现。

## 获取代码

```bash
git clone --recurse-submodules https://github.com/George3215/LunarBench.git
cd LunarBench
```

已有 checkout 执行 `git submodule update --init --recursive`。DrQ-v2 和 GPT-as-Policy 以固定提交的 submodule 引用。地形、机器人网格、模型权重和运行产物不随源码上传，准备方式见 [资源说明](assets/README.md) 和 [源码与依赖范围](docs/REPOSITORY.md)。文档中的 `/home/lry/...` 是本机路径；新机器优先使用下面的仓库相对路径命令。

月面机器人仿真仓库。当前只保留五个核心目录，按实际需要逐步扩展。

```text
MoonSim/
├── assets/       # 统一资源：机器人、地形、物体、障碍物
├── ue/           # UE 场景导入、渲染和已有 demo 的显示逻辑
├── mujoco/       # 机器人动力学、碰撞和控制
├── tasks/        # TASK1 石头收集；TASK2 Qwen 直接控制与强化学习堆叠
├── tools/        # 离线资源转换、启动、部署和验证脚本
└── README.md
```

```text
             assets
            /      \
           UE     MuJoCo
            \      /
              tasks
```

**资源 → UE / MuJoCo → Task。**

## 当前能运行什么

已有月面地形 + Go2 demo。UE 显示完整月面和岩石，MuJoCo 计算机器人与局部地形、岩石的接触。既有 Go2 checkpoint 继续用于 demo 的步态控制，不新增算法接入。

- `assets/environments/`：既有月面等环境资源。
- `assets/robots/go2_demo/`：demo 的 Go2 MJCF、网格、checkpoint、许可与来源资料。
- `ue/import_data/`：已有 R16、岩石实例与材质等离线导入产物。
- `ue/importer/`：原有 UE 资产导入和视口控制源码，属于 UE 实现细节；不建设通用 Plugin 系统。
- `mujoco/generated/`：demo 的固定物理场景、局部高度场与显示数据。
- `mujoco/go2.py`：原有 Go2 仿真；`motion.py` 负责位移目标控制，`streaming.py` 负责局部碰撞窗口。

TASK1 使用 `bridge/` 统一传感器与动作边界，独立 baseline 不读取引擎对象。既有 UE 场景显示 UDP 仍保留在引擎内部。

TASK1 的 UE RGB-D 与 MuJoCo LiDAR 已接入 bridge，默认运行 Z-Mobile-manip 传感器与接近基线。`tasks/` 里目前有 TASK1 与 TASK2（见下两节），不提前创建导航、建造等空任务系统。TASK2 使用独立的 MuJoCo 环境与策略接口，不共用 TASK1 的 bridge。

## TASK3 地形导致的目标重新可观测性

本节介绍**任务与地形定义**本身。下方的实验章节是历史记录，其结论已经过审计修正，见文末说明。

### 任务

机器人**曾经看到过**目标，之后因为自身运动与地形几何而暂时看不到它。目标没有移动，传感器也没有失效——挡住视线的是地形。任务是在有限动作预算内选出一串物理动作，让目标重新进入相机视野。

成功由**相机**判定：目标在画面中达到最小可检测面积并连续保持若干帧。一个偶然露出的像素不算成功，因此区分 `re-observation`（核心指标）与 `usable re-observation`（面积足以支持后续使用）。

这不研究目标检测、目标搜索或地图探索。它也不假设目标位置未知——恰恰相反，**知道目标在哪并不足以决定该做什么动作**，这正是任务的定义性困难。

### 地形：8 类遮挡机制

地形按**物理遮挡机制**组织，不按表面名称，也不作为难度阶梯。每类是一个真实场景类比：

| 机制 | 轴 | 真实类比 | 可见性效应 |
| --- | --- | --- | --- |
| `barrier` | 遮挡方向 | 墙体、巨石、货架 | 单侧遮挡，绕到另一侧即恢复 |
| `trench` | 高程 | 沟渠、壕沟、沟壑 | 落入后视线被自身边缘挡住 |
| `stairs_up` | 高程 | 楼梯、平台 | 抬升过程中反复遮挡 |
| `pit` | 空间尺度 | 坑洞、陨石坑 | 局部凹陷产生短暂视野 |
| `basin` | 空间尺度 | 谷地、洼地 | 大尺度凹陷，与 `pit` 同轴异端 |
| `waves` | 可见性动态 | 沙丘、起伏地面 | 沿路径多次翻转 |
| `perlin` | 地形复杂度 | 自然越野地面 | 空间相关的随机起伏 |
| `perlin_rough` | 地形复杂度 | 碎石、岩场 | 同上的强起伏版本 |

**核心现象是可见性随运动的非单调性**：

$$\text{Closer} \not\Rightarrow \text{More Observable}$$

沿一条走向目标的路径，可见性序列会反复翻转（如 `stairs_up` 的 `V X V X`）。假设单调性的策略不是不够精确，而是瞄准了错误的目标。

**两条框架规则写进了代码，不靠自觉**：机制表遇到 `challenging` / `hard` / `difficult` / `adversarial terrain` 时**拒绝渲染**，因为困难在目标的可见性而不在驾驶；`perlin` 与 `perlin_rough` 文档中明确写为空间相关高程的**程序化表示**，不声称是真实地表。

所有地形用精确 box slab 构造（`slabs.py`），不是 height field——实测 MuJoCo `hfield` 碰撞体会让两份策略都走不动，而 box slab 的表面高度与解析可见性 oracle 完全一致（同一个物体）。

### 冻结接口

`api.py` 定义 `Environment` / `Observation` / `ReObservationPolicy` 三方契约，版本 `task3-reobservability/1.1`：

- **地形永远不知道哪个策略在跑**，所以新增一个 baseline 绝不需要改动地形代码。
- 动作是十个固定原语：`forward` `back` `left` `right` `turn_left` `turn_right` `toward` `heading_lock` `left_hold` `right_hold`。
- 一个 step 是一个原语，**不是一米，也不是一次独立决策**；路径长度从执行轨迹实测，不从指令值相加。

`episode.py` 定义多步回合、终止集（`success` / `budget_exhausted` / `fallen` / `out_of_bounds`）、以及直接在逐步可见性轨迹上求值的 `RSR@K`。契约测试见 `test_api.py`（11 组，全通过）。

### 查看地形

```bash
python tasks/task3_reobservability/terrain_viewer.py --list
python tasks/task3_reobservability/terrain_viewer.py --check --terrain all
python tasks/task3_reobservability/terrain_viewer.py --terrain perlin --speed 0.6
python tasks/task3_reobservability/test_api.py
```

![TASK3 地形](docs/images/task3_terrains.png)

### 平台与动作接口

环境是 Go2+Piper（`third_party/LeggedManip_Lab` 的完整模型）+ 运行时注入的机身 RGB-D 相机。**不训练、不连真机、不授权 Go2 运动。**

运动控制器有两份，都由 `--controller` 选择，都是第三方预训练权重原样使用：

| `--controller` | 来源 | 本机结论 |
| --- | --- | --- |
| `leggedmanip` | 仓库 `third_party/LeggedManip_Lab` 自带 | 平地正常行走，爬不上 0.15 m 台阶 |
| `go2rl` | [`wty-yy/go2_rl_gym`](https://github.com/wty-yy/go2_rl_gym) MoE-CTS（MIT） | 平地 0.88 m/s，**爬得完十级楼梯** |

### 历史实验（结论已修正）

> **以下 Experiment 1–4 是历史记录。**
>
> 后续审计确认：主实验的成功判定未以相机检测为准、目标记忆中含真值目标坐标、且 Random 与 Learned 的场景分母不一致。因此旧结果**不再支持**"非特权视觉闭环优于随机策略"这一结论。
>
> 仍然成立的：**H1 观测歧义、H2 记忆不足、H3 几何可见性 ≠ 物理可重观测**三组对照实验，以及地形与接口本身——它们不依赖上述缺陷。修正后的结论与相机闭环实验另见 `paper/` 目录，该目录不在本任务接口的发布范围内。

**Experiment 2（动作条件未来可观测性）** 在 5 个地形上，从冻结的丢失状态物理执行每个候选动作，按 10 Hz 采样完整曲线 $V(a,t)$，并对同一张表套四条决策规则（A 永远向前、B 朝记忆方位直走、D 保持丢失瞬间朝向、C 用仿真把每个候选动作跑一遍再选）：

| 场景 | forward | back | left | right | left_hold | right_hold | turn_l | turn_r | toward | heading_lock | 最佳 | A | B | D | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `stairs_up` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | back | 1 | 1 | 1 | 1 |
| `trench` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | toward | 1 | 1 | 1 | 1 |
| `pit` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | back | **0** | **0** | **0** | **1** |
| `barrier` | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | left | **0** | **0** | **0** | **1** |
| `waves` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | forward | 1 | 1 | 1 | 1 |
| `basin` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | forward | 1 | 1 | 1 | 1 |
| `perlin` | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | forward | 1 | 1 | 1 | 1 |
| `perlin_rough` | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | right | 1 | 1 | 1 | 1 |

**不存在一个通吃的固定动作**（forward 6/8、back 6/8、侧移 3/8、转视角 **0/8**），最佳动作随地形改变，`pit` 与 `barrier` 上 A/B/D 全部失败而只有 C 成功。但也要注意：**其余六个场景区分不了 IMU**——`heading_lock` 的成功/失败模式与 `forward` 一致。

**Experiment 1（History Necessity）** 才把这个区分开，并且给出了速率：

![TASK3 History Necessity](docs/images/task3_history_necessity.png)

每对世界局部完全相同（同一面墙、同一扇窗、同一丢失状态与朝向），差别只是目标藏在窗口哪一侧，因而"最后一次看到它"的位置不同。成对世界由 `history_pairs.py` 按 7 条**预注册的纯几何判据**自动搜索接受（49 个候选 → 8 对通过），不是手挑的。

| 指标 | 8 对 / 16 个世界上 |
| --- | ---: |
| 丢失状态渲染**逐像素完全相同**的对数 | **8 / 8**（平均像素差 0.0、最大像素差 0、100% 像素一致、目标像素 0/0） |
| 最优动作相反的对数 | **8 / 8** |
| **没有任何与历史无关的动作可行的对数** | **8 / 8** |
| A 反应式 / B History-only / D IMU 朝向锁定 / **C 反事实** | **0% / 0% / 0% / 100%** |

也就是说：**$O_A = O_B$ 逐像素相同，$a_A^* \neq a_B^*$，且不存在一个"只看现在"的策略能同时解决两个世界。** 这同时证明了 `heading preservation ≠ target-relative spatial preservation`（D 在两个世界都保持了丢失瞬间的朝向，仍只成功 1/16），以及**知道目标在哪 ≠ 知道该做什么动作**（B 朝记忆中的目标直走，成功率 0%——正确动作是向另一侧平移）。

**Experiment 3（学习式可观测性预测）** 把上面的 oracle 上界换成训练出来的预测器。数据直接复用已录好的物理曲线（2184 个样本 / 13 个组），leave-one-group-out 评估：

![TASK3 学习式可观测性预测](docs/images/task3_visibility_model.png)

| 特征集 | OOF AUC | 成对历史世界（16）RSR | 未见地形族（8）RSR |
| --- | ---: | ---: | ---: |
| 只用当前观测（无记忆） | 0.794 | 0.50 | 0.375 |
| 记忆被打乱的对照 | 0.791 | 0.4375 | 0.375 |
| **观测 + 目标记忆** | **0.898** | **1.00** | 0.50 |

在成对历史上，**有记忆的学习式预测器达到 oracle 水平（16/16）**，打乱记忆的对照塌回 0.4375（增益确实来自记忆）；只用当前观测的版本塌成近似常数策略（24 个世界里 18 次都选同一个动作）——这正是 H2 在学习式策略层面的复现。**但在没见过的地形族上它只有 0.50，比 A/B/D 的 0.75 还差**，所以目前**不能声称可泛化**。

**Experiment 4（History Horizon，P0）** 回答"到底需要多少历史"。$H_k$ 一次定义死（10 Hz 观测，特征维度对每个 horizon 都是 36 维，只有窗口回溯多远在变），并过滤掉"此刻目标可见"的样本以免标签泄漏：

![TASK3 History Horizon](docs/images/task3_history_horizon.png)

| horizon | H0 | H1 | H2 | **H4** | H8 | H16 | H32 | H64 | Hfull |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 决策 RSR | 0.500 | 0.500 | 0.458 | **0.833** | 0.792 | 0.833 | 0.750 | 0.792 | 0.792 |
| 预测 AUC | 0.816 | 0.816 | 0.832 | 0.835 | 0.847 | 0.846 | 0.896 | **0.925** | 0.927 |

**决策在 H4（0.4 秒）就跳到天花板 0.833 并再也不涨；预测 AUC 在 H64（6.4 秒）饱和，完整轨迹只多 0.002。** 所需 horizon 由"目标最后一次被看到有多久"决定（中位数 4.8 秒、最大 8.4 秒），不是一个任意数字。

一个反直觉的分解：**目标记忆单独要 H64 才有用，而"最近运动摘要"在 H4 就到 0.79–0.83**——真正起作用的不是显式的"目标在哪"，而是**机器人是从哪个方向走过来的**。

**必须说清楚的**：决策指标只在 **24 个世界**上评测，分辨率 1/24，所以 H4 之后 1–2 个世界的差别在噪声内，**不能**读成"哪个 horizon 最好"；可靠的只有 H0→H4 的跳变和 AUC 的饱和点。把决策世界数提到 ~100 是下一轮第一件事。

```powershell
# 地形：列出全部、自检、交互式查看并驱动机器狗
python tasks/task3_reobservability/terrain_viewer.py --list
python tasks/task3_reobservability/terrain_viewer.py --check --terrain all
python tasks/task3_reobservability/terrain_viewer.py --terrain perlin --speed 0.6

# ---------------------------------------------------------------------------
# 以下命令属于论文实验，脚本不随本任务接口发布。
# 本任务发布的是地形与任务定义（api.py / scenes.py / slabs.py / terrain.py /
# benchmark.py / episode.py），以及 terrain_viewer.py 和 test_api.py。
# 想复现下面这些实验，需要论文仓库中的实验脚本，直接运行会 FileNotFoundError。
# ---------------------------------------------------------------------------
python tasks/task3_reobservability/figure_terrains.py
python tasks/task3_reobservability/figure_scenes.py

# Experiment 2：V(x) profile（秒级）或完整物理基准
python tasks/task3_reobservability/benchmark.py --mode profile
python tasks/task3_reobservability/benchmark.py --mode all
python tasks/task3_reobservability/figure_benchmark.py

# Experiment 1：搜索成对世界 + 观测等价性 + 动作对照
python tasks/task3_reobservability/history_pairs.py --pairs 8 --seed 11
python tasks/task3_reobservability/history.py --pair all --render-dir tasks/task3_reobservability/generated/history_camera
python tasks/task3_reobservability/figure_history.py

# Experiment 3：学习式可观测性预测器
python tasks/task3_reobservability/visibility_model.py
python tasks/task3_reobservability/figure_model.py

# Experiment 4：History Horizon
python tasks/task3_reobservability/history_horizon.py
python tasks/task3_reobservability/figure_horizon.py

# Experiment 0：几何 oracle + 机载相机证据
python tasks/task3_reobservability/experiment.py --terrain staircase10 --vision-mode full --save-terrain --save-camera

# ---------------------------------------------------------------------------
# 上面这段"不随本任务发布"的范围到此结束。
# realsense.py 随任务发布（独立真实 RealSense 探针，不动机器人）。
# ---------------------------------------------------------------------------
python tasks/task3_reobservability/realsense.py
```

**统一的 Benchmark API（接口冻结）**：从这一轮开始所有实验共用同一套接口，不再各自定义 action interface，地形与策略彻底解耦。

```text
Environment → Observation → ReObservationPolicy.act → action_id → Environment.rollout
```

`Observation` = 当前 RGB + 当前 depth + 本体感受 + 动作历史 + 目标记忆；真值（可见性、目标世界坐标）只存在于 `PrivilegedState`，只有评估器和 `Oracle` 能读。八个 baseline（Reactive / HeadingLock / HistoryOnly / LostTargetSearch / ObjectPermanence / NBV / Learned / Oracle）实现同一个 `ReObservationPolicy`，各自只保留论文的**核心决策机制**。

**"地形不感知策略"不是靠约定，是靠测试**：

```powershell
python tasks/task3_reobservability/test_api.py        # 契约测试，失败退出码非零
python tasks/task3_reobservability/benchmark_api.py   # 24 个地形 × 8 个策略
```

契约测试会 AST 扫描地形模块的 import、在子进程里单独 import 地形确认没把策略栈拖进来、检查 `Observation` 没有特权字段、并验证每个策略在每个地形上都返回合法动作。它已经抓到一个真 bug：`Environment.rollout` 原本会推进仿真状态，也就是说 **oracle 自己的探测会把机器人挪走**——现在 `rollout` 保存/恢复状态。详见 [TASK3 文档](tasks/task3_reobservability/README.md) §9。

**统一 baseline 的结果（24 地形 × 7 策略）**：

| 策略 | 多地形场景（8） | **成对历史世界（16）** |
| --- | ---: | ---: |
| Reactive / HeadingLock / HistoryOnly / LostTargetSearch / ObjectPermanence / NBV | 0.75 | **0.00** |
| Oracle | 1.00 | 1.00 |

**在真正有区分力的 16 个成对世界上，六个经典 baseline 全部为零，只有 oracle 满分。** 正确答案是把目标那一侧**反方向**的横向平移，而 `HistoryOnly` 直奔记忆中的目标、`LostTargetSearch` 原地转向、`ObjectPermanence` 对静止目标塌成同一条规则、`NBV` 信了一个物理不认的几何重投影。

**但当前协议是单步决策，对多步算法不公平**：`LostTargetSearch` 和 `ObjectPermanence` 只被评了第一步（转向），它们的 0.00 是失败率上界而非公平复现。扩成多步 episode 是统一 benchmark 的第一优先修复项。

**多步 episode 协议（已完成）**：单步协议被**多步 episode** 取代——接口不变，只改 evaluator。两个协议共用同一预算、同一观测模型、同一终止规则：

- **Protocol A 开环**：只问一次策略，之后重复同一动作、不再观测
- **Protocol B 闭环**：每一步后用新观测重新决策（$\pi(a_t \mid H_t)$），**以它为主**

公平性由测试强制：所有 primitive 每步**等时长**；预算必须**真的给满**（测试把可见性强制成"永远看不见"来验证）；终止条件统一为 `success` / `budget_exhausted` / `fallen` / `out_of_bounds`；`RSR@N` 单调，所以一次跑最大预算就得到整条曲线。

指标分开报：`RSR@1/@3/@5/@10`、`T_reobs`、`action cost`、`yaw change`、`steps`。$J = \mathbf{1}_{\text{reobs}} - \lambda C$ **故意不报**——权重会在看到数据前就被定死。

**步长是量出来的，不是拍的**：第一版取 1.5 s，结果**连 oracle 都只有 0.46**，benchmark 没有 headroom。查下去发现 1.5 s 时 7 个场景里有 4 个"没有任何动作能在一步内重新看到目标"。3.0 s 时 7/7 场景都有可用动作、贪心 oracle 五步内解 6/7，所以定 3.0 s。详见 [TASK3 文档](tasks/task3_reobservability/README.md) §9.4。

**必须同时说清楚的**：反事实基线 C 是把每个候选动作在 MuJoCo 里真跑一遍再选，是上界 / teacher，不是可部署策略。学习式预测器目前只在自己见过的地形族上达到 oracle 水平，**在没见过的地形族上低于朴素基线**。Experiment 1 的 8 对世界共用同一个 `window_wall` 地形族，结构与历史深度都有限。控制器能力也限制了可走地形复杂度，从而限制可见性翻转次数——详见 [TASK3 文档](tasks/task3_reobservability/README.md)。

`oracle` 是评估用的特权几何真值，不是学习到的世界模型；也**不能**声称已经首次研究目标丢失后的重新搜索、首次用主动感知解决重新观察。`go2rl` 权重不是本仓库训练的，也未在真机验证。

详见 [TASK3 文档](tasks/task3_reobservability/README.md)。

## TASK1 石头收集

![TASK1 石头收集示意](docs/images/task1.png)

此图用于任务展示，不是自主收集成功的实验记录。

已实现独立的 [TASK1](tasks/task1_collect/README.md)：20×20 m、1 cm 原始地形采样、150 块封闭三维实体石头（最长边 3.9–5.4 cm，逐块在基准体积的 1.5–2 倍之间随机，Piper 夹爪抓得住）、0.3 m 白色实体立方体收集框、默认 Go2-Piper 四足带臂机器人，以及任务内部的 observation/action/state/reward 接口。当前默认关闭得分、持续运行；RGB-D/LiDAR 与独立策略说明见 [baseline 文档](baselines/z_mobile_manip/README.md)。TASK1 使用原始 27.94 m 地形尺度；下文的旧 demo 保留原有场景尺度。两个入口的产物和物理进程分别管理，不同时启动。

```bash
bash /home/lry/MoonUnrealEnv/task1.sh
```

## TASK2 十块石头堆叠

固定 UR5e + Robotiq 2F-140、十块 paper 石头与可达料区，目标为 **4+3+2+1**。
当前两条主线共用场景与接触支撑成功判据：

| 策略 | 观测 | 控制 |
| --- | --- | --- |
| Qwen 直接控制 | `eye_in_hand` / `top` / `front` RGB 与本体状态 | 模型选择 TCP 与夹爪目标，Python 校验和执行 |
| 强化学习 | DrQ-v2 三视角 RGB；SAC 特权状态对照 | 7 维末端增量与夹爪动作，5 cm / 0.35 rad 门限 |

在已有依赖和资产的机器上，从仓库根目录运行：

```bash
# Qwen：需要可用的 VLM 服务，地址配置在 YAML 中
python tasks/task2_stack/run.py --config tasks/task2_stack/configs/upstream_qwen_trials.yaml --view

# RL：8 个 CPU MuJoCo 环境，共享一个 CUDA DrQ-v2 learner，总计 20 万步
WANDB_MODE=online bash tasks/task2_stack/rl.sh --config tasks/task2_stack/configs/rl_stack10_drqv2_200k_parallel.yaml --mode train
```

首次安装与 W&B 登录见 [RL 文档](tasks/task2_stack/docs/RL.md)。每回合上限 1000 步；所有环境合计 200000 个 transition。CPU 物理多进程采样已经接入，**不是 MJX GPU 物理**。训练输出、checkpoint、W&B 本地文件均留在本机。

已完成短训练、网络参数更新、保存/加载与并行运行验证；**尚无十块石头完整堆叠成功的已验证策略**。历史脚本保留作对照，不作为 Qwen 或 RL 的在线抓取兜底。

详细说明：[TASK2](tasks/task2_stack/README.md) · [RL 训练](tasks/task2_stack/docs/RL.md) · [Qwen 提示词与控制](tasks/task2_stack/docs/UPSTREAM_DIRECT.md)。

## 启动已有 demo

以下命令可在本机任意目录执行。切换模式前，先关闭上次启动的窗口。

仅 UE：静态显示月面和 Go2，不启动物理。

```bash
python3 /home/lry/MoonUnrealEnv/MoonSim/tools/run_demo.py --engine ue
```

仅 MuJoCo：显示出生点附近 10×10 米物理区域，运行机器狗。

```bash
python3 /home/lry/MoonUnrealEnv/MoonSim/tools/run_demo.py --engine mujoco
```

保留原有联合 demo：同时打开两个窗口，MuJoCo 计算物理，UE 接收位姿；两个窗口同步相机。

```bash
python3 /home/lry/MoonUnrealEnv/MoonSim/tools/run_demo.py --engine both
```

联合模式使用英文输入法，点击 UE 视口控制，无需点击 Play。单独运行 MuJoCo 时，在 MuJoCo 窗口中控制。

| 按键 | 功能 |
| --- | --- |
| W / S / A / D | 前 / 后 / 左 / 右移动 0.5 米 |
| Q / E | 左 / 右旋转 15° |
| H / L | 降低 / 提高速度档位 |
| 空格 | 停止 |
| R | 重置 |
| P | 暂停 / 继续 |

UE 视口失焦或右键相机操作会停止移动。联合模式的 MuJoCo 窗口用于观察，不接受机器人运动指令。关闭 UE 会停止它创建的物理进程。

## 显示、精度与相机同步

联合模式下，两个窗口共享相机位置、朝向、垂直视场角与宽高比：UE 右键拖动旋转相机；MuJoCo 左键拖动旋转、右键拖动平移、滚轮缩放，Shift 切换水平操作。任一侧操作都会更新另一侧，相机也跟随机器狗平移。UE 用临时相机约束到 MuJoCo 窗口的宽高比，因此可能出现黑边。两者材质、光照及远景覆盖不同，保证的是视角和投影一致，不是 RGB 像素一致。

通信沿用本机 UDP：19400 发送机器人和相机状态，19401 传递既有机器人键盘控制，19402 回传 UE 相机操作。接收端只解码最新状态，不重播积压帧；相机编辑用序号确认，避免旧状态把用户视角拉回。物理固定 500 Hz、状态/显示目标 50 Hz、UE 视口上限 45 FPS，彼此不等待渲染完成确认。操作系统调度和启动加载仍会影响瞬时帧时间；这不是逐帧锁步录制接口。

UE 默认后台节流会在切换到 MuJoCo 窗口后明显降帧。demo 激活期间临时关闭此选项，离开地图或退出时恢复，不保存全局编辑器偏好。MuJoCo 使用原生 GLFW 窗口，动态碰撞区域切换时只重建渲染资源，保留窗口与相机。

精度约束：

- 地形原始网格为 **2795×2795**，原始单位换算后的间距是 1 cm；现有 UE 场景 XY 缩放为 50，所以世界间距是 **0.5 m**。MuJoCo 的 10×10 m 局部窗口为 21×21，逐点截取源数据，没有抽样或插值，也不人为平整地形。
- 岩石传入完整顶点与原始三角面；TASK1 的 **123 块原型**（Apollo 扫描件 + lunar 程序化岩石，5000–6146 顶点、9520–12288 三角面）全部水密，带 UV 与基色贴图。显示不再由顶点凸包替代。
- 机器人使用原始 OBJ；隐藏仅用于碰撞的 group 3 简化形体，不修改碰撞计算。
- 岩石碰撞仍采用 MuJoCo 原生凸网格碰撞。完整显示面保留不代表已经实现凹形三角网格碰撞。MuJoCo 显示局部物理区域，UE 显示完整环境。

`tools/validate_precision.py` 审计原始 USD、R16 和编译后岩石面数（需要 `usd-core`）；`tools/validate_camera_sync.py` 通过真实窗口鼠标操作及 UE 实际投影矩阵验收双向相机同步。相机验收会临时切换窗口焦点及输入法，并关闭它启动的进程。

运行中的诊断在 `mujoco/generated/latency_current.json`、`runtime_metrics.json` 及对应 JSONL：可分别查看 UE 帧间隔、最新状态年龄、回调耗时、相机误差、物理实时率和渲染资源重载耗时。初次加载与正常运行应分开统计。

## 环境与路径

本机环境：UE 5.7.4；外部工程 `/home/lry/文档/Unreal Projects/Moon/Moon.uproject`；Python `/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python`。MuJoCo 依赖固定版本见 `mujoco/requirements.txt`。

换机器可设置 `LUNARBENCH_UE_EDITOR`、`LUNARBENCH_UE_PROJECT`、`LUNARBENCH_PYTHON`。这些环境变量保留旧 demo 命名以兼容部署；启动器自动将 `LUNARBENCH_WORKSPACE` 设置为当前 MoonSim 目录。不需要安装 Python package。

`LUNARBENCH_UE_MAP` 决定 UE 加载哪张地图，默认为原 demo 地图 `/Game/MoonGo2/Maps/MoonTerrain_Go2_FullRender`。它同时会被 C++ 侧读取（`MoonPaths::IsGo2Map`），用来判断当前世界是否该由插件接管。TASK1 也可用 `--ue-map` 传入同一个值。

UE 工程与已导入的地图仍在仓库外。更新 UE C++ 源码后需关闭编辑器并部署构建：

```bash
python3 /home/lry/MoonUnrealEnv/MoonSim/tools/deploy_ue_plugin.py --project '/home/lry/文档/Unreal Projects/Moon/Moon.uproject' --engine /home/lry/UE/UE_5.7.4
```

工具先备份已有插件源码，不重建地图、不覆盖原始地形。

## 地形换皮地图

`/Game/MoonMacro/Maps/MoonTerrain_Macro01_FullRender` 是原 demo 地图的副本，只把地形材质换成了 `moon_macro_01_4k` 贴图集。几何完全一致：高度数据、transform、岩石、Go2 显示部件全部原样带过去，源地图与源材质一律只读。

```bash
python3 /home/lry/MoonUnrealEnv/MoonSim/tools/build_moon_macro_map.py     # 生成
python3 /home/lry/MoonUnrealEnv/MoonSim/tools/validate_moon_macro_map.py  # 验收
LUNARBENCH_UE_MAP=/Game/MoonMacro/Maps/MoonTerrain_Macro01_FullRender \
  bash /home/lry/MoonUnrealEnv/task1.sh
```

材质分两层：母材质 `M_MoonMacro_Landscape` 放 `UV_TileCm` / `NormalScale` / `Specular_Level` 三个参数，每张地图对应一个 MIC 烘自己的值（当前 `MI_MoonMacro01_Landscape`）。以后加地图 = 新 MIC + 新地图资产，母材质不动。

**UV 用世界位置投影，不是 Landscape UV0，这一点有实际后果。** `ALandscape` 的 UV0 是地形空间的 quad 坐标、与 actor scale 无关，而 TASK1 运行时 `task_scene.py` 会把 scale 从地图里烘焙的 `[50,50,25]` 覆盖成 `[1,1,1]`——两种状态下世界尺寸差 50 倍（1397 m vs 27.94 m）。按地形空间平铺的话，demo 态下 4K 贴图会被铺到 1397 m 上变成约 0.1 px/cm，彻底糊掉。世界投影则只是重复次数随尺寸变、永远不会糊。

`UV_TileCm` 默认 1000 cm（4K 贴图约 2.4 mm/像素）：TASK1 态重复约 2.8 次，demo 态重复约 140 次。**这个值是按 TASK1 态标定的**，demo 全景下重复感明显——改 MIC 参数即可调整，不必重建材质。

代价：新 `.umap` 约 185 MB；三张 4K 贴图 `NeverStream=true` 常驻显存约 53 MB。

源贴图里的 `moon_macro_01_disp_4k.png` **未入库也未导入**：本仓库的地形物理是 1 cm 原始高度场、UE 必须渲染同一份几何，位移贴图会破坏这条不变量。也没有生成 `_nor_dx` 派生文件，改用 `UTexture::bFlipGreenChannel` 翻转 OpenGL 约定的绿通道。

## 资源工具与检查

`tools/prepare_unreal_landscape.py`、`prepare_scene_v2.py` 是原有 USD 离线处理工具；`prepare.py`、`export_patch.py`、`export_visual.py` 准备 demo 的物理/显示产物。转换工具按需运行，不在每次启动时重新生成地形。USD 工具额外需要 OpenUSD（`usd-core`）；策略导出工具额外需要 PyTorch，普通 demo 不需要这两项。

`assets/robots/` 下另有五个自包含机械臂目录（`ur10`、`ur10e`、`ur12e`、`so101`、`piper`），每个含 MuJoCo 模型 `arm.xml`、给 UE 的 `.usdz`、上游来源与逐文件校验记录。`tools/prepare_arms.py` 从上游 xacro/URDF/MJCF 生成，`tools/validate_arms.py` 验收 MuJoCo 载入与步进、home 位姿自接触与重力下垂、末端随动，以及 `.usdz` 的连杆坐标系与网格摆位是否逐一对上 MuJoCo 自身运动学。这几套臂目前只是可加载可仿真的资产，没有注册进 TASK1 的机器人 profile，也没有 UE 侧物理或地图改动。

`assets/robots/franka/` 与 `assets/robots/robotiq_2f85/` 不是生成的，是从 MuJoCo Menagerie 原样搬运的 Menagerie 模型（`panda.xml` 与 `2f85.xml`），只有 MuJoCo、没有 `.usdz`。`franka`（Panda）已经作为 TASK2 的可切换机械臂之一接入（见 [TASK2 文档](tasks/task2_stack/README.md)）；`robotiq_2f85` 目前只在 `.local/backups/task2_arm-20260921-183512/` 那一版空白环境里用过。

无界面物理与碰撞检查：

```bash
PYTHONPATH= /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python /home/lry/MoonUnrealEnv/MoonSim/mujoco/run.py --headless --seconds 6 --distance 0.5
PYTHONPATH= /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python /home/lry/MoonUnrealEnv/MoonSim/tools/validate.py
```

`tools/validate_demo_gui.py ue|mujoco|both` 用于本机有界图形验收，会关闭它创建的验证进程。联合验收会短暂操作窗口和输入法并恢复；正常使用请执行 `run_demo.py`。

## 扩展原则与当前边界

新增资源放 `assets/`；引擎相关功能放对应引擎目录；明确的任务逻辑放 `tasks/`；离线工具放 `tools/`。出现实际重复或接入需求时再抽象，不提前建设公共 API。

demo 保持原有地球重力 `-9.81 m/s²`；月球低重力稳定步态未验证。UE 只负责显示，MuJoCo 是唯一物理源。当前是固定资产专用 demo，不承诺任意 USD / 机器人自动可运行。第三方资产公开再分发权尚未确认。

## 本次目录迁移后的实际验证

2026-09-17，本机 UE 5.7.4 / MuJoCo 3.13.0：

- UE 插件重新构建成功；仅 UE 模式加载月面与 Go2，未启动物理子进程。
- 仅 MuJoCo 图形模式运行 6.002 秒，3001 个物理步，正常退出，未跌倒或越界。
- 联合模式收到真实 UE W 按键后，机器人前移 0.48931 米，目标余量 0.00954 米；33 个 UE 显示部件的位姿读回误差为 0 厘米。
- 无界面动态窗口测试运行 12 秒、6000 步，完成一次窗口切换，状态有限且未跌倒。
- 441 个地形高度检查和 7 个岩石接触检查通过。

本机证据保存在外层工作区 `.local/validation/moonsim_*` 日志及 `ue_process.json`、`mujoco_process.json`、`both_process.json`；这些是 demo 工程验收，不是算法或 Benchmark 成绩。

## 通信、精度和相机修复验收（2026-09-17）

本机实际窗口测试：后台节流时典型帧间隔约 334 ms；修复后双向鼠标相机测试的预热后采样中位数为 22.62 ms（约 44 FPS），状态年龄中位数 16.25 ms、UE Python 回调中位数 0.74 ms。相机投影检查通过，三个参考点在 1280×720 归一化画幅上的最大误差小于 0.00004 像素。这里的误差是同一相机状态的投影计算误差，不表示跨进程屏幕刷新零延迟；该短测试按秒采样，也不是完整帧时间分布。

原始 USD 到 R16 的全量高度对比误差为 0，两类岩石编译后面数与原始数据一致。完整 mesh 的 MuJoCo 图形运行完成 12 秒 / 6000 步、一次碰撞区域切换，前进约 1.93 m，未跌倒；切换时渲染资源重载约 31 ms。最终联合模式真实 W 按键回归前移约 0.468 m，剩余目标距离约 0.031 m，未跌倒。

本机详细证据：外层 `.local/validation/performance_camera_precision.json`、`camera_sync_live.json`、`viewer_stream.json` 与 `both_process.json`。修复涉及的源文件和原场景备份位于外层 `.local/backups/performance-camera-20260917-205201/`，UE 部署工具另行保留插件源码备份。
