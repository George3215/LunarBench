# TASK2：独立 Isaac Lab GPU 石头堆叠

MuJoCo 用于定义和离线导出场景；Isaac Lab 独立负责 PhysX GPU 仿真、批量 EEF 控制、课程奖励和 PPO 训练。运行时只读本目录的 `assets/task2/scene.usda` 与 `manifest.json`，不导入 MuJoCo 或原 TASK2 代码。

## 当前方案：完整任务的局部残差 PPO

`configs/visual_ppo_512.yaml` 现已切换为 `control: residual, stage: stack10`。同一个策略和同一个回合覆盖抓取、抬升、搬运、放置；抓取成功不结束回合，也无需手工切换课程。下面的直接 EEF/单阶段配置保留用于基线比较。

```bash
WANDB_MODE=online bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/visual_ppo_512.yaml --mode train
```

- 程序：按底层到顶层选择石头，先抬高到安全高度，再移动到石头上方、下降、闭合夹爪；真实接触确认后抬升，验证保持后搬运至目标上方，下降放置、松开、撤离和检查稳定性。
- 策略：6D `[dx, dy, dz, rx, ry, rz]`，只有抓取/放置局部区域接通。平移是相对程序目标的绝对偏置，默认范数不超过 1 cm；旋转为环境坐标轴角残差，默认不超过 0.15 rad。不会逐步累加成大范围移动。夹爪开合不交给策略。
- 抓取局部区域按指垫中点到石头中心的距离计算；放置按被抓石头中心到目标的距离计算，均为 5 cm。区域外即使网络输出非零也会被环境屏蔽。程序路点也经过EEF 4 cm/0.1 rad 每动作门控和原局部 DLS 控制。
- 视觉 Actor：三路 64×48 RGB + 50 维状态。其中原本体 25 维，新增当前石头中心3、放置目标3、程序 TCP 路点3、路点四元数4、10阶段 one-hot、石头编号1、局部开关1。石头中心是明确允许的仿真状态输入，不再宣称纯视觉估计。Critic 为210维。
- 初始策略均值为零残差，局部探索标准差为0.1（归一化动作）。程序阶段输出严格为零，并屏蔽 Actor 的策略/熵梯度；Critic 仍学习完整任务回报。PPO 仍在局部探索，不是无探索的监督学习。
- `residual_stack.py` 负责路点、阶段与残差奖励；`task2.py` 继续负责资产、DLS、真实接触和完整墙判据。旧7维动作/25维视觉本体模型不兼容，首次运行用 `checkpoint: null`。

阶段顺序为 `clear → approach → descend → grasp → lift → transport → place → release → retract → settle`。接触、保持、支撑和释放均检查实际仿真状态，不根据夹爪命令假定已经抓住，不吸附/瞬移石头。确认抓持后按已知石头质量通过关节力矩补偿载荷重力，松爪或失去双侧接触即撤销；这不会向石头直接施加吸附力。失败可重试当前石头，默认最多3次；不在回合中重置整个仿真。已完成部分明显移位会结束本次尝试。

残差配置放在 YAML 的 `residual` 下：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `local_radius_m` | 0.05 | 开启残差的局部球半径 |
| `position_residual_m` | 0.01 | 绝对平移残差范数上限 |
| `rotation_residual_rad` | 0.15 | 旋转残差范数上限 |
| `grasp_z_offset_m` | 0.012 | 指垫中点相对石头中心的抓取高度 |
| `support_preload_m` | 0.003 | 放置参考下调3 mm，确保能建立支撑接触 |
| `travel_z_m` | 0.45 | 搬运时石头中心的安全高度；空手接近时用于 TCP |
| `phase_steps` | 120 | 单阶段卡住的步数门槛 |
| `max_retries` | 3 | 当前石头最大失败重试次数 |
| `progress_weight` | 5.0 | 稠密进度差分的总权重 |
| `residual_cost` | 0.002 | 残差幅度惩罚 |
| `retry_penalty` | 2.0 | 一次重试惩罚 |

槽位的 x/y 保留，名义放置高度根据当前石头厚度和下层支撑中心估计，再减去毫米级预压；残差负责不规则表面的局部修正。原始完整墙成功判据不变。

稠密项包含程序接近距离、局部对中、双侧接触、接触保持、抓持抬升高度、搬运平面距离，以及放置中心误差、连续支撑高度误差、姿态误差、正确支撑和低速度。使用阶段内势函数差分；切换阶段重新建立基线，不靠停留刷固定正奖励。一次性抓取/抬升/放置奖励仍分别为10/15/25，完整墙成功为100，每步成本0.01，失败惩罚10。抓取和放置奖励随同一次完整任务共同训练，不使用旧的抓取成功即重置逻辑。

最终成功仍要求所有10块形成4+3+2+1、上层双石支撑、释放、低速度、位于堆叠区并稳定1.2秒。中间阶段不算任务成功。1000动作步上限保留；需要更长完整任务预算时只改 `episode_steps`。

## 旧直接 EEF 基线的日常使用

本机已安装 Isaac Lab **2.3.2**，复用 `/home/lry/isaac-sim` 的 **Isaac Sim 5.1**；Python 依赖隔离在 `../IsaacLab/.venv`，任务依赖记录在 `requirements.txt`；不安装与本任务无关的遥操作/其他 RL 算法扩展。没有修改本机另一份 `/home/lry/IsaacLab`。

训练：

```bash
bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/task2.yaml --mode train
```

所有实验设置修改 `configs/task2.yaml`，不增加命令行参数：

| 字段 | 含义 |
|---|---|
| `num_envs: 1024` | GPU 并行环境数；已另验收 4096 环境，正式训练建议先用 1024 |
| `stage: grasp` | `grasp → lift → place → stack4 → stack7 → stack9 → stack10` |
| `iterations: 2000` | PPO 采样/更新轮数；每轮收集 `24 × num_envs` 条 transition |
| `episode_steps: 1000` | 单环境回合动作上限；与全部环境累计 transition 数不同 |
| `physics_dt: 0.0015` | 物理积分步长，秒 |
| `control_steps: 160` | 一个策略动作执行 160 个物理步，即 0.24 秒 |
| `cameras: false` | 是否启用三路 RGB；状态 PPO 默认关闭渲染 |
| `checkpoint: null` | 新训练；填 `outputs/.../model.pt` 可恢复训练或评估 |
| `logger: tensorboard` | 改成 `wandb` 使用 W&B，项目名 `task2-isaac-stack` |
| `steps` | 评估或显式验证的动作步数，不限制训练预算 |

例如 1024 环境、2000 轮是 **49,152,000 条 transition**，不是 2000 步。要做约 20 万 transition 的试跑，可用 1024 环境、`iterations: 8`，即 196,608 条；这不足以保证学会十石堆叠。

使用 W&B：先在 YAML 改 `logger: wandb`，然后执行：

```bash
WANDB_MODE=online bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/task2.yaml --mode train
```

输出在 `outputs/时间戳/`：原始配置、TensorBoard/W&B 记录、定期 `model_*.pt`、最终 `model.pt`、`report.json`。训练中按 Ctrl+C 停止时，使用最近的定期检查点恢复；不保证尚未保存的更新被保留。

评估：把 `configs/view.yaml` 的 `checkpoint` 改为实际模型路径，`steps` 至少设为 `episode_steps`，再运行：

```bash
bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/view.yaml --mode eval --view
```

`--view` 打开 Isaac Sim 窗口。配置中的 `cameras` 控制三路传感器，两者是独立开关。评估保存最后一帧 `eye_in_hand.png`、`top.png`、`front.png`，并分别报告课程成功率和完整十石墙成功率。分母仅包含已结束回合，未结束的回合不算失败或成功。


## 视觉策略：PPO 可以读取相机

原 `configs/task2.yaml` 是 **185 维状态 PPO**。仅把 `cameras` 改成 true 只会启用渲染，不会改变旧 Actor 的输入。视觉训练使用新增的 `configs/visual_ppo.yaml`：

```bash
WANDB_MODE=online bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/visual_ppo.yaml --mode train
```

- Actor：`eye_in_hand / top / front` 三路 RGB + 25 维本体（关节角/速度、TCP 位姿、夹爪净开口）。不读取石头真实位置、姿态、速度或堆叠目标坐标。
- 三视角共享小型 CNN，每个视角得到 128 维特征；按上述顺序拼接，再与本体信息一起生成 7 维 EEF 动作。CNN 与 Actor 一起接受 PPO 梯度更新。
- Critic：训练时使用完整 185 维状态，帮助价值估计；推理动作不依赖 Critic。这是非对称 Actor–Critic。
- RGB 在 GPU 上渲染，rollout 保存 uint8；更新时转成 float 并重新经过 CNN。没有把图像展平成数万维后直接交给 MLP。
- 默认三路各 **128×96**，保持原 4:3 画面比例；修改 `camera_width / camera_height` 可以提高分辨率，训练与评估使用一致设置。
- 默认 **128 环境**。更大并行使用 `configs/visual_ppo_256.yaml`：256 环境、三路 96×72。之前 4096 环境的性能是关闭渲染的状态版结果，不能用于预测视觉版。
- `rollout_steps` 控制每轮每环境采样数，默认 24；总 transition = `iterations × rollout_steps × num_envs`。
- `num_mini_batches` 默认 4，单批样本数为 `num_envs × rollout_steps / num_mini_batches`；显存紧张时可增大到 8。环境数翻倍而 iterations 不变，总采样预算也翻倍。
- 旧状态策略的权重与视觉网络不兼容。视觉训练从 `checkpoint: null` 开始，后续只加载视觉版检查点。

这是使用 CNN 的视觉 PPO，不是 Qwen/VLM，也没有替换成 DrQ-v2。PPO 是优化算法，能否输入相机由观测管线和策略网络决定。当前选择保留 PPO 是为了复用已经跑通的 Isaac Lab GPU 并行训练入口。

视觉网络在 `visual_policy.py`，观测隔离在 `task2.py::_get_observations`。图像敏感性与 Actor 特权信息隔离的离线检查在 `tools/validate_visual_policy.py`，正常训练不调用它。

256 环境训练：

```bash
WANDB_MODE=online bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/visual_ppo_256.yaml --mode train
```

优先直接降低 `camera_width / camera_height`：96×72 的像素数是 128×96 的 56.25%，同时节省渲染、图像缓存和 CNN 计算。先高分辨率渲染再下采样只能减少后两者，不节省原始渲染成本。两个配置都保留三路相机、原相机位姿与 4:3 比例。更低分辨率可能损失石头边缘与夹爪接触细节；短训练吞吐验证不代表抓取成功率相同。

继续扩容可使用 `configs/visual_ppo_512.yaml`：512 环境、三路 64×48、4 个 PPO 小批次。64×48 是 96×72 像素数的 44.44%，直接以此尺寸渲染，不额外增加下采样链路。相机位姿、视野和物理控制参数不变。

```bash
WANDB_MODE=online bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/visual_ppo_512.yaml --mode train
```

旧直接 EEF 配置的本机短训练实测约 284 transition/s，整卡显存采样峰值约 12.13 GiB；当前512配置已改为残差完整任务，不能直接沿用旧吞吐数字。768 和 1024 环境均在初始化时出现 CUDA 非法内存访问，尚未跑通，不建议直接使用这两档。512 环境的 2000 轮共 24576000 transitions；若要与 256 环境的 2000 轮保持相同采样预算，将 iterations 改为 1000。性能证据见 `VALIDATION.md`，不代表已收敛或抓取成功率不受分辨率影响。

## 任务与控制合同

- 同一 UR5e、Robotiq 2F-140、十块程序石头、初始关节/石头位姿、工作区和相机外参，来自当前 `tasks/task2_stack/configs/rl_curriculum_01_grasp.yaml` 构建的场景。
- 状态配置使用 185 维 **特权状态**：六关节角/速度、TCP 位置与旋转矩阵、实测指垫净间距、十个目标位置、十石刚体位置/四元数/速度。所有位置减去各环境原点。该 PPO 基线不等同于原来的像素 DrQ-v2，旧 DrQ-v2 权重不能直接加载。
- 动作 7 维：环境轴下的 TCP 平移增量、轴角旋转增量、夹爪开合。平移向量范数最多 **2 cm**，旋转最多 **0.1 rad**，低于原 5 cm / 0.35 rad 门限；夹爪 -1 关闭、+1 打开。改变上限的位置在 `task2.py` 的 `Task2Cfg`。
- 每 8 个真实物理步，GPU DLS 根据实测状态重新计算小幅关节目标；PhysX 隐式 PD 与重力前馈执行目标。没有抓取轨迹脚本、home 分支搜索、步进过程中的物体传送或策略外的动作决策。
- 抓取需要双指垫真实接触、石头离开地面/其他石头、相对 TCP 稳定、抬高 2 cm 并保持至少 0.36 秒；抬升课要求 8 cm / 0.6 秒。
- 放置与各层课程要求已抓取过该石头、接近对应目标、正确支撑层、释放、低速和持续稳定。上层需由至少两块下层石头支撑。
- `stack10` 要求全部十石满足 4+3+2+1、工作区、释放与稳定支撑条件并保持 1.2 秒。课程成功、石头掉落或步数到限结束当前环境，随后只重置该环境。
- 稠密奖励使用 `0.99 Φ(s') - Φ(s)`，包含接近、居中、双指接触、保持、抬升、搬运、放置；抓取/抬升/放置各石头每回合仅奖励一次。完整成功与阶段奖励不重复叠加。

可在 YAML 中只覆盖需要修改的奖励项，不必修改导出资产：

```yaml
reward:
  reach: 2.0
  grasp: 3.0
  lift: 6.0
  transport: 5.0
  place: 15.0
curriculum:
  grasp_bonus: 10.0
  grasp_height_m: 0.02
  grasp_hold_seconds: 0.36
```

课程切换是明确的实验设置：先评估，再改 `stage` 并填上一阶段检查点。代码不会因未经验证的训练回报自动升级课程。

## 迁移边界

- 原始 1,473 个几何体、网格三角形、尺寸、质量/惯量、12 个关节和 4 个线性 tendon 关系保留。无质量固定节点合并到所属刚体；固定世界基座使用不参与运动的占位惯量。tendon 转为 PhysX mimic；四个被动关节不另加电机。
- 有效碰撞形状使用源模型的碰撞开关/位掩码与显式 body 排除；石头仍用凸碰撞。PhysX 接触按手指刚体过滤后，再按真实接触点是否落入指垫 box 过滤，避免把指尖接触直接当作抓取。
- 源 RGBA 转为简化漫反射外观，没有导入法线贴图，也没有复制源纹理。相机外参、垂直 FOV 保留，RGB 输出设为 640×480。
- **MuJoCo Newton 与 PhysX TGS 的接触动力学不同。** 不声称同一动作轨迹会产生数值相同的结果。MuJoCo `solref/solimp`、滑动/扭转/滚动摩擦不能逐项等价搬运：当前映射滑动系数到静/动摩擦、恢复系数设为 0，原三项摩擦值保存在 manifest。扭转/滚动系数未作为等价 PhysX 参数应用。
- PhysX 使用 1 mm contact offset、0 rest offset、16/4 求解迭代。机器人使用重力前馈以防重复局部 EEF 指令下沉；这是后端控制适配，不是改变石头重力或关闭接触。
- 原 MuJoCo 无限地面在本 TASK2 中导出为覆盖工作区的大平面；各环境有独立碰撞隔离。未来的大范围导航场景应重新定义地面范围。

## 文件职责与未来任务

```text
IsaacLab/                    上游发行包与隔离环境，不混入任务实现
isaac_tasks/
  install.sh                 固定 Isaac Lab 版本，连接本机 Isaac Sim
  run.sh / run.py            训练、评估入口（仅 config / mode / view）
  task2.py                   GPU 场景、局部 EEF、观测、物理判据、重置
  residual_stack.py          完整任务状态机、局部残差、稠密奖励
  configs/                   实验条件
  tools/mujoco_usd.py         可复用的编译后 MuJoCo 模型 → USD 转换
  tools/export_task2.py       当前 TASK2 离线适配器
  tools/validate_task2.py     显式验收工具，训练不会调用
  assets/task2/              自包含 USD + manifest + 仅溯源用 source.xml
  outputs/                   模型、日志、报告和画面
```

未来任务先在 MuJoCo 定义，再写一个小适配器调用 `export_model(model, initial_data, output, free_body_names, camera_names)`。转换器支持当前使用的固定/hinge/free、mesh/box/sphere/plane 和双关节线性 tendon；其他机器人关节/约束必须增加对应转换，不能把此接口理解为支持任意 MJCF 的无损转换器。任务奖励和观测仍需写成批量 Torch 运算。

本机重新导出（只有这一步依赖原 TASK2 和 MuJoCo）：

```bash
/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/.venv-rl/bin/python /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/tools/export_task2.py
```

迁移到另一台机器：先安装 Isaac Sim 5.1，设置 `ISAAC_SIM_PATH` 并运行 `install.sh`；复制本目录及 `assets/task2/`，无需复制原 MuJoCo 仓库。`ISAACLAB_PATH` 可指定其他安装位置。生成资产和上游安装目录被 Git 忽略；复制/归档项目时要单独带上资产，原第三方资产授权不变。

显式验证/导出当前画面（不是正常运行时的自动自检）：

```bash
bash /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/run.sh validate --config /home/lry/MoonUnrealEnv/MoonSim/isaac_tasks/configs/view.yaml
```

当前运行证据、短训练的适用边界见 [VALIDATION.md](VALIDATION.md)。
