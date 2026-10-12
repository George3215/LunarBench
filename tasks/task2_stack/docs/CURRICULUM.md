# TASK2：从抓取到十石堆叠

固定十块原有石头、料位和机器人初始状态不变。课程只修改奖励、阶段终止与动作尺度；不摆石头、不演示抓取、不自动产生策略动作。

## 为什么重做这次训练

暂停的 `rl_drqv2_20260922_213411_147551` 共记录 113024 条 transition：

- 24 条双指接触奖励为正，最长连续 2 个动作边界，约 0.24 秒。旧日志未记录石头身份，不能确认这两步夹着同一块石头。
- 已结束的 275 个回合全部因石头掉落结束，没有完整墙成功。
- 27 条记录的 `placed_count=1`，都来自 env 78；该回合没有双指接触奖励。这只能说明曾有石头进入几何目标条件，不能证明完成抓取再放置。
- TCP 目标距离误差中位数约 3.69 cm，99 分位约 1.07 m。

离线分析入口：

```bash
.venv-rl/bin/python tools/analyze_rl_trace.py outputs/rl_drqv2_20260922_213411_147551
```

报告写入该输出目录的 `grasp_trajectory_analysis.json`。分析读取日志，不连接模拟器。

## 先修复控制状态

旧 `RobotController` 在未提供 scratch 时把 `ik_data` 指向真实 `data`，而 IK 迭代写入 qpos。现在默认新建独立 `MjData`，避免 IK 求解污染真实积分状态。

RL 默认 `controller: local_dls`：每 8 个物理子步更新局部阻尼 IK，不尝试 home/限位中点构型；每次关节命令增量裁剪为 0.03 rad，命令距实测关节限制为 0.15 rad。它不做抓取规划，所有任务目标仍来自 actor。`legacy` 仅用于控制对照。

在 env 41 的同一组前 64 个归一化历史动作上，旧共享状态控制的最大 TCP 目标误差约 0.978 m，局部控制约 0.070 m；最大单动作实测关节变化约 6.846 rad 降到 0.254 rad。另一段历史动作中局部控制的最大目标误差仍约 0.118 m，因此不宣称所有轨迹都改善。这是有界控制对照，不证明学会抓取。新控制仍有跟踪滞后，不能把目标门限当作真实轨迹门限。

旧 checkpoint 来自不同且存在状态污染的控制过程，**第一阶段从头训练，不接续旧 checkpoint**。

## 奖励的含义

`stone_stack/rl/reward.py` 负责以下内容，所有权重和门限均在 `task2.yaml → policy.rl`。

1. `reach`：指垫中心接近尚未完成的石头，连续指数距离反馈。
2. `center`：接近且两侧指垫到石心距离接近，提供接触之前的居中反馈；不是成功条件。
3. `touch`：单指受力接触为 0.5，双指为 1。
4. `grasp`：同一石头两片指垫都受力，默认各至少 0.05 N。
5. `hold`：同一石头离地至少 2 cm、双指受力、没有地面/其他石头支撑且相对 TCP 稳定，累计到 0.36 秒。
6. `lift`：有双指接触、脱离支撑且相对 TCP 稳定时，抬升高度连续增加，8 cm 饱和。
7. `transport`：夹持抬起后，接近对应放置目标。堆叠课程先奖励当前未完成层，再进入上层。
8. `place`：本回合成功夹起过该石头，并在正确支撑层释放、低速、接近目标。推到目标但未成功夹起，不获得该项反馈。

势函数仍用 `gamma * Phi(next) - Phi(current)`；步长成本、动作成本和掉落惩罚保留。权重为 reach=2、center=1、touch=2、grasp=3、hold=4、lift=6、transport=5、place=15。

额外事件奖金按**每块石头、每回合、每种事件只发一次**：稳定抓取 +10，抬起保持 +15，成功放置保持 +25。初级课程只启用对应的奖金。阶段成功 +30；完整墙成功仍用 +100，二者不叠加。

抓取/抬升要求同一石头；相邻动作观测中，石头在 TCP 坐标系的位置变化不得超过 2 cm。保持时间在动作边界累计，不宣称物理子步之间每一瞬间都满足条件。一次抓取失败后重新抓同一块不会再发首次抓取奖金。

## 七个阶段

| 配置 | 结束条件 |
| --- | --- |
| `rl_curriculum_01_grasp.yaml` | 任一石头满足双指受力、离地 2 cm、脱离支撑、相对位置稳定，保持至少 0.36 秒 |
| `rl_curriculum_02_lift.yaml` | 同一石头抬起至少 8 cm 并保持至少 0.60 秒 |
| `rl_curriculum_03_place.yaml` | 至少一块底层石头先抓起，再放入对应目标 4 cm 内，释放、低速、有正确支撑，保持 0.60 秒 |
| `rl_curriculum_04_stack4.yaml` | 底层四块全部满足上述放置条件 |
| `rl_curriculum_05_stack7.yaml` | 前七块满足 4+3 的放置与跨接支撑条件 |
| `rl_curriculum_06_stack9.yaml` | 前九块满足 4+3+2 条件 |
| `rl_curriculum_07_stack10.yaml` | 原生完整墙判据：十块 4+3+2+1、跨接支撑、释放、低速、墙区内，保持 1.2 秒 |

课程成功记录为 `stage_success`，原生十石墙成功仍为 `is_success`。W&B 分别显示 `episode/stage_success` 和 `episode/success`；初级课程通过不会被算成完整堆叠成功。

默认各阶段：16 环境、CUDA、200000 个合计动作步、每回合上限 1000、2 cm / 0.1 rad 动作尺度、每动作 0.24 秒、探索 8000 步、replay 50000。2 cm / 0.1 rad 是更小的训练尺度，仍受原来 5 cm / 0.35 rad 上限约束。所有课程保持同一观测/网络维度，方便迁移。

## 启动与晋级

先从第一阶段开始：

```bash
WANDB_MODE=online bash rl.sh --config configs/rl_curriculum_01_grasp.yaml --mode train
```

不要一次串行启动全部七个阶段。先检查当前阶段的 `evaluation.json`、`stage_success_rate` 与新记录的石头身份/接触力/高度/保持时长，再决定晋级。固定场景的多次确定性评估也不代表随机初始条件泛化；建议结合可视化核对真实抓取。

进入第二阶段时，编辑 `configs/rl_curriculum_02_lift.yaml`：

```yaml
policy:
  rl:
    initialize_from: outputs/上一阶段运行目录/checkpoint.pt
    checkpoint: null
```

然后更换配置路径运行。`initialize_from` 只复制 encoder/actor；critic、优化器、replay、探索调度和阶段步数重新开始，避免把上一阶段的价值函数当作新奖励的价值函数。后续阶段使用相同方式。

同一阶段中断后续训使用 `checkpoint`，不要同时设置 `initialize_from`。现有续训仍会重新收集 replay。

## 修改入口

- 奖励权重：`policy.rl.reward`。
- 接触力、抬升高度、保持时长和阶段奖金：`policy.rl.curriculum`。
- 并行数量、探索步数、replay 容量：各阶段 YAML。
- 动作与日志：`stone_stack/rl/env.py`。
- 局部控制增益、阻尼和关节限制：`stone_stack/rl/control.py`。

本次完成 34 项回归（环境/回放 11、课程 4、Direct 15、上游判据 4），历史动作对照及 35 步课程训练验证。35 步完成 13 次更新，保存/加载评估与 W&B 离线记录正常，阶段成功率仍为 0。新课程小尺度动作的短测 TCP 误差中位数 2.6 mm、最大 7.7 mm；20 次零动作保持总漂移约 0.07 mm。这些短测不能代表长期接触任务表现。未启动长训练，未证明课程策略已经成功抓取。
