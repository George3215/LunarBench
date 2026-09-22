# TASK2 第二条策略：强化学习

## 固定任务 01

- 十块原有 paper 石头，seed=17，固定形状与可达料区，UR5e/Robotiq 初始状态不变。
- 每个 episode 仅在 reset 恢复同一份完整 MuJoCo 积分状态。step 不重置、移动或吸附石头。
- 目标是全部十块形成 4+3+2+1。没有脚本规划器、专家动作、自动摆石或抓取兜底。
- `evaluation.py` 的实际接触/层数/墙区/速度/机器人接触判据连续在动作边界达标 1.2 秒才成功。它是本地任务判据，不冒称上游 benchmark。
- `terminated`：成功或石头掉到桌下；`truncated`：episode 步数耗尽。截断必须保留 value bootstrap。

## 算法与来源

主实现为 facebookresearch/drqv2，MIT，提交 `c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef`。
`policy/drqv2/drqv2.py` 与 `utils.py` 未修改；原生 actor/critic、随机平移增强、目标网络与更新算法直接使用。
本项目替换的是 DMC 环境、采样循环和 replay 接口，接入 Gymnasium 的石头任务。

DrQ-v2 观测为 `uint8 (9,84,84)`：eye_in_hand/top/front 各 RGB 三通道拼接，默认 frame_stack=1；增加 frame_stack 后通道数相应增加。
不把石头真值传给图像 actor。奖励可读取真值，与 pixel-only policy 的观测权限区分。
多视角的随机平移共享上游增强参数；这是本任务的输入适配，不等于复现上游 DMC 成绩。

另提供 SB3 SAC 特权状态对照：实测机器人关节/速度/TCP/夹爪，以及十块石头的几何中心、姿态、速度和奖励目标。
它用于检查奖励和控制是否可学，不能把其成绩当作纯视觉策略成绩。

## 动作

`Box(-1,1,(7,),float32)`：世界系平移 xyz、世界系旋转向量 xyz、夹爪开口。
前两组三维量映射到单位球再缩放，目标平移范数 ≤0.05 m，旋转范数 ≤0.35 rad。
第七维 -1 表示闭合，+1 表示全开；0 表示半开，不是保持夹爪不动。
每个动作持续默认 0.12 秒真实物理积分，IK/位置伺服只转换网络目标，不产生任务路径。
目标门限不保证实体每一步没有跟踪误差或超调，也不保证无碰撞。

## 稠密奖励

`reward.py` 返回接近、双指接触、抬起、搬运靠近目标、释放且有支撑的放置项。
夹持必须两片指垫实际接触同一石块，不能用“命令闭合”代替；上层必须由正确的下层支撑。
固定目标沿 X 按 4/3/2/1 排列，层高来自现有石头平均厚度，默认槽间距 0.17 m。
这些只是可修改的奖励锚点，不是 IK 路径；不规则石头的最终稳定几何以真实接触判据为准。

势函数：

```text
Phi = 0.5 reach + 1 grasp + 2 lift + 3 transport + 10 place
reward = gamma * Phi(next) - Phi(current)
         - step_cost - action_cost * mean(action²)
         + success_bonus * success - fall_penalty * fallen
```

真正终止时下一势能为 0；时间截断保留下一势能和 bootstrap。没有“每步保持抓取不断领正奖励”。
默认 success_bonus=100，fall_penalty=10。所有权重在 `task2.yaml: policy.rl.reward`，可以在单次配置覆盖。
最终成功不依赖 reward 大小；放置奖励锚点允许调整，但不能绕过原生物理判据。

## 环境与运行

使用原有 MuJoCo/PyTorch 环境作为依赖来源，新增 RL 包安装在项目自己的虚拟环境：

```bash
cd /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack
/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python -m venv --system-site-packages .venv-rl
.venv-rl/bin/python -m pip install -r requirements-rl.txt
```

默认 device=cpu，cpu_threads=4，不停止或占用 Qwen 的 GPU 服务。空闲 GPU 可在 YAML 改为 cuda。

```bash
bash rl.sh --config configs/rl_stack10_validate.yaml --mode train
bash rl.sh --config configs/rl_stack10_drqv2.yaml --mode train
bash rl.sh --config configs/rl_stack10_sac.yaml --mode train
```

实际命令中的相对 config 路径相对当前工作目录；跨目录运行请传绝对路径。
训练步数为环境动作次数，不是物理子步。验证配置 160 步包含真实反向传播；正式配置 500000 步只是起始训练预算，不能保证收敛。

## 保存、加载与记录

每次生成独立 `outputs/rl_<algorithm>_<timestamp>/`：

- `config.yaml`、`scene.xml`、`task.json`：可复现实验条件与目标。
- DrQ-v2 `train.jsonl`：动作、奖励分项、物理结果、actor/critic loss。
- SAC `progress.csv`：SB3 训练统计。
- `checkpoint.pt`（DrQ-v2）或 `checkpoint.zip`（SAC）：权重/优化器。
- `training.json`：真实更新次数与 actor 参数是否变化。
- `evaluation_steps.jsonl`、`evaluation.json`、`eval_*_front.png`：从磁盘重新加载后的评估。
- `summary.json`：本次训练与评估摘要。

训练结束自动新建 actor、从 checkpoint 加载再评估。独立评估需在配置里设置 `policy.rl.checkpoint`，运行 `--mode eval`，可加 `--view`。相对 checkpoint 路径按 TASK2 根目录解析。
`--mode train` 加 checkpoint 表示恢复权重/优化器，replay 重新采集；不宣称逐位相同的完整训练恢复。
评估用与训练一致的场景、观测、网络尺寸配置，推荐复制该次输出的 config.yaml 再填写 checkpoint。

仓库中的 tools 测试仅离线运行，启动时不做相机或 IK 自检。

### 本机图形初始化顺序

本机 PyTorch 2.13/Triton 与已建立的 EGL 上下文存在初始化顺序问题：创建 Adam 时首次加载 Triton 可能段错误。
环境只在首次 reset 取图时建立渲染器和可选 viewer；训练/评估先构造模型与优化器，再 reset。reset 内先创建 EGL 离屏渲染器，再打开 GLFW viewer。
这是资源生命周期安排，不是每次启动的自动检测或失败重试。直接使用环境后再构造外部训练器时，也应遵循这一顺序。

## 本次验证结果

共 26 项离线测试通过：7 项 RL 环境/replay、15 项 Qwen 契约、4 项上游场景/trial。
DrQ-v2：160 个环境动作、68 次真实更新，critic loss 全部有限，actor 参数发生变化，checkpoint 加载评估成功。
SAC：64 个环境动作、48 次更新，actor 参数变化，checkpoint 加载评估成功。
本机 DrQ-v2 独立 `--mode eval --view` 也已验证，退出码 0。评估成功率仍为 0；落石按失败终止，预算耗尽按截断处理。
证据见 `reports/rl_integration_validation.json`。完整学习曲线与权重在报告中列出的 outputs 目录。

这不是已收敛模型。正式 500000 步预算需要另外运行正式配置，且可能需要更长训练、奖励调节或课程学习；当前没有降低十石四层任务目标。

## 20 万步与 W&B

`configs/rl_stack10_drqv2_200k.yaml` 使用三相机 DrQ-v2，训练 200000 个环境动作步，回合上限 1000 步。默认仍为 CPU；尚未接入 MJX。

首次运行先用 `.venv-rl/bin/wandb login` 登录，然后执行：

```bash
WANDB_MODE=online bash /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/rl.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/rl_stack10_drqv2_200k.yaml --mode train
```

W&B 项目为 `task2-stone-stack`，DrQ 每 `log_every` 步记录奖励分项及最近一次更新指标，回合结束记录回报、长度与成功/掉落/超时。最终评估写入 run summary；模型保存在本地输出目录，不自动上传模型。`policy.rl.wandb.enabled` 控制是否启用；账户/team 可通过 `WANDB_ENTITY` 设置。

## 多环境采样与 CUDA 训练

`configs/rl_stack10_drqv2_200k_parallel.yaml` 设置 `num_envs: 8`、`device: cuda`。八个独立 CPU MuJoCo 进程各自渲染三路图像，一个 CUDA DrQ-v2 策略批量产生动作，共用 replay 与 learner。这不是 MJX GPU 物理。

```bash
WANDB_MODE=online bash /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/rl.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/rl_stack10_drqv2_200k_parallel.yaml --mode train
```

`total_steps` 是所有环境的 transition 总和，必须能被 `num_envs` 整除。200000 步、8 环境对应 25000 次向量推进；每个环境的回合上限仍是 1000 步。探索、学习率外的探索噪声调度、更新间隔和 checkpoint 均按全局 transition 计数，默认每两条样本更新一次。各环境独立累积 n-step，终止末帧先写入 replay，再仅重置结束的环境。训练不打开 viewer，使用单环境 `--mode eval --view` 查看模型。

W&B 额外记录 `perf/transitions_per_second`、累计采样和更新耗时；JSONL 每行带 `env_id`。多进程环境数只对 DrQ 生效，SAC 多环境尚未接入。并行配置是独立新训练；不会自动接续或停止其他进程。
