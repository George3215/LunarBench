# TASK2：十块石头多层堆叠

当前固定任务：同一台 UR5e + Robotiq、原有十块 paper 石头、固定可达料位，搭成 **4+3+2+1** 干砌墙。

两条主要策略共享场景和物理成功判据：

| 策略 | 输入 | 决策与执行 |
|---|---|---|
| `qwen-direct` | 三视角 RGB + 机器人本体状态 | Qwen 选择 TCP/夹爪动作，Python 校验、执行、记录 |
| `rl` | DrQ-v2：三视角像素；SAC 对照：特权仿真状态 | 训练得到的 actor 输出 7 维动作，Gymnasium 环境执行 |

RL 使用 [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2) 的原生网络、图像增强和更新代码；SAC 使用 Stable-Baselines3。
环境接入与短训练跑通不代表已经学会十块堆叠。成功必须由物理状态判定，不能只看 reward 或进程退出码。

## Qwen 运行

```bash
bash /home/lry/MoonUnrealEnv/task2.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/upstream_qwen_trials.yaml --view
```

仅查看当前场景相机，使用同一命令并将 `--view` 换为 `--mode camera`。
相机为 `eye_in_hand`、`top`、`front`，被自身遮挡的额外 `wrist` 不再用于上游场景。

## 强化学习

RL 使用项目内 `.venv-rl`，不修改 Qwen 环境或模型服务。首次安装见 [RL.md](docs/RL.md)。

短训练、保存 checkpoint、加载后评估：

```bash
bash /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/rl.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/rl_stack10_validate.yaml --mode train
```

正式固定任务训练（8 环境、CUDA learner、20 万步；先执行 `.venv-rl/bin/wandb login`）：

```bash
bash /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/rl.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/rl_stack10_drqv2_200k_parallel.yaml --mode train
```

状态 SAC 对照使用 `configs/rl_stack10_sac.yaml`。评估时在 YAML 填入 `policy.rl.checkpoint`，把 mode 改为 `eval`；需要窗口再加 `--view`。
命令行仍只有 `--config`、`--mode`、`--view` 三个参数，训练步数/奖励/模型路径都在 YAML。

## 代码位置

```text
run.py                          统一入口；train/eval 交给 RL 模块
rl.sh                           RL 独立 Python 环境启动脚本
configs/                        Qwen、DrQ-v2、SAC、短训练配置
stone_stack/
  upstream_scene.py             固定机器人、十块石头、可达料区、相机
  evaluation.py                 两条主线共用物理成功判据
  tools/stones.py               程序生成 / mesh / MoonSim 石头资产
  direct_control.py             Qwen 动作白名单与严格门控
  policy/                       策略注册与 Qwen/ RL 描述
  rl/
    env.py                      Gymnasium reset/step/观测/终止
    reward.py                   稠密奖励与固定任务目标
    replay.py                   各环境独立 n-step，共享采样池
    parallel.py                 spawn 多进程采样，结束后逐环境重置
    drq.py                      上游 DrQ-v2 装配、保存和加载
    train.py                    训练、评估、日志与产物
policy/drqv2/                   原样保留的上游 MIT 代码
policy/GPT-as-Policy/            Qwen/Direct Astra 参考实现
prompts/                        Qwen 提示词
outputs/                        每次训练/评估/rollout 的独立目录
reports/                        验证记录
```

奖励权重修改 `task2.yaml → policy.rl.reward`；奖励计算修改 `stone_stack/rl/reward.py`。
机器人动作映射修改 `stone_stack/rl/env.py`；最终成功标准修改 `stone_stack/evaluation.py`。

原有 scripted、vlm、qwen-agent 保留为历史对照，不属于这两条当前主线，也不为 RL 提供自动抓取兜底。
更多说明：[RL 环境与训练](docs/RL.md) · [Qwen/上游场景](docs/UPSTREAM_DIRECT.md) · [石头导入及旧说明](docs/README_BEFORE_RL.md) · [旧 MoonSim 架构](docs/LEGACY_MOONSIM.md)。

## 已完成的运行验证

本机验证证据位于 `reports/rl_integration_validation.json`（运行产物不入库）。
DrQ-v2 完成 160 个环境动作、68 次更新；SAC 完成 64 个动作、48 次更新；两者 actor 参数均变化，保存后重新加载评估通过。
另已用真实桌面打开 DrQ-v2 checkpoint 执行评估，正常退出。当前短训练成功率均为 0，不能作为完成堆叠的模型使用。

直接查看本机短训练模型：

```bash
bash /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/rl.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/rl_stack10_eval.yaml --mode eval --view
```

该 eval 配置指向本次本机输出；迁移仓库后请填写自己的 checkpoint。
