# 仓库与依赖范围

GitHub：<https://github.com/George3215/LunarBench>。本机 checkout 位于
`/home/lry/MoonUnrealEnv/MoonSim`；外层工作区不是 Git 仓库。

## 目录

- `assets/`：资源说明、来源、许可证、轻量模型描述；大型网格和地形在本地准备。
- `ue/`、`mujoco/`、`bridge/`：显示、物理与跨进程传感器/动作接口。
- `tasks/task1_collect/`：月面石头收集。
- `tasks/task2_stack/`：固定十石堆叠，Qwen 直接控制和强化学习。
- `baselines/`、`third_party/`：已有独立策略接入及其上游许可。
- `tools/`：资源准备与离线验证，不在每次任务启动时自动运行。
- `docs/images/`：首页架构图与 TASK1 展示图，原文件仍保留在外层工作区。

不上传 `.local/`、虚拟环境、模型权重、训练输出、W&B 本地记录、外部月岩数据集、
UE 生成资产或论文 PDF。已有训练进程和输出目录不会因整理仓库而移动或删除。

## 上游代码

| 用途 | 来源 | 固定版本或记录 |
| --- | --- | --- |
| TASK2 场景和历史堆叠脚本 | <https://github.com/Xundendong/stone-stacking-mujoco> | `tasks/task2_stack/docs/UPSTREAM_README.md`；当前代码有本地适配 |
| DrQ-v2 原生网络和更新 | <https://github.com/facebookresearch/drqv2> | submodule，`c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef`，MIT |
| Direct Astra 工作流参考 | <https://github.com/anonymous-report-421/GPT-as-Policy> | submodule，`8f3d362b077d8efb77e2a7274d5b2c20e2243846` |
| UR5e / Robotiq 模型 | <https://github.com/ARISE-Initiative/robosuite> | `assets/robots/robosuite/source.json` 与 `UPSTREAM_LICENSE` |

保留各来源自己的许可与署名；引用一个上游仓库不意味着为其内容重新授权。
GPT-as-Policy 是参考实现，Qwen 直接控制与 RL 日常入口不会调用 Codex 执行任意模型生成代码。

## 新机器准备 TASK2

从仓库根目录初始化引用：

```bash
git submodule update --init --recursive
```

机器人网格未包含在源码仓库。根据 `assets/robots/robosuite/source.json`，
取 robosuite 提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` 下的
`robosuite/models/assets/robots/ur5e/`、`grippers/robotiq_gripper_140.xml`、
`grippers/meshes/robotiq_140_gripper/` 与 `grippers/meshes/robotiq_s_gripper/`，
保留层级放入 `assets/robots/robosuite/models/assets/`。
`source.json` 保存了本机文件校验信息。其他机器人和 TASK1 地形见 `assets/README.md`。

Python 3.11 为当前已验证版本。新环境先安装适合本机 CUDA 的 PyTorch，再安装项目依赖：

```bash
cd tasks/task2_stack
python3.11 -m venv .venv-rl
.venv-rl/bin/python -m pip install torch
.venv-rl/bin/python -m pip install -r requirements.txt -r requirements-rl.txt
.venv-rl/bin/wandb login
WANDB_MODE=online bash rl.sh --config configs/rl_stack10_drqv2_200k_parallel.yaml --mode train
```

GPU 配置需要 NVIDIA 驱动、可用 CUDA PyTorch 与 EGL；无 GPU 时使用单环境 CPU 配置。
Qwen 权重与服务单独部署，其地址、模型名在 YAML 中设置，不随仓库发布。
`configs/rl_stack10_eval.yaml` 中的 checkpoint 是本机示例，其他机器需要改为自己的模型路径。

## 结果边界

短训练与并行验证证明采样、梯度更新、保存/加载和评估流程能运行，不证明策略已完成十石堆叠。
8 环境表示 CPU MuJoCo 多进程，网络使用 CUDA；MJX/Warp 仍处于可行性评估阶段。
首页图片是设计与任务展示，不是成功 rollout 的记录。
