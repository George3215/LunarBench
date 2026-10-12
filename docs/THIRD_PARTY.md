# 第三方组件、上游来源与许可

本文件集中记录 MoonSim / LunarBench 用到的**非本仓库代码与资产**：哪些以 submodule
引用、哪些真正 vendored 进仓库、哪些只在本机工作区作为参考克隆存在、各自固定在哪个
提交。整理目标是让"仓库里有什么、缺什么、怎么补齐"一眼可查，而不是把上游代码复制
进仓库。

**总原则：引用一个上游仓库不等于为其内容重新授权。** 每个来源保留自己的 LICENSE 与
署名；第三方资产（地形、岩石、机器人网格、预训练权重）的公开再分发权尚未确认，默认
只在本机保存。

关联文档：[源码与依赖范围](REPOSITORY.md) · [资源说明](../assets/README.md) ·
[VLM 实验安排](../paper/vlm_goal_navigation/EXPERIMENT_PLAN.md)。

## 1. Git submodule（克隆时按固定提交拉取）

`git clone --recurse-submodules` 或 `git submodule update --init --recursive` 获取。

| 用途 | 上游 | 固定提交 | 许可 |
| --- | --- | --- | --- |
| TASK2 DrQ-v2 原生网络与更新 | <https://github.com/facebookresearch/drqv2> | `c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef` | MIT |
| Direct Astra 工作流参考实现 | <https://github.com/anonymous-report-421/GPT-as-Policy> | `8f3d362b077d8efb77e2a7274d5b2c20e2243846` | 见其 `THIRD_PARTY_NOTICES.md` |

GPT-as-Policy 只作为参考实现；Qwen 直接控制与 RL 的日常入口**不会**调用 Codex 执行
模型生成的任意代码，训练与评估也不依赖它。

## 2. 真正 vendored 进仓库的代码

| 路径 | 上游 | 固定提交 | 说明 |
| --- | --- | --- | --- |
| `third_party/z_mobile_manip/` | <https://github.com/Z-Robotics-Lab/Z-Mobile-manip> | `db76d242cf9210fae6068d63ce6e6e50a7ab0069` | TASK1 感知/抓取基线与鲁棒 IK。逐文件 SHA-256 记录在 `UPSTREAM.json`，仅精简了 `__init__.py`，实现文件未改；`THIRD_PARTY_NOTICES.md` 与 `LICENSE` 随目录发布。 |

### 本机 vendored、不入库的第三方机器人资产

| 路径 | 上游 | 状态 |
| --- | --- | --- |
| `third_party/LeggedManip_Lab/` | <https://github.com/zzzJie-Robot/LeggedManip_Lab> | 只落地了 Go2+Piper 的 MuJoCo 模型（`mujoco/robots/go2_piper/`）与 TorchScript 策略（`mujoco/deploy/policy/go2_piper/policy.pt`），约 480 MB，被 `.gitignore` 排除，由使用方自行获取。**本机目录只含 `mujoco/` 子集，未记录精确上游提交**，`controllers.py` 里标注为原样使用的第三方权重。 |

TASK3 的运动控制器有两份，都由 `--controller` 选择，都是第三方预训练权重原样使用、不训练
不微调：`leggedmanip`（上表）与 `go2rl`
（[`wty-yy/go2_rl_gym`](https://github.com/wty-yy/go2_rl_gym) MoE-CTS，MIT）。两者只提供
步态策略，不提供高层的目标重观测决策。

## 3. 工作区级参考克隆（外层 `third_party/`，不在 Git 仓库内）

这些仓库位于工作区 `/home/lry/MoonUnrealEnv/third_party/`，**不是** `MoonSim` 仓库的
一部分，也不会被提交（合计约 60 GB）。它们被 VLM 目标导航对照实验按其 conda/venv
环境直接调用；固定提交同时记录在
`paper/vlm_goal_navigation/v4_multi_model_20261009/models.json` 与 `REPOSITORIES.json`，
便于换机器后重新克隆。

| 目录 | 上游 | 固定提交 | 在本项目中的角色 | 当前状态 |
| --- | --- | --- | --- | --- |
| `NaVILA/` | <https://github.com/AnjieCheng/NaVILA> | `76b98f233dd0fff05dfcd69435eec6740febff9d` | `navila-llama3-8b-8f` 基线；RGB-D 为适配输入，动作映射到共同 1 s 原语 | 已排队运行 |
| `Taga-VLM/` | <https://github.com/APEX-BJUT/Taga-VLM> | `db2f2884e2ca29a0d85219a637435415b94e0a93` | `tagavlm-qwen2-0.5b/7b-camera-adapted` 基线；禁用仿真拓扑与 `graph_sprels`，只做相机骨干适配 | 已排队运行，**不代表完整原生 Taga 方法** |
| `ABot-Navigation/` | <https://github.com/amap-cvlab/ABot-Navigation> | `2a0aefb56f1e2d315bba924239e9e8ad9dca9d92` | ABot 导航评测接口与 agent 示例参考 | 参考 |
| `ABot-N0/` | <https://github.com/amap-cvlab/ABot-Navigation>（ABot-N0 分支） | `77ceab459a801f48d186b360604eca7a7dd95811` | 官方 ABot-N0 | **不可运行**：当前只有论文说明/媒体，无推理代码、权重或 release assets |
| `Qwen-VLA/` | <https://github.com/QwenLM/Qwen-VLA> | `fa25ab1e69fd9d4318aa0fc5d3f2114a443f78d3` | Qwen-VLA 对照 | **不可运行**：仓库只有说明/媒体，无推理代码与权重 |
| `Qwen-RobotNav/` | <https://github.com/QwenLM/Qwen-RobotNav> | `f74cf1f6079300aee6e40e03858da2f98b1b7261` | Qwen-RobotNav-4B 对照 | **不可运行**：官方 README 声明暂无权重发布计划 |
| `object-rel-nav/` | <https://github.com/oravus/object-rel-nav> | `6da34c30871e9b8a8cdc46c8607fe3683763020e` | Object-relative navigation 基线与 HM3D 数据准备参考（`.tmp/hm3d_cfg`、`.tmp/annots`） | 参考 |
| `navigation_inference/` | 本地自建（无 git 远端） | — | 无状态、仅相机的推理 HTTP 服务（`camera_server.py`），基于 LLaVA 常量与对话模板，不 import 仿真器、不读写文件 | 参考 |

模型权重（`NaVILA/checkpoints/`、`Taga-VLM/model_zoo/`、`MoonSim/assets/model/Qwen3-VL-8B-Instruct/`
等）与 Hugging Face 本地缓存同样不入库，按各自的上游下载方式准备。

## 4. 资产与数据集上游

| 资产 | 来源 | 记录位置 |
| --- | --- | --- |
| UR5e / Robotiq 2F-140 网格 | <https://github.com/ARISE-Initiative/robosuite> | `assets/robots/robosuite/source.json`、`UPSTREAM_LICENSE`（提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`） |
| Panda / Robotiq 2F-85 模型 | MuJoCo Menagerie | `assets/robots/franka/`、`assets/robots/robotiq_2f85/`，原样搬运 |
| 五个自包含机械臂（ur10/ur10e/ur12e/so101/piper） | 各自上游 xacro/URDF/MJCF | 由 `tools/prepare_arms.py` 生成，逐文件校验见对应目录与 `tools/validate_arms.py` |
| Go2 demo 步态权重 | 第三方 checkpoint | `assets/robots/go2_demo/`，含来源与许可 |
| 月面地形、岩石、材质、DEM | 外部扫描与程序化生成 | `assets/environments/`，来源与许可说明见 `assets/README.md` |
| 月岩数据集、UE 生成资产、15 GB 资产库 | 外部 | 不在仓库内，按 `assets/README.md` 自行准备 |

## 5. 发布边界

- 不随源码发布：`.local/`、虚拟环境、模型权重、训练输出、W&B 本地记录、外部月岩数据
  集、UE 生成资产、论文 PDF、原始实验产物。
- 实验原始产物的排除规则与复现方式见
  [VLM 实验安排](../paper/vlm_goal_navigation/EXPERIMENT_PLAN.md)。
- 若要把任何第三方资产从"本机可用"提升为"公开分发"，必须先确认其许可允许再分发，
  再单独提交许可与署名，不能借本文件的引用关系默认授权。
