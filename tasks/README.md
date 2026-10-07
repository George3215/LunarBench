# Tasks

任务负责观测、动作、奖励和成功/失败条件；资产和引擎实现留在对应目录。

| 任务 | 当前内容 | 入口 |
| --- | --- | --- |
| [TASK1](task1_collect/README.md) | Go2-Piper 月面石头收集，UE/MuJoCo 与 bridge 传感器/动作边界 | `task1_collect/run.py` |
| [TASK2](task2_stack/README.md) | 固定十石 4+3+2+1；Qwen 直接控制、DrQ-v2 图像训练与 SAC 状态对照 | `task2_stack/run.py` |
| [TASK3](task3_reobservability/README.md) | 地形导致的目标重新可观测性；8 类地形遮挡机制、冻结的观测/动作接口、多步回合协议与 RSR@K | `task3_reobservability/benchmark.py`、`task3_reobservability/terrain_viewer.py`、`task3_reobservability/test_api.py` |

TASK2 已接入多进程 CPU MuJoCo 采样和共享 CUDA DrQ-v2 策略，配置、20 万步命令及 W&B 见 [RL 文档](task2_stack/docs/RL.md)。尚未接入 MJX，也没有已验证的完整十石堆叠策略。

TASK3 目前提供**任务与地形定义**本身：8 类地形遮挡机制、冻结的 `Environment`/`Observation`/`ReObservationPolicy` 契约（`api.py`，版本 1.1）、十个固定动作原语、多步回合协议与终止集、以及直接在逐步可见性轨迹上求值的 `RSR@K`。契约测试 11 组全通过。地形用精确 box slab 构造，表面高度与解析可见性 oracle 完全一致。可交互查看地形并驱动机器狗的工具是 `terrain_viewer.py`（含无显示器自检：15 个地形里 14 个通过，12 个 slab/box 地形的接触面与 oracle 误差全为 0.0）。核心现象：同一条路径上目标可见性会 `Visible → Invisible → Visible`（`perlin` 达 6 次翻转），即 `Closer ⇏ More Observable`。

> **实验结论已过审计修正。** 下述 Experiment 1–4 是历史记录。审计确认主实验的成功判定未以相机检测为准、目标记忆中含真值目标坐标、且 Random 与 Learned 的场景分母不一致，因此旧结果**不再支持**"非特权视觉闭环优于随机策略"。仍然成立的是 H1/H2/H3 三组对照实验与地形/接口本身。修正后的结论另见 `paper/`。

历史实验简述：Experiment 2 在 5 个地形上物理执行每个候选动作并套四条决策规则——不存在通吃的固定动作（forward 6/8、back 6/8、侧移 3/8、转视角 0/8），`pit` 与 `barrier` 上 A/B/D 全失败而只有反事实基线成功；但其余六个场景区分不了 IMU 朝向锁定。Experiment 1 用 8 对**渲染逐像素完全相同**的世界（平均像素差 0.0、100% 像素一致、目标像素 0/0）证明：最优动作相反（8/8），且没有任何与历史无关的动作能同时解决两个世界（8/8）；16 个世界上 A 0%、B 0%、D 0%、C 100%。成对世界由 7 条预注册几何判据自动搜索接受（49 候选 → 8 通过）。Experiment 3 把 oracle 上界换成训练出来的预测器：有目标记忆时在成对世界上达到 oracle 水平（16/16），打乱记忆的对照塌回 0.4375；但**在没见过的地形族上只有 0.50，低于朴素基线 0.75**。Experiment 4 回答"需要多少历史"：决策 RSR 在 H4（0.4 秒）跳到 0.833 且不再涨，预测 AUC 在 H64 饱和；但决策指标只有 24 个世界（分辨率 1/24），H4 之后的排序在噪声内。详见 [TASK3 文档](task3_reobservability/README.md)。

Go2 demo 仍通过 `tools/run_demo.py` 独立运行。旧 TASK2 空白机械臂环境不再是活动入口，历史脚本与设计记录见 TASK2 的 `docs/`。
