# Tasks

任务负责观测、动作、奖励和成功/失败条件；资产和引擎实现留在对应目录。

| 任务 | 当前内容 | 入口 |
| --- | --- | --- |
| [TASK1](task1_collect/README.md) | Go2-Piper 月面石头收集，UE/MuJoCo 与 bridge 传感器/动作边界 | `task1_collect/run.py` |
| [TASK2](task2_stack/README.md) | 固定十石 4+3+2+1；Qwen 直接控制、DrQ-v2 图像训练与 SAC 状态对照 | `task2_stack/run.py` |

TASK2 已接入多进程 CPU MuJoCo 采样和共享 CUDA DrQ-v2 策略，配置、20 万步命令及 W&B 见 [RL 文档](task2_stack/docs/RL.md)。尚未接入 MJX，也没有已验证的完整十石堆叠策略。

Go2 demo 仍通过 `tools/run_demo.py` 独立运行。旧 TASK2 空白机械臂环境不再是活动入口，历史脚本与设计记录见 TASK2 的 `docs/`。
