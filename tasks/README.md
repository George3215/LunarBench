# Tasks

任务负责目标、成功/失败条件和评价；资产与引擎实现留在各自目录。

| 任务 | 说明 | 入口 |
| --- | --- | --- |
| （无） | 已有 Go2 月面移动 demo，本身不是任务 | `../tools/run_demo.py` |
| [task1_collect](task1_collect/) | 石头样本收集：20×20 m 场地 + 红边界 + 白色实体围栏收集区，石头进收集区加分，参数由 YAML 控制 | `task1_collect/run.py` |

已有 demo 仍是独立的可运行 demo，不是任务，也不受 TASK1 影响：TASK1 的场景写在
`task1_collect/generated/`，与 demo 的 `mujoco/generated/` 分开。

任务按需逐个实现，不提前建设通用的 Task 基类、任务注册表或奖励框架。目前只有
TASK1 一个任务，因此它的结构就是当前任务的实际写法：一个目录、一个 YAML、
一个 `run.py`，需要复用的部分直接从 demo 和 `tools/` 取。
