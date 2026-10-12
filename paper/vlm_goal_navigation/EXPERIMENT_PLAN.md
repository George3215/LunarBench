# Task3 / VLM 目标导航实验安排

本文件是 TASK3 与 VLM 目标导航这一研究线的**实验总索引**：有哪些协议、按什么顺序执行、
当前跑到哪一步、结果目录里什么入库什么不入库。协议正文、研究设计与地形定义各自有独立
文档，本文件不复制它们，只做导航与状态汇总。

上游来源与第三方模型固定提交见 [docs/THIRD_PARTY.md](../../docs/THIRD_PARTY.md)。
任务与地形定义（对外可复现的部分）见
[tasks/task3_reobservability/README.md](../../tasks/task3_reobservability/README.md)。

## 1. 研究目标

相机是策略观察环境的实际接口；目标曾经可见、之后被地形几何暂时遮挡。主任务要求机器人
最终**走到目标附近**，重新看到目标只是中间事件。历史记忆、动作条件后果预测与 VLM 的
结合是候选算法方向。完整问题定义、权限分层与算法矩阵（B0–B8）见
[RESEARCH_DESIGN.md](RESEARCH_DESIGN.md)；该文件是拟议设计，不代表已有结果。

## 2. 协议清单

| 协议 | 内容 | 文档 |
| --- | --- | --- |
| 全地形图像上下文与深度对照 | 31 场景 × 3 图像上下文 × 2 传感器条件 = 186 个确定性 trial；`current` / `first_current` / `all_history` | [ALL_TERRAIN_CONTEXT_PROTOCOL.md](ALL_TERRAIN_CONTEXT_PROTOCOL.md) |
| 完整评测与有限难度校准 | 186 trial 完整结束后做几何难度与经验难度划分，最多两次地形版本调整 | [DIFFICULTY_CALIBRATION_PROTOCOL.md](DIFFICULTY_CALIBRATION_PROTOCOL.md) |
| 严格恢复准入 | G1–G4 几何/物理判据，失败项不放宽、不补方块；恢复可见性 ≠ 到达 1 m 内 | `recovery_benchmark_20261009/`、`recovery_mechanism_v4_20261009/PROTOCOL.json` |
| 机制独立与递进消融（v4） | 12 个地形族 × 3 个设计（a0/a1/a2）= 36；正式矩阵 3 模型 × 2 传感器 × 2 历史 = 12 条件 | `recovery_mechanism_v4_20261009/DESIGN.md` |
| 地形设计 v3 | 38 个固定种子配置，16×12 m 场地，生成 + 真实物理检查 | [TERRAIN_DESIGN_V3.md](TERRAIN_DESIGN_V3.md) |

v4 正式矩阵的模型是 `Qwen3-VL-8B`、`VLMnav`、`NaVILA`，传感器 `rgb` / `rgbd`，历史
`first_current` / `all_history`，共 12 个条件；`current`-only 的 Qwen 结果只是补充对照。
尚未安装、因此**不可运行**的模型（ABot-N0、Qwen-VLA、Qwen-RobotNav）已记录在
`v4_multi_model_20261009/models.json`，不能算作缺席的对照结果。

## 3. 执行顺序与当前状态（截至 2026-10-11）

| 阶段 | 入口 | 结果目录 | 状态 |
| --- | --- | --- | --- |
| ① 旧版全地形 186 trial 基线（Qwen3-VL-8B） | `./task3_qwen.sh` | `runs/20261007_222608_qwen3vl8b_all_terrain_context` | **complete**：186/186，成功 19，主要终止 `premature_stop` 156 |
| ② 难度校准与完整评测 | `./task3_calibrate.sh` | `runs/20261008_task3_difficulty_calibration_v2_medium`、`workflows/20261008_task3_difficulty_calibration` | **complete 且已审计**：186/186，成功 14，整轮成功率 7.53%，校准方向 `keep`，2 次修订后停止 |
| ③ 地形设计 v3 生成 + 物理检查 | `MUJOCO_GL=egl bash ./task3_terrains.sh` | `terrain_design_v3/` | 已完成 38 个配置的建模与物理检查；**准入三条件（初始可见 / 移动丢失 / 近距到达）尚未在新版执行**，因此不能用旧版 7.53% 作为新版结果 |
| ④ v4 六条件矩阵（Qwen3-VL-8B） | `tasks/task3_reobservability/v4_qwen3vl_trial.py` | `runs/20261009_v4_qwen3vl_six_condition` | **trial 完成待分析**：132/132，成功 6，`premature_stop` 114 |
| ⑤ v4 六条件矩阵（GPT-6.1 Sol） | `./task3_sol61.sh` | `runs/20261009_v4_sol61_six_condition{,_r2}` | 首轮 `interrupted_incomplete`；`_r2` **paused**：12/132，成功 12 |
| ⑥ v4 多模型矩阵（NaVILA / Taga-VLM） | `v4_multi_model_20261009/` | `v4_multi_model_20261009/` | **未完成**：NaVILA `paused`，62/132，成功 23，审计违规 0；Taga-VLM 已排队未开跑 |
| ⑦ v4 机制消融准入 | `recovery_mechanism_v4_20261009/` | 同目录 `formal_manifest.json`、`ADMISSION_REPORT.md` | 设计完成、严格准入执行中；正式模型消融**尚未启动** |

当前没有已完成的新算法结果，也没有完整到达（走到目标附近）的主表成绩。上表数字来自各目录
的 `RUN_STATUS.json` / `WORKFLOW_STATUS.json` / `FORMAL_PROGRESS.json`，是运行状态，
不是论文结论。

### 与路线图的关系

[RESEARCH_DESIGN.md](RESEARCH_DESIGN.md) §7 列出必做主表（完整到达、B1–B5 消融、地形×方法
配对、可见性条件、受控观测歧义、未见目标外观）与规模预算（先导 144 episodes/方法，正式测试
草案 1080 episodes/方法）。这些是工程预算建议，不是已完成的实验。当前 ①–⑥ 属于该路线图的
地形准入、基线与模型对照阶段，主表尚未开始。

## 4. 结果目录约定

所有 VLM/导航运行写在 `paper/vlm_goal_navigation/` 下：

- `runs/<时间戳>_<名称>/`：每次执行一个目录，含 `RUN_STATUS.json`、`REPORT.md`、
  逐 trial 的 `result.json`。
- `v4_multi_model_20261009/<模型id>/`：多模型矩阵，模型与环境固定提交见 `models.json`、
  `REPOSITORIES.json`。
- `recovery_benchmark_20261009/`、`recovery_mechanism_v4_20261009*/`：准入搜索、设计清单与
  正式 manifest。
- `all_terrain_validation/`：31 场景的实际物理验收证据与场景审查页。
- `terrain_design_v3/`：每个种子配置的 `terrain.json`、`slabs.json`、`inspection.json`、
  `physical_state.json`、`rock_contacts.json`。
- `workflows/`：校准工作流的阶段记录与最终报告。

### 入库范围

仓库只提交**协议、脚本、配置、小体积地形元数据与汇总报告**。以下原始产物保留在本机、
由 `.gitignore` 排除，合计约 10.9 GB / 109 万个文件：

- 逐 step / 逐次模型调用日志（`*.jsonl`、`measurement.json`、`network_input.json`、
  `output.json`、`context.json`、`messages.json` 等）；
- 渲染转储（`rgb/`、`rgbd/`、`depth.npy`、逐位深度归档）与提示词原文
  （`raw_output.txt`、`rendered_prompt.txt`）；
- 浏览器导出（`trial_browser.html`）与 `archives/`、`isaac_tasks/outputs/`。

这不是数据丢失：原始产物按上表路径留在本机。**换机器后无法直接从仓库重放这些对照**，
需要按对应协议重新执行准入与 trial，因此引用这些数字时必须同时引用协议、模型提交和日期。

## 5. 入口命令

```bash
# 地形：列出 / 自检 / 交互查看
python tasks/task3_reobservability/terrain_viewer.py --list
python tasks/task3_reobservability/terrain_viewer.py --check --terrain all

# 地形设计 v3：生成全部配置与物理检查
MUJOCO_GL=egl bash ./task3_terrains.sh
bash ./task3_terrains.sh --list

# 全地形 186 trial 基线（需要本地 Qwen3-VL-8B 权重）
./task3_qwen.sh

# 难度校准工作流
./task3_calibrate.sh

# v4 六条件矩阵（GPT-6.1 Sol API）
./task3_sol61.sh
```

本地模型路径、token 上限、图像预处理与成功判据在各协议文档中固定；实验脚本不随 TASK3
对外任务接口发布，缺失依赖时直接运行会失败，这是刻意设置的范围边界。

## 6. 指标与必须说明的边界

- 主指标是**完整到达成功率**；重新可见率、恢复后到达率、错误 STOP、重复丢失次数是诊断指标。
- 成功必须同时满足：到达可通行邻域、最近真实相机证据确认正确目标且策略发出 STOP、
  机器人稳定未触发失败、在统一预算内完成。
- 路径效率使用**实测可行参考路线**，不是最短路径 SPL；不能穿墙算欧氏距离。
- 失败不从效率统计中消失；资源终止（`context_limit`、`gpu_memory_limit`）单独报告，
  不能据此宣称模型任务能力失败。
- 当前 ①–⑥ 的成功率都来自**旧版或未完成**的配置；v3/v4 新地形在通过全部准入前不得引用。
- `oracle` 是评估用的特权真值，不是学习到的世界模型；反事实基线 C 是把候选动作真跑一遍的
  上界 / teacher，不是可部署策略。

## 7. 下一步

1. 确认 v3 地形与 v4 机制的起点/目标配对（或明确采用非配对测试），完成全部准入。
2. 在新地形上按 v4 正式矩阵（12 条件）跑完 Qwen3-VL-8B、VLMnav、NaVILA，补齐 Taga-VLM。
3. 提高决策世界数量（`RESEARCH_DESIGN.md` §8 的预算），再冻结确认测试。
4. 主表：同 VLM 的 B1–B5 消融、强传统方法与最接近文献方法对比。
