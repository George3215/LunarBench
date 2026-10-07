> **研究方向更新（2026-10-04）**：用户已明确新论文为 VLM/世界模型算法研究，任务以实际到达目标附近为完成条件。当前设计见 [VLM 目标导航研究](../vlm_goal_navigation/README.md)。本目录稿件保留为旧单步任务的审计与探索材料；相机/几何差异不再作为拟议算法论文主贡献。

# TASK3 审计、补充实验与论文（2026-10-04）

先读 [AUDIT.md](AUDIT.md)，再读 [RESULTS.md](RESULTS.md) 与 [manuscript.md](manuscript.md)。LaTeX 源为 [manuscript.tex](manuscript.tex)。

**最终数据只有 `strict_camera/` 这一批。** 根目录 `camera_rollouts.json`、`camera_analysis.json`、`camera_pilot.json` 以及 `strict_camera_observer_mutation_incomplete/` 均为保留的试跑/诊断材料，不可混进最终表格。

| 内容 | 文件 |
|---|---|
| 最终协议与所有帧的记录 | `strict_camera/camera_rollouts.protocol.json`、`strict_camera/camera_rollouts.json` |
| 最终分组留出分析与逐折预测 | `strict_camera/camera_analysis.json`、`strict_camera/camera_oof_predictions.npz` |
| 最终 seen / zero-pixel-loss / reacquired 原始图片 | `strict_camera/frames/` |
| 运行源码快照与数据完整性 | `strict_camera/source_snapshot/`、`strict_camera/integrity_check.json` |
| 运行环境与策略/XML 哈希 | `runtime_manifest.json`、`asset_manifest.json` |
| 修复后无相机副作用验证 | `observer_check_corrected.json`（坐标差、路径差均 0） |
| 相机渲染及语义逻辑检查 | `renderer_checks.json`、`regression_checks.json` |
| 历史数据重算 | `legacy_reanalysis.json`、`legacy_oof_predictions.npz` |
| 相关工作一手来源核查 | `LITERATURE.md`、`literature_sources.json`、`literature_fulltext.json` |
| 自动生成结果表 | `camera_results.csv`、`RESULTS.md` |
| 最终统计图 | `audit_results.png` / `.pdf` |
| 最终真实渲染图像序列 | `camera_evidence.png` / `.pdf` |
| 论文 | `manuscript.md`、`manuscript.tex` |
| 编译状态 | `latex_compilation.json`；内置 LaTeX 编译器平台环境错误，**没有生成或验证论文 PDF** |

最终运行共 24 个候选世界，23 个有效世界，15 个构造组，138 个三秒恢复分支。相机为运行时注入的 MuJoCo RGB-D，目标为红色标记；恢复动作不使用目标世界坐标。单步相机成功率：Random 0.362，Forward 0.739，Image+Action 0.725，Image+Memory+Action 0.725。学习数字平均三个拟合 seed，不是三次独立物理实验。

英文稿是一份完整的探索性研究稿，保留负结果与局限；不能把它表述为已证明通用新方法优越性或已可直接提交的最终投稿稿。作者、单位、基金等信息保持待填。
