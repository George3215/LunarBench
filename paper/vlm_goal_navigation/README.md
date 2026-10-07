# VLM 驱动的目标恢复与到达

用户确定的目标：以相机为实际观测接口，研究增强 VLM/世界模型驱动机器人能力的新算法；任务最终需要到达目标附近，先完成仿真。

已完成未经训练的本地 VLM 基线：161 次原任务动作选择与 24 回合连续物理闭环。完整结论见 [VLM_BASELINE_RESULTS.md](VLM_BASELINE_RESULTS.md)，复现见 [REPRODUCE.md](REPRODUCE.md)。

- [模型与本机可行性](MODEL_FEASIBILITY.md)
- [本轮执行协议](runs/20261004_qwen35/EXECUTION_PROTOCOL.md)
- [原六动作任务结果](runs/20261004_qwen35/original_six_action/RESULTS.md)：模型离线选动作，使用已真实执行且固定哈希的物理分支评分。
- [4B 连续闭环结果](runs/20261004_qwen35/closed_loop_v2/RESULTS.md)：模型实时决定下一动作，MuJoCo 执行后返回新的相机图像。
- [9B 连续闭环结果](runs/20261004_qwen35/closed_loop_9b/RESULTS.md)
- [研究设计](RESEARCH_DESIGN.md)与[相关工作](RELATED_WORK.md)：后续新算法计划，不能视为已实现的贡献。

正式结果目录为 original_six_action/ 与 closed_loop_v2/，9B 在线结果另存 closed_loop_9b/。single_step_v1/v2 为混合到达目标的单步诊断，不能混入原任务主表。早期原生多图结果也仅用于诊断。所有实际完成数量以 result.json 与最终报告为准。
