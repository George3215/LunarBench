# 两轮五模型实验成果（2026-10-19核对）

两轮各5模型×22个冻结场景，共220个正式回合已完成；接口短测试与失败的SpatialRGPT兼容性尝试另存，不计正式分母。当前无GPU推理任务。

| 模型 | 原复杂JSON：到达/22 | 新动作接口：到达/22 | 新接口：重观测/22 | 新接口：模型主动STOP成功 |
| --- | --- | --- | --- | --- |
| qwen3vl8b | 5 | 8 | 18 | 3 |
| spatial_interactor8b | 0 | 7 | 19 | 2 |
| spatialreasoner | 0 | 7 | 21 | 0 |
| spatialrgpt | 3 | 0 | 15 | 0 |
| robobrain25_8b_nv | 5 | 0 | 13 | 0 |

## 得到的有效结论

1. Qwen由5/22（22.7%）变为8/22（36.4%），新增成功为 perlin_a2, pit_a2, window_a0，没有旧成功变失败；配对McNemar双侧p=0.25。这是小样本、多个actor协议组件共同变化下的描述性改善，不能宣称显著提升或空间推理学到新能力。
2. 新接口五模型均无invalid_model_output终止，解决了这组测试的动作可执行性障碍。Spatial-Interactor和SpatialReasoner各由0变为7/22；两者成功场景完全相同，且均只执行FORWARD/STOP。Qwen也有7/8成功属于此类，仅window_a0成功包含转向和后退。当前不能据成功率证明复杂遮挡规划能力。
3. SpatialReasoner重观测21/22，但仅7/22到达。RoboBrain重观测13/22、SpatialRGPT15/22，却都0到达。全新接口下，RoboBrain和SpatialRGPT各1408个自主动作全部是左右转向，零平移动作；转动可能让目标进入视野而不接近它，不能把重观测次数直接解释为空间推理收益。
4. 不能断言RoboBrain或SpatialRGPT本身没有空间能力：同样prompt不保证不同权重/架构对受约束动作标签有同样响应。零到达应先作为当前接口下的动作退化负结果。SpatialRGPT仍未测试原生区域/深度分支。

## 哪些成功了

- qwen3vl8b: basin_a1, perlin_a2, pit_a2, slope_up_a2, stairs_up_a2, trench_a0, waves_a0, window_a0。
- spatial_interactor8b: basin_a1, perlin_a2, pit_a2, slope_up_a2, stairs_up_a2, trench_a0, waves_a0。
- spatialreasoner: basin_a1, perlin_a2, pit_a2, slope_up_a2, stairs_up_a2, trench_a0, waves_a0。
- spatialrgpt: 无。
- robobrain25_8b_nv: 无。

## 已验证的实验边界

新协议五模型共用22冻结起点、64自主动作预算、事件关键帧、RGB-D、自身理想里程计、首帧视觉计划及实际动作效果。全部完成输入配对审计；无目标真实坐标/地图/参考路线进入模型文本。不同架构的原生tokenizer与视觉预处理仍不同。
新协议同时改变actor提示词、闭集动作输出约束、输出预算和取消可选逐步计划更新；跨版本变化不能归因于单一组件。
确认到达是评估器连续3采样满足可见、<1m、直立和界内，不等于全部由VLM主动STOP完成。Qwen8次到达中3次伴随合法模型STOP，其余按自动到达终点记录；所有STOP仍受当前RGB-D近距门槛限制。
V4准入证明目标能恢复可见，尚未认证每场景完整<1m路线。结果限于红球、理想模拟传感器和单次greedy，不代表开放词汇、噪声传感器或真机。

## 下一步最有区分度的工作

先加入始终前进和固定转向等简单策略对照，筛出必须绕行、需要改变接近方向的案例；分别报告这些案例的遮挡恢复和最终接近。该建议尚未执行，不能提前给结果。
若继续学习方法，应针对“恢复可见后的稳定接近”和“给定动作后能否重现、还需多远”的可验证预测设计监督，而不是仅以空间模型替换或复杂JSON作为论文创新。

原始RGB-D、输入token、模型输出、动作与状态均在本机保留；GitHub备份协议、源码、逐回合结果、精简逐步证据、分析和带成功标记轨迹。ACTION_BEHAVIOR_AUDIT.json给出本次计数与校验。
