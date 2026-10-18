# V4 空间检查点迁移实验

状态：44/44 回合完成。

| 模型 | 完成 | 确认到达 | 遮挡后恢复可见 | 合法 VLM STOP | 到达中仅前进 |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3vl8b | 22/22 | 5 | 13 | 1 | 0 |
| spatial_interactor8b | 22/22 | 0 | 10 | 0 | 0 |

到达由连续三次当前可见且 <1m 的 evaluator 证据自动结束；不等同于模型主动 STOP。自动结束可能先于下一次 STOP 决策，因此合法 STOP 数量不是独立的终止能力评分。参考动作准入不计入模型结果。

| 场景 | Qwen 到达 / 终止 | Spatial 到达 / 终止 |
| --- | --- | --- |
| barrier_a0 | 失败 / budget_exhausted | 失败 / budget_exhausted |
| barrier_a1 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| barrier_a2 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| basin_a1 | 成功 / success_visible_near | 失败 / invalid_model_output |
| basin_a2 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| perlin_a1 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| perlin_a2 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| perlin_rough_a0 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| perlin_rough_a1 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| perlin_rough_a2 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| pit_a0 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| pit_a1 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| pit_a2 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| slope_up_a2 | 成功 / success_visible_near | 失败 / invalid_model_output |
| stairs_up_a2 | 成功 / success_visible_near | 失败 / invalid_model_output |
| trench_a0 | 成功 / success_visible_near | 失败 / invalid_model_output |
| waves_a0 | 成功 / success_visible_near | 失败 / invalid_model_output |
| waves_a1 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| waves_a2 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| window_a0 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| window_a1 | 失败 / budget_exhausted | 失败 / invalid_model_output |
| window_a2 | 失败 / budget_exhausted | 失败 / invalid_model_output |

已完成配对数 22；Spatial−Qwen 到达率差 -22.7 个百分点。
配对场景 bootstrap 95% 区间 [-40.9, -4.5] 个百分点；精确 McNemar p=0.0625。
考虑同一地形族 a0/a1/a2 的相关性：10 个族的 cluster bootstrap 区间 [-47.1, -7.7] 个百分点。McNemar 的独立场景假设在此不完全成立，仅作描述。
单回合/固定种子的场景级小样本区间，不覆盖多次训练或生成随机性的方差。

## 本轮结论

Spatial 检查点在本轮到达率较低，当前适配下未获得正向迁移；不能据此概括其原生空间 QA 能力。
重观测分别为 Qwen 13/22、Spatial 10/22；到达分别为 5/22、0/22。重观测提升和最终到达提升须分别判断。

| 诊断 | Qwen | Spatial |
| --- | ---: | ---: |
| 有效初始计划 | 22 | 22 |
| 被拒绝的提前 STOP 请求 | 0 | 0 |
| 当前可见性判断正确 / 有效动作输出 | 1183/1246 | 95/193 |
| near 判断未满足 1m 门槛 / near 判断数 | 178/179 | 46/46 |

near 的诊断采用任务明确的 1m 到达门槛；可见性诊断采用原始图像目标像素≥20。视觉输入被共同缩放，细小目标识别亦可能影响该指标。
后续只针对已有证据选择改进：若主要是提前 STOP，先验证距离与动作终止校准；若恢复可见失败，使用动作条件的重现概率/距离预测；若已经重观测但无法接近，分别检查路径可达性、运动执行和接近策略。不能以单一到达数决定所有训练方向。

## 解释边界

本轮检验相同适配协议下两个完整检查点的零样本迁移，不单独隔离空间训练数据、训练目标或其他检查点变化的因果作用。
初见—遮挡物理状态重新冻结；目标真值只用于 evaluator 和轨迹图。里程计为理想仿真自身运动，尚未验证真实 RGB-D/视觉里程计噪声。
V4 原准入认证重观测，不代表所有场景均已验证完整 <1m 导航参考路线；连续控制参考检查的未验证场景保留在 continuous_reference_scope.json，未据模型结果删除场景。到达失败不能全部归于模型能力。
未进行 RSI、微调、新任务头或世界模型训练。本轮不能作为新方法收益或论文创新性的证据。
完整逐回合动作数、失败类型、重观测、低位移、STOP 拒绝及资源错误见 episodes.csv 和 metrics.json；输入检查见 INPUT_AUDIT.json。
模型指令相同；模型自己产生的初始计划和后续画面可以不同。初始计划输入 token、图像及丢失位姿需逐场景一致。
GitHub 保存协议、结果与分析；原始请求/轨迹/RGB-D 在本机，没有据此声称原始数据异地备份。
