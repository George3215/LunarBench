# TASK3 新增空间模型测试

仅统计已完成回合；接口/资源失败与导航失败分别报告。

| 模型 | 正式完成 | 确认到达 | 恢复可见 | 终止类型 |
| --- | --- | --- | --- | --- |
| qwen3vl8b | 22/22 | 5 | 13 | {"budget_exhausted": 17, "success_visible_near": 5} |
| spatial_interactor8b | 22/22 | 0 | 10 | {"budget_exhausted": 1, "invalid_model_output": 21} |
| spatialreasoner | 0/22 | 未测 | 未测 | {} |
| spatialrgpt | 2/22 | 0 | 0 | {"invalid_model_output": 2} |
| robobrain25_8b_nv | 0/22 | 未测 | 未测 | {} |

## 两场景接口诊断（每场景最多2动作，不纳入正式成功率）

| 模型 | 完成案例 | 有效动作总数 | 终止 |
| --- | --- | --- | --- |
| spatialreasoner | 2/2 | 0 | {"invalid_model_output": 2} |
| spatialrgpt | 2/2 | 4 | {"budget_exhausted": 2} |
| robobrain25_8b_nv | 2/2 | 4 | {"budget_exhausted": 2} |

SpatialReasoner 的前两例没有执行自主动作：首次输出为不完整自然语言字段；纠正后仍缺 action，或 evidence_frames 类型错误。
这属于当前复杂结构化接口的直接迁移失败，不能推出它没有空间能力；两个接口样例不能代替22场景正式评测。
SpatialRGPT、RoboBrain 通过接口诊断后接续正式场景。SpatialRGPT 的通用多图配置与原生区域深度输入不同。

Qwen/Spatial-Interactor 是前轮冻结协议记录；新增模型有原生 tokenizer/vision/chat template 差异。
RoboBrain 使用作者链接的官方 ModelScope NV 权重，分片与 HF 哈希一致。SpatialRGPT 首轮为无额外region的通用多图配置，不能代表其原生区域深度分支完整能力。
未完成全场景且未通过输入审计前，不给出最终模型排序。即使到达率不同，也不能直接归因于空间训练数据。
当前目标是红球；结果不能单独证明开放词汇语义理解或跨目标类别泛化。
原V4准入证明可重观测，不保证全部场景已经认证完整<1m到达路线。

新方法及开源导航对照的取舍见 [RELATED_WORK.md](RELATED_WORK.md)。
