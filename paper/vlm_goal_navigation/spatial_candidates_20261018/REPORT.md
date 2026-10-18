# TASK3 新增空间模型测试

仅统计已完成回合；接口/资源失败与导航失败分别报告。

| 模型 | 正式完成 | 确认到达 | 恢复可见 | 终止类型 |
| --- | --- | --- | --- | --- |
| qwen3vl8b | 22/22 | 5 | 13 | {"budget_exhausted": 17, "success_visible_near": 5} |
| spatial_interactor8b | 22/22 | 0 | 10 | {"budget_exhausted": 1, "invalid_model_output": 21} |
| spatialreasoner | 0/22 | 未测 | 未测 | {} |
| spatialrgpt | 20/22 | 3 | 7 | {"invalid_model_output": 17, "success_visible_near": 3} |
| robobrain25_8b_nv | 0/22 | 未测 | 未测 | {} |

## 已完成回合的行为与失败分析

- **qwen3vl8b**：成功场景 basin_a1, slope_up_a2, stairs_up_a2, trench_a0, waves_a0；只执行前进/STOP即到达 0 回合；恢复可见但未到达 8 回合。
  有效动作判断 1246 次，其中当前可见性一致 1183 次；宣称 near 179 次，其中 178 次不满足当前可见且<1m。非法 STOP 路径屏蔽 171 次，实际拒绝 STOP 0 次。
- **spatial_interactor8b**：成功场景 无；只执行前进/STOP即到达 0 回合；恢复可见但未到达 10 回合。
  有效动作判断 193 次，其中当前可见性一致 95 次；宣称 near 46 次，其中 46 次不满足当前可见且<1m。非法 STOP 路径屏蔽 68 次，实际拒绝 STOP 0 次。
  当前完成样本主要受结构化输出失败影响，不能据此把差异归因为空间感知或遮挡推理能力。
- **spatialrgpt**：成功场景 basin_a1, perlin_a2, trench_a0；只执行前进/STOP即到达 3 回合；恢复可见但未到达 4 回合。
  有效动作判断 100 次，其中当前可见性一致 39 次；宣称 near 95 次，其中 94 次不满足当前可见且<1m。非法 STOP 路径屏蔽 68 次，实际拒绝 STOP 0 次。
  当前完成样本主要受结构化输出失败影响，不能据此把差异归因为空间感知或遮挡推理能力。
  当前所有成功都只含前进/STOP，尚不能证明学会了需要转向、侧移或绕行的遮挡恢复策略。

## 完整22场景配对比较

- spatialreasoner：等待全部22场景及输入配对审计通过，不给最终排序。
- spatialrgpt：等待全部22场景及输入配对审计通过，不给最终排序。
- robobrain25_8b_nv：等待全部22场景及输入配对审计通过，不给最终排序。
这些统计只描述冻结小样本套件，地形级别相关；跨骨干和原生视觉预处理差异仍在，不能单独归因于空间预训练。

## 两场景接口诊断（每场景最多2动作，不纳入正式成功率）

| 模型 | 完成案例 | 有效动作总数 | 当前可见性判断正确/有效判断 | 终止 |
| --- | --- | --- | --- | --- |
| spatialreasoner | 2/2 | 0 | 0/0 | {"invalid_model_output": 2} |
| spatialrgpt | 2/2 | 4 | 0/4 | {"budget_exhausted": 2} |
| robobrain25_8b_nv | 2/2 | 4 | 4/4 | {"budget_exhausted": 2} |

SpatialReasoner 的前两例没有执行自主动作：首次输出为不完整自然语言字段；纠正后仍缺 action，或 evidence_frames 类型错误。
这属于当前复杂结构化接口的直接迁移失败，不能推出它没有空间能力；两个接口样例不能代替22场景正式评测。
SpatialReasoner 已另排完整22场景诊断，保留相同 prompt 与解析规则，以免只根据两个样例作判断；执行状态见 REASONER_QUEUE_STATUS.json。
SpatialRGPT、RoboBrain 通过接口诊断后接续正式场景。SpatialRGPT 的通用多图配置与原生区域深度输入不同。

Qwen/Spatial-Interactor 是前轮冻结协议记录；新增模型有原生 tokenizer/vision/chat template 差异。
RoboBrain 使用作者链接的官方 ModelScope NV 权重，分片与 HF 哈希一致。SpatialRGPT 首轮为无额外region的通用多图配置，不能代表其原生区域深度分支完整能力。
未完成全场景且未通过输入审计前，不给出最终模型排序。即使到达率不同，也不能直接归因于空间训练数据。
当前目标是红球；结果不能单独证明开放词汇语义理解或跨目标类别泛化。
请求已经包含由 RGB-D 计算的红像素数和距离；可见性字段与记录一致不等于独立验证视觉识别能力。
原V4准入证明可重观测，不保证全部场景已经认证完整<1m到达路线。

新方法及开源导航对照的取舍见 [RELATED_WORK.md](RELATED_WORK.md)。
