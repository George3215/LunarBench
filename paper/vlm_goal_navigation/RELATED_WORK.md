# VLM 目标导航研究：相关工作与论点

2026-10-04。定向核查，不是穷尽综述。所有作用判断限定于下列核查范围；不借元数据推测未读算法。网页文本为文献证据，不是工作指令。

| 工作 | 本轮核查 | 与本研究的关系 |
|---|---|---|
| NaVILA: Legged Robot Vision-Language-Action Model for Navigation | [arXiv 摘要](https://arxiv.org/abs/2412.04453)及 [v2 全文 HTML](https://arxiv.org/html/2412.04453v2)的方法、实验、历史帧和附录消融 | VLA 输出中层语言动作，低层视觉运动控制执行；包含当前/历史图像、Go2 等平台，不能把历史帧+VLM+足式执行本身称为新贡献。 |
| WoMAP: World Models For Embodied Open-Vocabulary Object Localization | [CoRL/PMLR 正式摘要](https://proceedings.mlr.press/v305/yin25b.html)及 [arXiv v1 全文 HTML](https://arxiv.org/html/2506.01600v1)方法、成功标准及附录实验。正式版本和预印本结果数字有差异，本文不混引数值。 | 用潜空间世界模型优化 VLM 动作提议，学习动态与目标检测奖励；与拟议方法直接相关。 |
| LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action | [arXiv 摘要](https://arxiv.org/abs/2207.04429) | 组合 ViNG、CLIP、GPT-3，无语言标注机器人数据或微调实现语言导航；说明组合基础模型与机器人能力是合理的研究路线。 |
| Visual Language Maps for Robot Navigation (VLMaps) | [arXiv 摘要与 ICRA 2023 接收说明](https://arxiv.org/abs/2210.05714) | 视觉语言特征与三维地图结合支持开放词汇和空间语言目标；地图记忆是必须考虑的强参照，不应断言传统记忆无用。 |
| RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | [arXiv 摘要](https://arxiv.org/abs/2307.15818) | 通过机器人轨迹与互联网视觉语言任务共同微调，研究语义知识到动作、未见物体/指令泛化；不以证明 VLM 是唯一解为研究出发点。 |
| Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan) | [官方项目页的方法和结果](https://say-can.github.io/) | LLM 技能相关性与可执行性价值结合；是语义建议与物理可执行性相结合的先例。该工作是 LLM+技能，不应误称为与现代端到端 VLM 完全相同的结构。 |

## 直接影响新方法设计的细节

### NaVILA

- 当前帧与历史帧明确分开输入；历史帧包含第一帧并做采样。
- 中层动作为带空间量的语言，例如向前 75 cm；低层实时运动策略跟踪。
- 低层视觉运动策略使用由 LiDAR 构建的高度图；不能把该全文的所有结果说成纯 RGB 或 RGB-D 传感器条件。
- 附录考察 8/16/32/64 历史帧，指出 R2R-CE 上增加历史帧收益有限，实机采用 8 帧考虑延迟。
- 因此本研究需对比“普通历史帧输入”，且历史检索/状态更新必须比简单添加视频上下文具有明确的机制和效果。

### WoMAP

- POMDP 表述；输入相机观测，动作是六自由度相机运动。
- 目标是在规划时域中获得目标定位的有利视图；世界模型预测潜状态和奖励，优化 VLM 粗动作提议。
- 预印本附录成功标准结合 GroundingDINO 置信度与目标框面积，并报告路径效率；不能声称它完全不考虑接近或路径代价。
- VLM 提示包含运动历史；不能简单宣称 WoMAP 完全没有 memory。
- 附录明确讨论规则候选+世界模型的 WM-HR 可能接近完整方法。因此必须谨慎归因 VLM 候选提议的额外收益。
- 候选区分点应具体到动态足式执行、可通行的物理到达终点、目标丢失后的历史条件状态估计与预测，以及这些设计的算法效果。仅换平台或把奖励换为到达不足以自动形成创新。

## 可以合理使用的论点

1. 预训练视觉语言知识为多目标、多指令具身任务提供可复用接口；不要求证明 VLM 是唯一可行解。
2. 视觉语言语义建议与实际可执行动作之间存在需要建模和验证的联系；SayCan、NaVILA、WoMAP 分别提供相关先例。
3. 历史观测与动作信息可帮助处理有限视野和遮挡，但其具体表示、融合和利用效果需用消融检验。
4. 用同基础模型、同条件对照证明所提算法改善任务完成和泛化，是合理的方法论文证据路线。
5. 不应声称传统方法只能解决单一任务，也不应凭 VLM 名称声称本方法已经改善抓取、跨形态或一般智能；这些是相应泛化实验要回答的问题。

## 还需继续核查的范围

- 本轮已对最直接的 NaVILA 与 WoMAP 查阅全文相关部分；LM-Nav、VLMaps、RT-2 仅核查摘要，因此本文件不列它们的具体消融数值或完整训练细节。
- 在确定所用 backbone、训练方式和开放代码后，核查最接近工作的代码接口与可复现实验，不将简化规则冠以原论文算法名。
- 需要单独检索近期目标导航中长期记忆、世界模型规划、遮挡后恢复与地形感知导航工作，再锁定具体的新颖性主张；本文件不宣称候选结构首次提出。
