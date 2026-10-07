# 研究问题与文献核查（2026-10-04）

**结论：问题合理，但“曾经看到目标、丢失后主动恢复可见性”本身不是新问题。** 本项目合理的研究范围是：在足式平台与复杂地形中，图像中的目标再捕获与几何视线、运动原语执行后果之间有哪些可测量的差距，历史图像信息能否在公平协议下提高动作选择质量。当前证据不支持宣称必要性、首创性或优于现有完整系统。

## 实际核验的来源

| 文献 | 当前核查范围 | 支持什么 | 对本稿的约束 |
|---|---|---|---|
| R. Bajcsy, *Active perception*, Proceedings of the IEEE 76:966–1005, 1988. DOI [10.1109/5.5968](https://doi.org/10.1109/5.5968) | Crossref 书目信息 | 主动感知是既有领域 | 不将动作选择改善感知作为新概念 |
| Kaelbling, Littman, Cassandra, *Planning and acting in partially observable stochastic domains*, Artificial Intelligence 101:99–134, 1998. DOI [10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X) | Crossref 书目信息 | 历史与信念状态有经典理论基础 | 不把“单帧不足”本身称为新理论 |
| Radmard & Croft, *Active target search for high dimensional robotic systems*, Autonomous Robots 41:163–180（在线发表于 2015，卷期 2017）. DOI [10.1007/s10514-015-9539-8](https://doi.org/10.1007/s10514-015-9539-8) | Springer 全文 HTML 与 Crossref | 明确研究跟踪丢失后利用最后目标信息、考虑运动约束、在线规划使目标回到视场 | 本项目“朝记忆方向一直走”的简单规则不等于该论文算法，不能据其低分否定该研究方向 |
| Radmard, Meger, **Little**, Croft, *Resolving Occlusion in Active Visual Target Search of High-Dimensional Robotic Systems*, IEEE T-RO 34:616–629, 2018. DOI [10.1109/TRO.2018.2796577](https://doi.org/10.1109/TRO.2018.2796577) | Crossref 书目信息，未声称读到全文 | 遮挡恢复与主动视觉搜索已有直接先例 | 旧稿作者 “Law” 应为 “Little”；不臆测具体方法弱点 |
| Hu et al., *OA-NBV: Occlusion-Aware Next-Best-View Planning for Human-Centered Active Perception on Mobile Robots*, [arXiv:2603.11072v2](https://arxiv.org/html/2603.11072v2), 2026 | 摘要及全文 HTML | 将目标遮挡、目标完整度、可行/可达视点和真实运动约束结合，并报告仿真与实机 | 旧稿“NBV 只考虑视点、不考虑能否到达”是过度概括，应删除。本文简化几何规则不代表 OA-NBV |
| Yin et al., *WoMAP: World Models For Embodied Open-Vocabulary Object Localization*, CoRL, PMLR 305:3605–3630, 2025. [正式论文页](https://proceedings.mlr.press/v305/yin25b.html) | PMLR 正式索引与论文页摘要 | 已有动作条件世界模型用于主动目标定位及真实平台 | 不把世界模型加规划作为创新；本轮随机森林是小型动作评分器，不称完整世界模型 |
| *LEAP-NBV*, [arXiv:2609.23974](https://arxiv.org/abs/2609.23974) | 摘要 | 边缘设备部署与主动感知存在相关工作 | 本轮不测边缘延迟/能耗，因此不比较其部署性能 |

检索响应保存在 `literature_sources.json`、`literature_fulltext.json` 和对应 `.source.txt`。第一次尝试 `yin25a` 返回另一篇论文；通过 PMLR 卷索引确认正确地址是 **yin25b**，只引用后者。`OAR-Nav` 等旧会话提及名称未在本轮获得可靠的一手对应记录，故不进入参考文献；这不是其不存在的结论。此核查为定向相关工作核查，不是穷尽式系统综述。

## 可检验的问题与允许的结论

1. **测量一致性**：同一动态机器人状态下，几何可见标签是否等于图像检测？用配对帧混淆矩阵回答，不能用几何标签给相机性能背书。
2. **动作信息价值**：当前图像、最后检测与动作信息，哪些提高留出构造组上的预测与选择？分组留出、固定模型、统一动作和预算；报告负结果。
3. **动作集充分性**：仅前进/左/右是否足够？在同一失去目标状态的执行分支上，与加入后退/转向的可达上界比较。上界来自特权枚举，不是可部署策略。
4. **历史结果可信度**：2240 行只是 224 状态、24 世界、16 构造组。成对世界一起留出；同一状态不同动作不能随机拆分。

“历史有用”与“历史不可或缺”不同；“最佳动作不同”与“成功动作集合不相交”不同；单步不可同时成功也不能推导任意多步无记忆策略不可能成功。该问题值得做，但若投稿主张是通用新方法，还需要新场景、完整强基线、真实检测器和多步闭环评价。
