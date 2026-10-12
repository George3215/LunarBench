# 本地模型正式测评进度

Sol 已按用户要求暂停；请求次数不设配额。归档的裁剪异常试验不进入以下统计。

| 模型 | 协议 | 完成/计划 | 成功 | 终止原因 | 输入审计违规 |
|---|---|---:|---:|---|---:|
| navila-llama3-8b-8f | matrix | 62/132 | 23 | {'fallen': 2, 'out_of_bounds': 15, 'success_visible_near': 23, 'context_limit': 18, 'budget_exhausted': 4} | 0 |

未完成结果仅为部分描述性记录；context_limit 表示模型输入资源限制，不作为导航失败率的直接证据。
主矩阵不丢弃历史帧；NaVILA native8 为明确标注的八帧采样补充实验。Taga 为相机骨干适配，不能代表原生拓扑方法。
