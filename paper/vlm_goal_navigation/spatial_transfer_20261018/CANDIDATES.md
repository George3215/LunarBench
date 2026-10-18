# 空间底座候选与本轮取舍

核对日期：2026-10-18。作者宣传或离线空间 QA 能力不等于本任务成功；实际运行状态以本目录记录为准。

| 模型 | 本轮状态 | 比较中需要控制的差异 | 官方依据 |
| --- | --- | --- | --- |
| Qwen3-VL-8B-Instruct | 已下载完整 BF16；V4 正式评测中 | 原始对照，共同 processor、prompt、动作及预算 | [权重](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Spatial-Interactor-Qwen3-VL-8B | 已下载完整 BF16；按队列接续 GPU 烟测与正式评测 | 同架构空间训练检查点；统一缓存、EOS 和预处理；自身计划可以不同 | [权重](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-8B)、[官方代码](https://github.com/ZJU-OmniAI/Spatial-Interactor) |
| RoboBrain2.5-8B-NV | 未下载、未测试；访问条件需要登录接受并共享联系信息，本轮未代为接受 | Qwen3-VL 架构；三维理解及操作任务进度不等于持续导航记忆；官方 navigation 示例是图像选点，并非本环境物理闭环 | [权重与访问条件](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV)、[代码](https://github.com/FlagOpen/RoboBrain2.5) |
| SpatialReasoner | 未下载、未测试；后续备选 | Qwen2.5-VL-7B 及空间推理适配不同，需自己对应的 processor，不能宣称只改变训练空间能力一个因素 | [权重](https://huggingface.co/ccvl/SpatialReasoner) |
| SpatialRGPT-VILA1.5-8B | 未下载、未测试；后续备选 | VILA 架构及 region/depth 输入适配，不能把检测/分割/区域模块收益混入原始图像底座比较 | [权重](https://huggingface.co/a8cheng/SpatialRGPT-VILA1.5-8B)、[代码](https://github.com/AnjieCheng/SpatialRGPT) |

优先比较同一 Qwen3-VL-8B 架构的两个完整检查点，以减少工程接口差异。其余模型后续可作为跨底座迁移验证，不能在未运行前填写成功率或作为失败样本。
