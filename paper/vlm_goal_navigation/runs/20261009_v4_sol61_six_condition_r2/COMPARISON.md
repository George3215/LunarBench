# 第四版地形：GPT-6.1 Sol 相机观测测试

当前状态：`paused`；完成 12/132，成功 12。

| 输入 | 帧条件 | Sol 成功/已完成 | Qwen3-VL-8B 成功/已完成 |
|---|---|---:|---:|
| rgb | current | 1/1 | 1/22 |
| rgb | first_current | 3/3 | 0/22 |
| rgb | all_history | 3/3 | 0/22 |
| rgbd | current | 0/0 | 0/22 |
| rgbd | first_current | 3/3 | 3/22 |
| rgbd | all_history | 2/2 | 2/22 |

未完成时，不能将两列成功数直接当作最终成功率比较。

相机输入审计：检查 678 个实际请求、677 个模型回复，发现 0 个违规项。

仅发送相机 RGB；RGBD 增加对齐理想深度和 RGB 红色区域的深度测量。模型无工具、无目标坐标、无评分器信息、无先前回复或动作文本。

历史帧保留每个动作边界的相机观测，包含共同接近过程；不进行历史抽样。

渲染一致性限制：Sol 原始相机为 Mesa llvmpipe；恢复和后续本地模型锁定相同后端。现有 Qwen 对照使用不同渲染后端：barrier_a0 初始物理状态完全相同，但 RGB 像素不同，因此现有 Qwen 列只作参考，不能声称逐像素配对比较。

比较限制：Different image encoder, reasoning, token budget and decoding from Qwen; same camera area cap, task, action and step budget。每组每地形仅一次运行；结果属于描述性比较。

第一轮六路并发遇到 HTTP 429，未执行模型动作；中断记录保留在相邻的无 r2 后缀目录。正式重跑使用串行模型请求。
