# V4 spatial-checkpoint transfer comparison (2026-10-18)

研究问题：目标最初可见，运动后被地形遮挡时，已有空间训练的 VLM 检查点能否提高重观测与接近目标的能力？本轮不训练新模型，也不预设正结果。

## 恢复边界

旧工作区被误删后，从 LunarBench `a63a83e8a976663565408467955a0fb3f924c77d` 恢复。
V4 地形代码、36 个候选配置、22 个准入记录及历史小型结果仍在 Git 中。
原始大体量图像、轨迹、模型权重和 10 月 17 日记忆导航脚本未随 Git 恢复。
因此本轮是**重新建立的配对评测**，不声称复现已删除脚本，也不将历史成功率直接并入本轮。

## 预先固定的比较

| 检查点 | 固定 HF revision | 用途 |
| --- | --- | --- |
| Qwen/Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 基座对照 |
| kagakouko/Spatial-Interactor-Qwen3-VL-8B | `455e99af74252e6be8fa572f6b469fd1f53fb242` | 空间训练检查点迁移 |

共 22 个 V4 场景，每模型一回合，共 44 回合。初始位姿、固定遮挡前缀、任务 prompt、RGB-D、事件记忆选择、理想自运动观测、动作接口、生成参数和 64 次自主动作预算相同。初始计划只读第一帧，由当前检查点生成；行动由同一 VLM 闭环决定，无外部路线规划器或动作回退。

- 事件记忆最多四帧：第一帧、最后可见帧、最近的可见→不可见转折帧、当前帧；去重后按时间排序。
- 里程计是**仿真提供的理想自身运动**，以第一帧机器人位置和水平朝向为原点；不是视觉里程计，也不是已估计 SLAM。不向模型提供真实目标位置或障碍物几何。
- 两模型统一使用原始 Qwen processor、tokenizer 与 chat template，图像面积上限 65536。另行检查 Spatial tokenizer 的共有词表兼容性与相同初始请求的 tokens/图像哈希。
- 下载后检查发现 Spatial 保存了训练时的 `use_cache=false` 和不同 EOS 列表；在首次 Spatial 推理前改为使用原始 Qwen 的 inference generation config（cache 开启、同一 EOS/pad、greedy）。已运行的 Qwen 使用的正是这些原始设置；它的源快照保持原样，Spatial 的源快照包含这项显式统一补丁。不得把两次源哈希不同隐瞒成字节级同一脚本。
- 每次行动最多两次格式/非法 STOP 修正请求；两次仍非法时记录明确终止，不执行隐藏的替代动作。
- STOP 只在**当前** RGB 红色像素 ≥20，且当前红区有效表面距离严格在 (0,1) m 内开放；历史证据不开放 STOP。
- 到达是独立 evaluator 对真实目标身份、可见面积、距离、直立和边界连续三次 10Hz 检查通过，自动结束回合。**到达不能写成 VLM 主动 STOP。**
- 记录无效输出、STOP 拒绝、资源错误、遮挡后重观测、转向/侧移、无位移与 forward-only 轨迹；小样本阴性结果如实报告。

它检验的是检查点在本适配协议上的迁移收益。即使 Spatial 更好，也不能仅凭换检查点把因果全部归于某一种训练能力，更不能把 prompt/工具组合写成已验证的新方法。

## 数据与备份

`models/*.json` 保存版本、上游文件元数据、本地 SHA256；权重放在 Git 忽略的 `.local/models/`。
`terrain_validation/verify_report.json` 保存重新实测的初见—遮挡结果。
`reference_validation_r2/validation.json` 保存历史参考动作的重新验证和输入图像哈希：22/22 通过，属于地形准入，**不是模型成功率**。
参考分支按历史准入代码在丢失时重置低层控制器；VLM 自主回合不重置。第一次未重置的 `reference_validation/` 保留为诊断，不能当作参考回放失败的正式结论。
22 个场景认证的是**可恢复可见**，并非全部具有已验证的 `<1m` 完整导航参考路线；到达率需结合这一边界解释，不能把未到达全归于模型能力。
`runs/<checkpoint>/` 保存独立结果与协议。大体量原始 RGB-D、逐次模型请求和物理轨迹留在本机；GitHub 保存脚本、协议、逐回合结果、审计与分析。GitHub 记录不等于原始数据已异地备份。

状态以 `STATUS.json`、各 run 的 `RUN_STATUS.json` 和最终 `REPORT.md` 为准。不存在的文件或未完成的回合不能作为结果。
`case_records/` 保存可推送的精简决策、初始计划、输入哈希和评价证据摘要；`trajectories.svg` 中 `*` 标记确认到达，`x` 标记失败。完整图像与逐 token 请求不在这些摘要中。

## 运行

在仓库根目录，用具备 PyTorch CUDA、Transformers Qwen3-VL 和 MuJoCo 的 Python：

```bash
python tools/fetch_task3_spatial_models.py --root .local/models --record paper/vlm_goal_navigation/spatial_transfer_20261018/models
MUJOCO_GL=egl python tasks/task3_reobservability/validate_spatial_transfer_v4.py --out paper/vlm_goal_navigation/spatial_transfer_20261018/reference_validation_r2
MUJOCO_GL=egl python tasks/task3_reobservability/spatial_transfer_v4.py --model .local/models/qwen3vl8b --processor .local/models/qwen3vl8b --out paper/vlm_goal_navigation/spatial_transfer_20261018/runs/qwen3vl8b
MUJOCO_GL=egl python tasks/task3_reobservability/spatial_transfer_v4.py --model .local/models/spatial_interactor8b --processor .local/models/qwen3vl8b --out paper/vlm_goal_navigation/spatial_transfer_20261018/runs/spatial_interactor8b
```

若直连不可用，可在命令进程上设置 HTTP(S) proxy，不写入全局配置。`STOP` 文件在下一次动作边界生效；未完成回合保留，不能直接当作完整结果或无验证续跑。

## 来源

- [Spatial-Interactor 官方仓库](https://github.com/ZJU-OmniAI/Spatial-Interactor)
- [Spatial-Interactor 完整检查点](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-8B)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

RoboBrain2.5-8B-NV 的访问门槛、SpatialReasoner 的 Qwen2.5 底座差异、SpatialRGPT 的 region/depth 输入差异需要独立适配；本轮主比较不把这些额外差异混入相同 Qwen3 协议。
