# 冻结 V4：遮挡恢复分支预测 pilot

本实验执行已批准的 [量化评测设计](../native_diagnostics_20261019/OCCLUSION_EVALUATION_DESIGN.md) 的第一阶段：在同一个真实几何遮挡状态，预测四个短动作分支的重现概率、条件重现时间，选择一个分支。两个模型使用相同允许观测、候选、生成上限和确定性解码；不微调模型，不把分支恢复当作最终导航成功。

开始前已锁定 `protocol.json`。22 个 V4 准入布局各取一个丢失状态，按字母顺序前 8 个布局再取延后 0.4 秒状态，计划 30 个。`basin_a1` 延后状态不再满足持续真实遮挡，因此在执行候选前排除；最终 29 个状态、116 条正式分支，另外 29 条重复 `advance` 校验，不扩充统计样本。未按候选结果更换状态或候选。

主要入口：

- `RESULTS.md`：结果与解释边界；`SUMMARY.json`：全部指标及按布局聚类的区间。
- `index.html`、`plots/`：输入画面、四条物理轨迹、恢复标记、模型原始回答。
- `forecast_scores.csv`：58 份模型回答对应的 232 条预测—标签记录；其中只有 116 条不同物理分支。
- `INPUT_AND_LABEL_AUDIT.json`：每份请求的输入哈希、解析方式和生成上限。
- `PROVENANCE.json`、`source_snapshots/`：环境、模型与控制器文件哈希、源码。

## 输入与评分隔离

每个 `states/<id>/request.json` 在生成分支标签前封存，包括初始、最后可见和当前 RGB-D，自身相对里程计，以及由 RGB 红色区域、相机深度、自身标定/运动变换得到的粗略目标表面点。当前目标不可见，不提供仿真目标坐标或隐藏几何。理想自身位姿来自仿真自身状态，因此这是**理想里程计条件**，不是实际视觉里程计系统。

`tools/task3_occlusion_forecasts.py` 只读取状态清单、这些公开请求及对应图像，保存最终送入模型的文本、图像哈希、token 和图像张量哈希。该脚本没有读取分支评分文件的代码。

只有离线评分器读取 `gate_evaluator_only.json`、候选映射和 `branches/*/label.json`。轨迹图上的目标星标同样只用于离线解释，不曾提供给模型。原始米制深度保存为 NPY，输入图是固定 0–10m 灰度深度。粗略目标点是已观察表面点；没有假装给它添加经校准的不确定性分布。

恢复定义：精确目标身份对应的 RGB 可见区域至少 20 像素，upright≥0.5、在界内，连续三个 10Hz 采样满足。6 秒未恢复记为窗口内未恢复，不推断永远不可恢复。失稳、越界提前结束的分支仍计入分母。接触数只作原始诊断，脚接地不当作碰撞。

## 状态重放

分支使用完整 `mj_copyData`，并恢复 Go2 TorchScript 策略内部的 `history`、上次 action、PD 目标、控制 tick 和随机数状态。只恢复 qpos/qvel 会遗漏策略历史。四条候选前均核对物理/控制器状态、RGB 和深度哈希；执行全部候选后再重复 `advance`，要求全部轨迹标签和终点哈希完全一致。29 个状态均通过。

归档保存完整 integration 向量、控制器张量与确定性前缀。实际候选恢复经过验证的是进程内 `mj_copyData` 路径；没有声称仅靠 NPZ 就已经验证了跨进程恢复。采集脚本支持前缀重放后核对状态哈希再续跑。

## 原始输出与运行事件

首个 Spatial-Interactor 回答在生成后触发解析器 TypeError，当时原始生成尚未写盘；保持同样输入和解码重新运行。另一次启动在生成前因 tuple/list 的配置序列化比较失败。日志及初始源码保留，没有按标签选择重试答案，最终保存 58 份回答；已知共尝试 59 次生成。

修复仅包括：跳过没有 choice 的内层 JSON 对象、配置比较规范化，以及在解析前先持久化生成。prompt、权重、图像、候选和生成参数保持一致。初始模型协议保留其原始源码哈希，后续 `runtime_revisions/` 和各结果记录实际源码哈希。

模型有时在 EOS 前漏写末尾花括号。离线统一解析器只允许补齐最多两个末尾 `}`，并将 `strict_json`、`append_*_final_brace` 和 `invalid` 分别报告；不修改候选选择、概率、时间或添加缺失字段。原始 `result.json` 中的严格解析结果不改写。不能把这类语法修复算作模型严格遵守输出格式。

最终按字段独立计分：S10_0 明确给出的选择和概率计分，但三个不合规的 0 秒时间保持原值、标记无效，不用于时间误差。语义可解析、严格 JSON、完整 schema 合规和有效时间覆盖分别记录。详细解释及后续计划见 `ANALYSIS_AND_NEXT_STEPS.md`。

## 复现

在仓库根目录，用安装了 MuJoCo、CUDA PyTorch 和 Transformers 的环境运行。模型从 `.local/models/qwen3vl8b` 和 `.local/models/spatial_interactor8b` 本地读取，不把权重提交到仓库。

```bash
MUJOCO_GL=egl python tools/task3_occlusion_branches.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/task3_occlusion_forecasts.py --model qwen3vl8b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/task3_occlusion_forecasts.py --model spatial_interactor8b
MPLCONFIGDIR=/tmp/task3_branch_mpl python tools/task3_occlusion_analysis.py
```

当前本机解释器是 `/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python`。上述脚本遇到已完成状态会保留已有结果；若要完全重新采集，应复制代码并使用新的输出目录，保留当前原始数据。后续训练/调参把本批视为开发集，正式测试按完整物理布局隔离。
