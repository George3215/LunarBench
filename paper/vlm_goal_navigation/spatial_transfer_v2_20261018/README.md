# V4 空间检查点配对评测：合法动作解码版

本轮比较原始 Qwen3-VL-8B-Instruct 与 Spatial-Interactor-Qwen3-VL-8B，在**相同**视觉输入、空间观测、prompt、动作接口和预算下，是否提高地形遮挡后的重观测和接近能力。没有训练新模型，也不预设提升。

## 为什么另建 v2

[上一批接口诊断](../spatial_transfer_20261018/README.md)完成了 11 个 Qwen 回合，0 个 `<1m` 到达，其中 6 个因反复请求非法 STOP 提前终止，5 个耗尽预算。旧门控拒绝执行 STOP，但仍让拒绝导致提前终止，掩盖了真正的遮挡决策能力。原始数据、未完成回合和 STOP 标记全部保留，**不把这批作为完整模型比较或合并分母**。

v2 在生成 JSON 的 `action` 字段时，直接屏蔽当前不合法的动作 token。STOP 没开放时，模型只能生成六个运动动作之一；合法候选的相对 logits 不变，方向和动作仍由 VLM 自己选择。它不使用绕障启发式或外部规划器，不代模型选择动作。理由、空间判断和计划文字保持自由生成。此改动是公平、可用接口所需的工程控制，**不是论文创新点**。

Spatial 首次烟测还出现 `<think>`、`Answer:` 与显式标签字段格式。两模型统一使用相同适配：仅移除完整 think 包装并读取完整 JSON 或明确命名字段，不从自然语言猜动作、不补造缺失观测。JSON `action` 和显式 `Action:` 两种字段都执行相同动作合法性解码。初始计划也可读取同名的五个明确标签。证据来源也允许已实际提供的自身里程计、最近执行反馈和自身计划；缺失必填字段仍拒绝执行。严格 JSON 的旧烟测保留为接口记录，不当作导航失败结论。最终共同接口下仍可能出现缺失必填字段，正式评测按冻结规则计为接口终止，并与动作后的导航失败分别报告；不为特定模型补造字段或动作。烟测至少须证明模型可执行动作且无资源／STOP 约束故障。

## 冻结协议

- 22 个原 V4 准入场景 × 2 个完整 BF16 检查点 = 44 回合；每回合最多 64 次自主动作，batch=2，greedy，seed=7。
- 每个场景先真实执行相同前缀，确认最初看见目标、随后由地形遮挡；自主动作从同一丢失状态开始，不把未经历遮挡的直走到达混入恢复成功。
- 最多四帧 RGB-D 事件记忆：初始、最后可见、最近可见→丢失转折、当前，去重排序；图像统一缩放至面积上限 65536。
- 理想仿真自身里程计：初始机器人坐标系中的相对位置、朝向、俯仰/滚转，及最近六次实际动作效果。**没有目标真值坐标、地图、障碍物坐标或参考动作输入。**
- 当前检查点仅根据第一帧生成初始视觉计划；后续 VLM 可以在动作 JSON 中更新自己的计划。计划不是 oracle，固定遮挡前缀不是它的导航成绩。
- 两模型统一使用原始 Qwen processor、tokenizer、chat template 和 inference generation config；cache 开启、同一 EOS/pad、同一输出预算（actor512、plan640）。词表 151,669 个 ID 完全一致。v2 两模型源代码快照亦应一致。
- STOP 只在**当前**红色像素 ≥20、有效红区表面距离严格在 `(0,1)` m 内开放；历史可见和历史距离不开放 STOP。动作解码限制在生成期间执行，生成后仍有防御检查。
- 独立 evaluator 以真实目标身份、可见像素≥20、表面距离<1m、直立且在界内连续三次10Hz采样确认到达并自动结束。到达和 VLM 主动 STOP 分开报告，自动结束可能先于下一次 STOP 决策。
- 无自动运动回退、无回原点、无重训练、无 RSI、无世界模型训练。

## 场景与恢复边界

工作区从 LunarBench `a63a83e8a976663565408467955a0fb3f924c77d` 恢复。地形、配置和准入记录仍在；误删前的大体量原始实验及最近记忆脚本未恢复，因此不与其成功率直接合并。

新鲜实测的 22 个丢失 qpos 与原清单**逐项完全一致**，初见—遮挡门槛22/22通过，见 [重放核对](loss_state_replay_check.json)。历史参考在原准入的控制器重置语义下22/22恢复可见，见[参考核验](../spatial_transfer_20261018/reference_validation_r2/validation.json)。VLM 主实验保持低层控制器连续。连续控制参考有20/22获得证据，`pit_a1`、`waves_a2` 尚未验证，不能据有限搜索失败称其不可解；全部22仍保留。

原准入认证**可重观测**，没有认证所有场景均有完整 `<1m` 导航参考路线。重观测与到达分别报告，不能把全部未到达归于模型能力。同族 a0/a1/a2 有相关性，区间采用地形族 cluster bootstrap，单固定种子结果仍需后续泛化验证。

## 数据与 GitHub

- `models/`、`weight_integrity_audit.json`：固定版本、文件哈希；权重在 Git 忽略的 `.local/models/`，8 个分片均与上游 SHA256 一致。
- `smoke/`：接口烟测，不计入正式分母。
- `runs/{qwen3vl8b,spatial_interactor8b}/`：协议、初见/丢失门槛、每回合结果、运行状态及本机原始证据。
- `case_records/`：可推送的精简计划、逐步动作/理由、输入哈希与评价摘要。
- `episodes.csv`、`metrics.json`、`REPORT.md`、`INPUT_AUDIT.json`：配对分析和逐场景核对。
- `trajectories.svg`：两模型轨迹；`*` 标记到达，`x` 标记失败，红色 `+` 是仅供评价绘图的目标真值。
- `QUEUE_STATUS.json`：单 GPU 队列与 GitHub 推送记录。阶段记录每约四个完成回合备份至 `experiments/task3-spatial-transfer-20261018` 分支。

原始 RGB-D、逐 token 输入和完整物理轨迹保留本机，受 `.gitignore` 排除；**GitHub 摘要备份不等于原始数据异地备份**。结果以状态文件为准，不把未完成回合写成完成。

## 复现

在仓库根目录，使用有 CUDA、Transformers Qwen3-VL、MuJoCo 和 Matplotlib 的 Python：

```bash
python tools/fetch_task3_spatial_models.py --root .local/models --record paper/vlm_goal_navigation/spatial_transfer_v2_20261018/models
MUJOCO_GL=egl python tasks/task3_reobservability/spatial_transfer_v4.py --model .local/models/qwen3vl8b --processor .local/models/qwen3vl8b --out paper/vlm_goal_navigation/spatial_transfer_v2_20261018/runs/qwen3vl8b
MUJOCO_GL=egl python tasks/task3_reobservability/spatial_transfer_v4.py --model .local/models/spatial_interactor8b --processor .local/models/qwen3vl8b --out paper/vlm_goal_navigation/spatial_transfer_v2_20261018/runs/spatial_interactor8b
python paper/vlm_goal_navigation/analyse_spatial_transfer_v4.py paper/vlm_goal_navigation/spatial_transfer_v2_20261018 --plots
```

## 模型依据

[Spatial 官方方法](https://github.com/ZJU-OmniAI/Spatial-Interactor)、[Spatial 完整权重](https://huggingface.co/kagakouko/Spatial-Interactor-Qwen3-VL-8B)、[Qwen 基座权重](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。其他候选与限制见[名单](../spatial_transfer_20261018/CANDIDATES.md)，尚未测试者不填成功率。
