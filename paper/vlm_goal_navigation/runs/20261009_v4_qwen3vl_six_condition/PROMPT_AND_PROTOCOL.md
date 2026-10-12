# Qwen3-VL-8B 试验提示词与协议（terrain_design_v4）

本文件固定并记录本次实验送给模型的全部文本与输入布局。**提示词不包含任何地形信息**：
没有地形名称、族别、几何参数、材质、坡度、坑深、方块数量、场景 ID 或任何从准入真值派生的量。
模型只能看到相机 RGB（及可选深度图）和一份与地形无关的任务说明。

## 固定 system prompt（逐字，`SYSTEM_PROMPT.txt` 为同一份）

```
Control the quadruped to find the red sphere, approach it, and finish with the sphere clearly visible and within 1 metre of the camera. Use only the observations supplied in this request. No hidden earlier frames, target coordinates, scene name, or previous assistant messages are available.
Actions last 1 second: FORWARD vx=0.6m/s, BACKWARD vx=-0.5m/s, STRAFE_LEFT vy=0.5m/s, STRAFE_RIGHT vy=-0.5m/s, TURN_LEFT wz=0.6rad/s, TURN_RIGHT wz=-0.6rad/s. STOP ends the trial after a short stand. Actual motion can differ on terrain or against obstacles. Look for a traversable route, avoid repeated collisions, recover sight when the target is occluded, and avoid overshooting. STOP only when the current target is visible and reliably within 1 metre.
Each supplied RGB image is tagged with an observation frame ID. In RGB-D requests it is followed by aligned ideal camera depth: fixed grayscale white=0m axial depth, black=10m or farther/invalid. A numeric red-region range is calculated from RGB red pixels and depth back-projection, without ground-truth object identity; null means no reliable red-region depth, not arrival. Historical depth belongs to its historical frame and is not the current range.
For each action, give a brief public explanation of the strategy and cite the supplied frame IDs and evidence channels that support it. Explain whether you use current observations or remembered supplied images/depth. Do not provide a long internal chain of thought. If there is no earlier frame in the request, do not claim to have observed one.
Return one JSON object, no markdown:
{"target_visible":true or false,"target_location":"left|centre|right|absent|uncertain","proximity":"near|far|uncertain","confidence":number from 0 to 1,"visual_assessment":"brief visible evidence","decision_rationale":"brief strategy and reason for this action","evidence_frames":[integer frame IDs],"evidence_sources":["current_rgb|past_rgb|current_depth|past_depth"],"action":"FORWARD|BACKWARD|STRAFE_LEFT|STRAFE_RIGHT|TURN_LEFT|TURN_RIGHT|STOP"}
```

任务三要素（红球、接近、1 m 内停止）与动作接口都在提示词里说清楚；唯一未说明的是地形本身，
这是刻意保留的：模型必须从图像推断可通行性与遮挡，不能依赖我们告诉它“这是坑/窗/波浪”。

## 每次请求的 user 文本（模板）

```
Target: red sphere. Available image context: <horizon>. Sensor condition: <mode>.
```

并按条件附加图像与逐帧读数：

- 每个图像前有一行标签：`Observation frame <id>, simulation time <t>s RGB`；
- `rgbd` 条件在该帧 RGB 后紧跟 `Observation frame <id> ... aligned depth (white 0m, black 10m+)`，
  以及一行相机测量 `{"frame_id":..,"red_pixels":..,"red_region_axial_depth_m":..,"red_region_surface_range_m":..}`
  （由 RGB 红色区域 + 深度反投影得到，不是真值物体身份）。

## 条件矩阵（6 组）

| sensor | history | 选帧规则 |
|---|---|---|
| rgb / rgbd | current | 只给当前帧 |
| rgb / rgbd | first_current | 第 0 帧 + 当前帧 |
| rgb / rgbd | all_history | 从第 0 帧到当前帧的全部已记录动作边界观测，不抽样、不丢弃 |

`history` 只影响给模型的图像集合；不提供任何过去的助手输出或动作历史（`context.json` 中
`past_assistant_outputs_in_context=false`、`past_action_history_in_context=false`）。

## 预算与判定

- 每回合自主动作上限 **3000**（`--max-steps 3000`）；公共 approach 段是固定物理动作，不是模型决策。
- 成功定义沿用 task3：目标分割 ≥20 px、目标表面中值距离 <1 m 连续 3 帧、机器人直立且在界内。
- 资源类终止（`context_limit`/`gpu_memory_limit`/`disk_space_limit`）不算导航失败证据。

## 记录内容

每次请求都落盘：`messages.json`（含实际 system/user 内容与图像路径）、`rendered_prompt.txt`
（chat template 渲染结果）、`context.json` / `network_input.json`（帧 ID、图像数、token 数、是否含历史）、
`output.json` 与 `raw_output.txt`（模型原始输出与解析结果）。地形真值只存在于：`scene_setup.json`、
每条 `initialization.json`、`approach_gate.json`，以及准入记录里，绝不进入模型请求。
