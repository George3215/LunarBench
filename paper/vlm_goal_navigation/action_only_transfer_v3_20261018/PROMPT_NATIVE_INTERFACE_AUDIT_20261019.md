# 实际Prompt与模型原生任务接口核对（2026-10-19）

本次是源码、已有逐步输出和官方说明的核对，没有新增GPU推理或导航回合。以下prompt从已完成实验的protocol.json直接导出，保留原文。

## 结论修订

已有空间能力权重可以作为迁移候选，但多数并非训练成直接输出移动动作的闭环导航策略。前两轮应称为“统一自定义导航接口下的零样本迁移测试”，不能称为各模型官方导航能力复现。

保留原生tokenizer/chat template不等于保留原生任务接口；统一prompt测量即插即用迁移，尚不足以比较各模型在其正确原生接口下的能力。前面在原生任务诊断之前先做全量测试，实验顺序需要修正。

## 官方任务与本轮差异

| 模型 | 官方可核验用途/输出 | 本轮主要差异 |
| --- | --- | --- |
| Spatial-Interactor | 图像/视频+问题，学习状态转换、自身运动和长时序空间问答 | 改成直接选择一秒移动动作，没有确认其有此动作策略训练 |
| SpatialReasoner | 三维空间问答，原生系统提示使用think/answer标签；原生评估允许512输出token | 替换官方任务system，直接输出动作名、最多16token，不保留原生显式推理/答案格式 |
| RoboBrain2.5-8B-NV | 通用问答、pointing、grounding、操作轨迹等；官方导航示例调用pointing，输出图像点坐标 | 未用pointing选点接口，而要求混合历史RGB/灰度深度和里程计直接输出移动动作 |
| SpatialRGPT | 区域定位的空间问答；空间评测传入images/depths/masks及区域/深度token | 本轮仅通用多图输入，灰度深度当普通图像，未传原生depths/masks；不能代表完整空间分支 |
| NaVILA | 学习过高层语言导航动作，再由低层运动策略执行 | 本轮未重测；后续需按官方视觉历史与动作句式适配 |

官方核对来源：

- [Spatial-Interactor](https://github.com/ZJU-OmniAI/Spatial-Interactor)：视觉输入+问题，以及QA训练和评估范围。
- [SpatialReasoner官方推理入口](https://github.com/johnson111788/SpatialReasoner/blob/main/src/eval/infer.py)；[官方system prompt](https://github.com/johnson111788/SpatialReasoner/blob/main/src/spatial_reasoner/prompt.py)。
- [RoboBrain模型卡的导航示例](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV#5-usage-for-navigation-tasks-embodied)；[官方inference.py](https://github.com/FlagOpen/RoboBrain2.5/blob/main/inference.py)。官方pointing分支要求单图并补充二维点坐标输出模板；general分支支持多图。
- [SpatialRGPT官方空间评测](https://github.com/AnjieCheng/SpatialRGPT/blob/main/llava/eval/eval_spatial.py)。
- [NaVILA官方代码](https://github.com/AnjieCheng/NaVILA)。

## 只转向的已证实事实

- 新接口RoboBrain和SpatialRGPT各1408个自主动作全部为转向，没有前进、后退或侧移。
- RoboBrain这1408次生成的masked_argmax_tokens均为0：在当前greedy设置下，没有首选token被我们的动作约束改写。因此不能解释为解码器把其本来首选的FORWARD强行改成TURN。
- SpatialRGPT这1408次生成的masked_argmax_tokens均为1。现有日志只记录每次生成的总次数，不能据此确定屏蔽发生在动作选择还是结束符位置，也不能直接归因转向。
- 原复杂JSON协议下，RoboBrain有275次FORWARD并到达5/22；SpatialRGPT有127次FORWARD并到达3/22。这说明没有证据支持它们的动作接线根本不能前进；结果对任务/输出协议明显敏感。
- TURN动作的实际反馈有少量机器人漂移，但没有向前平移指令；“原地”应理解为转向为主，不能声称世界坐标绝对不变。

## 合理假设与未验证项

任务分布不匹配、去掉原生推理/答案格式、把专用深度当普通图像、纯贪心下的重复动作、首帧计划误判、图像分辨率与小目标信息不足，均可能影响表现。不能仅从现有220回合确定各原因的因果贡献。

当前最多四个事件RGB-D关键帧（最多八张普通图片），不是连续原生视频；里程计和运动效果为文本，没有经过运动到空间状态转换的任务专门训练。仅把数值写进prompt不保证模型掌握其决策语义。

下一步应先做每模型少量原生空间/定位任务诊断，随后比较保留原生回答和转换成动作的方案。原生空间能力复现与统一导航动作评估分开报告。对于RoboBrain可试原生pointing候选点；其选点接口本身不能解决完全遮挡的目标，仍需保持历史目标状态并预测可行子目标，不能把不可见目标当成可直接grounding的对象。

## 新接口完整actor system prompt

```text
Control the quadruped to find the red sphere, approach it, and finish with the sphere clearly visible and within 1 metre of the camera. Use only the observations supplied in this request. No hidden earlier frames, target coordinates, scene name, or previous assistant messages are available.
Actions last 1 second: FORWARD vx=0.6m/s, BACKWARD vx=-0.5m/s, STRAFE_LEFT vy=0.5m/s, STRAFE_RIGHT vy=-0.5m/s, TURN_LEFT wz=0.6rad/s, TURN_RIGHT wz=-0.6rad/s. STOP ends the trial after a short stand. Actual motion can differ on terrain or against obstacles. Look for a traversable route, avoid repeated collisions, recover sight when the target is occluded, and avoid overshooting. STOP only when the current target is visible and reliably within 1 metre.
Each supplied RGB image is tagged with an observation frame ID. In RGB-D requests it is followed by aligned ideal camera depth: fixed grayscale white=0m axial depth, black=10m or farther/invalid. A numeric red-region range is calculated from RGB red pixels and depth back-projection, without ground-truth object identity; null means no reliable red-region depth, not arrival. Historical depth belongs to its historical frame and is not the current range.

You also receive ideal simulated SELF odometry relative to the initial robot frame,
recent measured action effects, and up to four event-selected RGB-D observations.
These are robot measurements, not a map or hidden target coordinates. Body forward
is +x and left is +y; positive yaw turns left. Depth is ideal camera depth.
The initial target evidence remains available. Maintain an uncertain spatial belief
when the target disappears; choose a route to recover its view and approach it.
The initial plan is your own earlier visual plan, not an oracle route. Correct it
when fresh observations or actual motion contradict it. Do not assume commanded
displacement was achieved. No external planner executes or corrects your actions.
STOP is forbidden unless CURRENT RGB has at least 20 red pixels and CURRENT finite
red-region surface range is strictly between 0 and 1 metre. A historical sighting
never permits STOP. If STOP is unavailable, choose one of the six motion actions.

Return exactly ONE legal action name, with no other text: FORWARD, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT, TURN_RIGHT, or STOP. Use the images, measured motion and own initial plan to select the next action. No assessment JSON or explanation is requested.
```

## 两轮共享的首帧plan system prompt

```text
You are the visual route planner for a simulated quadruped.
From ONLY its initial forward RGB-D observation, propose a tentative route toward
the red sphere. Terrain behind visible surfaces is unknown. Estimate likely target
direction, obstacles, a broad motion strategy, and when to reconsider it. Mark
uncertainty explicitly. Do not invent hidden geometry. This plan is advisory and
will be revised using later images and measured self-motion. No map or target
world coordinates are given. Return one JSON object with keys goal_evidence,
tentative_route, occlusion_hypothesis, uncertainty, revision_trigger (short strings).
```

## 原复杂JSON接口完整actor system prompt

```text
Control the quadruped to find the red sphere, approach it, and finish with the sphere clearly visible and within 1 metre of the camera. Use only the observations supplied in this request. No hidden earlier frames, target coordinates, scene name, or previous assistant messages are available.
Actions last 1 second: FORWARD vx=0.6m/s, BACKWARD vx=-0.5m/s, STRAFE_LEFT vy=0.5m/s, STRAFE_RIGHT vy=-0.5m/s, TURN_LEFT wz=0.6rad/s, TURN_RIGHT wz=-0.6rad/s. STOP ends the trial after a short stand. Actual motion can differ on terrain or against obstacles. Look for a traversable route, avoid repeated collisions, recover sight when the target is occluded, and avoid overshooting. STOP only when the current target is visible and reliably within 1 metre.
Each supplied RGB image is tagged with an observation frame ID. In RGB-D requests it is followed by aligned ideal camera depth: fixed grayscale white=0m axial depth, black=10m or farther/invalid. A numeric red-region range is calculated from RGB red pixels and depth back-projection, without ground-truth object identity; null means no reliable red-region depth, not arrival. Historical depth belongs to its historical frame and is not the current range.
For each action, give a brief public explanation of the strategy and cite the supplied frame IDs and evidence channels that support it. Explain whether you use current observations or remembered supplied images/depth. Do not provide a long internal chain of thought. If there is no earlier frame in the request, do not claim to have observed one.
Return one JSON object, no markdown:
{"target_visible":true or false,"target_location":"left|centre|right|absent|uncertain","proximity":"near|far|uncertain","confidence":number from 0 to 1,"visual_assessment":"brief visible evidence","decision_rationale":"brief strategy and reason for this action","evidence_frames":[integer frame IDs],"evidence_sources":["current_rgb|past_rgb|current_depth|past_depth"],"action":"FORWARD|BACKWARD|STRAFE_LEFT|STRAFE_RIGHT|TURN_LEFT|TURN_RIGHT|STOP"}
You also receive ideal simulated SELF odometry relative to the initial robot frame,
recent measured action effects, and up to four event-selected RGB-D observations.
These are robot measurements, not a map or hidden target coordinates. Body forward
is +x and left is +y; positive yaw turns left. Depth is ideal camera depth.
The initial target evidence remains available. Maintain an uncertain spatial belief
when the target disappears; choose a route to recover its view and approach it.
The initial plan is your own earlier visual plan, not an oracle route. Correct it
when fresh observations or actual motion contradict it. Do not assume commanded
displacement was achieved. No external planner executes or corrects your actions.
STOP is forbidden unless CURRENT RGB has at least 20 red pixels and CURRENT finite
red-region surface range is strictly between 0 and 1 metre. A historical sighting
never permits STOP. If STOP is unavailable, choose one of the six motion actions.
Optional JSON fields: "spatial_belief" (brief uncertain bearing/distance/occlusion
assessment), "plan_update" (brief revised route), "expected_reappearance_m"
(a nonnegative number or null; an uncertain prediction, never a known fact).
Additional permitted evidence_sources: "self_odometry", "recent_executed_motion",
"own_initial_or_revised_plan", matching the measurements and own plan provided.
```

## 实际user文本样例（RoboBrain、barrier_a0首个自主决策）

```text
Target: red sphere. Available image context: all_history. Sensor condition: rgbd.
Camera measurement: {"frame_id": 0, "red_pixels": 96, "red_region_axial_depth_m": 6.676217079162598, "red_region_surface_range_m": 6.6836620038217305}
Camera measurement: {"frame_id": 1, "red_pixels": 92, "red_region_axial_depth_m": 6.6885762214660645, "red_region_surface_range_m": 6.714783330327675}
Camera measurement: {"frame_id": 2, "red_pixels": 12, "red_region_axial_depth_m": null, "red_region_surface_range_m": null}
Camera measurement: {"frame_id": 3, "red_pixels": 0, "red_region_axial_depth_m": null, "red_region_surface_range_m": null}

Measured self-motion and own plan: {"self_odometry_source": "ideal simulated base odometry, initial body frame; not SLAM", "observation_poses": [{"frame_id": 0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0, "roll_rad": -0.0012, "pitch_rad": -0.0239}, {"frame_id": 1, "x_m": -0.0221, "y_m": 0.3766, "z_m": -0.002, "yaw_rad": 0.0159, "roll_rad": -0.0008, "pitch_rad": -0.0261}, {"frame_id": 2, "x_m": -0.0429, "y_m": 0.7838, "z_m": 0.002, "yaw_rad": 0.0114, "roll_rad": -0.0181, "pitch_rad": -0.0382}, {"frame_id": 3, "x_m": -0.0528, "y_m": 0.9439, "z_m": -0.0065, "yaw_rad": 0.0139, "roll_rad": -0.0136, "pitch_rad": -0.0085}], "recent_executed_motion": [], "own_initial_or_revised_plan": {"goal_evidence": "A small red region is detected at the right edge of the dark cube, with axial depth ~6.68m and surface range ~6.68m.", "tentative_route": "Move forward from the current position toward the dark cube, then turn right to approach the red region on the cube's surface.", "occlusion_hypothesis": "The red region is on the cube's surface, so it is not occluded by other objects in the current view.", "uncertainty": "The exact location of the red region on the cube is unknown; it could be on any face.", "revision_trigger": "If the red region is not visible after moving closer to the cube, or if the cube's position changes."}, "stop_available": false, "allowed_actions": ["FORWARD", "BACKWARD", "STRAFE_LEFT", "STRAFE_RIGHT", "TURN_LEFT", "TURN_RIGHT"]}
```

图像标签（RGB与对应灰度深度按标签依序进入模型）：

- Observation frame 0, simulation time 1.800s RGB
- Observation frame 0, simulation time 1.800s aligned depth (white 0m, black 10m+)
- Observation frame 1, simulation time 2.800s RGB
- Observation frame 1, simulation time 2.800s aligned depth (white 0m, black 10m+)
- Observation frame 2, simulation time 3.800s RGB
- Observation frame 2, simulation time 3.800s aligned depth (white 0m, black 10m+)
- Observation frame 3, simulation time 4.200s RGB
- Observation frame 3, simulation time 4.200s aligned depth (white 0m, black 10m+)
