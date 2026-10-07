"""Build a human-readable report from verified canonical result summaries."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'paper/vlm_goal_navigation';RUN=DOC/'runs/20261004_qwen35'
one=json.loads((RUN/'original_six_action/summary.json').read_text())
mode_order=['current_direct','current_reasoned','actions_reasoned','images_reasoned','images_actions_reasoned']
labels=dict(zip(mode_order,['当前画面，直接输出','当前画面＋推理','当前画面＋动作历史＋推理','当前画面＋图像历史＋推理','当前画面＋两类历史＋推理']))
lines=['# 本地 VLM 基线：原 benchmark 与连续目标恢复实验','',
'2026-10-04。目的：先直接测试现有 VLM，无新训练、无自研记忆模块、无世界模型，观察现有 benchmark 能否区分能力，再决定算法研究切入点。相机观测是实现约束；本实验不试图证明 VLM 是解决此任务的唯一选择。','',
'## 实际接入与算力','',
'本机 RTX 2080 Ti，11,264 MiB 显存；约 63.7 GiB 系统内存；Xeon W-2223，8 逻辑线程。使用本地 Ollama 0.32.15 和 Q4_K_M 量化的 Qwen3.5-4B、Qwen3.5-9B，context=8192。9B 已完成真实图片推理，首次图像推理后整卡显存采样为 7839 MiB，约 7.66 GiB（瞬时值，不是峰值）；Ollama 报告模型驻留显存约 5.73 GB，其他占用包括运行时、显示和相机渲染。模型摘要、精确 digest 和运行证据见运行目录 hardware_manifest.json 与 runtime_qwen3.5_9b.json。','',
'这是两种当前代际、适合本机的紧凑 VLM 基线，不是全球最强 VLN/VLA 声明。所有模型推理本地运行，没有使用付费云接口或真实机器人。模型接收的是模拟机器人上实际渲染的 RGB，未向模型传递目标坐标、几何可见性、红色像素检测值、场景名称或未来动作结果。底层现有 Go2 控制器执行固定速度原语，模型不直接生成关节力矩。','',
'## A. 原有六动作单步任务','',
'23 个有效世界、15 个构造组。重新执行了 24 个场景的接近过程，所有有效性标签和 23 个有效场景的丢失时刻与旧严格相机数据一致。每次 VLM 从相机及该条件允许的历史中选一个动作，再用旧实验已经物理执行的六个 3 秒分支评分。分支表和所选分支固定哈希，评分结果不进入模型请求。这是离线动作选择评估，与下节新执行的在线闭环实验分开报告。','',
'| 模型/条件 | 重新看到目标 | 7 个基础地形世界 | 16 个配对窗口世界 | 无效输出 |','|---|---:|---:|---:|---:|']
for model in ['qwen3.5:4b','qwen3.5:9b']:
 for mode in mode_order:
  row=next((s for s in one['summary'] if s['model']==model and s['mode']==mode),None)
  if row:
   lines.append(f"| {model}，{labels[mode]} | {row['success']}/{row['n']} ({row['success']/row['n']:.1%}) | {row['base']['success']}/{row['base']['n']} | {row['window']['success']}/{row['window']['n']} | {row['invalid']} |")
lines+=['| 固定前进（旧物理分支） | 17/23 (73.9%) | 1/7 | 16/16 | — |','',
'均匀随机选择六动作的期望成功率为 36.23%；这是从六分支的真实结果精确计算的期望，不是新采样的在线随机策略实验。每个有效世界至少存在一个成功动作。9B 只测试了表中两个条件；其余三个条件不能填成零或推测成绩。','',
'4B 直接决策在全部 23 个世界选择 forward，因此其 73.9% 成功率完全等同固定前进。16 个配对窗口世界都能靠该动作成功，揭示了当前单步任务的动作分布偏置。4B 两类历史条件仅 5/23，但在基础地形组由 1/7 变为 2/7：总体更差不能直接解读为每类场景中历史都无效。需要查看分组结果与具体轨迹。','',
'![原任务结果](figures/original_six_action.png)','',
'## B. 新执行的连续相机—动作闭环','',
'每回合从相同丢失状态恢复，VLM 看图选择动作，MuJoCo 真正执行，再返回新图。最多 10 次、每次 3 秒，STOP 终止。4B 的四场景为 stairs_up、trench、pit、hist001_left；9B 为 stairs_up、hist001_left。两模型分母不同，不把原始比例当作公平规模比较。','',
'| 模型/条件 | 回合 | 曾重新看到 | 曾进入平面 1m 邻域 | 接近并正确 STOP |','|---|---:|---:|---:|---:|']
for folder in ['closed_loop_v2','closed_loop_9b']:
 f=RUN/folder/'detailed_analysis.json'
 if not f.exists():continue
 data=json.loads(f.read_text())
 for mode in mode_order:
  for key,s in data['summary'].items():
   model,m=key.split('/')
   if m==mode:lines.append(f"| {model}，{labels[mode]} | {s['n']} | {s['reacquired']}/{s['n']} | {s['within_1m_exploratory']}/{s['n']} | {s['stop_near']}/{s['n']} |")
lines+=['',
'“重新看到”要求 10 Hz 相机中连续 3 帧至少 3 个目标像素，且机器人未跌倒/越界。“平面 1m”是轨迹的探索性距离指标。“正确 STOP”额外要求停止时目标可见、平面距离不超过 1m、目标深度不超过 1.25m、机器人直立。这两个到达指标尚未验证同一可通行表面，不能作为最终 terrain-aware navigation 成功定义。仅“正确 STOP”为零，不代表机器人从未接近过目标。','',
'4B 当前画面＋推理在四个试验场景均曾重新看到目标。stairs_up 在第二个 forward 动作过程中、丢失后 4.502 秒恢复视野，随后仍前进并越界。因此不能继续以“当前观察策略无法重新找回目标”作为通用论断。但本次也未证明它完成了用户要求的到达任务。','',
'![相机轨迹](figures/camera_reacquisition_then_failure.png)','',
'这些是暂停仿真等待推理的同步闭环。模型延迟已保存，不能据此声称能在真实机器人上实时运行。旧分支只支持单步评分，连续轨迹没有用旧分支拼接。','',
'## 输入与执行审计','',
'- 4B 在单张真实 benchmark 图片的目标存在性诊断中正确 46/46（23 seen＋23 lost）。9B 为 45/46，漏检 barrier 的 seen 帧，所有 lost 帧均判断为不可见。这支持所选帧上的基本红球辨识，但不能排除小目标感知误差，也不代表全过程感知无误。',
'- 原生多图输入未通过简单的图像身份/顺序检查；明确标注 HISTORY/CURRENT 的单张拼图在 4B、9B 的简单身份检查中均通过 3/3 对照。正式图像历史实验使用该拼图。历史表现不佳仍可能与时间理解、图像布局及缩放有关，不能直接归因于“记忆没有价值”。',
'- 推理条件统一为 512 token 有界思考，再用相同图像和思考内容进行 180 token 非思考 JSON 输出；直接条件仅一次 180 token 输出。思考阶段可能截断，因此没有测试无限制或充分长的推理。最初格式诊断结果保留而不进入正式表。',
'- 温度为 0、名义 seed=7；没有独立模型种子重复，不能把每张图或每个分支当作独立机器人任务样本。表中为描述性统计。',
'- 调用期间某些相同模型任务共用串行服务，墙钟延迟含排队；原始服务端时长另存。不能直接用不同批次平均响应时间做性能优劣结论。',
'- 输入图片 SHA256、所选物理分支 SHA256、原始输出和实际轨迹均保留。前缀复现与第一步轨迹比对见 artifact_verification.json、prefix_reproduction_check.json、first_step_physics_check.json。',
'- 一条 4B 在线记录在运行期间遇到源文件编辑，旧逻辑在回合结束读取磁盘哈希；已通过进程起始源码快照给出 PROVENANCE_CORRECTION.json 注释，不改变原始结果。新 runner 在开始时固定源码哈希。十条无图像历史记录逐字节复用旧批次，见 REUSED_EPISODES.json。','',
'## 对研究方向的实际影响','',
'现有数据支持保留 VLM 算法研究路线，但不支持先假定 Image＋Memory＋Action 必然最好，再寻找解释。无需证明 VLM 是唯一解；需要在同一基础模型、输入和预算下，证明所提算法改善到达和泛化。','',
'下一阶段应按优先级完成以下工作：','',
'1. **校准到达任务接口。** 保留本轮原 benchmark 不变，另开完整导航协议。先比较较短移动距离/时长（如 0.25m、0.5m、1m 或对应标定时长）及短角度转向，明确 STOP 是终止、静止观察是另一动作，并测实际位移和相机变化。当前 3 秒速度原语与终止理解可能导致越过目标或提前退出；先消除这些接口因素，再判断算法缺口。',
'2. **在相同基础模型上检查廉价改善。** 比较更充分的推理预算、同布局的当前画面对照、历史时间戳/动作对齐、完整视频或经过验证的多图输入，以及简单文字记忆。预留未参与调参的场景。先确认显著失败并非输入排布或提示语问题，再加入学习的状态估计、动作条件预测或 world model。',
'3. **建立能检验泛化的导航 benchmark。** 在开发集均衡初始朝向和成功动作分布，增加同地形多目标位置、多种丢失状态、可行绕行与不可行路径；按地形生成种子/布局构造组划分测试集。加入自然目标、相似干扰物与语言目标描述，再扩展光照、噪声、移动模糊。对各方法使用一致动作和预算，报告到达成功、路径效率、失败类型、推理开销及未见目标/地形结果。不要根据最终测试中模型做错的位置反复改题。','',
'通过这些校准后，若 VLM 能恢复视野却仍不能稳定、安全地到达，就可以聚焦“历史与执行反馈如何帮助 VLM 估计可到达性并选择动作”。新方法必须相对简单历史输入、检索记忆和相关强基线提供增量，而不是把接上 VLM 本身作为创新。','',
'## 专用 VLN/VLA 与尚未验证的模型','',
'| 模型/路线 | 本轮状态 | 具体限制 |','|---|---|---|',
'| Qwen3.5-4B / 9B Q4_K_M | 已本地接入并进行原任务及闭环试验 | 量化通用 VLM，不是完整专用 VLN 系统 |',
'| NaVILA-8B | 参考全文已核查，未运行 | 8B FP16 语言权重本身约 16GB，高于 11GB 显存；量化/CPU offload 尚未验证。原系统视觉低层控制依赖 LiDAR/高度图，替换为现有 Go2 控制器需要明确适配，不能冒称原论文复现 |',
'| WoMAP | 参考方法已核查，未部署 | 需要相应表征、预测模型、训练与动作接口适配；没有证据表明只受显存限制 |',
'| Qwen3-VL-4B/8B、其他家族 VLM | 可作跨模型检查，未下载运行 | 本轮优先完成同代际 4B/9B 容量检查；不把未测模型填入成绩表 |',
'| 更大多模态模型 | 未运行 | CPU 卸载可能可行，但吞吐与图像上下文预算尚未测量；不是已证明不可运行 |','',
'来源：[Qwen3.5-4B 官方运行包](https://ollama.com/library/qwen3.5:4b)、[9B 官方运行包](https://ollama.com/library/qwen3.5:9b)、[NaVILA 全文](https://arxiv.org/html/2412.04453v2)、[WoMAP 正式论文](https://proceedings.mlr.press/v305/yin25b.html)。本次不是穷尽 2026 VLN/VLA 排行榜检索，不能声称已验证“当前全球最 SOTA”。','',
'## 复现与产物','',
'模型服务启动见 tools/Start-VlmBenchmarkServer.ps1；命令见 REPRODUCE.md。正式单步数据：runs/20261004_qwen35/original_six_action/；正式在线：closed_loop_v2/、closed_loop_9b/。single_step_v1/v2 和原生多图诊断不进入上述主表。','',
'本报告可支持论文的基线与问题分析部分。尚无新算法实现、训练或优于基线的证据，不能将这些实验写成新算法效果。']
(DOC/'VLM_BASELINE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(DOC/'VLM_BASELINE_RESULTS.md')
