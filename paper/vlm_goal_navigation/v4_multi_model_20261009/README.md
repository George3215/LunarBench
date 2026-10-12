# 第四版地形多模型测评

最新正式顺序（用户指示 2026-10-10）：先暂停 GPT-6.1 Sol，保留全部结果与断点，立即依次运行 NaVILA、TagaVLM-0.5B 相机适配、TagaVLM-7B 相机适配。Sol 延后进行，本轮先完成后续模型；后续模型不再等待 Sol 完成。环境和权重准备不等于测评完成。

共同主矩阵：22 个第四版 strict-admitted 地形 × RGB/RGBD × current/first_current/all_history = 每模型 132 次。共同物理起始状态、遮挡接近、1 秒动作原语、3000 步总预算（包括共同接近）、相同相机评分器。历史保留所有动作边界帧，不使用先前回复或动作文本。

用户已撤销之前每批两个请求的限制，不设请求总次数或每批请求数上限。当前本地调度批次为六个独立试验，每个模型的推理服务串行处理请求以适配单卡显存；这属于运行调度，不是用户请求配额。每个请求仍保留其实验条件要求的完整相机帧。

Sol 在 2026-10-10 凌晨因 HTTP 503 中断，已完成的 12/132 次结果保留。恢复通过真实动作重放，逐帧核对 RGB PNG、深度 PNG、浮点深度、相机测量和仿真时间，输入完全一致的已保存决策才复用。临时服务错误保持相机状态不动，间隔 60 秒重试。ERROR.json 的旧错误及 RUN_STATUS.json 原状态另存于 recovery；当前进度以 RUN_STATUS.json 和 RECOVERY_STATUS.json 为准。

Sol 原始相机使用 Mesa llvmpipe（LLVM 15.0.7），主机恢复显式选同一软件 EGL 设备，CUDA 模型推理仍在主机 GPU 执行。后续本地模型同样锁定此相机后端。现有 Qwen 对照的 barrier_a0 初始物理状态相同，但相机像素与 Sol 不同；现有 Qwen 对照仅作参考，不宣称逐像素公平配对。恢复试验的 wall_s_including_scheduler_wait 只覆盖恢复会话，另用 wall_time_scope 标记，不能与完整未中断试验作总耗时比较。

模型只接收所选 RGB 图像；RGBD 增加所选帧的理想对齐深度与 RGB 红色区域测距。禁止目标真值坐标、分割身份、仿真图结构、评分器信息和参考路线进入决策。推理服务不导入仿真器，没有工具调用。

## 测评适配

- NaVILA：主矩阵支持变长图像上下文，不采样历史。原模型训练输入为八帧 RGB，RGBD 是灰度深度图片与相机数值测量的适配。原生动作语言只覆盖前进、左转、右转、停止，输出距离/角度映射到共同 1 秒原语。另运行 native8 输入协议 22 次，采样/黑帧补齐至八帧，只含 RGB。
- TagaVLM-0.5B/7B：使用官方权重、官方 patched transformers 与 SigLIP 视觉编码器，但原生 Taga 全景/拓扑候选节点协议无法直接满足本轮单前相机条件。主矩阵仅测试 camera-adapted 骨干，graph_sprels=None，无候选节点真值或图距离。提示改为七动作 JSON。这些结果不能代表完整原生 Taga 方法。
- 原生未支持 RGBD 的模型收到深度图属于实验适配，不能写成模型原生 RGBD 能力。
- 模型上下文长度、视觉编码、生成预算不同。保持观测来源与动作条件；上下文超限明确结束，保留完整输入，不静默截断。

## 可复现性问题

- ABot-Navigation 默认分支是 ABotN-Bench（对应 N1）；已另下载官方 ABot-N0 分支到 third_party/ABot-N0。N0 分支当前只有说明与媒体，没有可运行推理代码或权重发布；保持 unavailable。
- Qwen-VLA：对应 2605.30280，已下载官方 QwenLM/Qwen-VLA 仓库；当前没有可调用推理代码、权重下载入口或 release assets。
- Qwen-RobotNav-4B：对应 2606.18112，已下载官方 QwenLM/Qwen-RobotNav 仓库。README 明确表示当前没有发布权重计划。
- 没有用通用 Qwen VLM 替代上述专用模型；未能运行的项目不记为 0% 成功率。
- NaVILA 原有环境缺少 transformers、配套扩展及其他依赖，已在仓库内补齐；按官方安装说明覆盖其私有环境的 transformers replacement。
- Taga 使用自己的 venv，复用 NaVILA 的 torch 安装，独立覆盖 patched transformers 4.40、tokenizers 与相关依赖。未修改当前 MuJoCo/Qwen 主环境。
- Taga 上游在无 nav_data 时返回空列表，并可能传给图注意力。局部补丁将无图输入的返回值改为 None；原生有图路径不变。补丁保存于 Taga-VLM_local_changes.patch。

## 运行记录

- models.json：具体模型、仓库 commit、权重路径、实验协议和不可测原因。
- REPOSITORIES.json：所有实际下载仓库的 URL 和 commit。
- WEIGHT_PREFLIGHT.json：safetensors 头与数据长度、分片索引存在性检查；不代表已经运行模型。
- STATUS.json：当前本地模型队列状态；本轮不设置 Sol 前置条件。
- 每个模型目录保存 server.log、matrix.log、protocol.json、逐步 network_input.json/response.json、相机观测、独立 evaluator、result.json 与 summary.json。
- 推理错误、模型载入失败、上下文超限及格式错误分别记录；队列继续尝试其他可用模型。

队列由 queue_v4_models.py 管理。STOP 文件可在前置等待、模型间或动作边界结束队列；已有结果保留。最后 phase=queue_finished 只表示队列处理结束，各模型是否成功完成以 models 中的 state 和完整 episode 数为准。

NaVILA 单卡推理采用 NAVILA_SINGLE_GPU_INFERENCE=1，将未使用的序列并行通信模块切为 torch.distributed，避免推理主机上 DeepSpeed 导入时强制探测 CUDA 编译工具链；不初始化并行组，不改模型权重。补丁见 NaVILA_local_changes.patch。

2026-10-10 10:34 恢复中断：六个未完成试验的相机一致性检查全部通过后，prepare() 创建已经存在的待执行步骤目录触发 FileExistsError，未发送新请求、未执行新动作。10:54 已修复：仅对经过恢复验证且无 transition.json 的未执行步骤允许复用目录；拒绝复用已执行步骤。六个真实断点的准备输入及五个缓存回复通过离线预检，记录在 Sol recovery/PENDING_RESUME_PREFLIGHT.json；重新启动完整动作重放恢复，正式新增结果仍以 result.json 为准。

2026-10-10 用户暂停 Sol：在重放旧动作阶段设置 STOP 并通过 SIGINT 退出，确认 Sol 进程停止。正式完成数仍为 12/132；RUN_STATUS.json 和 RECOVERY_STATUS.json 已标记 paused，保留旧状态历史。第三方队列移除 Sol 前置条件，立即启动本地模型。

NaVILA 首次真实推理发现上游输入裁剪：实际检查读取的是 model.llm.config.tokenizer_model_max_length，而非外层 config。首次 matrix 产物完整保存在 matrix_attempt_*_truncation_excluded，明确排除正式统计。服务修复为同时关闭两层裁剪，使用原配置 4096 作为有效上下文预算（架构位置上限 8192），在完整视觉嵌入计数后明确返回 context_limit，禁止静默丢弃历史帧。模型权重不变。

本地正式输入审计与报告由 analyse_camera_models_v4.py 持久观察更新，FORMAL_PROGRESS.json 与 FORMAL_REPORT.md 只统计 models.json 指定的 matrix/native8 正式目录，排除归档 attempt。检查相机输入白名单、RGB/RGBD 图像数、三种帧选择、实际模型标签与图片占位符、模型身份及显式 context_limit；逐协议保存 INPUT_AUDIT.json。
