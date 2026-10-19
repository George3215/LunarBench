# TASK3 原生格式小样本诊断（2026-10-19）

## 目的和预注册范围

先验证模型对当前图像和跨帧变化的理解，再决定是否值得继续接入导航。此次为 **自定义 MoonSim 原生格式诊断**，不是官方基准复现，不计入过去 220 个导航回合，不产生新的导航成功率。所有问答均无动作词表约束、无 JSON 动作约束，不反馈正确答案，不重试以挑选好结果。

固定使用先前 Qwen action-only 轨迹的 barrier_a0、pit_a0、window_a0 三场景。每场景固定取初始帧、首个 RGB 红色像素为零的帧，以及红色像素至少 64、横坐标最接近 .2/.8/.8 的可见帧。选帧不读取回合成败。偏离帧可能在后续重新看到目标时取得；它只用于图像定位诊断，不作为首次丢失时可获得的在线记忆。

9 张 640×480 原始相机图不加标注送入模型。评分标签由 RGB 红色阈值计算，独立存放在 labels_evaluator_only.json。模型输入不含目标世界坐标、仿真物体 ID、分割真值、路线或未来帧；指定的两帧诊断会明确提供两帧图像，只回答二者之间的变化。

## 模型和输入输出

|模型|协议|调用数|
|---|---|---:|
|Qwen3-VL-8B|checkpoint chat template；RGB 问答；单帧 15 题、双帧 6 题|21|
|Spatial-Interactor-Qwen3-VL-8B|同上，无额外训练状态轨迹|21|
|SpatialReasoner|作者 system prompt、单 RGB、Question/Options；512 输出 token，保留 think/answer|15|
|RoboBrain2.5-8B-NV|单 user、图像在前；general 21 题；作者 pointing、grounding 模板各 3 题；768 输出 token|27|
|SpatialRGPT-VILA1.5-8B|作者 llama_3 对话；单图问答 15 题；原生 region mask 3 题；原生 region+depth 分支 3 题|21|

全模型 greedy、seed=7、BF16、单 GPU 顺序运行。使用 checkpoint 原始图像处理配置，不沿用前次固定 256² 像素设置；记录实际 image_grid、展开 token、处理器配置和输入哈希。Qwen/Interactor 512 输出 token，SpatialRGPT 128。输出上限触发单独记录，不伪装成完整回答。SpatialReasoner 采用确定性解码，是公开推理代码的近似确定性变体，非逐位复现官方评测。

SpatialRGPT 的 RGB region mask 来自可见红球阈值和预设近处地面矩形，使用作者 mask processor；不是 simulator segmentation。深度是**受控 RGB-D 变体**：将既有相机编码深度近似解码为米、取倒数并逐图归一化为近白远黑，再送入原生 depths 参数。作者官方使用 DepthAnything 估计深度；因此此 3 题只能验证传感器深度分支的连通与行为，不能写成完整官方深度流程复现，也不能与 RGB-only 模型合并排行。RGB mask 问题不添加 depth token；深度问题按作者代码添加 <depth> token。

## 评分和边界

* 可见性 9 题；水平三分区 6 题；双帧可见性变化、图像水平位移各 3 题。选项位置预先轮换。只提取最终答案，不从推理中捞正确字母。
* RoboBrain 点坐标按作者约定 0–1000 还原，分别报告格式可解析、是否落入可见红球紧框、中心像素误差；grounding 报 IoU 和 IoU≥.5。
* SpatialRGPT 区域左右问题使用自然回答 yes/no；深度题提取答案所指区域，记录解析失败与原始回答。
* 不依据单 RGB 要求精确米制距离。目标消失并不能仅凭当前图唯一判断遮挡/出视野，更不能标注障碍背面的不可知几何。
* 此轮跨帧题只测图像变化，**不是里程计更新、物体恒存、遮挡后方位推断或重现距离预测**。这些能力需要下一阶段可辨识、带运动输入的诊断和闭环检验。
* 只含三种理想化仿真场景、红色球、一次确定性推理，小样本不能证明泛化或显著收益。

## 决策规则

完整跑完固定诊断并检查所有原始回答。先区分接口/截断、可见目标定位、跨帧观察变化的失败。当前 15 道共同单帧题、6 道双帧题只支持接口和视觉理解判断，不能据此宣布获得导航规划能力。原生 pointing 若通过，可进入“像素目标→相机深度/方位→短时动作”的独立控制器试验；目标不可见时必须保留明确的不确定状态，禁止虚构目标点或使用仿真真值补位。

## 可复现材料

生成与推理：tools/task3_native_diagnostics.py；单 GPU 队列：tools/run_task3_native_diagnostics.py。
inputs.json 冻结题目；frames.json 保存来源与 RGB 标签来源；runs/ 保存逐题 serialized_request、input_audit、原始结果和 token ID；scores.json/metrics.json 保存评分；日志和 QUEUE_STATUS.json 保存退出状态。新增 assets/ 中本次 9 张图和少量 mask、depth 将显式加入 Git 防误删，不提交模型权重或旧实验大图像目录。

官方接口依据：

* https://github.com/FlagOpen/RoboBrain2.5/blob/main/inference.py
* https://github.com/johnson111788/SpatialReasoner/blob/main/src/spatial_reasoner/prompt.py
* https://github.com/johnson111788/SpatialReasoner/blob/main/src/eval/infer.py
* https://github.com/ZJU-OmniAI/Spatial-Interactor
* https://github.com/AnjieCheng/SpatialRGPT/blob/main/llava/eval/eval_spatial.py
