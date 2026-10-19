# 原生格式诊断结果（2026-10-19）

这是 TASK3 自定义小样本诊断；不是官方基准复现，也不是新增导航成功率。

|模型|可见性（9）|水平位置（6）|两帧可见性变化（3）|两帧图像位移（3）|
|---|---:|---:|---:|---:|
|qwen3vl8b|9/9|5/6|3/3|3/3|
|spatial_interactor8b|9/9|6/6|3/3|3/3|
|spatialreasoner|2/9|1/6|不测（原生单图）|不测（原生单图）|
|robobrain25_8b_nv|9/9|5/6|3/3|3/3|
|spatialrgpt|8/9|4/6|不测（原生单图）|不测（原生单图）|

选择题只按最终答案评分：支持字母、boxed 字母或任一选项的精确文字。未解析和截断单列在 scores.json；不从推理段寻找正确答案。

## 原生坐标和区域接口

* robobrain25_8b_nv / pointing: 3/3，可解析 3。
* robobrain25_8b_nv / grounding: 1/3，可解析 3。
* spatialrgpt / region_left: 3/3，可解析 3。
* spatialrgpt / region_depth_sensor_variant: 3/3，可解析 3。

点坐标命中定义为落入可见红球紧框，框定位正确定义为 IoU≥0.5。
点中心误差（像素）：barrier_a0_pointing=1.32, pit_a0_pointing=2.29, window_a0_pointing=1.62。
框 IoU：barrier_a0_grounding=0.444, pit_a0_grounding=0.433, window_a0_grounding=0.551。

SpatialRGPT 深度题是原生分支的 RGB-D 传感器变体，未使用作者 DepthAnything；不可声称完整官方流程复现。

## 输入、资源和输出质量

完成 105 / 105 次固定推理；最终审计异常 0。
* qwen3vl8b: 未解析 0，输出上限 0，含 ComponentPlacement 异常串 0。
* spatial_interactor8b: 未解析 0，输出上限 0，含 ComponentPlacement 异常串 0。
* spatialreasoner: 未解析 0，输出上限 0，含 ComponentPlacement 异常串 0。
* robobrain25_8b_nv: 未解析 0，输出上限 0，含 ComponentPlacement 异常串 0。
* spatialrgpt: 未解析 0，输出上限 0，含 ComponentPlacement 异常串 14。

最初四个 Qwen 系模型因缺少 qwen-vl-utils 在加载和推理前退出，补齐隔离依赖后续跑；保留首次退出记录，没有把环境错误算成能力失败。

实际输入处理尺寸、token、pixel 哈希见 runs/*/*/input_audit.json。每题均保留完整问题、原始输出和 token ID。

## 结论边界

这些题只检查视觉定位、当前可见性、简单图像变化和接口行为。双帧位移不是带里程计的空间记忆更新；区域深度题不是遮挡背后几何推断。三场景红球结果不能推断开放词汇、家庭/野外泛化、遮挡重现预测或导航成功率。

对之前只转圈结果的解释应限制在证据范围：原生坐标/区域接口可回答，说明直接强制六动作会漏掉模型原有输出能力；它不能单独证明原来的失败完全由 prompt 造成。还需使用明确的动作转换器，比较相同感知、预算、初始状态下的闭环行为。

复现：先运行 tools/task3_native_diagnostics.py build（已有冻结集勿重建），再以合适 Python 环境运行 tools/run_task3_native_diagnostics.py，最后运行 tools/analyse_task3_native_diagnostics.py。
