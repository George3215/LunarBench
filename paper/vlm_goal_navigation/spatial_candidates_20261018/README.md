# 新增空间模型与遮挡恢复文献

用户授权：2026-10-18，下载并测试 RoboBrain2.5-8B-NV、SpatialReasoner、SpatialRGPT-VILA1.5-8B；检索室内外物体搜索及目标短暂丢失恢复的开源工作。

- [相关工作、实现机制与对比取舍](RELATED_WORK.md)：优先 EZREAL、LOVON；补充 Nav-R2、VLMnav、VLFM、ApexNav、AstraNav-Memory、Memory Over Maps、GOAT。
- `*_inventory.json`：Hugging Face 不可变版本、文件元数据与访问状态。
- `*_download.json`：实际下载状态；只有 complete 且分片哈希全部匹配才允许进入推理。
- `ADAPTER_PROTOCOL.json`（各运行目录）：架构、预处理、模板、源码、深度接口等差异，不宣称跨骨干比较是空间训练的单变量消融。

RoboBrain 的 Hugging Face 访问检查失败后，已核对作者仓库链接的官方 ModelScope NV 分发：可公开下载，两个权重分片 SHA256 与 HF 同名 NV 版本一致。已改从该来源下载；无需用户为了本次测试登录 HF，也没有代替用户共享联系信息。具体文件版本及哈希保留在库存和下载记录。

SpatialReasoner 使用自己的 Qwen2.5-VL tokenizer/processor；不能强装 Qwen3-VL processor。SpatialRGPT 用隔离旧版 Transformers 环境和官方 Llama/VILA/SigLIP 实现；不更改现有 Qwen 的运行依赖。

评测复用 V4 的22个同一丢失起点、相同语义 prompt、四帧事件 RGB-D、理想自身里程计、动作与64步预算、相同停止/到达条件。原生 chat template 和视觉编码分辨率必须按模型分别记录，不要求跨架构 token 完全相同。SpatialRGPT 首轮使用通用多图输入：深度以相同灰度图呈现；**没有给它额外的目标真值 mask/region，也没有把该配置称为原生区域深度接口完整评测**。若后续测试原生 region/depth 分支，必须另立协议并统一从观测生成区域。

先做两场景两动作接口烟测，记录解析成功、非法 STOP 屏蔽、资源错误；随后在单 GPU 队列接续正式场景。尚未完成的模型不填完整导航成功率。已有 Qwen/Spatial-Interactor 队列保持独立，先完成其冻结实验。

三个新增模型均已完成实际 GPU 接口诊断。SpatialRGPT、RoboBrain 通过解析检查；SpatialReasoner 两例均未产生可执行的完整动作输出。为避免以两例代替全场景结论，SpatialReasoner 另外排入22场景正式接口迁移诊断，接在前两个模型之后；`REASONER_QUEUE_STATUS.json` 记录等待/执行状态。其格式失败保留为格式失败，不据此声称空间能力无效。测试不追加外部动作猜测或目标真值。

权重、上游源码、完整原始图像/轨迹保留 `.local/` 或运行目录且不入 Git；协议、状态、摘要、参考文献和结论备份到当前 GitHub 实验分支。GitHub 摘要备份不等于原始数据异地备份。
