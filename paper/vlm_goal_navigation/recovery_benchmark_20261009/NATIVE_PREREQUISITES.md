# 原生环境审查与延期条件

用户已说明暂时没有HM3D授权场景和模型API，允许先不考虑安装。因此VLMnav环境创建在元数据获取阶段停止；没有运行VLMnav/NaVILA模型，没有下载两者的checkpoint，没有修改已有机器人环境，也没有把自建回合冒充官方测试。

## 已固定的官方源码

- VLMnav：`https://github.com/Jirl-upenn/VLMnav`，commit `82d59684c15a6694ea73b969e76a57157c8740b8`。
- NaVILA：`https://github.com/AnjieCheng/NaVILA`，commit `76b98f233dd0fff05dfcd69435eec6740febff9d`。
- 独立目录：`/home/lry/vlm_recovery_benchmark`。源码采用浅层稀疏获取；不下载媒体或评估数据。固定commit及Git blob校验记录保留在native_logs。

## VLMnav

官方环境为Python3.9、Habitat-Sim0.3.1、withbullet、headless。官方ObjectNav需要HM3D v0.2 val场景和ObjectNav episodes。HM3D场景需要用户取得Matterport数据访问并配置token，本次未尝试获取授权数据。

官方配置使用Gemini后端，也提供OpenAIVLM扩展接口。后端模型及可用API需用户配置；环境中同名变量的存在不作为用户选择后端或授权调用的替代。本次没有发出模型API请求。

官方requirements没有列出OpenAI包，但OpenAIVLM会导入它；后续选择该后端时须在独立环境中补充并冻结版本。`magnum==0.0.0`需检查Conda提供的编译包元数据，不应为满足pip而覆盖已安装的Habitat/Magnum。

官方ObjectNav默认`navigability_mode=depth_sensor`：深度进入动作候选规划。RGB-only组必须禁用这条路径，并分别记录RGB模型输入与规划器传感器输入。官方配置`context_history=0`也不是目标恢复实验的First+Current，不能直接把官方分数当Task3分数。

## NaVILA

官方README当前已标记模型、训练和评估发布完成，不能沿用旧笔记的TODO状态。原生VLN-CE安装需要Python3.10、从源码构建Habitat-Lab/Sim v0.1.7、Torch2.3对应的FlashAttention2.5.8、Transformers4.37.2及仓库补丁、WebDataset0.1.103。不得在现有MuJoCo或VLMnav环境中安装。

评估需要MP3D场景、R2R/RxR annotations和官方`a8cheng/navila-llama3-8b-8f`checkpoint。官方`environment_setup.sh`还包含训练依赖及整个CUDA toolkit安装，不直接盲目执行。未来按评估所需逐项安装、保存版本与错误。

该仓库中的Habitat评估只证明高层动作在Habitat中的执行，不证明实体机械狗或MuJoCo腿式底层控制已跑通。README指向独立的legged-loco及NaVILA-Bench；腿式控制层需另立实测记录。不得将底层执行失败混同于语言动作选择失败。

## 恢复顺序

1. 获得用户提供的已授权HM3D数据路径和可用模型后端配置。
2. VLMnav独立环境导入/依赖检查 → 一个官方短回合 → 原生环境中目标丢失恢复验证。
3. 准备NaVILA数据/checkpoint，再建立独立环境并完成原生评估、恢复及控制层审查。
4. 两者通过后才构建MuJoCo模型适配器并运行12条件。

当前54个场景是恢复可见性集；正式最终到达指标还需补充近距可达检查。窗口类是否存在历史必要性，需要同当前图像的配对审计和相反恢复动作验证。不能以本次G1–G4通过替代这些额外条件。
