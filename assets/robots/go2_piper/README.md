# Unitree Go2 搭载 Agilex Piper 机械臂

`go2_piper.xml` 是四足底盘带臂的接入模型：Go2 机体与 Piper 臂都取自本仓库已有资源，本目录只新增两者的组合。来源为 LeggedManip_Lab 的 GO2-PIPER 平台，但只取该平台的**安装变换**（`base_link` 上方 6 cm，无旋转）——臂的几何、关节变换与惯性来自 `assets/robots/piper/arm.xml`，腿的几何、惯性与执行器来自 `assets/robots/go2_demo/go2/go2.xml`。两个源文件都没有被修改。

**没有复制任何网格。** `<compiler meshdir="../">` 指向 `assets/robots/`，Go2 网格写成 `go2_demo/go2/assets/*.obj`、Piper 网格写成 `piper/assets/*.STL`，因此本目录只有约 48 KB 的描述文件。`source.json` 记录两个源文件的 SHA256、上游安装变换来源、各 Piper mesh 的 SHA256 和改动清单；`UPSTREAM_LICENSE` 与 `piper/` 同一份 Apache-2.0 文本，不代表单独核实了所有 mesh 的公开再分发授权。`tools/prepare_go2_piper.py` 可从两个本地模型重新生成。

组合时必须自包含而不能用 `<include>`：MuJoCo 只认顶层 `<compiler>` 的 `meshdir`，被 include 文件的 `meshdir` 会被忽略，两套网格会按错误目录解析。

新增/改动内容：把 `arm_base` 从 worldbody 直接子节点改为 `base_link` 的子节点；删除代表参考底盘甲板的 `mounting_pedestal`（Go2 背部本身就是安装面）；把 `base_link` 经 `childclass="go2"` 继承的 `geom margin 0.001` 与 `joint frictionloss 0.2` 在臂上显式固定为 0，使臂的行为与 `piper/arm.xml` 一致；把 demo 的 `home` keyframe 补到扩大后的 `nq`/`nu`。腿的 12 个 `<motor>`、4 组默认类别和传感器原样保留。

腿部由**既有 Go2 步态策略**驱动，臂与夹爪是 7 路位置伺服，由 TASK1 的动作向量直接给目标。上游 GO2-PIPER 的**全身策略没有移植**：它是腿臂联合的 210 维观测 / 18 维动作 TorchScript 网络，且上游模型里夹爪关节是注释掉的、根本没有夹爪，无法完成 TASK1 的收集任务。本模型与上游策略、上游力矩接口都不兼容。

`tools/prepare_go2_piper.py` 会断言 `nu==19`（12 腿 + 6 臂 + 1 夹爪）、Go2 的 16 个 mesh 占据 id 0–15 且名字与 UE 的 `SM_Go2_<id>` 编号一致（见 `tasks/task1_collect/display.py`）、`joint1`/`joint7`/`joint8`/`tool_tip` 存在、臂关节无继承摩擦、臂几何无继承 margin、keyframe 长度与模型匹配。

上述组合、伺服与安装是仿真接入配置，未经实机标定；当前没有全身控制、抓取规划、IK 或训练策略。夹爪净开口 6.996 cm（两指内表面实测），能夹住 TASK1 当前约 3.5–4.5 cm 的石头；抓取本身由 `tools/validate_task1_grasp.py` 验证，臂/夹爪的接入由 `tools/validate_task1_manipulators.py` 验证。注意抓取只是物理上成立，**没有任何东西负责找到并够到石头**。配置和 Python 控制示例见 `tasks/task1_collect/README.md`。
