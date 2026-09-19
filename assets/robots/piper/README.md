# Piper 机械臂与参考移动操作机器人

`arm.xml` 是固定基座 Piper；`rover_piper.xml` 是本项目参考差速底盘搭载同一机械臂。两者保留来源模型的 STL 网格、关节变换、关节范围和惯性数据，不降低 mesh 面数。

来源为本机 ATEC_UE_sim 的 LeggedManip_Lab/go2_piper。`source.json` 记录原路径和原 XML、各 mesh 的 SHA256；`UPSTREAM_LICENSE` 保留上游仓库 Apache-2.0 文本，不代表单独核实了所有 mesh 的公开再分发授权。无需安装或修改 ATEC 项目即可使用已复制的本地资源。

`tools/prepare_piper.py` 可从原本地资源重新生成。新增内容：6 路有限力矩位置伺服、两个通过 equality 耦合的夹爪 slide joint、1 路夹爪位置伺服、tool_tip site、固定底座/参考底盘安装。显式排除底座与第一关节重叠轴承外壳，以及两个指爪彼此的碰撞；它们与环境/石头的碰撞保留。MuJoCo mesh 碰撞采用原生凸包语义。

上述伺服、夹爪结构和组合底盘是仿真接入配置，未经实机标定。当前没有抓取规划、IK 或训练策略。夹爪净开口 6.996 cm（两指内表面实测），能夹住 TASK1 当前约 3.5–4.5 cm 的石头；`tools/validate_task1_grasp.py` 用实际合爪抬起验证了这一点。实机标定仍然没有做。

配置和 Python 控制示例见 `tasks/task1_collect/README.md`；验证脚本为 `tools/validate_task1_manipulators.py`。
