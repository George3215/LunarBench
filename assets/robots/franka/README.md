# Franka Emika Panda 机械臂（MuJoCo Menagerie）

固定基座七轴机械臂 + 平行夹爪。`panda.xml` 是 MuJoCo 模型，`assets/` 是它引用的全部网格。

来源为 `google-deepmind/mujoco_menagerie` 的 `franka_emika_panda`，commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`。`source.json` 记录上游仓库、commit、
镜像、逐文件 SHA256，以及本次改动清单；`fetch_report.json` 是抓取时的原始记录，
其中每个文件的 `git_blob_sha1` 是**对着 GitHub API 给的 blob 哈希校验过的**——
只记自己下载结果的哈希只能证明"下载没坏"，对不上上游就无法证明"拿到的是这个版本的字节"。

## 这个目录里没有一个字节是自己写的

与 `ur10`/`ur10e`/`ur12e`/`so101` 不同：那四个是 `tools/prepare_arms.py` 从上游 URDF
**生成**的（补伺服增益、`tool_tip` site、`home` 关键帧、显示用 `.usdz`），所以它们的
`source.json` 里有一条"相对上游改了什么"。这里是**原样搬运**：`panda.xml`、`LICENSE`
与 67 个网格与上游逐字节相同，本目录只多了 `source.json` / `README.md` /
`fetch_report.json` 三个说明文件。

Menagerie 上游本身就已经是可仿真的 MJCF（7 个臂关节 + 1 个夹爪执行器、位置伺服、
`home` 关键帧、`link0`/`link1` 的接触对排除都是上游写好的），所以不需要再生成一遍。

## 模型内容

上游 `panda.xml` 本身就已经是可仿真的 MJCF（7 个臂关节 + 1 个夹爪执行器、位置伺服、
`home` 关键帧、`link0`/`link1` 的接触对排除都是上游写好的），所以不需要再生成一遍。
载入实测 `nbody=12`、`nq=9`、`nu=8`、`nsite=0`、`nkey=1`，全部执行器 `ctrllimited=True`。

固定基座是 `link0`：它是 `worldbody` 的直接子节点、没有关节，坐在平地上（基座碰撞
网格最低点 z=0）；`link1` 在它上方 **0.333 m**，即肩部高度。

臂展上界实测 **1.2383 m**：在关节限位内随机采 20 万组关节角，取手指 body 原点相对
`link0` 原点的最大半径。注意这不是手册上的 855 mm——手册给的是工作半径，而这里量的
是从基座原点算起的最大半径，基座原点到肩部本身就有 0.333 m。

## 与 `assets/robots/sensors/d435.*` 无关

`sensors/d435.obj` / `.dae` 是另一份资产（机载深度相机网格），与这里的 `franka_hand`
无关，也与 `assets/robots/manipulators/manipulator/franka_arm.usdz`（只给 UE 显示的
USD，没有 MuJoCo 自由度）无关。

## 验收

`tools/validate_arms.py` 不覆盖这个模型——它检查的是 `prepare_arms.py` 生成的那四个
（`arm.xml` + `.usdz` 成对）。这份模型目前唯一的凭证是**来源**：`fetch_report.json` 里
69 个文件逐个对着 GitHub API 的 git blob SHA1 校验，全部一致。

**它现在没有接入任何任务。** 旧的 TASK2（`task2_stack`）已整体删除；新的
[`task2_arm`](../../tasks/task2_arm/README.md) 空白环境用的是 UR12e + Robotiq 2F-85，
不是这个模型。所以除了"能载入、能步进、几何量与上游一致"以外没有别的验收，
也没有任何 reach / 工作空间的校验在用上面的数字——那两个数只是这份模型的几何事实。

## 未声明

没有做硬件标定，没有训练或接入任何策略，没有 UE 侧物理或显示资产。
上游 `panda.xml` 没有 `tool_tip` site（实测 `nsite=0`），要接抓取规划时需要自己补一个。
