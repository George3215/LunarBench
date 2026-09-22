# Robotiq 2F-85 自适应两指夹爪（MuJoCo Menagerie）

`2f85.xml` 是 MuJoCo 模型，`assets/` 是它引用的 8 个 STL。**只有 MuJoCo 模型，
没有 UE 显示资产**（没有 `.usdz`）——它和 `franka/` 一样是从 MuJoCo Menagerie 原样
搬运的，不是本仓库用 `tools/prepare_arms.py` 生成的。

来源为 `google-deepmind/mujoco_menagerie` 的 `robotiq_2f85`，commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`（**与 `franka/` 同一个 commit**）。
`source.json` 记录仓库、commit、镜像、逐文件 SHA256 与本次改动清单；
`fetch_report.json` 是抓取时的原始记录，其中每个文件的 `git_blob_sha1` 是对着 GitHub
API 给的 blob 哈希校验过的——13 个文件全部逐字节等于上游。

## 这个目录里没有一个字节是自己写的

`2f85.xml`、`LICENSE`、`CHANGELOG.md`、`scene.xml` 与 `assets/` 下 8 个 STL 都是上游
原文。本目录只多三个说明文件：`source.json`、`README.md`、`fetch_report.json`。
上游自己的 `README.md` 也保留了，改名为 `UPSTREAM_README.md` 以免和本文件撞名；
上游的 `2f85.png`（示意图）没有入库，所以 `UPSTREAM_README.md` 里的图片链接是死的。

上游 `scene.xml` 是 2F-85 的独立查看/自测场景（含一个吊着的方块），保留原样，可用于
单独打开夹爪：

```bash
python -m mujoco.viewer --mjcf MoonSim/assets/robots/robotiq_2f85/scene.xml
```

## 模型内容

上游这份 MJCF 是 URDF 转来的，已经带齐仿真需要的东西：

| 项 | 值 |
| --- | --- |
| body / mesh | 14 个 body、8 个 mesh（左右各一套 driver / coupler / spring_link / follower / pad / silicone_pad） |
| 自由度 | 8 个铰链关节（每侧 4 个四连杆），`nq=8` |
| 执行器 | 1 个：`fingers_actuator`，作用在 `split` 肌腱上 |
| 等式约束 | 3 条：两条 `connect` 把 follower 绑到 coupler，一条 `joint` 把左右 driver 绑成反向同步 |
| 接触 | 6 条 `exclude`，避免连杆之间自碰撞；指面另有 `pad_box1/2` 两段高摩擦 box |
| `<option>` | `cone="elliptic" impratio="10"`（上游为稳定抓取加的） |
| 总质量 | 1.0526 kg（其中 `base` 0.7774 kg） |
| 抓取中心 | `pinch` site，在 `base` 坐标系 `(0, 0, 0.145)`，即 `base_mount` 原点上方 **0.1558 m** |

控制量是上游定制的：`ctrlrange` 是 **0–255**，而不是弧度。上游注释说明
`gainprm[0] = 0.8 * 100 / 255`，把 0–255 重标定到 driver 关节的 0–0.8 rad。
实测（无重力、跑 4 s 收敛）两指内表面间距：

| `ctrl` | 指间距 |
| ---: | ---: |
| 0 | **0.0984 m**（全开） |
| 255 | **0.0150 m**（闭合） |

`tasks/task2_arm/scene.py` 里的 `GRIPPER_OPEN` / `GRIPPER_CLOSED` 就是这两个值。

## 装到机械臂上

**这个目录不含任何安装变换**，它只是夹爪本身。夹爪根部 body 是 `base_mount`
（`pos="0 0 0.007"`，它的 +Z 指向手指方向）。挂到 UR12e 上由
`tasks/task2_arm/scene.py` 负责：在那个文件里给 `wrist_3_link` 加一个单位位姿的
`tool_flange` frame，再把 `base_mount` 所在的整棵子树 attach 上去。组合后的模型里
法兰到 `pinch` 的距离仍是 **0.1558 m**。

组合用 MuJoCo 的 procedural attachment（`MjSpec.attach`），不是 `<include>`；
原因和 `assets/robots/go2_piper/README.md` 里记的是同一个坑（`<include>` 会丢掉
子文件的 `meshdir`）。另外注意 attach 时两个子模型的 `<option>` 必须对齐，否则
MuJoCo 会保留父模型的值并打警告——见 `scene.py` 的说明。

## 未声明

没有硬件标定；没有抓取规划、IK 或训练策略；没有 UE 侧物理或显示资产；
没有标定真实的 2F-85 开合力/行程，上面的指间距是仿真里量出来的几何量。

上游这份模型按 BSD-2-Clause 发布（见 `LICENSE`），本目录原样保留该许可文本，
不代表已经单独核实了所有网格的公开再分发授权。
