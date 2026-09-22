# 三条机械臂的实测几何事实（ur5e / panda / piper）

本文所有数字都由 `tasks/task2_stack/tools/probe_arm_geometry.py` 在运行时从**编译后的 MuJoCo 模型**里量出来，
不抄 README、不抄注释。原始数据在 `tasks/task2_stack/reports/arm_geometry_<arm>.json`（合并版 `arm_geometry.json`）。

* 三条臂都只用运动学（`mj_kinematics` / `mj_forward`）与前向动力学的短时 settle，不使用 Renderer / viewer（本机没有 GPU/GL）。
* 夹爪都是耦合机构（ur5e 是 4-bar linkage + tendon equality，panda 是 tendon `split` + equality，piper 是 joint equality），
  **直接写 `data.qpos` 会得到物理上不可能的构型**；因此每个采样点都用**全新的 `MjData`**、设好 `qpos`/`ctrl` 后步进到
  `max|qvel| < 1e-5` 再测量。开合曲线与"参考状态"两条路径是同一套初值 + 同一套 ctrl，交叉校验偏差 ≤ 6.3e-7 m。
* 探针场景把基座放在世界原点、删掉桌面、**把所有 geom 的 contype/conaffinity 置 0**（本次不测接触）——
  否则 Robotiq 的两块 fingerpad 在闭合时自碰撞，量到的就不是机构本身的运动学关系。

---

## 0. 复现命令（每条命令都必须带 PYTHONPATH）

```bash
cd /home/lry/MoonUnrealEnv/MoonSim
PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack \
  /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python \
  tasks/task2_stack/tools/probe_arm_geometry.py --arm ur5e

PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack \
  /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python \
  tasks/task2_stack/tools/probe_arm_geometry.py --arm panda

PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack \
  /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python \
  tasks/task2_stack/tools/probe_arm_geometry.py --arm piper

# 合并三份 JSON 到 reports/arm_geometry.json
PYTHONPATH=/home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack \
  /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python \
  tasks/task2_stack/tools/probe_arm_geometry.py --merge
```

可选参数：`--out PATH`（默认 `tasks/task2_stack/reports/arm_geometry_<arm>.json`）、
`--samples N`（可达性采样数，默认 20000）、`--seed`（默认 0）、`--quiet`。
脚本从任意 cwd 可运行（路径全部由 `__file__` 推导），固定种子、不含时间戳，因此可逐字节复现。

ur5e 的模型不是"读一个 XML"：它调用 legacy 的
`scripts/run_official_ur5e_robotiq_wall_stack.py::build_wall_stack_scene([], {})` 组合
robosuite UR5e + Robotiq 2F-140（mesh 绝对化、gripper 挂到 `right_hand`、`ur_pos_*` kp=650/kv=55、
gripper 执行器 kp=150/kv=5、tendon/equality 原样拷贝），然后只做本探针必需的改动（基座归零、删桌面、关碰撞）。

---

## 1. 关键数字总表

### 1.1 ur5e（robosuite UR5e + Robotiq 2F-140）

| 项目 | 实测值 |
| --- | --- |
| 模型规模 | `nq=12 nv=12 nu=8 nbody=18 nsite=7 neq=4 ntendon=4 nkey=0`，`dt=0.0015` |
| 运动链（base→TCP body） | `base → fixed_base_link → shoulder_link → upper_arm_link → forearm_link → wrist_1_link → wrist_2_link → wrist_3_link → right_hand → right_gripper → eef` |
| 臂关节 | `shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint`，qpos/dof 地址 0..5，限位见 JSON `arm_joints` |
| 臂执行器 | `ur_pos_<joint>` position，kp=650、kv=55，ctrlrange=关节限位，**forcerange ±280 N·m** |
| TCP site | `grip_site`（挂在 body `eef` 上，site 局部 pos=0） |
| 最大开口（pad 中心距） | **0.128444 m** @ ctrl `(0, 0)`（全开） |
| 最小开口（pad 中心距） | **0.007455 m** @ ctrl `(0.4375, -0.4375)` |
| 最大内表面间距 | 0.126433 m；两指点云**最近距离**最小 0.005688 m @ ctrl `(0.35, -0.35)` |
| 闭合轴（TCP 帧） | **`[+1.0, -0.000797, +0.000004]`**（≈ `+x_site`；用开合曲线数值微分得到的"两指真正靠近方向"与它相差 0.003°） |
| 接近轴（TCP 帧） | **`[0.0, 0.0, +1.0]`**（`+z_site`；法兰→pad 中点方向，顶抓时它朝下） |
| TCP→pad 中点方向 | `[0.0, 0.0, -1.0]`（与接近轴反平行 179.99°：`grip_site` 落在两指之外 32.3 mm） |
| pad 中点偏移（TCP 帧） | `[0.000003, 0.0, -0.032265]`，模长 **0.032265 m** |
| TCP 相对法兰（`right_hand`） | 0.207500 m，在 `right_hand` 帧下 `[0, 0, 0.2075]`；相对 `wrist_3_link` 见 JSON |
| 可达半径（2 万采样，全部姿态） | min 0.0173 / 中位 0.6382 / p95 1.0736 / max **1.3072 m** |
| 可达半径（接近轴朝下 ≤35°，n=1216，6.1%） | min 0.0173 / 中位 0.6101 / **p95 0.9511** / max 1.0683 m |
| 质量 | 总 **20.3595 kg**（臂 19.6291 + 夹爪 0.7304；夹爪运动件 0.1688 kg） |
| 载荷粗估（肩 ±280 N·m，r=0.6·1.3072=0.7843 m） | **36.39 kg**（上界，见 §6 的重要警告；用上游真实肩力矩 150 N·m 则为 19.50 kg） |
| home（本脚本 IK） | qpos `[-0.302359, -1.702680, 1.520051, -1.388147, -1.570804, -0.301560]`，TCP `[0.45, 0, 0.35]`，接近轴 `[0,0,-1]`（离竖直 0.0002°），肘高 0.584 m |
| legacy `Q_HOME_ELBOW_UP` 实测 TCP | `[0.348552, 0.500045, 0.188072]`，四元数 `[-0.000641, 0.707378, 0.706815, -0.005375]`，接近轴 `[-0.00851, -0.006674, -0.999942]`（离竖直 0.62°），pad 中点 `[0.348826, 0.500263, 0.220334]` |

legacy 位姿的补充实测：把同一组 qpos 作为**位置伺服目标**让它稳定下来后，TCP 会掉到 `[0.339483, 0.491764, 0.146006]`，
比纯运动学结果低 **0.043822 m**（kp=650 下的重力下垂）。规划器如果"发 ctrl 就以为到位"，会有 4 cm 级别的偏差。

### 1.2 panda（MuJoCo Menagerie franka_emika_panda）

| 项目 | 实测值 |
| --- | --- |
| 模型规模 | `nq=9 nv=9 nu=8 nbody=12 nsite=0 neq=1 ntendon=1 nkey=1`，`dt=0.002` |
| 运动链（base→TCP body） | `link0 → link1 → link2 → link3 → link4 → link5 → link6 → link7 → hand` |
| 臂关节 | `joint1..joint7`，qpos/dof 地址 0..6；`joint4 ∈ [-3.0718, -0.0698]`、`joint6 ∈ [-0.0175, 3.7525]`、其余 ±2.8973 |
| 臂执行器 | `actuator1..7` general（kp=4500/4500/3500/3500/2000/2000/2000，kv=kp/10=450/450/350/350/200/200/200），forcerange：1-4 为 ±87 N·m，5-7 为 ±12 N·m |
| 夹爪执行器 | `actuator8`，作用在 fixed tendon `split`（0.5·q1+0.5·q2），`ctrlrange 0 255`，`forcerange ±100`；平衡时 tendon 长度 = `0.04*ctrl/255` |
| TCP | **无 site**（实测 `nsite=0`）→ 本脚本经验测定的合成 TCP，见 §3 |
| 最大开口 | **0.085400 m**（pad 中心距，ctrl=255，全开）；内表面间距 0.080000 m（= Panda 手册的 80 mm） |
| 最小开口 | **0.005400 m**（ctrl=0，两指端部小 pad 已贴合，最小点距 0.0 m） |
| 闭合轴（合成 TCP 帧） | **`[0.0, +1.0, 0.0]`**（= `hand` 帧 +y，指向 left_finger） |
| 接近轴（合成 TCP 帧） | **`[0.0, 0.0, +1.0]`**（= `hand` 帧 +z，指向指尖；顶抓时它朝下） |
| pad 中点偏移（TCP 帧） | `[0, 0, 0]`（**按定义** TCP 就是全开时两 pad 面的中点） |
| TCP 相对法兰（`hand`） | 0.103100 m，在 `hand` 帧下 `[0, 0, 0.103100]` |
| 可达半径（2 万采样） | min 0.0245 / 中位 0.8111 / p95 1.1958 / max **1.2762 m**（TCP=pad 中点口径） |
| 可达半径（接近轴朝下 ≤35°，n=1864，9.3%） | min 0.0890 / 中位 0.6603 / **p95 0.9165** / max 0.9941 m |
| 质量 | 总 **17.4519 kg**（臂 16.6919 + 夹爪 0.7600；两指运动件 0.0300 kg） |
| 载荷粗估（joint1 ±87 N·m，r=0.6·1.2762=0.7657 m） | **11.58 kg**（上界）；若按全臂最弱关节 joint5/6/7 的 12 N·m 算则 **1.60 kg** |
| home | keyframe `home`：qpos `[0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]`，ctrl `[..., 255]`，TCP `[0.5545, 0, 0.521402]`，接近轴 `[0,0,-1]` |

keyframe home 的另一个读数：按 keyframe ctrl 让它稳定后 TCP 落在 `[0.554652, -0.000071, 0.514680]`，
比 keyframe qpos 的纯运动学结果低 **0.006725 m**（重力下垂）。

### 1.3 piper

| 项目 | 实测值 |
| --- | --- |
| 模型规模 | `nq=8 nv=8 nu=7 nbody=11 nsite=1 neq=1 ntendon=0 nkey=0`，`dt=0.002` |
| 运动链（base→TCP body） | `arm_base → link1 → link2 → link3 → link4 → link5 → link6 → end_effector` |
| 臂关节 | `joint1..joint6`，qpos/dof 地址 0..5；限位 `[-2.618,2.168] [0,3.14] [-2.967,0] [-1.745,1.745] [-1.22,1.22] [-2.0944,2.0944]` |
| 臂执行器 | `arm_joint1..6` position，kp=100、kv=10，forcerange `±20 ±20 ±15 ±7 ±5 ±5` N·m |
| 夹爪 | `joint7`(link7)/`joint8`(link8) 两个 slide joint，equality 耦合；唯一执行器 `gripper`（joint7，kp=500、kv=20，`ctrlrange 0..0.035`，forcerange ±20） |
| TCP site | `tool_tip`（body `end_effector`，局部 pos=0） |
| 最大开口 | **0.086659 m**（pad 中心距，ctrl=0.035=全开）；内表面间距 **0.069348 m**；两指点云最近距离 0.070000 m |
| 最小开口 | **0.016770 m**（ctrl=0，两指已合上，最近点距 0.0 m） |
| 闭合轴（TCP 帧） | **`[0.0, -0.999693, +0.024778]`**（≈ `-y_tcp`，指向 link8）；用曲线数值微分得到的方向是 `[-0.000007, -1.0, -0.0001]`，两者差 1.43°（见 §4 说明） |
| 接近轴（TCP 帧） | **`[+1.0, 0.0, 0.0]`**（`+x_tcp`；`link6`→pad 中点；顶抓时它朝下） |
| TCP→pad 中点方向 | `[-1.0, 0.0, 0.0]`（与接近轴反平行 179.9996°：`tool_tip` 在两指之外 33.4 mm） |
| pad 中点偏移（TCP 帧） | `[-0.033446, 0.0, 0.0]`，模长 **0.033446 m** |
| TCP 相对法兰（`link6`） | 0.130000 m，在 `link6` 帧下 `[0, 0, 0.130000]` |
| 可达半径（2 万采样） | min 0.0078 / 中位 0.6167 / p95 0.8404 / max **0.8805 m** |
| 可达半径（接近轴朝下 ≤35°，n=2437，12.2%） | min 0.0078 / 中位 0.3166 / **p95 0.5624** / max 0.6541 m；该子集 **TCP 高度中位数 −0.12 m、最大仅 0.271 m，只有 0.6% 高于 0.20 m** |
| 质量 | 总 **3.9125 kg**（臂 3.8625 + 夹爪 0.0500；运动件 0.0500 kg） |
| 载荷粗估（joint1 ±20 N·m，r=0.6·0.8805=0.5283 m） | **3.86 kg**（上界）；按最弱关节 joint5/6 的 5 N·m 算则 **0.96 kg** |
| home（本脚本 IK，**目标不可达的尽力解**） | qpos `[-0.013841, 1.446273, -1.498509, 0.024962, 1.128144, -0.556947]`，TCP `[0.36, 0, 0.28]`，接近轴 `[0.550, 0.015, -0.835]`（离竖直 **33.4°**），离名义目标 (0.45,0,0.35) 还差 0.115 m |

> **piper 的顶抓能力有一条硬约束**：在「工具(接近轴)朝下」的约束下，5 万组随机采样的关节角里 TCP 高度最大只有
> **0.292 m**（≤5° 时只有 0.121 m）。也就是说 piper 做不了"在 0.35 m 高的桌面上竖直下抓"——名义 home 目标
> (0.45, 0, 0.35) 及其 0.9/0.8/0.7/0.6/0.5 倍收缩版本全部不可达，脚本只能给出 33.4° 斜下压的尽力解。
> 规划工作台时必须用这个数，而不是 §1.3 的 0.562 m 半径。

---

## 2. 开合曲线（可直接粘进代码的线性拟合）

`width` 的定义：两指 pad geom 的 **AABB 中心距离**（= 沿逐采样点闭合轴的投影，恒 ≥ 0）。
`inner_surface_gap` = 两指全部表面点（box 取 8 角点 / mesh 取全部顶点）沿固定闭合轴的极值差，
`min_surface_distance` = 两指点云最小距离（KD-tree，恒 ≥ 0，真实最近接近量）。

### ur5e（`ctrl = (g, -g)`，g 从 0 全开到 0.7）

| g | ctrl 向量 | 指关节 qpos | pad 中心距 | 内表面间距 | 最近点距 |
| --- | --- | --- | --- | --- | --- |
| 0.0 | `[..., 0.0, -0.0]` | `(0.000100, -0.000132)` | 0.128444 | 0.126433 | 0.126433 |
| 0.0875 | `[..., 0.0875, -0.0875]` | `(0.087569, -0.087602)` | 0.102202 | 0.090626 | 0.090626 |
| 0.175 | `[..., 0.175, -0.175]` | `(0.175039, -0.175072)` | 0.076657 | 0.055761 | 0.055761 |
| 0.2625 | `[..., 0.2625, -0.2625]` | `(0.262510, -0.262543)` | 0.052151 | 0.022377 | 0.022377 |
| **0.30**（legacy `--close`） | `[..., 0.3, -0.3]` | `(0.299997, -0.300031)` | 0.042044 | 0.008648 | 0.008648 |
| 0.35 | `[..., 0.35, -0.35]` | `(0.349981, -0.350015)` | 0.028993 | -0.009027 | **0.005688** |
| 0.4375 | `[..., 0.4375, -0.4375]` | `(0.437455, -0.437487)` | **0.007455** | -0.038004 | 0.035021 |
| 0.525 | `[..., 0.525, -0.525]` | `(0.524930, -0.524961)` | 0.012231 | -0.064165 | 0.037138 |
| 0.6125 | `[..., 0.6125, -0.6125]` | `(0.612407, -0.612437)` | 0.029879 | -0.087186 | 0.025336 |
| 0.7 | `[..., 0.7, -0.7]` | `(0.699886, -0.699915)` | 0.045350 | -0.106815 | 0.014540 |

* 关节耦合（实测与解析式一致）：`q(finger_joint)=g`，`q(left_inner_finger_joint)=-g/1.5`，
  `q(left_inner_knuckle_joint)=g/5.25`，右指镜像。
* **可用抓取段 g ∈ [0, 0.4375]** 的线性拟合：`width ≈ -0.277955·g + 0.126593`（rms 1.456e-3 m，max 2.468e-3 m，R²=0.99863，n=7）。
* 全量程 g ∈ [0, 0.7] 的拟合：`width ≈ -0.135886·g + 0.099421`（rms 2.274e-2 m，R²=0.6223）——
  **不要用这一条**：4-bar 过了最小间距后两 pad 中心会越过中线，间距重新变大，是分段非单调的。
* `0.5·max_opening = 0.064222 m` 对应 g ≈ **0.224391**（拟合）/ **0.219400**（实测曲线插值）。
* 物理闭合极限：g≈0.32–0.35 附近两指表面已经互相穿插（内表面间距转负，最近点距最小值 5.688 mm @ g=0.35），
  legacy 的 `--close 0.32` 正好落在这个"刚好夹住"的位置。

### panda（`actuator8` ctrl 0..255，255 全开）

`width ≈ 3.13725e-4 · ctrl + 0.005400`（**精确线性**：max/rms 残差 = 0.0 m，R²=1.0，n=9）。
等价写法：`width = 0.0054 + 0.08 · ctrl/255`，内表面间距 `= 0.08 · ctrl/255`。

| ctrl | 0 | 31.875 | 63.75 | 95.625 | 127.5 | 159.375 | 191.25 | 223.125 | 255 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 指关节 qpos | 0 | 0.005 | 0.010 | 0.015 | 0.020 | 0.025 | 0.030 | 0.035 | 0.040 |
| pad 中心距 | 0.0054 | 0.0154 | 0.0254 | 0.0354 | 0.0454 | 0.0554 | 0.0654 | 0.0754 | 0.0854 |
| 内表面间距 | 0.0 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 |

`0.5·max_opening = 0.0427 m` → ctrl ≈ **118.89375**（拟合与实测插值一致）。
驱动方式：只能发 `data.ctrl[7] = ctrl`，靠 tendon+equality 让 `finger_joint1/2` 同步走到 `0.04*ctrl/255`。

### piper（`gripper` ctrl 0..0.035，**0.035 全开、0 闭合**）

`width ≈ 1.997291·ctrl + 0.016738`（max 残差 3.25e-5 m，rms 1.53e-5 m，R²=0.9999995，n=9）。
内表面间距 `≈ 1.999371·ctrl − 0.000630`（全开时 0.069348 m）。

| ctrl | 0.035 | 0.030625 | 0.02625 | 0.021875 | 0.0175 | 0.013125 | 0.00875 | 0.004375 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `joint7=joint8` | 0.035 | 0.030625 | 0.02625 | 0.021875 | 0.0175 | 0.013125 | 0.00875 | 0.004375 | 0 |
| pad 中心距 | 0.086659 | 0.077912 | 0.069165 | 0.060420 | 0.051677 | 0.042936 | 0.034199 | 0.025473 | 0.016770 |
| 内表面间距 | 0.069348 | 0.060601 | 0.051854 | 0.043106 | 0.034359 | 0.025612 | 0.016864 | 0.008117 | -0.000630 |

`0.5·max_opening = 0.043329 m` → ctrl ≈ **0.013314**（拟合）/ **0.013322**（实测插值）。

---

## 3. TCP 帧与四个规划用量（怎么量的）

| 量 | ur5e | panda | piper |
| --- | --- | --- | --- |
| TCP 帧 | site `grip_site`（原样用资产里的 site） | **合成**：`hand` 帧下 `pos=[0,0,0.1031]`、`quat=[1,0,0,0]` | site `tool_tip`（原样用资产里的 site） |
| `closing_axis_tcp` | `[1.0, -0.000797, 0.000004]` | `[0.0, 1.0, 0.0]` | `[0.0, -0.999693, 0.024778]` |
| `approach_axis_tcp` | `[0.0, 0.0, 1.0]` | `[0.0, 0.0, 1.0]` | `[1.0, 0.0, 0.0]` |
| `pad_center_offset_tcp` | `[0, 0, -0.032265]` | `[0, 0, 0]` | `[-0.033446, 0, 0]` |
| TCP 相对法兰偏移 | 0.207500 m（`right_hand`） | 0.103100 m（`hand`） | 0.130000 m（`link6`） |
| 法兰帧下的 TCP 偏移 | `[0, 0, 0.2075]` | `[0, 0, 0.1031]` | `[0, 0, 0.13]` |

**panda 合成 TCP 的推导（经验测定，因为上游 `nsite=0`）**

1. 把模型 reset 到 keyframe `home`、手指全开（`q=0.04`），并在同一 ctrl 下 settle 到静止；
2. 对 `left_finger` / `right_finger` 上全部 5 个 `fingertip_pad_collision_*` box（`model.geom_aabb` 给出它们在
   geom 帧内的包围盒中心）求世界 AABB 中心均值，得到两个"pad 面中心"：
   `left = [0.597199, -0.000004, 0.521402]`、`right = [0.511800, 0.000004, 0.521402]`；
3. 取中点 → 世界 `[0.554500, 0.0, 0.521402]`；转到 `hand` 帧就是 `[0, 0, 0.103100]`
   （两指在 `hand` 帧下分别位于 `y=+0.0427` / `-0.0427`，所以中点必然落在两指中线上）；
4. 姿态取与 `hand` 对齐（`quat = 1 0 0 0`），因为 `hand` 帧本身就是"`+z` 朝指尖、`+y` 沿开合方向"的正交基。
   可直接用的 site：

   ```xml
   <site name="tcp_panda" pos="0.000000 0.000000 0.103100" quat="1 0 0 0" size="0.005" rgba="0 1 0 1"/>
   ```

   若要和 ur5e 的 `grip_site` 同约定（`+x`=闭合轴、`-z`=接近轴），改用
   `quat="0 0.707107 0.707107 0"`（同一 `pos`）。

**各量的定义与推导方式**

* `closing_axis_tcp`：**全开构型**下 `finger[0]` 的 pad 中心 → `finger[1]` 的 pad 中心（`finger[0]` 是 ur5e 的
  `left`/panda 的 `finger_joint1`/piper 的 `link7`），单位向量，表达在 TCP 帧；
* `approach_axis_tcp`：`normalize(pad 中点 − 法兰 body 原点)`，单位向量，表达在 TCP 帧。
  这是**抓取最后一段下压的方向**（工具从腕部伸出去的方向），§5 的"顶抓"过滤用的就是它；
* `pad_center_offset_tcp`：`pad 中点 − TCP 原点`，在 TCP 帧下；
* TCP 相对法兰偏移：`|TCP 原点 − 法兰 body 原点|`，同时给出它在法兰帧下的分量。

⚠️ **符号坑（必须知道）**：ur5e 的 `grip_site` 与 piper 的 `tool_tip` 都落在两指**之外**
（分别超出 pad 中点 32.3 mm / 33.4 mm），所以"从 TCP 指向手指工作空间"的方向与 `approach_axis_tcp`
**反平行**（实测夹角 179.99°）。也就是说：这两个 site 的 `-z`（ur5e）/`-x`（piper）指向 pad，
而**下压方向是 site 的 `+z`/`+x`**。legacy 的 `top_down_gripper_rotation()` 正是让 site z 朝下来实现顶抓，
与本文的 `approach_axis_tcp` 完全一致。panda 的合成 TCP 定义在 pad 中点上，`TCP→pad 中点` 是零向量
（JSON 里 `tcp_to_pad_mid_axis_degenerate=true`），此时只能看 `approach_axis_tcp`。

---

## 4. 耦合机构：怎么正确驱动夹爪（不要直接写 qpos）

| 臂 | 耦合结构（实测 `neq/ntendon`） | 正确驱动方式 | 实测关节关系 |
| --- | --- | --- | --- |
| ur5e | 4 fixed tendon + 4 tendon equality（`neq=4 ntendon=4`） | 两个 position 执行器 `finger_1`(joint `finger_joint`)、`finger_2`(joint `right_outer_knuckle_joint`)，ctrl `(g, -g)`，然后 settle | `finger_joint=g`，`left_inner_finger_joint=-g/1.5`，`left_inner_knuckle_joint=g/5.25` |
| panda | 1 fixed tendon `split` + 1 joint equality（`neq=1 ntendon=1`） | 只发 `actuator8`（作用在 tendon 上）的 ctrl ∈ [0,255]，然后 settle | `finger_joint1=finger_joint2=0.04·ctrl/255` |
| piper | 1 joint equality（`neq=1`，`joint7=joint8`） | 只发执行器 `gripper`（joint7）的 ctrl ∈ [0,0.035]，然后 settle | `joint7=joint8=ctrl` |

⚠️ 一个容易踩的坑：**equality/tendon 约束是软约束，刚度取决于关节 armature**。
legacy 的 ur5e 场景有 `<default><joint damping="1.2" armature="0.01"/></default>`，
此时 4-bar 耦合误差 ~1e-4；如果去掉这个 default（armature=0），同样的 ctrl 下 tendon 长度可以偏离 0 达 0.25，
夹爪会"散架"（实测 `ten_length=[-2.66, -2.68, -0.18, -0.55]`）。本脚本沿用 legacy 的 default，
每个采样点都检查 `settle.converged`（三条臂全部 True），耦合残差写进 `opening_curve.coupling_check`：
ur5e 指间残差 ≤ **3.4e-5 rad**、tendon 长度残差 ≤ **1.23e-4 m**（目标是 0）；panda / piper 的
`|q(finger0) − q(finger1)|` 在全部采样点上都是 **0.0**（panda 的 tendon 长度是执行器指令的平衡值，不是残差）。

`closing_axis` 在开合过程中的稳定性（固定全开轴 vs 逐采样点自身轴的**直线**夹角）：
ur5e 0.031°、panda 0.000°、piper 5.936°。piper 偏大是因为两个手指 mesh 的 **AABB 中心并不镜像对称**，
连线的倾角会随开合变化；用开合曲线数值微分得到的真实靠近方向与全开连线差 1.43°，
所以 piper 建议直接用 `[-0.000007, -1.0, -0.0001]`（≈ `-y_tcp`）作为闭合轴。

---

## 5. 可达性（工作空间）

口径：在臂关节限位内**均匀随机采样 qpos**（`seed=0`，2 万组），FK 到 TCP 原点，统计到世界原点
（= 基座帧原点）的距离。这是**关节限位内的运动学包络**，不含自碰撞与灵巧性约束，所以比手册工作空间大。
"顶抓"子集 = `approach_axis_tcp` 在世界系中与竖直向下夹角 ≤ 35°。

| 臂 | 全姿态 min / 中位 / p95 / max | 顶抓(≤35°) 占比 | 顶抓 min / 中位 / p95 / max | 顶抓子集 TCP 高度中位 / 最大 |
| --- | --- | --- | --- | --- |
| ur5e | 0.0173 / 0.6382 / 1.0736 / **1.3072** m | 6.08% (1216) | 0.0173 / 0.6101 / **0.9511** / 1.0683 m | 0.161 / 0.744 m（全姿态）；−0.122 / 0.744 m（顶抓） |
| panda | 0.0245 / 0.8111 / 1.1958 / **1.2762** m | 9.32% (1864) | 0.0890 / 0.6603 / **0.9165** / 0.9941 m | 0.534 / 1.273 m（全姿态）；0.250 / 0.925 m（顶抓） |
| piper | 0.0078 / 0.6167 / 0.8404 / **0.8805** m | 12.19% (2437) | 0.0078 / 0.3166 / **0.5624** / 0.6541 m | 0.344 / 0.880 m（全姿态）；**−0.120 / 0.271 m（顶抓）** |

工作台尺寸该用"顶抓 p95"这一列：ur5e 0.951 m、panda 0.917 m、piper 0.562 m。
但 piper 的顶抓子集 TCP 高度几乎都在 0.27 m 以下（只有 0.6% 高于 0.20 m），所以它实际上不能做
"高桌面 + 竖直下抓"，选型时要额外看 `tcp_z_*` 字段。

---

## 6. 质量、执行器上限与载荷粗估

| 臂 | 总质量 | 臂 / 夹爪 / 夹爪运动件 | 肩关节上限 | 全臂最弱关节 | 载荷粗估（肩） | 载荷粗估（最弱关节） |
| --- | --- | --- | --- | --- | --- | --- |
| ur5e | 20.3595 kg | 19.6291 / 0.7304 / 0.1688 kg | `shoulder_pan` ±280 N·m | 同上 ±280 N·m | **36.39 kg** | 36.39 kg |
| panda | 17.4519 kg | 16.6919 / 0.7600 / 0.0300 kg | `joint1` ±87 N·m | `joint5` ±12 N·m | **11.58 kg** | 1.60 kg |
| piper | 3.9125 kg | 3.8625 / 0.0500 / 0.0500 kg | `joint1` ±20 N·m | `joint5` ±5 N·m | **3.86 kg** | 0.96 kg |

公式（脚本里 `mass.payload_estimate.formula`）：

```
r    = 0.6 * max_reach                      # 60% 水平臂展
m_max = tau_shoulder / (g * r),  g = 9.81   # 只算静态力矩
```

**这个数是上界，不能当规格用**：它忽略了臂自重（真机肩部力矩大部分用来托住自己的臂）、忽略了负载在其它关节上的分量、
忽略腕部力矩与摩擦。ur5e 还有额外一层问题：legacy 组合把臂执行器统一写成 `forcerange ±280 N·m`，
而上游 UR5e 的真实肩/肘力矩是 ±150 N·m、腕是 ±28 N·m（`assets/robots/robosuite/models/assets/robots/ur5e/robot.xml`
的 `torq_j*` motor ctrlrange）。用文档值重算同一个公式得到 **19.50 kg（肩 150 N·m）/ 3.64 kg（最弱 28 N·m）**，
即组合模型的 forcerange 把 ur5e 的载荷估算放大了 **1.87 倍**。要拿载荷结论做设计，请用文档值那一栏，或直接做动力学仿真。

---

## 7. home 位姿

| 臂 | 来源 | qpos | 实测 TCP 位姿 | 接近轴（世界） |
| --- | --- | --- | --- | --- |
| panda | keyframe `home` | `[0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]`（ctrl 尾项 255） | `[0.5545, 0, 0.521402]`，`quat=[0, 0.707141, 0.707072, 0]` | `[0, 0, -1]`（竖直向下） |
| ur5e | 本脚本 DLS IK（32 起点，取肘部最高解，`scale=1.0` 精确命中） | `[-0.302359, -1.702680, 1.520051, -1.388147, -1.570804, -0.301560]` | `[0.45, 0, 0.350001]`，pos 误差 5.6e-7 m、姿态误差 0.00026° | `[-0.000004, -0.000001, -1.0]` |
| piper | 本脚本 DLS IK（名义目标不可达 → 采样+位置精修的尽力解） | `[-0.013841, 1.446273, -1.498509, 0.024962, 1.128144, -0.556947]` | `[0.36, 0, 0.28]`（离名义目标 0.115 m） | `[0.550, 0.015, -0.835]`（离竖直 33.4°） |

IK 目标：TCP 落在基座前方 0.45 m、高 0.35 m，接近轴竖直向下、闭合轴沿世界 +x（`closing=+x` 只是选一个水平朝向）。
ur5e 一次命中；panda 不需要 IK（有 keyframe）；piper 在"工具朝下"约束下够不到，退让策略与残差都写在 JSON 的
`home.home_pose_kind / unreachable_reason / best_effort_scan` 里。

legacy ur5e 参考位姿 `Q_HOME_ELBOW_UP = [0.74, -1.30, 1.50, -1.76, -1.57, -0.83]`（`run_official_ur5e_robotiq_wall_stack.py:56`）
的实测结果：

```json
{"tcp_world_pos": [0.348552, 0.500045, 0.188072],
 "tcp_world_quat": [-0.000641, 0.707378, 0.706815, -0.005375],
 "approach_axis_world": [-0.00851, -0.006674, -0.999942],   // 离竖直 0.62°
 "pad_center_midpoint_world": [0.348826, 0.500263, 0.220334],
 "distance_from_base_origin_m": 0.637891,
 "at_servo_equilibrium": {"tcp_world_pos": [0.339483, 0.491764, 0.146006],
                          "offset_from_exact_qpos_m": 0.043822}}
```

pad 中点比 TCP（`grip_site`）高 32.3 mm、法兰比 TCP 高 207.5 mm——三者共线，方向由 `approach_axis_world` 给出。

---

## 8. 实测与文档不一致的地方

1. **piper 的"净开口 6.996 cm"方向反了（量值对）** —— `assets/robots/piper/README.md` 写
   "夹爪净开口 6.996 cm（两指内表面实测）"，但没说 ctrl 方向。实测：**ctrl=0.035 才是全开**
   （内表面间距 6.935 cm、两指点云最小距离 **7.000 cm**，与 6.996 cm 相符），**ctrl=0 是闭合**
   （内表面间距 −0.063 cm、最近点距 0.0 m）。按 ctrl=0 理解会把夹爪当成"全开 7 cm"，实际它已经合上。
   我信任实测值（`inner_surface_gap_m` 与 `min_surface_distance_m` 两条独立口径互相印证）。
   另外 README 说"能夹住 TASK1 当前约 3.5–4.5 cm 的石头"，从曲线看 ctrl≈0.0179–0.0229 之间对应 3.5–4.5 cm，
   机制上成立。
2. **ur5e 组合模型的臂力矩上限被放大 1.87 倍** —— legacy 组合给 `ur_pos_*` 写死 `forcerange ±280 N·m`，
   而上游 `robot.xml` 的 `torq_j*` 是肩/肘 ±150 N·m、腕 ±28 N·m。任何用这个模型做的载荷/力矩结论都会偏乐观
   （§6 的 36.39 kg vs 19.50 kg）。这是**模型与上游文档不一致**，不是 README 写错。
3. **panda README 的"臂展上界 1.2383 m"需要补一个前提** —— 实测复现：把两个 finger slide joint 固定在
   `qpos0=0`（等价于只采 7 个臂关节）时 **1.238367 m**，与 README 完全一致；但**手指全开（q=0.04）时是
   1.258625 m**，9 个关节一起随机采是 1.256458 m。手指沿半径方向能多伸 40 mm，所以这个数只有在"手指闭合"口径下才成立。
   另外本脚本主指标（TCP = pad 中点）的半径上界是 1.276241 m，与 README 的"手指 body 原点"口径不同。
4. **panda 没有 TCP site，且 home 位姿存在 6.7 mm 重力下垂** —— README 明确说没有 `tool_tip`（实测 `nsite=0`，一致），
   所以本脚本给出合成 TCP（`hand` 帧 `[0,0,0.1031]`，见 §3）。额外发现：keyframe `home` 的 qpos 是纯运动学的，
   按 keyframe 的 ctrl 实际稳定后 TCP 会低 6.725 mm；ur5e 的 legacy home 更明显，低 43.822 mm。
   文档没有写这个偏差，规划器如果混用"目标 qpos"和"实际 TCP"会引入厘米级误差。
5. **piper 做不了"0.35 m 高、工具竖直向下"的抓取** —— README 只说"没有抓取规划、IK 或训练策略"，
   没提这条硬约束。实测：在接近轴朝下 ≤35° 的 2437 个采样里 TCP 高度中位数 −0.12 m、最大 0.271 m
   （只有 0.6% 高于 0.20 m）；把角度收紧到 ≤5° 时，5 万组采样里 TCP 最高只有 0.121 m。
   所以 piper 的 home 只能给出 33.4° 斜下压的尽力解。规划器 / 工作台高度必须按这个来。
6. **ur5e 的"32 mm"注释是对的**（不是不一致，列在这里做交叉确认）：`grip_site_from_pad_center()` 注释说
   fingerpad 中心在 `grip_site` 局部 −z 方向约 32 mm，实测 `[0.000003, 0, −0.032265]`（32.265 mm）；
   `top_down_gripper_rotation()` 注释说手指开合方向是 `grip_site` 局部 x 轴，实测闭合轴 `[1.0, −0.0008, 0]`。

---

## 9. JSON 字段索引

每个 `reports/arm_geometry_<arm>.json` 都含：`model`（规模、运动链、质量明细、来源与组合方式）、
`arm_joints`（关节名/范围/qpos/dof 地址/执行器 ctrlrange、forcerange、kp、kv）、`tcp`（TCP 定义与 pos/quat/site 写法）、
`opening_curve`（每个采样点的 raw ctrl、指关节 qpos、pad 中心、三种间距、settle 诊断 + 两套线性拟合 + 0.5 倍开口点）、
`frames`（§3 的四个量 + 世界系对照 + 轴稳定性）、`reach`（全部/顶抓的距离与高度统计）、
`mass`（质量拆分、执行器上限、载荷粗估与文档值对照）、`home`（keyframe 或 IK 结果、legacy 位姿及其伺服平衡态）、
`documentation_checks`（本文 §8 的每条逐项对照）、`generated_by`（脚本、解释器、命令、mujoco/numpy 版本、种子）。
