# Go2 爬楼梯策略（go2_rl_gym MoE-CTS）

本目录只放一份**第三方预训练策略**，用来给 TASK3 的 Go2+Piper 提供能上台阶的运动控制器。

| 项 | 值 |
| --- | --- |
| 上游仓库 | [wty-yy/go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) |
| 上游提交 | `30e74dc507bec7a642a8c98be26081f2c6f0822d` |
| 上游文件 | `deploy/pre_train/go2/go2_moe_cts_high_slope_thre_164k_0.6715.pt` |
| 本地文件 | `policy/go2_moe_cts_high_slope_thre_164k_0.6715.pt` |
| SHA256 | `9d9ad783a1017b6eced5984eb95279cc5b36db8cc84d21e646f46ba2a8023d9d` |
| 许可 | MIT（上游新增部分）；原文见 `policy/LICENSE` |
| 训练项目 | go2_rl_gym `go2_moe_cts_high_slope_thre`，16.4 万次迭代，上游自报 RoboGauge 0.6715 |

## 部署契约（不要改这些数）

观测 45 维，**纯本体感知，没有高度图**：

```text
[0:3]    基座角速度 * 0.25
[3:6]    投影重力
[6:9]    速度指令 * [2.0, 2.0, 0.25]
[9:21]   (腿部关节角 - 默认角) * 1.0     12 维
[21:33]  腿部关节角速度 * 0.05
[33:45]  上一帧动作
```

关节顺序是 `FL_hip, FL_thigh, FL_calf, FR_*, RL_*, RR_*`，和本仓库 Go2 系 MJCF 的 MuJoCo 顺序一致，**不需要重排**。上游 `go2_rl_gym` 的 `model_joint_names` 与 `mujoco_joint_names` 本来就相同，本机 rollout 也证实了这一点（另一套 Isaac 顺序直接走不起来）。

物理步长 0.002 s、控制抽取 10 步 → 策略 50 Hz；`kp=20`、`kd=0.5`、`action_scale=0.25`、默认角 `[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]`。这些值写在 `tasks/task3_reobservability/controllers.py` 的 `Go2RlGymController` 里。

## 本机实测（本仓库的 Go2+Piper 模型，不是上游的纯 Go2）

| 场景 | 结果 |
| --- | --- |
| 平地，指令 1.0 m/s | 0.88 m/s，基座高度稳定在 0.327 m，`upright_z = 1.000` |
| 十级楼梯（0.15 m 高 / 0.30 m 深） | 从 x = −2.0 m 走到 x = +2.99 m 平台上（基座 z 从 0.36 升到 1.81），全程未摔倒 |

上游用这个检查点在随机地形（等级 7–9，含 0.20–0.23 m 台阶）上做过 MuJoCo sim2sim；本仓库的楼梯高度在它验证过的范围内。

## 我们改了什么 / 没改什么

- 策略文件**逐字节拷贝**，没有重训练、微调、蒸馏或量化。
- Go2+Piper 比纯 Go2 多一条 Piper 臂：策略只输出 12 个腿部动作，**六个臂关节不在策略里**，由位置伺服固定在零位（本机实测用 `kp=20 / kd=0.5`，与腿部同一组增益）。臂的存在没有阻止策略行走或爬楼。
- TASK3 只提供地形、速度指令和臂关节保持。

## 重新获取权重

权重 5.2 MB，`.gitignore` 忽略 `*.pt`，所以新克隆的 checkout 里没有这个文件。任选一种方式补齐，然后核对 SHA256：

```bash
# 方式一：只取这一个文件
curl -L -o assets/robots/go2_rl_gym/policy/go2_moe_cts_high_slope_thre_164k_0.6715.pt \
  https://raw.githubusercontent.com/wty-yy/go2_rl_gym/30e74dc507bec7a642a8c98be26081f2c6f0822d/deploy/pre_train/go2/go2_moe_cts_high_slope_thre_164k_0.6715.pt

# 方式二：完整克隆后拷贝
git clone --depth 1 https://github.com/wty-yy/go2_rl_gym /tmp/go2_rl_gym
cp /tmp/go2_rl_gym/deploy/pre_train/go2/go2_moe_cts_high_slope_thre_164k_0.6715.pt \
   assets/robots/go2_rl_gym/policy/
cp /tmp/go2_rl_gym/LICENSE assets/robots/go2_rl_gym/policy/LICENSE
```

```powershell
(Get-FileHash assets\robots\go2_rl_gym\policy\go2_moe_cts_high_slope_thre_164k_0.6715.pt -Algorithm SHA256).Hash
# 期望 9D9AD783A1017B6ECED5984EB95279CC5B36DB8CC84D21E646F46BA2A8023D9D
```

## 不能声称的东西

- 没有在真机 Go2+Piper 上验证过。
- 没有验证月球低重力步态。
- 没有验证边走边操作机械臂。
- 上游检查点的再分发权以 MIT 许可为准，本目录不做额外授权。
