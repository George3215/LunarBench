# 观察 Qwen 的推理与动作、并诊断抓取

一条命令：

```bash
cd MoonSim/tasks/task2_stack
bash scripts/run_qwen_debug.sh                 # ur5e，1 次抓取，开 MuJoCo 窗口
PLACEMENTS=0 bash scripts/run_qwen_debug.sh    # 完整 10 次
NO_VIEW=1 bash scripts/run_qwen_debug.sh       # 无显示器时不开窗口
TASK2_VLM_TRACE_FULL=1 bash scripts/run_qwen_debug.sh   # 连完整 prompt 也打印
```

先确保 vLLM 在跑（`--max-model-len 16384`，4k/8k 会因 160 条候选而 400）：

```bash
vllm serve /home/lry/MoonUnrealEnv/Asset/model/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen3-VL-8B --host 0.0.0.0 --port 3001 \
  --gpu-memory-utilization 0.95 --max-model-len 16384 --max-num-seqs 4 \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 1024
```

## 输出怎么读

| 前缀 | 含义 |
| --- | --- |
| `[VLM>]` | Qwen 收到的请求：候选数、图像数、相机、prompt 字符数 |
| `[VLM<]` | **Qwen 的原始回复**——它的"思考"（JSON 及其前面的说明文字） |
| `[VLM=]` | 解析出的动作：HIGH 给 (石头, 槽位, 目标)；LOW 给 TCP 目标/夹爪/done |
| `· pick stepN` | 执行器每一步实际下发的 TCP 目标、夹爪指令、伺服残差、**实测开口** |
| `[抓取诊断]` | 每次抓取尝试的合拢量、实测开口、接触数、拾起高度差、石头位移 |

日志同时写到 `outputs/qwen_debug_<时间>.log`。

## 本次改动

1. **`--view` 以前是坏的**：`viewer` 建了却从不 `sync()`，窗口不刷新。现在
   `Executor` 持有 viewer，并在 `settle` / `servo_to` / `servo_translate` 里按
   ~30 Hz 调 `viewer.sync()`。
2. **`vlm.py`** 增加 `TASK2_VLM_TRACE` 开关，打印请求 / 原始回复 / 解析出的动作。
3. **`execution.py`** 增加 `TASK2_TRACE`，逐步打印低层动作，并在每次抓取后打印诊断。
4. **修掉一个误导性诊断**：`controller.measured_opening()` 量的是
   `profile.gripper.pad_geoms`（Robotiq 旧指垫），但场景用的是**自造平行夹爪**
   （`built.pad_geoms = pg_jaw_geom_*`，执行器驱动 `pg_jaw_act_0/1`）。旧指垫仍在
   模型里但没人驱动，读数一直停在 ~128 mm，看起来像"夹爪没闭合"。执行器现在用
   自己那份 `_pad_geom_ids`（`Executor.measured_opening()`）。

## 首次诊断已经看到的两种失败模式

用 ur5e + `--policy vlm` 跑 1 次（Qwen 正常决策，`model_choice=0`）：

**尝试 1**（`close_fraction=0.00` → 夹爪指令 2.0 mm）
```
stone_moved=[-20.0, +85.7, -4.2] mm      # 石头被推走 8.6 cm
contacts=1  lifted_z < pick_z  lift_gain=-4.2mm
```
→ 合拢过程把石头**推走**了，指垫最后夹在空档里。

**尝试 2**（`close_fraction=0.20` → 指令 13.3 mm）
```
stone_moved=[-0.0, +0.1, -6.7] mm        # 石头没被推动
contacts=3  lift_gain=-6.7mm             # 夹住了，但抬不起来
```
→ 指垫确实接触（3 个接触点），但**提不起来**：夹持力/摩擦不足，石头打滑。

`task2.yaml` 里那条注释也印证了摩擦很边际：
“实测有效摩擦随步长改善（0.0015 -> mu 0.082；0.0005 -> >0.10），夹持很边际”。

## 建议的下一步实验

对着 viewer 逐个改，一次只动一个量：

1. `execution.grasp_close_fraction`：现在默认 **0.0**（= 最紧），执行器注释却说
   “0.45 能到 ~100 N 并稳定吊住石头”。先试 0.45 看接触力与抬升。
2. `robot.timestep`：摩擦随步长改善，0.0005 已比 0.0015 好；可试更小步长。
3. 石头的摩擦系数 / 质量：`stone_stack/moonsim_rocks.py`、`scene.py` 里的摩擦设置。
4. `execution.side_standoff_m` / `nest_align`：解决"合拢把石头推走"。
