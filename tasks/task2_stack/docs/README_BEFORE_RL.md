# TASK2：QwenVL 控制机械臂堆石

命令行只保留三个参数：

| 参数 | 用途 |
| --- | --- |
| `--config PATH` | 选择 YAML；省略时读取 `task2.yaml` |
| `--mode run\|camera\|list` | 运行、保存相机图、列出机械臂和策略；默认 `run` |
| `--view` | 运行时打开 MuJoCo 窗口 |

机械臂、策略、石头数量、层数、缩放、seed、trial 预算和输出路径都在 YAML 中设置。
旧的 `stack ur5e`、`--arm`、`--policy`、`--report` 等命令行写法已移除。
`task2.sh` 直接转发这三个参数，没有第二套参数解析。

## 运行

上游 UR5e/Robotiq、可达料区、全部十块石头、4+3+2+1、失败复盘：

```bash
bash /home/lry/MoonUnrealEnv/task2.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/upstream_qwen_trials.yaml --view
```

该配置默认三次 trial，每次最多 400 次工具调用。无窗口时去掉 `--view`。
两次短 trial 的链路验证使用 `configs/upstream_trials_smoke.yaml`。
不指定配置时仍使用 `task2.yaml` 中的 MoonSim/scripted 设置。

列出机械臂和策略：

```bash
bash /home/lry/MoonUnrealEnv/task2.sh --mode list
```

保存初始相机图，不请求模型：

```bash
bash /home/lry/MoonUnrealEnv/task2.sh --config /home/lry/MoonUnrealEnv/MoonSim/tasks/task2_stack/configs/upstream_qwen_trials.yaml --mode camera
```

图片默认存到 `outputs/camera/`，也可在 YAML 的 `output.save_frames` 设置目录。
启动和拍照都不执行相机或 IK 自检。

## 配置和修改位置

[task2.yaml](../task2.yaml) 是唯一默认配置来源，`--config` 文件只需写要覆盖的字段。例如：

```yaml
robot:
  arm: ur5e
  stones: 10
  stone_pool: 10
  courses: [4, 3, 2, 1]
scene:
  layout: upstream
  supply_layout: reachable
policy:
  name: qwen-direct
  direct:
    trials: 3
    max_steps: 400
output:
  report: /tmp/task2_result.json
  log: /tmp/task2.log
```

| 修改内容 | 文件 |
| --- | --- |
| 石头位置、间距和工作区 | `task2.yaml` 的 `scene.supply_*` |
| 模型、传输方式、图像、历史和预算 | `task2.yaml` 的 `policy.direct` |
| 4+3+2+1 提示词 | [prompts/direct_stack.md](../prompts/direct_stack.md) |
| Qwen 请求、对话和 trial 复盘 | [stone_stack/policy/direct.py](../stone_stack/policy/direct.py) |
| EE pose、夹爪、计划和失败记录工具 | [stone_stack/direct_control.py](../stone_stack/direct_control.py) |
| trial 恢复、检查与经验回流 | [stone_stack/trials.py](../stone_stack/trials.py) |
| 上游场景、料区布置 | `stone_stack/upstream_scene.py` |

## 石头准备工具

生成和导入统一由 [stone_stack/tools/stones.py](../stone_stack/tools/stones.py) 负责。
`run.py` 与上游场景只调用 `load_stones()`，不实现生成或导入逻辑。

| `stones.source` | 来源 | 设置 |
| --- | --- | --- |
| `procedural` | 程序生成 | `robot.rock_style` 选 box/paper/rough/natural |
| `moonsim` | MoonSim 月岩 USDZ 库 | `stones.directory` 为空时用本地默认月岩目录 |
| `mesh` | 仓库中的 OBJ/STL/PLY/GLB/GLTF | 设置目录和匹配规则，或直接列出 paths |

例如，从自己的资产目录导入十块石头，在任务 YAML 中添加：

```yaml
stones:
  source: mesh
  directory: /absolute/path/to/rock_meshes
  pattern: "*.obj"
  unit_scale: 0.001   # 源坐标为毫米；源坐标为米则填 1.0
  density: 2200.0    # kg/m³
```

目录文件按名称排序；也可以清空 directory，并在 paths 中按顺序列出文件。
通用网格导入会居中、转凸包、换算单位与质量，保留源坐标轴，不保留源材质/纹理。
MoonSim USDZ 使用已有的月岩导入流程，会按堆叠石头的尺寸先验对齐和缩放。
`scene.upstream_report` 有值时，优先恢复该报告选中的石头。

其他脚本可以直接调用：

```python
from stone_stack.tools.stones import import_stones, load_stones

stones = load_stones(config, scale=1.0, count=10)
# 或独立导入；无需运行 run.py、启动模型或打开窗口。
stones = import_stones(["/absolute/path/rock.obj"], unit_scale=1.0, density=2200.0)
```

这是场景准备工具，trial 内仍不能创建或重置石头。

窗口在 `run.py` 中直接调用 `mujoco.viewer.launch_passive()`，没有额外的 viewer 包装函数。
运行输出统一使用 Python `logging`，`output.log` 可以指定日志文件。

## 控制与 trial

```text
相机/本体观测 → Qwen 计划抓取和放置 EE pose → 短段运动 → 新观测
                                    ↓
                        记录失败 → trial 末复盘
                                    ↓
                   恢复同一初始场景 → 下一 trial 加载经验
```

工具为 `plan_stone`、`move_tcp_delta`、`move_tcp`、`record_failure`、`finish`。模型控制运动全过程；
trial 内不调用脚本抓取器，不重定位石头。每次运动目标最多平移 5 cm、旋转 0.35 rad，
`move_tcp_delta` 接收模型选择的世界系局部位移并保持姿态；`move_tcp` 接收绝对目标，姿态填 `null` 表示保持实测姿态。两者共用门控，实际运动仍有伺服跟踪误差。工具只做一次结构化动作解码，不抽取自然语言中的 JSON，不自动重试。
默认 `json_schema` 传输支持当前本机 vLLM；原生 `tool_calls` 需要服务启用工具解析器。

每次运行的 `outputs/direct_<时间戳>/` 保存场景、trial 状态、图像、动作、
失败与复盘。`trial_002/context.json.previous_trial_lessons` 来自第一轮 `review.json`。
`finish` 只请求检查，不能结束 trial；Python 在控制观测边界检查本地几何/接触条件，成功或预算耗尽才终止；进程正常退出不代表堆叠成功。

上游来源、坐标、工具契约、trial 判据和局限见 [UPSTREAM_DIRECT.md](../docs/UPSTREAM_DIRECT.md)。
已有真实短 trial 验证了可达布局、控制与复盘回流，尚未证明完整抓取或堆叠成功。
证据见 [reachable_trials_validation.json](../reports/reachable_trials_validation.json)。

旧策略和上游独立复现脚本保留供对照；`docs/LEGACY_MOONSIM.md`、`STATUS.md` 等旧记录中的
命令属于历史接口，不应用于当前 `run.py`。

控制不动或请求失败时，先看日志中的 `[direct N result]`：`validation_rejected` 没有执行，`executed` 已推进物理。
请求超限等 HTTP 错误现在保留服务端响应正文；默认只保留两轮动作历史，旧观测压缩，当前计划和最近失败随新观测传回。
完整轨迹仍在磁盘；上下文 token 用量在 `trajectory.jsonl` 的 call.usage 中。完整运行验证见 `reports/motion_context_validation.json`。

上游正式配置的 Qwen 相机为 `eye_in_hand`、`top`、`front`；额外 `wrist` 已移除。`front` 从工作区 -Y 侧水平沿 +Y 正视墙面，相机位置可在 `task2.yaml` 的 `scene.front_camera_offset` 修改。
