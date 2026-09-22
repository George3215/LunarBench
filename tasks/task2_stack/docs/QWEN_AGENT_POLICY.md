# TASK2 `qwen-agent`：把 GPT-as-Policy 换成 QwenVL

记录时间：2026-09-22。目标是把 `policy/GPT-as-Policy` 的 **agent 式策略**接进 task2，
并把背后的 GPT-6 换成本机 vLLM 上的 **Qwen3-VL-8B**。

## 1. 为什么不能只改一个模型名

`GPT-as-Policy` 的"策略"不是一个 API 调用，而是 **Codex CLI 的 agent**：

```text
skill/run.py ──JSON-RPC──▶ codex app-server ──▶ GPT-6 (responses API)
      │                                              ▲
      └── 三个动态工具 (robodojo_start/pi05_infer/robodojo_execute) ──┘
```

模型身份被三处硬编码：`robodojo/settings.py`（`MODEL/EFFORT/WIRE_API`）、
`codex_backend/profiles.py`（`config_text()`）、`codex_backend/validate.py`（fail-closed 校验）。
而且它走的是 `/v1/responses`。本机 vLLM 0.29 **同时**提供 `/v1/chat/completions`
与 `/v1/responses`，所以对接的关键是让 Codex 用 Responses API（见下）。

同时 `GPT-as-Policy` 是给 RoboDojo/RoboLab 写的长连接控制器，**不控制 task2**。
task2 的执行器是同步调用 `HighLevelPolicy.decide()` / `LowLevelPolicy.step()`
（见 `stone_stack/policy/base.py`），所以正确做法是给 task2 写一座桥。

## 2. 本仓库的做法：`qwen-agent`

新增 `stone_stack/policy/agent.py`，把 Codex 当**每步一次**的高层规划器：

```text
task2 run.py ──▶ CodexAgentHighLevel.decide(observation, candidates)
                     │  写 3 路相机 JPEG + 候选文本
                     ▼
                codex exec -i ... -c model_providers.vllm=...   （一次性子进程）
                     │  provider 指向本机 vLLM（Responses API）
                     ▼
                Qwen3-VL-8B（vLLM, tool-calling）
                     │  最终消息 {"choice": <int>, "reason": "..."}
                     ▼
                Decision（选中的 (stone, slot)）→ 脚本低层执行
```

- **高层** = Codex + QwenVL；**低层**沿用 `ScriptedLowLevel`，避免每个控制周期起子进程。
- 失败路径与 `vlm.py` 一致且可观测：子进程/HTTP 失败、JSON 解析失败、choice 越界
  都计入 `stats`，先修复重试一次，再回退 `fallback`（脚本高层），最后退化为
  "取 score 最大的候选"。只有 `strict: true` 才外抛 `CodexAgentError`。
- Codex 用 `--ignore-user-config --ephemeral --skip-git-repo-check -s read-only`，
  不污染你的 `~/.codex`，也不让模型动文件系统。

关键文件：

| 文件 | 作用 |
| --- | --- |
| `stone_stack/policy/agent.py` | `CodexAgentHighLevel`、`build_codex_command` |
| `stone_stack/policy/__init__.py` | 注册 `qwen-agent`，`_section()` 读 `policy.<key>` |
| `stone_stack/task_config.py` | `policy.agent` 默认值 |
| `task2.yaml` | `policy.agent` 配置段 |
| `run.py` | `--list-policies` 增加说明 |
| `tools/test_agent_policy.py` | 离线测试（假 codex，无需 GPU） |
| `Asset/model/qwen3VL.sh` | vLLM 加 `--enable-auto-tool-choice --tool-call-parser hermes` |

> 顺带修正：原 `_vlm_config` 读的是顶层 `config["vlm"]`，而 YAML 把配置放在
> `policy.vlm` 下，导致 YAML 里的 VLM 参数实际被忽略。新的 `_section()` 按
> `policy.<key>` 读取，同时兼容旧的直接传法。

## 3. 怎么跑

```bash
# 1) 起 vLLM（Qwen3-VL-8B + tool calling），确认 /v1/models 可访问
bash assets/model/qwen3VL.sh
curl -s http://127.0.0.1:3001/v1/models | head

# 2) 离线测试（不需要 GPU/服务）
cd MoonSim/tasks/task2_stack
PYTHONPATH= MUJOCO_GL=egl \
  /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python tools/test_agent_policy.py

# 3) 真跑：qwen-agent 只做高层
bash task2.sh stack ur5e --policy qwen-agent --courses 3,2,1
```

### 必须注意：把策略超时调大

Codex 每步要起一个子进程、加载配置、做一次多模态对话，耗时通常 **几十秒**，
而 `task2.yaml` 默认 `execution.policy_timeout_s: 25.0`，会直接判定 `policy_timeout`。
用 `qwen-agent` 时请把它调到 **300** 左右（或在命令行配置里覆盖）。

### 可选：把 GPT-as-Policy 自己的 RoboDojo 后端也换成 QwenVL

如果你还想让 `hybrid_rollout/robodojo` 的 rollout 用 Qwen（而不是走本桥），
Codex 侧需要三处改动（本仓库未改，因为 task2 不用它）：

1. `codex_backend/profiles.py`：加一个 API profile，`base_url` 指向 vLLM，
   `config_text()` 里 `wire_api = "responses"`、`model = "Qwen3-VL-8B"`，并去掉
   `model_reasoning_effort`。
2. `robodojo/settings.py`：让 `MODEL / EFFORT / WIRE_API / PROVIDER` 由 profile 决定；
   `EFFORT` 对 Qwen 应为 `None`。
3. `codex_backend/validate.py` 与 `skill/run.py`：放宽对 `model_reasoning_effort`
   和 `response["reasoningEffort"]` 的等值校验（Qwen/vLLM 没有 reasoning effort）。

## 4. 对接 Codex 时踩到的两个硬约束（已修）

1. **`wire_api` 必须是 `responses`，不能用 `chat`。** 本机 `codex-cli 0.135.0` 直接报
   `wire_api = "chat" is no longer supported`。初版配置按 chat completions 写是错的。
   幸运的是 **vLLM 0.29.0 自带 Responses 端点**：
   `vllm/entrypoints/openai/responses/api_router.py` 提供 `POST /v1/responses`，
   由 `entrypoints/generate/api_router.py` 默认挂载，**和 `/v1/chat/completions` 同端口**，
   所以不需要 LiteLLM 之类的转换网关，provider 直接指 `http://127.0.0.1:3001/v1` 即可。
   实测把 `wire_api="responses"` 交给 codex，它确实向 `…/v1/responses` 发起请求
   （用桩服务验证时错误是 `error sending request for url (…/v1/responses)`，
   说明配置已被接受、请求已发出）。

2. **`CODEX_HOME` 必须可写。** 不设置时 codex 的 in-process app-server 会以
   `failed to initialize in-process app-server client: Read-only file system (os error 30)`
   崩溃。`agent.py` 现在每次决策在 workdir 下建一个私有 `codex_home/` 并注入
   `CODEX_HOME`，配合 `--ignore-user-config --ephemeral` 做到隔离且不留会话文件。

## 5. 已验证 / 未验证

- ✅ `tools/test_agent_policy.py`：16/16（命令拼装、JSON 解析、停机、越界回退、
  子进程失败回退、JPEG 落盘与魔数）。
- ✅ `tools/test_vlm_policy.py`：96/96（确认 `_vlm_config` 修正无回归）。
- ✅ `tools/test_kinematics_tools.py`：36/36。
- ✅ `policy/__init__.py` / `run.py` / `task_config.py` 编译与 `qwen-agent` 装配。
- ✅ `codex exec` 接受 `model_provider=vllm` + `wire_api=responses` 配置并发出请求
  （用 OpenAI 兼容桩服务验证，桩没有 `/v1/responses`，所以止步于连接层）。
- ❌ 未做真机端到端：当前 `nvidia-smi` 不可用（驱动无法通信），起不了 vLLM。
- ❌ 未验证 vLLM 的 `/v1/responses` 是否完全满足 Codex 的期望（工具调用、
  图像 `input_image`、流式事件格式）——**这是接入后最需要第一时间验证的点**。
- ❌ 未验证 Qwen3-VL-8B 在 Codex agent 协议下的实际工具调用质量；这是模型能力问题，
  与本次接线无关。
- ⚠️ 与策略无关的前置阻塞：`STATUS.md` 记录 task2 **物理抓取尚未成功**
  （夹爪能合上、抬起时石头不跟随）。在它解决之前，任何策略都不会有堆叠产出。

## 6. 实跑记录：vLLM 上下文必须 ≥ 16k（2026-09-22）

用 UR5e + `--policy vlm` 真跑后确认：**`--max-model-len 4096/8192 都不够**，
高层请求会被 vLLM 以 HTTP 400 拒绝，策略全程回退脚本，QwenVL 一次都没决策。

原因（实测抓包 + tokenizer 计数，`tools/` 下无此脚本，用临时 monkeypatch 取得）：

| 请求 | 文本 token | 图像 | 合计 |
| --- | --- | --- | --- |
| 高层 `decide()` | **9815**（`CANDIDATES (160)` 占 16.7k 字符） | 3×640×480 ≈ 900 | ≈ **10715** |
| 低层 `step()` | 662 | 同上 ≈ 900 | ≈ 1562 |

`CANDIDATES` 是 **16 块石头 × 10 个槽位 = 160 条**候选，把高层 prompt 撑到 ~10.7k token。
注意 vLLM 报错里的 "prompt contains at least N input tokens" 中 `N = max_model_len − max_tokens`，
**不是真实 prompt 长度**，别被误导。

可用启动参数：

```bash
vllm serve ... --gpu-memory-utilization 0.95 --max-model-len 16384 \
  --max-num-seqs 4 --enable-prefix-caching --enable-chunked-prefill \
  --max-num-batched-tokens 1024
# KV cache 23,808 tokens，16k 并发 1.45x，实测可跑
```

改完后高层决策正常：`model_choice=0 ... latency≈0.8–2.0s`，`images=3`，不再是 fallback。

**但 `placed 0/10` 仍未改善**：每次都是抓取失败（`lift_gain≈-0.007`，石头不随夹爪抬起），
即 `STATUS.md` 里那个与模型无关的物理抓取问题。**QwenVL 控制链路已通，堆叠产出卡在抓取物理。**
