# QwenVL 与 Codex 的实测连接

2026-09-22，本机 Codex CLI 0.135.0，Qwen3-VL-8B，http://127.0.0.1:3001/v1。
在独立临时 CODEX_HOME、只读工作目录、禁用 shell 工具的条件下实测：

- 自定义 provider 使用 `wire_api="responses"`，文本请求返回 `QWEN_CODEX_OK`。
- `-i` 传入真实 trial 的 top 图片，Qwen 返回场景描述，证明图片输入链路可用。
- 描述出现错误的 “green laser beam”，连接可用不代表视觉判断或抓取能力正确。

证据：`reports/qwen_codex_probe.json`。未将 Codex 作为正式 qwen-direct 后端，未验证 Codex 动态动作工具、裁剪或持久 agent rollout。
当前正式入口仍由 DirectPolicy 直接调用 Qwen，StoneTools 执行白名单动作，Python 保存计划、失败与跨 trial 笔记。

自定义 provider 的核心配置：

```toml
model = "Qwen3-VL-8B"
model_provider = "qwen"
[model_providers.qwen]
name = "qwen"
base_url = "http://127.0.0.1:3001/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
```

Codex 外壳可以保留看图、读文件等能力，但不能把普通 shell/任意写文件权限直接交给在线 Qwen。
只读沙箱也不等于禁止网络或第二条仿真连接。若进一步接入，应由 host 注册受限工具：
仅当前 trial RGB/允许的文本读取、图像裁剪、数值计算、host 代写 NOTES；禁止源码/配置/证据写入及任意代码执行。
这些辅助能力不产生或替换动作目标。当前未提供通用计算/裁剪工具，不把连接验证冒称完整工具迁移。

本次尝试读取 OpenAI 官方配置文档 https://developers.openai.com/codex/config-reference/ 时返回 Forbidden，
因此上述兼容性结论依据本机真实 CLI 请求，不声称官方保证所有自定义模型或工具兼容。
