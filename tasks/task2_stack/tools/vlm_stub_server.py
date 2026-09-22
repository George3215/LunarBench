"""VLM 策略测试用的**桩服务**：一个最小可用的 OpenAI 兼容 HTTP 服务（纯标准库）。

它不推理、不加载模型，只做三件事：

1. **扮演服务端**：实现 `POST /v1/chat/completions` 与 `GET /v1/models`，
   接受标准 OpenAI vision 消息体（见下），按脚本返回内容 / HTTP 错误 / 慢响应。
2. **记录请求**：每个请求的 prompt 文本、图像张数、每张图 base64 解码后的字节数、
   JPEG 魔数、时间戳等，全部进内存列表，可用 `GET /__stats` 取回或写到 JSON 文件。
3. **可编排**：用 `--script FILE` 预置响应队列，或运行时 `POST /__script` 换队列，
   于是测试可以让它先回合法 JSON、再回围栏 JSON、再回垃圾、再回 500、再回慢响应。

## 接受的请求体（OpenAI 兼容子集）

```json
{
  "model": "Qwen3-VL-8B",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}
    ]}
  ],
  "max_tokens": 256,
  "temperature": 0.0,
  "stream": false
}
```

`content` 既可以是字符串，也可以是 `{"type": "text"|"image_url"}` 组成的列表；
两种都会被记录（文本拼接进 `prompt_text`，图像解码后统计字节数与魔数）。
`stream: true` 不支持（桩服务只回非流式响应，遇到时仍按非流式返回）。

## 响应脚本格式

`--script FILE` 或 `POST /__script` 的 body：

```json
{
  "responses": [ <spec>, ... ],
  "fallback": <spec>,          // 可选：队列用完后一直用它
  "repeat_last": false         // 可选：队列用完后重复最后一个 spec
}
```

也可以直接给一个数组 `[<spec>, ...]`。单个 `<spec>`：

| 字段 | 含义 |
| --- | --- |
| `{"content": "..."}` | 200，助手文本就是 `content` |
| `{"status": 500, "body": {...}}` | 返回该 HTTP 状态码；`body` 默认 `{"error": ...}` |
| `{"delay_s": 2.0, "content": "..."}` | 先 sleep 再回，用来触发客户端超时 |
| `{"raw": {...}}` | 直接用这个 JSON 当响应体（替代默认的 choices 包装） |
| `{"model": "..."}` | 覆盖响应里的 model 字段 |

队列为空且没有 `fallback` / `repeat_last` 时，使用 `--default-content`（默认是一句
合法 JSON，方便手工 curl 试跑）。

## 其它端点

- `GET  /__stats` ：`{"n_requests": n, "requests": [...], "queue_remaining": k, ...}`
- `POST /__script`：替换响应队列（body 同上）
- `POST /__reset` ：清空已记录请求（可选 `{"script": ...}` 顺带换脚本）
- `POST /__dump`  ：把统计写到 JSON，body `{"path": "..."}`（缺省用 `--stats-out`）
- `POST /__shutdown`：优雅退出（测试收尾用）

`--port 0` 时监听临时端口，并在 stdout 打印一行
`STUB_READY {"host": "...", "port": 12345, "base_url": "http://..."}`（已 flush），
测试读到这一行就能拿到真实端口。`--ready-file PATH` 可以同时落一份 JSON。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

#: 默认回复：一句合法 JSON，方便不写脚本时手工试跑。
DEFAULT_CONTENT = '{"choice": 0, "reason": "stub default reply"}'

#: 记录列表默认上限（防止长时间运行时内存无限增长）。
DEFAULT_MAX_RECORDS = 500

#: JPEG 文件头（SOI + APP0 起始），用来验证"这真的是张 JPEG"。
JPEG_MAGIC = b"\xff\xd8\xff"


def _b64_payload_size(data_url: str) -> dict[str, Any]:
    """把 data URL 解码，返回字节数、sha256、魔数是否匹配、是否可解码。

    sha256 让测试可以逐字节核对"这一路相机图确实是它"，而不必把几 MB 的
    base64 存进记录里。
    """
    info: dict[str, Any] = {
        "data_url_prefix": "",
        "b64_chars": 0,
        "decoded_bytes": 0,
        "sha256": "",
        "jpeg_magic_ok": False,
        "decode_error": "",
    }
    if not isinstance(data_url, str):
        info["decode_error"] = f"not a string: {type(data_url).__name__}"
        return info
    head, _, payload = data_url.partition(",")
    info["data_url_prefix"] = head[:64]
    info["b64_chars"] = len(payload)
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as error:  # noqa: BLE001 - 桩服务要把坏输入记下来而不是崩掉
        info["decode_error"] = f"{type(error).__name__}: {error}"
        return info
    info["decoded_bytes"] = len(raw)
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    info["jpeg_magic_ok"] = raw.startswith(JPEG_MAGIC)
    return info


class StubState:
    """进程内共享状态：脚本队列 + 请求记录。所有读写都在锁内，多线程安全。"""

    def __init__(self, script: Mapping[str, Any] | None = None, max_records: int = DEFAULT_MAX_RECORDS):
        self.lock = threading.Lock()
        self.records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self.queue: list[dict[str, Any]] = []
        self.fallback: dict[str, Any] | None = None
        self.repeat_last = False
        self.last_spec: dict[str, Any] | None = None
        self.n_requests = 0
        self.n_completions = 0
        self.n_models = 0
        self.stats_out: str | None = None
        self.started_at = time.time()
        if script is not None:
            self.load_script(script)

    # -- 脚本 -------------------------------------------------------------------------
    def load_script(self, script: Any) -> dict[str, Any]:
        """装载脚本（list 或 dict 两种形态），返回队列长度摘要。"""
        if isinstance(script, list):
            responses, fallback, repeat_last = script, None, False
        elif isinstance(script, Mapping):
            responses = list(script.get("responses", []) or [])
            fallback = script.get("fallback")
            repeat_last = bool(script.get("repeat_last", False))
        else:
            raise ValueError("script 必须是数组或对象")
        with self.lock:
            self.queue = [dict(item) for item in responses]
            self.fallback = dict(fallback) if isinstance(fallback, Mapping) else None
            self.repeat_last = repeat_last
            return self.script_summary_locked()

    def script_summary_locked(self) -> dict[str, Any]:
        return {
            "queue_remaining": len(self.queue),
            "repeat_last": self.repeat_last,
            "has_fallback": self.fallback is not None,
        }

    def next_spec(self) -> dict[str, Any]:
        """取下一个响应 spec；队列空时按 repeat_last / fallback / 默认 依次退化。"""
        with self.lock:
            if self.queue:
                spec = self.queue.pop(0)
                self.last_spec = dict(spec)
            elif self.repeat_last and self.last_spec is not None:
                spec = dict(self.last_spec)
            elif self.fallback is not None:
                spec = dict(self.fallback)
            else:
                spec = {}
            return dict(spec)

    # -- 记录 -------------------------------------------------------------------------
    def record(self, entry: dict[str, Any]) -> int:
        """登记一条请求记录并返回它的序号（序号在锁内分配，不受并发影响）。"""
        with self.lock:
            index = self.n_requests
            entry["index"] = index
            self.records.append(entry)
            self.n_requests += 1
            return index

    def snapshot(self, include_prompts: bool = True) -> dict[str, Any]:
        with self.lock:
            records = []
            for entry in self.records:
                item = dict(entry)
                if not include_prompts:
                    item.pop("prompt_text", None)
                records.append(item)
            return {
                "n_requests": self.n_requests,
                "n_completions": self.n_completions,
                "n_models": self.n_models,
                "uptime_s": round(time.time() - self.started_at, 3),
                "requests": records,
                **self.script_summary_locked(),
            }

    def reset(self) -> None:
        with self.lock:
            self.records.clear()
            self.n_requests = 0
            self.n_completions = 0
            self.n_models = 0

    def dump(self, path: str | None = None) -> str | None:
        """把统计写成 JSON（先写临时文件再原子替换，读者不会看到半个文件）。"""
        target = path or self.stats_out
        if not target:
            return None
        payload = self.snapshot()
        directory = os.path.dirname(os.path.abspath(target)) or "."
        os.makedirs(directory, exist_ok=True)
        handle, tmp_path = tempfile.mkstemp(dir=directory, prefix=".stub_stats_", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target)
        return target


def _collect_message(message: Mapping[str, Any], text_chunks: list[str], images: list[dict[str, Any]]) -> None:
    """从一条 message 里抽出文本与图像（兼容 content 为字符串/列表两种形态）。"""
    content = message.get("content")
    if isinstance(content, str):
        text_chunks.append(content)
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "image_url":
            url = part.get("image_url")
            if isinstance(url, Mapping):
                url = url.get("url", "")
            images.append(_b64_payload_size(url if isinstance(url, str) else ""))
        else:
            text = part.get("text")
            if isinstance(text, str):
                text_chunks.append(text)


class StubHandler(BaseHTTPRequestHandler):
    """把 StubState 暴露成 OpenAI 兼容端点。"""

    server_version = "MoonSimVLMStub/1.0"
    protocol_version = "HTTP/1.1"

    # -- 基础工具 ---------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 覆盖基类签名
        """默认往 stderr 打日志会污染测试输出，这里改成可选（--verbose 时打开）。"""
        if getattr(self.server, "verbose", False):
            sys.stderr.write("[stub] " + fmt % args + "\n")

    @property
    def state(self) -> StubState:
        return self.server.state  # type: ignore[attr-defined]

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- 端点 -------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
        path = self.path.split("?", 1)[0]
        if path in ("/v1/models", "/models"):
            with self.state.lock:
                self.state.n_models += 1
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "Qwen3-VL-8B", "object": "model", "owned_by": "stub"},
                        {"id": "stub-echo", "object": "model", "owned_by": "stub"},
                    ],
                },
            )
            return
        if path == "/__stats":
            include_prompts = "summary=1" not in self.path
            self._send_json(200, self.state.snapshot(include_prompts=include_prompts))
            return
        if path in ("/", "/__health"):
            self._send_json(200, {"ok": True, "service": "vlm-stub", **self.state.script_summary_locked()})
            return
        self._send_json(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        raw = self._read_body()

        if path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_completion(raw)
            return
        if path == "/__script":
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
                summary = self.state.load_script(payload)
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": {"message": f"bad script: {error}"}})
                return
            self._send_json(200, {"ok": True, **summary})
            return
        if path == "/__reset":
            self.state.reset()
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, Mapping) and "script" in payload:
                    self.state.load_script(payload["script"])
            self._send_json(200, {"ok": True, **self.state.script_summary_locked()})
            return
        if path == "/__dump":
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                payload = {}
            target = payload.get("path") if isinstance(payload, Mapping) else None
            written = self.state.dump(target)
            self._send_json(200 if written else 400, {"ok": bool(written), "path": written})
            return
        if path == "/__shutdown":
            self._send_json(200, {"ok": True, "bye": True})
            self.state.dump()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self._send_json(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})

    def _handle_completion(self, raw: bytes) -> None:
        arrived = time.time()
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": {"message": f"bad json body: {error}", "type": "invalid_request"}})
            return
        if not isinstance(body, Mapping):
            self._send_json(400, {"error": {"message": "body must be an object", "type": "invalid_request"}})
            return

        text_chunks: list[str] = []
        images: list[dict[str, Any]] = []
        images_per_message: list[int] = []
        messages = body.get("messages") or []
        roles: list[str] = []
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, Mapping):
                    roles.append(str(message.get("role", "?")))
                    seen = len(images)
                    _collect_message(message, text_chunks, images)
                    images_per_message.append(len(images) - seen)

        spec = self.state.next_spec()
        delay = float(spec.get("delay_s", 0.0) or 0.0)
        status = int(spec.get("status", 200) or 200)
        content = spec.get("content")
        if content is None:
            content = spec.get("default_content", self.server.default_content)  # type: ignore[attr-defined]
        model = str(spec.get("model", body.get("model", "Qwen3-VL-8B")))
        raw_override = spec.get("raw")

        entry = {
            "timestamp": round(arrived, 4),
            "path": self.path,
            "model": str(body.get("model", "")),
            "roles": roles,
            "n_messages": len(roles),
            "images_per_message": images_per_message,
            "prompt_text": "\n".join(text_chunks),
            "prompt_chars": sum(len(chunk) for chunk in text_chunks),
            "image_count": len(images),
            "images": images,
            "image_decoded_bytes": [item["decoded_bytes"] for item in images],
            "image_sha256": [item["sha256"] for item in images],
            "image_magic_ok": [item["jpeg_magic_ok"] for item in images],
            "image_prefixes": [item["data_url_prefix"] for item in images],
            "max_tokens": body.get("max_tokens"),
            "temperature": body.get("temperature"),
            "stream": bool(body.get("stream", False)),
            "request_bytes": len(raw),
            "response_kind": "error" if status >= 400 else ("slow" if delay > 0 else "content"),
            "response_status": status,
            "response_delay_s": round(delay, 4),
            "script_spec": {key: value for key, value in spec.items() if key != "raw"},
        }
        # 先登记再 sleep：慢响应场景下客户端会超时，但"服务端收到过这个请求"必须可查。
        index = self.state.record(entry)
        if status < 400:
            with self.state.lock:
                self.state.n_completions += 1
        self.state.dump()

        if delay > 0:
            time.sleep(delay)

        if status >= 400:
            payload = spec.get("body")
            if not isinstance(payload, Mapping):
                payload = {
                    "error": {
                        "message": f"stub scripted error {status}",
                        "type": "stub_error",
                        "code": status,
                    }
                }
            self._send_json(status, payload)
            return

        if isinstance(raw_override, Mapping):
            self._send_json(200, raw_override)
            return

        self._send_json(
            200,
            {
                "id": f"chatcmpl-stub-{index}",
                "object": "chat.completion",
                "created": int(arrived),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, entry["prompt_chars"] // 4),
                    "completion_tokens": max(1, len(str(content)) // 4),
                    "total_tokens": max(2, (entry["prompt_chars"] + len(str(content))) // 4),
                },
            },
        )


class StubServer(ThreadingHTTPServer):
    """线程化 HTTP 服务；`daemon_threads` 保证慢响应不会拖住进程退出。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        state: StubState,
        default_content: str = DEFAULT_CONTENT,
        verbose: bool = False,
    ):
        super().__init__(address, StubHandler)
        self.state = state
        self.default_content = default_content
        self.verbose = verbose

    def handle_error(self, request: Any, client_address: Any) -> None:
        """客户端超时断连（BrokenPipe/ConnectionReset）是测试的常态，不该刷栈到 stderr。"""
        if self.verbose:
            super().handle_error(request, client_address)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI 兼容的 VLM 桩服务（纯标准库）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=0, help="监听端口，0 表示临时端口（默认 0）")
    parser.add_argument("--script", default=None, help="响应脚本 JSON 文件路径（可选）")
    parser.add_argument("--default-content", default=DEFAULT_CONTENT, help="队列用完后默认回复的文本")
    parser.add_argument("--stats-out", default=None, help="把请求记录写成 JSON 的路径")
    parser.add_argument("--ready-file", default=None, help="把就绪信息写成 JSON 的路径")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS, help="内存里保留的请求记录上限")
    parser.add_argument("--max-requests", type=int, default=0, help="处理满 N 个 completion 后自动退出（0=不限）")
    parser.add_argument("--verbose", action="store_true", help="把访问日志打到 stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    script = None
    if args.script:
        with open(args.script, "r", encoding="utf-8") as stream:
            script = json.load(stream)

    state = StubState(script=script, max_records=max(1, int(args.max_records)))
    state.stats_out = args.stats_out
    server = StubServer((args.host, int(args.port)), state, default_content=args.default_content, verbose=args.verbose)
    host, port = server.server_address[0], server.server_address[1]
    base_url = f"http://{host}:{port}/v1"

    ready = {
        "host": host,
        "port": port,
        "base_url": base_url,
        "pid": os.getpid(),
        "ready": True,
    }
    if args.ready_file:
        with open(args.ready_file, "w", encoding="utf-8") as stream:
            json.dump(ready, stream, ensure_ascii=False, indent=2)

    def _stop(_signum: int, _frame: Any) -> None:
        state.dump()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print("STUB_READY " + json.dumps(ready, ensure_ascii=False), flush=True)

    if args.max_requests and args.max_requests > 0:
        def _watch() -> None:
            while True:
                time.sleep(0.05)
                with state.lock:
                    done = state.n_completions
                if done >= int(args.max_requests):
                    server.shutdown()
                    return

        threading.Thread(target=_watch, daemon=True).start()

    interrupted = False
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:  # 没走 SIGINT handler 的兜底路径（例如被别处打断）
        interrupted = True
    finally:
        server.server_close()
        written = state.dump()
        if written:
            print(f"STUB_STATS {written}", flush=True)
        print(
            "STUB_STOPPED " + json.dumps({"n_requests": state.n_requests, "interrupted": interrupted}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
