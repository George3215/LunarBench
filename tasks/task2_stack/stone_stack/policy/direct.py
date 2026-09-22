"""QwenVL 直接策略：持久对话 + 结构化工具调用，不从自然语言里抽动作。

模型只接收 RGB、机器人本体状态和工具执行结果。工具由 direct_control 提供，
因此更换模型只改这里，更换仿真/机械臂只改工具适配器。
"""
from __future__ import annotations

import base64
import json
import math
from io import BytesIO
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError


@dataclass
class DirectPolicy:
    config: dict[str, Any]
    name: str = "qwen-direct"
    description: str = "QwenVL 看图并直接调用 TCP/夹爪工具"
    requires_network: bool = True

    def __post_init__(self):
        # 可以收紧门限，但不能通过配置绕过 5 cm / 0.35 rad 上限。
        if not 0 < self.config["max_delta_m"] <= 0.05:
            raise ValueError("max_delta_m must be in (0, 0.05]")
        if not 0 < self.config["max_rotation_rad"] <= 0.35:
            raise ValueError("max_rotation_rad must be in (0, 0.35]")
        if type(self.config["max_steps"]) is not int or self.config["max_steps"] < 1:
            raise ValueError("max_steps must be a positive integer")

    @property
    def needs_cameras(self) -> tuple[str, ...]:
        return tuple(self.config["cameras"])

    def reset(self, instruction: str, context: dict, tools: list[dict]) -> None:
        self.last_usage = {}
        self.tools = tools
        self.transport = self.config["transport"]
        # 两种传输共用同一个工具 schema，避免提示词、接口字段各维护一份。
        self.schema = {"oneOf": [
            {"type": "object", "properties": {
                "name": {"type": "string", "enum": [t["function"]["name"]]},
                "arguments": t["function"]["parameters"],
            }, "required": ["name", "arguments"], "additionalProperties": False}
            for t in tools
        ]}
        prompt = (Path(__file__).resolve().parents[2] / self.config["prompt_file"]).read_text()
        for filename in ("eef_control.md", "teacher_context.md"):
            prompt += "\n" + (Path(__file__).resolve().parents[2] / "prompts/context" / filename).read_text()
        self.messages = [{"role": "system", "content": (
            prompt + f"\nTask: {instruction}\nRobot and task context: {json.dumps(context, ensure_ascii=False)}\n"

        )}]

    def decide(self, state: dict, frames: dict) -> dict:
        # 每轮是 observation/assistant/result 三条消息；保留完整轮次，不拆散工具调用。
        # 更早的记录在 trajectory.jsonl；模型在 reason 中携带简短进度。
        self.messages = self.messages[:1] + self.messages[1:][-3 * self.config["history_turns"]:]
        # 旧观测只留测量值；计划/失败记忆由最新观测提供，避免重复累积到上下文溢出。
        for message in self.messages[1:]:
            if message["role"] == "user" and isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "text" and part["text"].startswith('{"step_id":'):
                        old = json.loads(part["text"])
                        part["text"] = json.dumps({k: old[k] for k in
                            ("step_id", "tcp_position", "tcp_quaternion_wxyz", "grip_width_m") if k in old})
        # 图像单独限量；避免三相机迅速占满上下文。
        recent = self.config["history_images"]
        observations = [m for m in self.messages if m["role"] == "user"
                        and any(p["type"] == "image_url" for p in m["content"])]
        for message in observations[:max(0, len(observations) - recent + 1)]:
            message["content"] = [p for p in message["content"] if p["type"] == "text"]
        state = dict(state, cameras={})
        content = []
        for name in self.needs_cameras:
            frame = frames[name]
            jpeg = frame.jpeg_bytes(max_side=self.config["image_max_side"])
            # 标定对应实际发送的 JPEG，不再把原图宽高配给缩小后的像素。
            with Image.open(BytesIO(jpeg)) as preview:
                width, height = preview.size
            focal = height / (2 * math.tan(math.radians(frame.fovy_deg) / 2))
            state["cameras"][name] = dict(
                width=width, height=height, source_width=frame.width, source_height=frame.height,
                position_world_m=frame.pos.tolist(), rotation_camera_to_world=frame.rot.tolist(),
                K=[[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
                fovy_deg=frame.fovy_deg)
            encoded = base64.b64encode(jpeg).decode()
            content.extend([
                {"type": "text", "text": f"Camera: {name}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ])
        self.last_observation = state
        content.insert(0, {"type": "text", "text": json.dumps(state, ensure_ascii=False)})
        self.messages.append({"role": "user", "content": content})
        payload = dict(model=self.config["model"], messages=self.messages,
                       temperature=0, max_tokens=self.config["max_tokens"])
        if self.transport == "tool_calls":
            payload.update(tools=self.tools, tool_choice="required", parallel_tool_calls=False)
        elif self.transport == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "stone_tool_call", "strict": True, "schema": self.schema}}
        else:
            raise ValueError(f"Unknown direct transport: {self.transport}")
        message = self.complete(payload)
        if self.transport == "json_schema":
            self.messages.append({"role": "assistant", "content": message["content"]})
            return json.loads(message["content"])
        call, = message["tool_calls"]
        self.messages.append({"role": "assistant", "content": message.get("content"),
                              "tool_calls": [call]})
        return dict(id=call["id"], name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]))

    def record_result(self, call: dict, result: dict) -> None:
        content = json.dumps(result, ensure_ascii=False)
        if self.transport == "tool_calls":
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        else:
            self.messages.append({"role": "user", "content": [{"type": "text", "text": "Tool result: " + content}]})

    def complete(self, payload):
        """单次请求；动作和 trial 复盘共享传输，不自动重试。"""
        request = Request(self.config["base_url"].rstrip("/") + "/chat/completions",
                          data=json.dumps(payload).encode(), headers={
                              "Content-Type": "application/json",
                              "Authorization": "Bearer " + self.config["api_key"],
                          })
        # HTTP、格式错误直接上报。不能在动作是否已执行不明确时自动重试。
        try:
            with urlopen(request, timeout=self.config["timeout_s"]) as response:
                reply = json.load(response)
        except HTTPError as error:
            # 保留服务端的真实原因；仍然单次请求，不自动重试。
            raise RuntimeError(f"VLM HTTP {error.code}: {error.read().decode()}") from error
        self.last_usage = reply.get("usage", {})
        return reply["choices"][0]["message"]

    def review(self, outcome, failures):
        """trial 结束后总结；不执行工具，也不修改仿真。"""
        schema = {"type": "object", "properties": {
            "summary": {"type": "string"},
            "failures": {"type": "array", "items": {"type": "object", "properties": {
                key: {"type": "string"} for key in ("evidence", "cause_hypothesis", "next_change")},
                "required": ["evidence", "cause_hypothesis", "next_change"], "additionalProperties": False}},
            "next_trial_plan": {"type": "string"},
        }, "required": ["summary", "failures", "next_trial_plan"], "additionalProperties": False}
        # 复盘使用相同图片历史，但把 host 的工具消息转成普通记录，以免开启动作通道。
        history = []
        for m in self.messages[1:]:
            if m["role"] == "tool":
                history.append({"role": "user", "content": "Executed tool result: " + m["content"]})
            elif "tool_calls" in m:
                history.append({"role": "assistant", "content": json.dumps(m["tool_calls"])})
            else:
                history.append(m)
        prompt = ("Trial ended. Do not act. Write a Chinese evidence-based review and a concrete next_trial_plan. "
                  "Host evaluation is a local geometric/contact check, not upstream native success. "
                  "Include cumulative useful lessons, actual failures and uncertainty. "
                  + json.dumps(dict(outcome=outcome, recorded_failures=failures), ensure_ascii=False))
        payload = dict(model=self.config["model"], messages=self.messages[:1] + history +
                       [{"role": "user", "content": prompt}], temperature=0,
                       max_tokens=self.config["review_max_tokens"], response_format={
                           "type": "json_schema", "json_schema": {"name": "trial_review", "strict": True, "schema": schema}})
        return json.loads(self.complete(payload)["content"])
