"""同一初始状态 -> Qwen 控制 -> 证据/复盘 -> 下一 trial。

只在 trial 边界恢复状态。评估使用物体状态，但不向在线动作策略提供物体真值；
在线只返回是否终止/成功，trial 结束后把详细检查结果交给 Qwen 复盘。此检查是 TASK2 本地判据，不是上游原生计分。
"""
import base64
import json
from pathlib import Path

import mujoco
import numpy as np

from PIL import Image

from .direct_control import StoneTools, run_direct


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


from .evaluation import evaluate_wall, support_layers


def run_trials(executor, policy, instruction, output):
    output.mkdir(parents=True, exist_ok=False)
    e = executor
    # 保存完整积分状态，包括速度、控制量和求解器热启动量。每 trial 起点完全相同。
    state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.empty(mujoco.mj_stateSize(e.model, state_kind))
    mujoco.mj_getState(e.model, e.data, initial, state_kind)
    np.save(output / "initial_integration_state.npy", initial)
    (output / "scene.xml").write_text(e.built.xml)
    write_json(output / "run_config.json", dict(instruction=instruction,
               policy={k: v for k, v in policy.config.items() if k != "api_key"}))
    lessons, results = {}, []
    for trial_index in range(policy.config["trials"]):
        mujoco.mj_setState(e.model, e.data, initial, state_kind)
        mujoco.mj_forward(e.model, e.data)
        directory = output / f"trial_{trial_index + 1:03d}"
        e._log(f"[trial {trial_index + 1}/{policy.config['trials']}] 从同一初始状态开始")
        try:
            result = run_direct(e, policy, instruction, directory, lessons=lessons)
        except Exception as error:
            # 不吞异常或重试动作；保留 trial 失败原因与此前已执行的轨迹。
            write_json(directory / "error.json", dict(type=type(error).__name__, message=str(error)))
            raise
        result["trial"] = trial_index + 1
        write_json(directory / "result.json", result)
        # 把最后的真实稳定性观测也保存并交给复盘；不提供额外隐藏物体位姿。
        tools = StoneTools(e, policy.config)
        tools.step_id = result["tool_calls"]
        tools.finished = True
        tools.success = result["success"]
        state, frames = tools.observe()
        content = [{"type": "text", "text": "Final observation: " + json.dumps(state)}]
        for name, frame in frames.items():
            Image.fromarray(frame.rgb).save(directory / f"final_{name}.jpg")
            encoded = base64.b64encode(frame.jpeg_bytes(max_side=policy.config["image_max_side"])).decode()
            content.extend([{"type": "text", "text": name}, {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + encoded}}])
        # 复盘只保留最后这组三张图，避免超过现有 vLLM 图像数限制。
        for message in policy.messages:
            if isinstance(message.get("content"), list):
                message["content"] = [p for p in message["content"] if p["type"] == "text"]
        policy.messages.append({"role": "user", "content": content})
        lessons = policy.review(result, result["failures"])
        write_json(directory / "review.json", lessons)
        (directory / "NOTES.md").write_text(lessons["summary"] + "\n\n" + lessons["next_trial_plan"] + "\n")
        write_json(output / "lessons.json", lessons)
        results.append(result)
        write_json(output / "summary.json", dict(trials=results))
        if result["success"]:
            break
    return dict(arm=e.profile.name, policy=policy.name, status="trials_finished",
                success=any(r["success"] for r in results), executed_actions=sum(r["executed_actions"] for r in results),
                trials=results, artifact_dir=str(output.resolve()))
