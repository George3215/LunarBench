"""NaVILA policy client for the task3 V4 evaluation.

The NaVILA (VILA/Llava) model runs in its own environment behind a local HTTP
service (third_party/NaVILA/local_inference/navila_server.py). This client
implements the same `decide_batch(requests)` interface as
`context_navigation_policy.Policy`, so it plugs into all_terrain_context_trial.

NaVILA emits mid-level language actions (stop / move forward N cm / turn left N
degree / turn right N degree). One model decision maps to exactly one task3
1-second primitive action, so the action budget is comparable with the
Qwen3-VL conditions:
  0 stop        -> STOP
  1 move forward-> FORWARD
  2 turn left   -> TURN_LEFT
  3 turn right  -> TURN_RIGHT

The instruction states the task only and never names or describes the terrain.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests

import context_navigation_policy as cnp
from vlm_benchmark import dump, sha

INSTRUCTION = "Find the red sphere and stop when it is clearly visible and within 1 metre of the camera."
MID_TO_TASK3 = {0: "STOP", 1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
NATIVE_HORIZON = "native8"


def request_for(frames, horizon, mode):
    """Extend context_navigation_policy.request_for with the NaVILA native horizon.

    native8 = NaVILA's own protocol: every recorded RGB frame through the current
    one, which the server samples/pads to the model's 8-frame input.
    """
    if horizon == NATIVE_HORIZON:
        base = cnp.request_for(frames, "all_history", "rgb")
        base["horizon"] = NATIVE_HORIZON
        base["input_description"] = "NaVILA native: all recorded RGB frames sampled to 8; no depth"
        return base
    return cnp.request_for(frames, horizon, mode)


class NavilaPolicy:
    def __init__(self, endpoint="http://127.0.0.1:8100", protocol="matrix", timeout_s=600):
        self.endpoint = endpoint.rstrip("/")
        self.protocol = protocol
        self.timeout_s = timeout_s
        self.generation_calls = 0

    def _paths(self, request):
        paths = []
        for label, path in request["images"]:
            if request["mode"] == "rgb" and "depth" in label.lower():
                continue
            paths.append(Path(path))
        return paths

    def decide_batch(self, requests):
        results = []
        for request, folder, attempt in requests:
            folder.mkdir(parents=True, exist_ok=True)
            paths = self._paths(request)
            blobs = [base64.b64encode(p.read_bytes()).decode("ascii") for p in paths]
            payload = {
                "instruction": INSTRUCTION,
                "images": blobs,
                "condition": f"{request['mode']}/{request['horizon']}",
            }
            start = time.perf_counter()
            error = None
            try:
                response = requests.post(self.endpoint + "/predict", json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001
                data = {"prompt": "", "raw_output": "", "mid_action": None, "generated_token_count": 0}
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - start
            self.generation_calls += 1

            raw = data.get("raw_output", "")
            mid = data.get("mid_action")
            action = MID_TO_TASK3.get(mid) if mid is not None else None
            parsed = None
            if action:
                parsed = {
                    "action": action,
                    "mid_action": mid,
                    "distance_cm": data.get("distance_cm"),
                    "degree": data.get("degree"),
                }

            saved = [{"type": "text", "text": data.get("prompt", "")}]
            for path in paths:
                saved.append({"type": "image", "image_path": str(path), "sha256": sha(path)})
            dump(
                folder / "messages.json",
                {
                    "messages": saved,
                    "mode": request["mode"],
                    "horizon": request["horizon"],
                    "protocol": self.protocol,
                    "instruction": INSTRUCTION,
                    "policy_inputs": request["input_description"],
                },
            )
            (folder / "rendered_prompt.txt").write_text(data.get("prompt", ""))
            dump(
                folder / "context.json",
                {
                    "horizon": request["horizon"],
                    "mode": request["mode"],
                    "protocol": self.protocol,
                    "instruction": INSTRUCTION,
                    "input_frame_ids": request["frame_ids"],
                    "input_image_count": len(paths),
                    "all_recorded_frame_count": request["all_recorded_frame_count"],
                    "first_frame_in_context": request["first_frame_in_context"],
                    "history_span_sim_s": request["history_span_sim_s"],
                    "past_assistant_outputs_in_context": False,
                    "past_action_history_in_context": False,
                },
            )
            record = {
                "raw_output": raw,
                "parsed": parsed,
                "mid_action": mid,
                "distance_cm": data.get("distance_cm"),
                "degree": data.get("degree"),
                "generated_token_count": data.get("generated_token_count", 0),
                "elapsed_s": latency,
                "batch_amortized_s": latency,
                "generation_skipped": False,
                "error": error,
                "termination": "inference_error" if error else None,
            }
            dump(folder / "output.json", record)
            (folder / "raw_output.txt").write_text(raw)
            results.append(record)
        return results
