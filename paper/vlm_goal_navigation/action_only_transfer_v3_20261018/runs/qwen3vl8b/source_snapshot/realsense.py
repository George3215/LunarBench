"""Optional RealSense probe for TASK3 sensor-boundary tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def probe(frames: int = 30) -> dict:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        return {"available": False, "reason": "pyrealsense2_not_installed", "error": str(exc)}
    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    try:
        profile = pipe.start(cfg)
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
        got = 0
        for _ in range(frames):
            pipe.wait_for_frames(timeout_ms=2000)
            got += 1
        return {"available": True, "serial": serial, "frames": got,
                "streams": {"color": [640, 480, 30], "depth": [640, 480, 30]}}
    except Exception as exc:
        return {"available": False, "reason": "stream_start_failed", "error": repr(exc)}
    finally:
        try: pipe.stop()
        except Exception: pass


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--frames", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("tasks/task3_reobservability/generated/realsense_probe.json"))
    a = p.parse_args(); result = probe(a.frames); a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
