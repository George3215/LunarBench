"""NaVILA trials on the terrain_design_v4 strict-admitted terrains.

Two protocols, both on the same 22 admitted terrains, same success definition
and same <=3000 autonomous-action budget as the Qwen3-VL run:

  matrix : sensor {rgb, rgbd} x history {current, first_current, all_history}
           (NaVILA is natively RGB-only, so rgbd feeds aligned depth images as
           additional frames; documented adaptation)
  native : NaVILA's own protocol - all recorded RGB frames sampled/padded to the
           model's 8-frame input, no depth

NaVILA inference runs in its own environment behind the local HTTP service
(third_party/NaVILA/local_inference/navila_server.py).
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

import all_terrain_context_trial as act
import navigation_trial as nt
import navila_policy
from context_navigation_policy import HORIZONS, MODES
from navila_policy import NavilaPolicy
from v4_qwen3vl_trial import DEFAULT_ADMISSION, build_setups, verify
from visible_start_trial import actions
from vlm_benchmark import dump

ROOT = Path(__file__).resolve().parents[2]


def run_protocol(root: Path, protocol: str, setups, terrain_map, max_steps: int, batch_size: int, endpoint: str, modes, horizons):
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    actions()
    act.MAX_STEPS = max_steps
    act.request_for = navila_policy.request_for
    policy = NavilaPolicy(endpoint=endpoint, protocol=protocol)

    total = len(setups) * len(modes) * len(horizons)
    dump(
        root / "protocol.json",
        {
            "version": "terrain_design_v4-navila/1",
            "protocol": protocol,
            "terrain_design_module": "terrain_design_v4",
            "admission_source": str(DEFAULT_ADMISSION),
            "scene_count": len(setups),
            "scenes": [s["terrain_id"] for s in setups],
            "trial_count": total,
            "sensor_conditions": modes,
            "context_conditions": horizons,
            "max_steps": max_steps,
            "model": "navila-llama3-8b-8f",
            "instruction": navila_policy.INSTRUCTION,
            "instruction_discloses_terrain": False,
            "action_mapping": navila_policy.MID_TO_TASK3,
            "rgbd_adaptation": "aligned depth PNGs interleaved after each RGB frame; NaVILA is natively RGB-only",
            "native_horizon": navila_policy.NATIVE_HORIZON,
            "decoding": "greedy",
            "seed": 7,
        },
    )
    dump(root / "scene_setup.json", setups)
    (root / "INSTRUCTION.txt").write_text(navila_policy.INSTRUCTION + "\n")
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("navila_policy.py", "v4_navila_trial.py", "v4_qwen3vl_trial.py", "all_terrain_context_trial.py"):
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())

    terrains = terrain_map
    tasks = iter([(s, m, h) for s in setups for m in modes for h in horizons])
    active = []
    start = time.perf_counter()
    try:
        while True:
            while len(active) < batch_size:
                task = next(tasks, None)
                if task is None:
                    break
                setup, mode, horizon = task
                active.append(act.Trial(setup, terrains[setup["scene_name"]], root, mode, horizon))
            if not active:
                break
            act.status(root, active, total, start, policy)
            if shutil.disk_usage(root).free < 5 * 1024**3:
                for trial in active:
                    trial.finish("disk_space_limit")
                raise RuntimeError("Explicit disk-space guard; remaining tasks not started")
            records = policy.decide_batch([t.prepare() for t in active])
            retry = []
            for trial, record in zip(active, records):
                trial.calls.append(record)
                if record["parsed"] is None and not record.get("termination"):
                    retry.append(trial)
            if retry:
                records = policy.decide_batch([(t.request, t.step_folder / "inference_2", 2) for t in retry])
                for trial, record in zip(retry, records):
                    trial.calls.append(record)
            for trial in active:
                trial.complete()
            active = [t for t in active if not t.done]
            act.status(root, active, total, start, policy)
        value = act.status(root, active, total, start, policy)
        value["status"] = "trials_complete_pending_analysis"
        dump(root / "RUN_STATUS.json", value)
    except Exception as exc:
        dump(root / "ERROR.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        for trial in active:
            if not trial.done:
                trial.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    p.add_argument("--terrains", nargs="*")
    p.add_argument("--protocol", choices=["matrix", "native", "both"], default="both")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--endpoint", default="http://127.0.0.1:8100")
    p.add_argument("--modes", nargs="*", default=list(MODES))
    p.add_argument("--horizons", nargs="*", default=list(HORIZONS))
    a = p.parse_args()

    setups, terrain_map = build_setups(a.admission.resolve(), a.terrains)
    out = a.out.resolve()
    if a.protocol in ("matrix", "both"):
        run_protocol(out / "matrix", "matrix", setups, terrain_map, a.max_steps, a.batch_size, a.endpoint, a.modes, a.horizons)
    if a.protocol in ("native", "both"):
        run_protocol(
            out / "native",
            "native",
            setups,
            terrain_map,
            a.max_steps,
            a.batch_size,
            a.endpoint,
            ["rgb"],
            [navila_policy.NATIVE_HORIZON],
        )


if __name__ == "__main__":
    main()
