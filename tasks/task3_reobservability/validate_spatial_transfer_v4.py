"""Replay historical evaluator reference routes on freshly restored assets.

These are privileged admission checks, never VLM outcomes or policy inputs.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from spatial_transfer_v4 import Trial, actions, sha, dump, build_setups, DEFAULT_ADMISSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(1)
    actions()
    setups, terrains = build_setups(DEFAULT_ADMISSION, None)
    manifest = {r["terrain_id"]: r for r in json.loads((DEFAULT_ADMISSION / "formal_manifest.json").read_text())}
    rows = []
    for setup in setups:
        trial = Trial(setup, terrains[setup["scene_name"]], a.out, 64)
        record = manifest[setup["terrain_id"]]
        reference = next(x for x in record["oracle"] if x.get("recovery"))
        try:
            # Historical admission resets low-level recurrent action memory at
            # each oracle branch. Reproduce that evaluator-only detail exactly.
            # Autonomous VLM trials do NOT reset the walking controller at loss.
            trial.env.controller.reset()
            for step, action in enumerate(reference["actions"], start=trial.prefix+1):
                terminal = trial.physics.execute(action, step)
                if terminal:
                    break
            trial.cam.read(trial.env.model, trial.env.data)
            trial.cam.save_evidence(trial.folder / "reference_final")
            trial.sf.flush()
            trial.ef.flush()
            data = [json.loads(line) for line in (trial.folder / "evaluator.jsonl").read_text().splitlines()]
            run = maximum = 0
            for sample in data:
                if sample["step"] <= trial.prefix:
                    continue
                visible = (sample["target_visible_rgb_pixels"] >= 20 and sample["upright"] >= .5 and sample["in_bounds"])
                run = run+1 if visible else 0
                maximum = max(maximum, run)
            row = {"terrain_id": setup["terrain_id"], "initial_and_loss_pass": True,
                   "reference_controller_reset_at_loss": True,
                   "reference_recovery_pass": maximum >= 3,
                   "consecutive_recovered_samples": maximum,
                   "privileged_reference_actions": reference["actions"],
                   "initial_rgb_sha256": sha(Path(trial.frames[0]["rgb_path"])),
                   "lost_rgb_sha256": sha(Path(trial.frames[-1]["rgb_path"])),
                   "initial_depth_sha256": sha(Path(trial.frames[0]["rgb_path"]).with_name("depth.npy")),
                   "lost_depth_sha256": sha(Path(trial.frames[-1]["rgb_path"]).with_name("depth.npy")),
                   "frame_count": len(trial.frames)}
            rows.append(row)
            print("REFERENCE_GATE", row["terrain_id"], row["reference_recovery_pass"], maximum, flush=True)
            dump(a.out / "validation.json", {"kind": "privileged terrain admission only, no model evaluated",
                 "renderer": os.environ.get("MUJOCO_GL"), "checked": len(rows),
                 "expected": len(setups), "records": rows, "updated_unix": time.time()})
        finally:
            trial.close()
    if not all(r["reference_recovery_pass"] for r in rows):
        raise RuntimeError("recovery admission drift; do not silently remove failed scenes")


if __name__ == "__main__":
    main()
