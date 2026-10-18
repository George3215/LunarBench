"""Continue the authorized paired experiment and back up records after milestones.

One GPU model at a time. No training, hidden actions, global git configuration,
force push, or deletion. A queue STOP marker prevents the next experiment.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    a = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = a.root.resolve()
    status_file = root / "QUEUE_STATUS.json"
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    expected_branch = "experiments/task3-spatial-transfer-20261018"
    if branch != expected_branch:
        raise RuntimeError(f"wrong backup branch: {branch}")
    env = dict(os.environ, MUJOCO_GL="egl")
    analysis = repo / "paper/vlm_goal_navigation/analyse_spatial_transfer_v4.py"
    driver = repo / "tasks/task3_reobservability/spatial_transfer_v4.py"
    state = {"pid": os.getpid(), "started_unix": time.time(), "phase": "waiting_for_qwen",
             "baseline": "already running outside this queue", "branch": branch,
             "push_authorized": a.push, "backups": []}
    save(status_file, state)

    def analyse(plots=False):
        args = [sys.executable, str(analysis), str(root)]
        if plots:
            args.append("--plots")
        subprocess.run(args, cwd=repo, env=env, check=True)

    def backup(reason):
        analyse()
        # Explicitly stage this experiment's metadata and the owned new scripts.
        paths = [str(root.relative_to(repo)), str(analysis.relative_to(repo)),
                 "tools/fetch_task3_spatial_models.py", "tools/run_task3_spatial_queue.py",
                 "tasks/task3_reobservability/spatial_transfer_v4.py",
                 "tasks/task3_reobservability/action_grammar.py",
                 "tasks/task3_reobservability/model_output_adapter.py",
                 "tasks/task3_reobservability/context_navigation_policy.py",
                 "tasks/task3_reobservability/test_spatial_transfer_v4.py",
                 "tasks/task3_reobservability/validate_spatial_transfer_v4.py"]
        subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo).returncode:
            subprocess.run(["git", "diff", "--cached", "--check", "--", *paths], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", f"Record TASK3 spatial transfer: {reason}", "--only", "--", *paths], cwd=repo, check=True,
                           stdout=subprocess.DEVNULL)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if a.push:
            subprocess.run(["git", "-c", "http.proxy=http://127.0.0.1:7890", "push", "origin", branch], cwd=repo, check=True, timeout=120)
        state["backups"].append({"reason": reason, "commit": head, "pushed": a.push, "unix": time.time()})
        save(status_file, state)

    def wait_completed(key, needed=22):
        milestone = -1
        while True:
            if (root / "STOP").exists():
                raise RuntimeError("queue STOP present; no next experiment launched")
            status = read(root / "runs" / key / "RUN_STATUS.json")
            completed = status.get("completed", 0)
            if completed // 4 > milestone and completed:
                milestone = completed // 4
                backup(f"{key} {completed} of {needed} episodes")
            if status.get("status", "").startswith("error"):
                raise RuntimeError(f"{key} failed; preserved as incomplete")
            if status.get("status") == "complete_pending_analysis":
                if completed != needed:
                    raise RuntimeError("inconsistent completion count")
                return
            state.update(phase=f"running_{key}", current=status, updated_unix=time.time())
            save(status_file, state)
            time.sleep(10)

    try:
        wait_completed("qwen3vl8b")
        # The CUDA owner should exit before a second 17.5GB model loads.
        pid = read(root / "runs/qwen3vl8b/process.json")["pid"]
        while Path(f"/proc/{pid}").exists():
            try:
                if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                    break
            except FileNotFoundError:
                break
            time.sleep(2)
        while read(root / "models/spatial_interactor8b.json").get("status") != "complete":
            if (root / "STOP").exists():
                raise RuntimeError("queue STOP before model load")
            state.update(phase="waiting_for_spatial_model_files", updated_unix=time.time())
            save(status_file, state)
            time.sleep(10)
        state.update(phase="spatial_gpu_smoke", updated_unix=time.time())
        save(status_file, state)
        command = [sys.executable, str(driver), "--model", str(repo / ".local/models/spatial_interactor8b"),
                   "--processor", str(repo / ".local/models/qwen3vl8b")]
        smoke = root / "smoke/spatial_interactor8b_final"
        if read(smoke / "RUN_STATUS.json").get("completed") != 2:
            with (root / "spatial_smoke.log").open("w") as log:
                subprocess.run(command + ["--terrains", "terrain_design_v4/barrier_a0", "terrain_design_v4/pit_a0",
                               "--action-budget", "2", "--out", str(smoke)], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        smoke_status = read(smoke / "RUN_STATUS.json")
        if smoke_status.get("completed") != 2:
            raise RuntimeError("spatial smoke incomplete")
        smoke_results = [read(p) for p in smoke.glob("rgbd/*/trials/*/result.json")]
        if len(smoke_results) != 2 or not any(r['autonomous_actions'] for r in smoke_results):
            raise RuntimeError("spatial smoke has no demonstrated executable model action")
        if any(r['termination'] in ('gpu_memory_limit', 'context_limit', 'repeated_forbidden_stop') for r in smoke_results):
            raise RuntimeError("spatial smoke resource/action-constraint failure")
        # A checkpoint's incomplete schema is a measured interface failure, not
        # a reason to hide it or invent fields/actions. Report it separately.
        state['spatial_smoke_interface_failures'] = sum(r['termination'] == 'invalid_model_output' for r in smoke_results)
        if (root / "STOP").exists():
            raise RuntimeError("queue STOP before spatial full run")
        state.update(phase="running_spatial_interactor8b", updated_unix=time.time())
        save(status_file, state)
        with (root / "spatial_full.log").open("w") as log:
            child = subprocess.Popen(command + ["--out", str(root / "runs/spatial_interactor8b")], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["spatial_pid"] = child.pid
            save(status_file, state)
            # Poll subprocess and persist four-episode checkpoints as it runs.
            milestone = -1
            while child.poll() is None:
                if (root / "STOP").exists():
                    (root / "runs/spatial_interactor8b/STOP").touch()
                status = read(root / "runs/spatial_interactor8b/RUN_STATUS.json")
                completed = status.get("completed", 0)
                state.update(current=status, updated_unix=time.time())
                save(status_file, state)
                if completed and completed // 4 > milestone:
                    milestone = completed // 4
                    backup(f"spatial_interactor8b {completed} of 22 episodes")
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f"spatial evaluation exited {child.returncode}")
        if read(root / "runs/spatial_interactor8b/RUN_STATUS.json").get("completed") != 22:
            raise RuntimeError("spatial run ended incomplete")
        analyse(plots=True)
        state.update(phase="complete", updated_unix=time.time())
        save(status_file, state)
        backup("complete paired evaluation and trajectory analysis")
    except BaseException as exc:
        state.update(phase="stopped_with_error", error=repr(exc), updated_unix=time.time())
        save(status_file, state)
        raise


if __name__ == "__main__":
    main()
