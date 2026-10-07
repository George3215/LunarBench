# Reproduce local VLM baseline

Use PowerShell in `D:\LunarBench` and the existing `C:\Users\all\miniconda3\python.exe` environment. Required packages already used successfully: requests, Pillow, numpy, MuJoCo, torch, matplotlib. No Transformers upgrade is required.

Start local Ollama only if port 11435 is not already serving this experiment:

```powershell
powershell -File tools/Start-VlmBenchmarkServer.ps1
```

Model files are stored in `.local/vlm/ollama-models`. Exact digests are recorded in the run manifest. Models already downloaded for this run; if recreating elsewhere, pull the same named model and verify the digest, since a tag may later change.

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11435'
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
```

Do not run different models concurrently on this single-loaded-model service; that forces repeated unload/load. Temperature 0 and seed 7 fix nominal settings but do not guarantee bitwise output across runtime versions or GPU/cache states.

Recreate real camera seen/lost prefixes (existing metadata is skipped):

```powershell
python tasks/task3_reobservability/vlm_benchmark.py --prepare
```

Exact original six-action evaluation, scored on pinned previously executed branches:

```powershell
python tasks/task3_reobservability/vlm_one_step.py --model qwen3.5:4b
python tasks/task3_reobservability/vlm_one_step.py --model qwen3.5:9b --modes current_direct images_actions_reasoned
```

These commands resume by skipping completed result.json files. They do not overwrite or create independent repeated trials. The one-step runner uses the archived run directory; use a separate workspace copy if genuinely repeating the entire one-step run with the same source and protocol. Keep original data intact.

Actual continuous physics, image decisions and STOP task:

```powershell
python tasks/task3_reobservability/vlm_benchmark.py --out paper/vlm_goal_navigation/runs/20261004_qwen35/closed_loop_v2 --prefix-root paper/vlm_goal_navigation/runs/20261004_qwen35/prefixes --model qwen3.5:4b --scenes stairs_up trench pit hist001_left --steps 10 --tokens 512 --history-layout sheet
python tasks/task3_reobservability/vlm_benchmark.py --out paper/vlm_goal_navigation/runs/20261004_qwen35/closed_loop_9b --prefix-root paper/vlm_goal_navigation/runs/20261004_qwen35/prefixes --model qwen3.5:9b --scenes stairs_up hist001_left --modes current_direct images_actions_reasoned --steps 10 --tokens 512 --history-layout sheet
```

Use a new `--out` directory for an independent online rerun. Source snapshots preserve the exact versions used in the canonical runs; current sources include a provenance-hash fix that does not change actions. Protocol files describe reused v1 episodes and one annotated old source-hash field.

Optional diagnostics (do not supply diagnostic outputs to policy inputs):

```powershell
python paper/vlm_goal_navigation/probe_model_runtime.py --model qwen3.5:9b
python paper/vlm_goal_navigation/vlm_perception_probe.py --model qwen3.5:9b
python paper/vlm_goal_navigation/probe_history_sheet.py --model qwen3.5:9b
```

Regenerate tables, verify stored input hashes and first-step physics, then produce report/figures:

```powershell
python paper/vlm_goal_navigation/summarize_one_step.py
python paper/vlm_goal_navigation/summarize_vlm.py paper/vlm_goal_navigation/runs/20261004_qwen35/closed_loop_v2
python paper/vlm_goal_navigation/summarize_vlm.py paper/vlm_goal_navigation/runs/20261004_qwen35/closed_loop_9b
python paper/vlm_goal_navigation/analyse_vlm_diagnostics.py paper/vlm_goal_navigation/runs/20261004_qwen35/closed_loop_v2
python paper/vlm_goal_navigation/analyse_vlm_diagnostics.py paper/vlm_goal_navigation/runs/20261004_qwen35/closed_loop_9b
python paper/vlm_goal_navigation/verify_vlm_artifacts.py
python paper/vlm_goal_navigation/plot_vlm_results.py
python paper/vlm_goal_navigation/build_baseline_report.py
```

No real robot is controlled. Inference pauses simulated time. Read VLM_BASELINE_RESULTS.md for task definitions and evidence limits before interpreting scores.
