#!/usr/bin/env bash
# 默认跑可达料区的 Qwen trial；配置/预算/日志路径改 YAML。
# 例：bash scripts/run_qwen_debug.sh --view
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYTHON="${LUNARBENCH_PYTHON:-/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python}"
exec "$PYTHON" run.py --config configs/upstream_qwen_trials.yaml "$@"
