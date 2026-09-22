#!/usr/bin/env bash
# 与 run.py 使用同样三个参数，隔离 RL 依赖，不改 Qwen 的 Python 环境。
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$TASK_DIR/.venv-rl/bin/python" "$TASK_DIR/run.py" "$@"
