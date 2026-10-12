#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="${ISAACLAB_PATH:-$TASK_ROOT/../IsaacLab}"
export PYTHONPATH="$TASK_ROOT:$LAB_ROOT/source/isaaclab:$LAB_ROOT/source/isaaclab_assets:$LAB_ROOT/source/isaaclab_rl:$LAB_ROOT/source/isaaclab_tasks"
export PYTHONEXE="$LAB_ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
SCRIPT="$TASK_ROOT/run.py"
if [[ "${1:-}" == "validate" ]]; then
    shift
    SCRIPT="$TASK_ROOT/tools/validate_task2.py"
fi
exec "$LAB_ROOT/_isaac_sim/python.sh" "$SCRIPT" "$@"
