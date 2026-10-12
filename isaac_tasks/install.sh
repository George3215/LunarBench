#!/usr/bin/env bash
# 已安装 Isaac Sim 5.1 后，建立固定版本的 Isaac Lab 和隔离 Python 环境。
# ISAAC_SIM_PATH / ISAACLAB_PATH 可迁移到另一台机器，运行时不依赖 MuJoCo。
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="${ISAACLAB_PATH:-$TASK_ROOT/../IsaacLab}"
SIM_ROOT="${ISAAC_SIM_PATH:-/home/lry/isaac-sim}"
if [[ ! -f "$SIM_ROOT/python.sh" ]]; then
    echo "请先安装 Isaac Sim 5.1，并设置 ISAAC_SIM_PATH。" >&2
    exit 1
fi
if [[ ! -f "$LAB_ROOT/isaaclab.sh" ]]; then
    mkdir -p "$LAB_ROOT"
    ARCHIVE=$(mktemp /tmp/isaaclab-v2.3.2.XXXXXX.tar.gz)
    curl -fL https://api.github.com/repos/isaac-sim/IsaacLab/tarball/v2.3.2 -o "$ARCHIVE"
    tar -xzf "$ARCHIVE" --strip-components=1 -C "$LAB_ROOT"
    rm "$ARCHIVE"
fi
[[ -e "$LAB_ROOT/_isaac_sim" ]] || ln -s "$SIM_ROOT" "$LAB_ROOT/_isaac_sim"
[[ -x "$LAB_ROOT/.venv/bin/python" ]] || "$SIM_ROOT/kit/python/bin/python3" -m venv --system-site-packages "$LAB_ROOT/.venv"
export PYTHONEXE="$LAB_ROOT/.venv/bin/python"
export PYTHONPATH=
"$SIM_ROOT/python.sh" -m pip install -r "$TASK_ROOT/requirements.txt"
"$SIM_ROOT/python.sh" -m pip install --no-deps --no-build-isolation \
    -e "$LAB_ROOT/source/isaaclab" -e "$LAB_ROOT/source/isaaclab_assets" \
    -e "$LAB_ROOT/source/isaaclab_tasks" -e "$LAB_ROOT/source/isaaclab_rl"
