#!/bin/bash
set -e

# ===== 1. 激活 conda 环境 =====
# 根据你的 conda 安装路径调整，常见是 ~/miniconda3 或 ~/anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moonunreal-mujoco

# ===== 2. 路径与配置 =====
MODEL_PATH="/home/lry/MoonUnrealEnv/Asset/model/Qwen3-VL-8B-Instruct"   # 改成你的实际绝对路径
SERVED_NAME="Qwen3-VL-8B"
PORT=3001
LOG_FILE="./vllm_qwen3vl.log"
PID_FILE="./vllm.pid"

# ===== 3. 启动 vLLM =====
# Codex/Qwen 当策略时需要函数调用（tool calling）：没有下面两行，vLLM 只回文本，
# 不会产生结构化 tool_calls，Codex 的 agent 循环就拿不到任何动作。
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --chunked-prefill-size 512 \
  --swap-space 8 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "✅ vLLM 已启动"
echo "   PID: $(cat $PID_FILE)"
echo "   日志: $LOG_FILE"
echo "   服务地址: http://0.0.0.0:$PORT"