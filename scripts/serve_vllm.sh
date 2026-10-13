#!/usr/bin/env bash
# Start the vLLM server with the settings fixed for every run from Step 3 on
# (docs/step2_decisions.md §13d). Run on the GPU box, inside ~/vllm-env:
#
#   source ~/vllm-env/bin/activate
#   bash scripts/serve_vllm.sh <gpu-index>        # e.g. bash scripts/serve_vllm.sh 2
#
# Pick a GPU with >= 41 GB free: nvidia-smi --query-gpu=index,memory.free --format=csv
# The log is also written to vllm.log (grep it for "KV cache size").
set -euo pipefail

GPU="${1:?usage: serve_vllm.sh <gpu-index>}"
echo "vLLM $(python -c 'import vllm; print(vllm.__version__)') on GPU ${GPU}" | tee vllm.log

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
vllm serve Qwen/Qwen3.5-27B-FP8 --port 8000 \
  --max-model-len 32768 \
  --language-model-only \
  --enable-prefix-caching --mamba-cache-mode align \
  --enable-prompt-tokens-details \
  --max-num-seqs 16 \
  --kv-cache-dtype fp8 \
  2>&1 | tee -a vllm.log
