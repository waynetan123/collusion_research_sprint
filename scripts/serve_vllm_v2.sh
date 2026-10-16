#!/usr/bin/env bash
# Start vLLM for the v2 experiment (docs/v2_step1_decisions.md §7). Run on the GPU
# box, inside ~/vllm-env:
#
#   source ~/vllm-env/bin/activate
#   bash scripts/serve_vllm_v2.sh <gpu-index>        # e.g. bash scripts/serve_vllm_v2.sh 2
#
# Pick a GPU with >= 41 GB free: nvidia-smi --query-gpu=index,memory.free --format=csv
# Before the first run, confirm every flag below exists in this vLLM version:
#   vllm serve --help=all | grep -E 'tool-call-parser|reasoning-parser|mamba-cache-mode'
#
# Changes from v1's serve_vllm.sh:
#   --max-model-len 40960            v2 history peaks near 30k tokens plus the reply
#   --served-model-name ...          the name must contain "qwen" so the fork replays
#                                    reasoning within an episode in Qwen's format
#   --enable-auto-tool-choice        v2 agents act only through tool calls
#   --tool-call-parser qwen3_coder   Qwen3.5's chat template emits <tool_call><function=...>
#   --reasoning-parser qwen3         splits <think> into reasoning_content (logged to disk)
#   --max-num-seqs 10                v2 runs 6-10 trajectories at once
# Sampling: requests set temperature; top_p/top_k come from the checkpoint's
# generation_config.json (vLLM's default --generation-config auto).
set -euo pipefail

GPU="${1:?usage: serve_vllm_v2.sh <gpu-index>}"
PORT="${PORT:-8000}"
echo "vLLM $(python -c 'import vllm; print(vllm.__version__)') on GPU ${GPU}, port ${PORT}" | tee vllm_v2.log

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
vllm serve Qwen/Qwen3.5-27B-FP8 --port "${PORT}" \
  --served-model-name qwen3.5-27b-fp8 \
  --max-model-len 40960 \
  --language-model-only \
  --enable-prefix-caching --mamba-cache-mode align \
  --enable-prompt-tokens-details \
  --max-num-seqs 10 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  2>&1 | tee -a vllm_v2.log
