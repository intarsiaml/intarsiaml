#!/bin/bash
# Lane M35P: the formal "no worse than the recorded Metal baseline (+ per-prompt noise envelope)" verdicts for the 35B, in one fleetq slot.
# Usage: baseline-gate.sh BIN OUT_DIR
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BIN=$1; OUT=$2; mkdir -p "$OUT"
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden; B=$G/baselines
echo "job $FLEETQ_JOB $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l)" > "$OUT/meta.txt"
$BIN accuracy --model $M --backend metal --golden $G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 \
  --baseline $B/qwen36-35b-a3b-nvfp4-vllm-top20.metal.baseline.json --noise-envelope $B/qwen36-35b-a3b-nvfp4-vllm-top20.metal.envelope.json --allow-unstamped > "$OUT/short-vs-metal-baseline.log" 2>&1
echo "short exit $?"
$BIN accuracy --model $M --backend metal --golden $G/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode both --decode-positions 64 --max-seq 8192 \
  --baseline $B/qwen36-35b-a3b-nvfp4-vllm-4k-top20.metal.baseline.json --allow-unstamped > "$OUT/4k-vs-metal-baseline.log" 2>&1
echo "4k exit $?"
# cross-backend: Metal run against the CUDA baseline (config mismatch allowed: backend/device differ), report only
$BIN accuracy --model $M --backend metal --golden $G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 \
  --baseline $B/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json --noise-envelope $B/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.envelope.json --allow-unstamped --allow-config-mismatch > "$OUT/short-vs-cuda-baseline.log" 2>&1
echo "short-vs-cuda exit $?"
