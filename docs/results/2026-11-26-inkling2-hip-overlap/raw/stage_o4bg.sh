#!/usr/bin/env bash
# Background stage of window O4 (lane ink-hip-overlap) on card 1: o4 at the recipe's default serve config (`--max-batch 16`; overlap
# on by default): prime, context line, KV capacity, VRAM; then o4 at the speed config (`--kv-fraction 0.9 --max-seq 4608`) to cache
# its extra cases for card 0's visits (touch logs/o4/o4-primed); then o4's greedy md5 with the DSpark default; hip tiny suites on the
# o4 tree; `kernels test --op` for the ops whose kernels changed (the gate kernel's ConvMix / LoopBlend, InkAttnDecode, the fused route).
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
L=$REPO/logs/o4; mkdir -p "$L"; rm -f "$L/o4-primed"
SENV="" WAITS=150 srv 1 def-o4 o4 --max-batch 16
stop 1
SENV="" WAITS=60 srv 1 cfg-o4 o4 --max-batch 16 --max-seq 4608 --kv-fraction 0.9
stop 1
touch "$L/o4-primed"
md5s 1 o4 ""
stop 1
tiny 1 o4 inkling2_batch inkling2_packed
admit 1 o4 conv_mix loop_blend ink_attn_decode ink_route
stop 1
echo "[bg] done $(now)"
