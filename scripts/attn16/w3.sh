#!/usr/bin/env bash
# Lane attn16, GPU window 3 (one fleetq slot on spark2, quiet on GPU and CPU), on main 781d21833 + this branch: batched == solo
# on the 35B again (merged tree), timing_35b_batched_step at 1024-token prompts, base vs dev binaries interleaved, then nsys
# profiles (prof.sh: serve --max-batch 16, 16-stream burst) base and dev for the per-kernel / bucket numbers.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 45 --expect 45 --label "attn16: W3 gate + step + nsys" -- scripts/attn16/w3.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-attn16-decode-attn/raw; mkdir -p $RAW
date +%T; "$H/batch-real.sh" dev > "$RAW/batch-real-merged.txt" 2>&1; cat "$RAW/batch-real-merged.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real-merged.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
date +%T
for b in base dev dev base; do BIN=$b "$H/steptime.sh" INTARSIA_STEP_PROMPT=1024 | sed "s/^/$b /"; done | tee "$RAW/steptime-w3.txt"
date +%T; "$H/prof.sh" base base-b16
date +%T; "$H/prof.sh" dev dev-b16
date +%T
