#!/usr/bin/env bash
# Lane attn16, GPU window 1 (one fleetq slot on spark2, quiet on GPU and CPU): batched == solo on the 35B (batch-real.sh, dev
# binary, default env = the fixed-grid row attention), row_tables + rows_exact (dev), then timing_35b_batched_step under
# INTARSIA_ATTN_ROWS_PERSIST unset / 0 (0 = main's kernels, SASS-identical), prompts of 64 and 1024 tokens, interleaved.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 40 --expect 40 --label "attn16: W1 batch_cuda_real + rows + step timing" -- scripts/attn16/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-attn16-decode-attn/raw; mkdir -p $RAW
date +%T; "$H/batch-real.sh" dev > "$RAW/batch-real.txt" 2>&1; cat "$RAW/batch-real.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
date +%T
ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-attn16/src/crates/intarsia-backend-cuda
  for t in row_tables rows_exact; do echo \"== \$t\"; INTARSIA_REQUIRE_GPU=1 timeout 600 ~/intarsiaml-attn16/bin/\$t-dev --test-threads=1 --nocapture 2>&1 | grep -aoE '^G-B.*|panicked.*|test result.*|^test .*(ok|FAILED)\$'; done" > "$RAW/rows.txt" 2>&1
cat "$RAW/rows.txt"
date +%T
for p in 64 1024; do
  echo "== prompt $p"
  "$H/steptime.sh" INTARSIA_STEP_PROMPT=$p INTARSIA_STEP_PROMPT=$p,INTARSIA_ATTN_ROWS_PERSIST=0 INTARSIA_STEP_PROMPT=$p,INTARSIA_ATTN_ROWS_PERSIST=0 INTARSIA_STEP_PROMPT=$p
done | tee "$RAW/steptime-w1.txt"
date +%T
