#!/usr/bin/env bash
# Lane step16, GPU window 1 (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. rows_exact on the dev tree (every form, M = 1..16);
#   2. batched == solo on the 35B (batch-real.sh, dev test binary);
#   3. served-token identity base vs dev (ident.sh: 21 cells, both `serve --max-batch 16`);
#   4. the interleaved serve A/B base vs dev, 3 rounds at 1 / 4 / 8 / 16 streams (le-window.sh, the #224 / #229 / #230 settings).
# Stops before the A/B if 1-3 fail. base = origin/main ebf8192a0 (~/intarsiaml-step16/bin/base), dev = this branch (bin/dev).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "step16: W1 exactness + serve A/B" -- scripts/step16/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/step16/docs/results/2026-10-29-step16-decode-step/raw; mkdir -p $RAW
cd "$H/../.."
date +%T; ssh -n -o BatchMode=yes spark1 "cd ~/intarsiaml-step16/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 ~/intarsiaml-step16/bin/rows_exact-dev --test-threads=1 2>&1 | grep -aE '^test |test result|panicked|mismatch'" > $RAW/rows-exact.txt
tail -1 $RAW/rows-exact.txt; grep -q "test result: ok. 9 passed" $RAW/rows-exact.txt || { echo "ABORT: rows_exact"; exit 2; }
date +%T; "$H/batch-real.sh" dev > "$RAW/batch-real.txt" 2>&1; cat "$RAW/batch-real.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
date +%T; "$H/ident.sh" base dev; ok=$?
echo "identity exit $ok"; [[ $ok == 0 ]] || { echo "ABORT: base vs dev served tokens differ"; exit 2; }
date +%T; "$H/le-window.sh" w1 3 base:base::--max-batch_16 dev:dev::--max-batch_16
date +%T
