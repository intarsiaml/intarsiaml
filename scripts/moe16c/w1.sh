#!/usr/bin/env bash
# Lane moe16c, the one GPU window (one fleetq slot on spark1, quiet on GPU and CPU):
#   1. rows_exact on the dev tree (every form, M = 1..16);
#   2. served-token identity base vs dev (ident.sh: 21 cells, both `serve --max-batch 16`);
#   3. batched == solo on the 35B (batch-real.sh, dev tree);
#   4. the interleaved serve A/B base vs dev, 3 rounds at 1 / 4 / 8 / 16 streams (le-window.sh, the #224 / #229 settings).
# Stops before the A/B if 1-3 fail. base = origin/main after #229 (~/intarsiaml-moe16c/bin/base), dev = this branch (bin/dev).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "moe16c: window" -- scripts/moe16c/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/moe16c/docs/results/2026-10-28-moe16c-fp8-convert/raw; mkdir -p $RAW
cd "$H/../.."
date +%T; ssh -n -o BatchMode=yes spark1 "cd ~/intarsiaml-moe16c/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 ~/intarsiaml-moe16c/bin/rows_exact-dev --test-threads=1 2>&1 | grep -aE '^test |test result|panicked|mismatch'" > $RAW/rows-exact.txt
cat $RAW/rows-exact.txt | tail -2; grep -q "test result: ok. 9 passed" $RAW/rows-exact.txt || { echo "ABORT: rows_exact"; exit 2; }
date +%T; "$H/ident.sh" base dev; ok=$?
echo "identity exit $ok"; [[ $ok == 0 ]] || { echo "ABORT: base vs dev served tokens differ"; exit 2; }
date +%T; "$H/batch-real.sh" run > "$RAW/batch-real.txt" 2>&1; cat "$RAW/batch-real.txt"
grep -aq "test result: ok. 2 passed" "$RAW/batch-real.txt" || { echo "ABORT: batch_cuda_real"; exit 3; }
date +%T; "$H/le-window.sh" w1 3 base:base::--max-batch_16 dev:dev::--max-batch_16
date +%T
