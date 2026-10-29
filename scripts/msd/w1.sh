#!/usr/bin/env bash
# Lane msd2, GPU window 1 (one fleetq slot): the corrected base profile (prof2.sh base base, burst.py with ignore_eos), then the
# dev tree's row-exactness tests and the evicted micro-timing (rows_exact timing_rows_evict, base forms included).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 40 --label "msd2: W1 profile + dev micro" -- scripts/msd/w1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/msd2/docs/results/2026-10-28-msd2-dense-gemv/raw
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
date; "$H/prof2.sh" base base
date; ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-msd2/src && INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release --test rows_exact -- --ignored --nocapture timing_rows_evict 2>&1 | grep -aE '^EVT|panicked|error|test result'" > "$RAW/micro-evict-1.txt"
date; ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-msd2/src && INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release --test rows_exact -- --test-threads=4 2>&1 | grep -aE '^test |G-B|panicked|FAIL|test result'" > "$RAW/rows-exact-1.txt"
date; tail -3 "$RAW/rows-exact-1.txt"
