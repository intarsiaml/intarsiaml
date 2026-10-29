#!/usr/bin/env bash
# Lane msd: ncu of the MoE kernels in the timing_moe_rows micro test (spark1). Inside a fleetq slot:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 10 --expect 12 --label "msd: ncu moe" -- scripts/msd/ncu.sh <kernel-regex> <skip> <count> [section args...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}; KR=$1; SKIP=$2; CNT=$3; shift 3
ssh -n -o BatchMode=yes "$HOST" "export PATH=/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB INTARSIA_REQUIRE_GPU=1; cd ~/intarsiaml-msd2/src && T=\$(ls -t target/release/deps/rows_exact-* | grep -v '\.d\$' | head -1); ncu --kernel-name regex:'$KR' --launch-skip $SKIP --launch-count $CNT ${*:---set full} \$T --ignored --nocapture timing_moe_rows 2>&1 | grep -av '^==PROF==.*Disconnected' | cut -c1-200 | tail -150"
