#!/usr/bin/env bash
# Lane cgv window W3b driver (inside the same held slot, after w3.sh's ids stage): kernel admission on the dev2 tree (it carries the new W4A16 M = 2..4
# sweep cases), the planted mutants, and the verify-rows probe variants. FLEETQ_JOB=<id> scripts/cgv/w3b.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
D=$(cd "$(dirname "$0")" && pwd); RAW=$D/../../docs/results/2026-10-27-cgv-cuda-mxfp4-gemv/raw; mkdir -p "$RAW/gates" "$RAW/mutants"
echo "== kernels test $(date +%T)"
ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'REMOTE'
B=$HOME/intarsiaml-cgv/trees/dev2/target/release/intarsia; O=$HOME/intarsiaml-cgv/out; mkdir -p $O
$B kernels test --backend cuda --op matmul_mxfp4 > $O/ktest-mxfp4-dev2.log 2>&1; echo "rc=$?"; tail -9 $O/ktest-mxfp4-dev2.log | cut -c1-200
$B kernels test --backend cuda --quick > $O/ktest-quick-dev2.log 2>&1; echo "rc=$?"; grep -aE "REJECTED|cases in|incomplete" $O/ktest-quick-dev2.log | tail -3 | cut -c1-200
REMOTE
scp -q 'spark1:intarsiaml-cgv/out/ktest-*-dev2.log' "$RAW/gates/" 2>/dev/null
echo "== verify variants $(date +%T)"
ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s" < "$D/verify-variants.sh" | tee "$RAW/gates/verify-variants.txt"
scp -q 'spark1:intarsiaml-cgv/out/gates/verify-rows-*.log' "$RAW/gates/" 2>/dev/null
echo "== mutants $(date +%T)"
ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB python3 - dev2" < "$D/mutants.py" | tee "$RAW/mutants/mutants.txt"
echo "== W3b done $(date +%T)"
