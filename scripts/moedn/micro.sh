#!/usr/bin/env bash
# Lane moedn: short check (< 3 min) on spark1: the T = 1..16 cold rows micro (quantmoe bench_moe_rows_cold: v2 down vs the dedup
# kernel's geometries, Zipf routing), optionally under nsys for per-kernel times (NSYS=1), then the MoE rows_exact gate.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 8 --expect 4 --label "moedn: micro" -- scripts/moedn/micro.sh <bin> <outname>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
B=${1:-dev}; OUTN=${2:-micro}
L='~/intarsiaml-moedn'
RAW=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../docs/results/2026-10-29-moedn-down-proj/raw; mkdir -p "$RAW"
PRE=""; [[ -n "${NSYS:-}" ]] && PRE="/usr/local/cuda/bin/nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $L/out/$OUTN"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; mkdir -p $L/out; cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 timeout 150 $PRE $L/bin/quantmoe-$B --ignored --nocapture --test-threads=1 --exact bench_moe_rows_cold 2>&1 | grep -avE '^\s*$|Collecting|Generating|Processing|Saving|nsys-rep'" | tee "$RAW/$OUTN.txt"
if [[ -n "${NSYS:-}" ]]; then
  scp -q "$(dirname "${BASH_SOURCE[0]}")/mk.py" spark1:intarsiaml-moedn/out/mk.py
  ssh -n -o BatchMode=yes spark1 "cd $L/out && /usr/local/cuda/bin/nsys stats -r cuda_gpu_trace -f csv --force-export=true -o $OUTN-t $OUTN.nsys-rep > /dev/null 2>&1; python3 mk.py ${OUTN}-t_cuda_gpu_trace.csv; rm -f $OUTN.sqlite ${OUTN}-t_cuda_gpu_trace.csv" | tee "$RAW/$OUTN-kern.txt"
fi
[[ -n "${SKIP_EXACT:-}" ]] || ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd $L/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 timeout 100 $L/bin/rows_exact-$B --test-threads=1 moe_decode_rows 2>&1 | grep -aE '^test |test result|panicked|G-B'" | tee "$RAW/$OUTN-rows.txt"
