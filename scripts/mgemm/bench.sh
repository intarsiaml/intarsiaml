#!/bin/bash
# Lane mgemm: NAX GEMM microbench on endor. Run under `fleetq run --host endor --quiet gpu,cpu`.
# usage: bench.sh <out-file> [shapes...]   env NAX_M / NAX_SWZ / NAX_ONLY / NAX_POOL_MIB / NAX_ROUNDS pass through.
W=/opt/code/intarsiaml/.worktrees/mgemm
out=$1; shift
cd "$W" || exit 1
"$W/target/release/examples/gemm_nax" ${MODE:-bench} "$@" > "$out" 2> "$out.err"
echo "exit $?" >> "$out"
