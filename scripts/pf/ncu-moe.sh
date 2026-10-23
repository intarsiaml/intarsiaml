#!/usr/bin/env bash
# Lane pf, spark side: ncu full-set profile of the w4a16 routed-expert kernels (gate_up = MODE 0, down = MODE 1) on the microbench
# shapes. Inside a fleetq slot. Usage: ncu-moe.sh <tree> <outname> <T> [ENV=V ...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
export PATH=/usr/local/cuda/bin:$PATH
P=$HOME/intarsia-decfuse; T=$1; N=$2; TT=$3; shift 3
BIN=$(ls -t $P/$T/target/test/release/deps/moe_nvfp4-* | grep -v '\.d$' | head -1)
O=$P/out/$N; mkdir -p $O
env "$@" INTARSIA_BENCH_T=$TT INTARSIA_BENCH_KERNEL=v2 INTARSIA_REQUIRE_GPU=1 ncu --set full -k regex:moe_w4a16_kernel -s 20 -c 2 -f -o $O/ncu-T$TT $BIN --ignored --nocapture --test-threads=1 bench_moe_nvfp4_real_shapes > $O/ncu-T$TT.log 2>&1
ncu --import $O/ncu-T$TT.ncu-rep --page details > $O/ncu-T$TT.txt 2>&1
ncu --import $O/ncu-T$TT.ncu-rep --page raw --csv > $O/ncu-T$TT.csv 2>&1
tail -3 $O/ncu-T$TT.log; wc -l $O/ncu-T$TT.txt
