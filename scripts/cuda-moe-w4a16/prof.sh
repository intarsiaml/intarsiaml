#!/usr/bin/env bash
# Per-kernel GPU time of one MoE kernel from the microbench (nsys, vLLM UP; the GPU is idle, memory shared):
#   fleetq run --host spark1 --mem 12 --label "w4a16 prof" -- scripts/cuda-moe-w4a16/prof.sh <T> <KERNEL> "<ENV=v ...>" ...
# KERNEL: bf16 (moe_experts_mma_bf16 only), fp4, or v2 (whatever INTARSIA_BENCH_KERNEL=<value> selects in the bench).
# Each further argument is one environment setting set, run in turn. TREE (default w4-dev) is the tree on spark1.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TS=${1:-2048}; WHICH=${2:-bf16}; shift; shift
HOST=${HOST:-spark1}
TREE=${TREE:-w4-dev}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
for T in ${TS//,/ }; do for cfg in "$@"; do
  echo "== T=$T kernel=$WHICH ${cfg:-default}"
  rsh "cd $R/$TREE && NSYS=\$(ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1) && BIN=\$(ls -t target/acc/release/deps/moe_nvfp4-* | grep -v '\\.d\$' | head -1) && env $cfg INTARSIA_BENCH_T=$T INTARSIA_BENCH_KERNEL=$WHICH INTARSIA_REQUIRE_GPU=1 \$NSYS profile -t cuda --force-overwrite true -o /tmp/w4prof_$FLEETQ_JOB \$BIN --ignored --nocapture bench_moe_nvfp4_real_shapes 2>&1 | tail -4; \$NSYS stats -r cuda_gpu_kern_sum --format csv /tmp/w4prof_$FLEETQ_JOB.nsys-rep 2>/dev/null | cut -d, -f1,3,4,9- | sed 's/(anonymous namespace):://; s/_Z[0-9A-Za-z_]*//; s/(int)//g; s/void <unnamed>:://' | cut -c1-110; rm -f /tmp/w4prof_$FLEETQ_JOB*"
done; done
