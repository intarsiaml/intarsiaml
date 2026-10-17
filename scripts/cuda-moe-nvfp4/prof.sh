#!/usr/bin/env bash
# Per-kernel GPU time of the MoE microbenchmark (vLLM UP: the kernels run alone on an idle GPU, memory is shared):
#   fleetq run --host spark1 --mem 12 --label "nvfp4-mma prof" -- scripts/cuda-moe-nvfp4/prof.sh [T=2048] [fp4|both]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
T=${1:-2048}; WHICH=${2:-fp4}
HOST=${HOST:-spark1}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "cd $R/nv-dev && NSYS=\$(command -v nsys || ls /opt/nvidia/nsight-systems*/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1) && echo nsys=\$NSYS && BIN=\$(ls -t target/acc/release/deps/moe_nvfp4-* | grep -v '\\.d\$' | head -1) && INTARSIA_BENCH_T=$T INTARSIA_BENCH_KERNEL=$WHICH INTARSIA_REQUIRE_GPU=1 \$NSYS profile -t cuda --force-overwrite true -o /tmp/nvfp4prof_$FLEETQ_JOB \$BIN --ignored --nocapture bench_moe_nvfp4_real_shapes 2>&1 | tail -8 && \$NSYS stats -r cuda_gpu_kern_sum --format csv /tmp/nvfp4prof_$FLEETQ_JOB.nsys-rep 2>/dev/null | cut -d, -f1-9 | sed 's/(anonymous namespace):://; s/_Z[0-9A-Za-z_]*//' | cut -c1-200; rm -f /tmp/nvfp4prof_$FLEETQ_JOB*"
