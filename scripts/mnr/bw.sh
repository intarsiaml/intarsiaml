#!/bin/bash
# Lane mnr microbench: matmul_mxfp4_gemv (M = 1, today's bitwise rows at M = 4 / 8 / 16) against the row kernel at the
# Qwen3.8-27B shapes. Run inside a fleetq slot:
#   fleetq run --host endor --expect 6 --mem 8 --quiet gpu --label "mnr: kernel microbench" -- scripts/mnr/bw.sh <out.log> [kernels] [Ms]
set -u
cd "$(dirname "$0")/../.."
OUT=${1:?out log}
KERNELS=${2:-matmul_mxfp4_gemv,matmul_mxfp4_naxr_rows@0}
MS=${3:-1,4,8,16}
: > "$OUT"
for shape in 17408x5120 5120x17408 12288x5120 5120x6144; do
  MXFP4_BW_SHAPE=$shape MXFP4_BW_M=$MS MXFP4_BW_KERNEL=$KERNELS target/release/examples/mxfp4_bw >> "$OUT" 2>&1
done
