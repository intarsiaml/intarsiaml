#!/usr/bin/env bash
# Lane mr, ON tatooine inside a fleetq slot that takes radiance down: the batched-step timing and per-op profile of the real
# Qwen3.8-27B MXFP4 on one R9700 (card 0), HIP. Needs build.sh dev first. Logs to ~/mr-logs/profile. The trap starts radiance again.
#   profile.sh [timing rows, default 1,2,3,4,5,6,8,12,16] [profile rows, default 1,2,4,8,12,16] ["timing profile micro"]
#   timing: unprofiled ms/step per row count; profile: per-op table (B = 1 is the solo step); micro: rows_exact timing_mxfp4_rows
set -uo pipefail
TROWS=${1:-1,2,3,4,5,6,8,12,16}; PROWS=${2:-1,2,4,8,12,16}; WHAT=${3:-timing profile micro}
S=$HOME/mr-hip/scripts
D=$HOME/mr-hip/deploy/tatooine/dev.sh
VD=intarsia-cargo-target-mr; RD=$HOME/mr-hip
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
O=$HOME/mr-logs/profile; mkdir -p "$O"
trap 'bash $S/mr/win.sh restore' EXIT
bash $S/mr/win.sh stop
run() { # run <rows> <test>
  INTARSIA_MODELS=$H INTARSIA_REPO=$RD INTARSIA_TARGET_VOL=$VD HIP_VISIBLE_DEVICES=0 $D env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 \
    INTARSIA_QWEN36_DIR=$M27 INTARSIA_TIMING_ROWS=$1 cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 "$2" \
    < /dev/null 2>&1 | grep -aE "^TIMING|^PROFILE|^test |panicked|^error|test result"
}
if [[ " $WHAT " == *" timing "* ]]; then
echo "[profile] timing $(date +%T)"
run "$TROWS" real_35b_batch_step_timing | tee "$O/timing.log"
fi
if [[ " $WHAT " == *" profile "* ]]; then
echo "[profile] op profile $(date +%T)"
run "$PROWS" real_35b_batch_step_profile | tee "$O/profile2.log"
fi
if [[ " $WHAT " == *" micro "* ]]; then
echo "[profile] GEMV and lm_head micro-timing $(date +%T)"
INTARSIA_REPO=$RD INTARSIA_TARGET_VOL=$VD HIP_VISIBLE_DEVICES=0 $D env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test rows_exact -- --ignored --nocapture timing_mxfp4_rows \
  < /dev/null 2>&1 | grep -aE "^TIME|^test |panicked|^error" | tee "$O/micro.log"
fi
echo "[profile] done $(date +%T)"
