#!/usr/bin/env bash
# W1 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): grouped MoE kernel, accumulation-mode experiment
# (docs/results/2026-10-17-moe-mma-small-t.md). The tree is ~/intarsia-decfuse/moe-mma-dev (scripts/cuda-decode/build.sh).
#   STAGES (default "sweep bench"):
#     sweep   quantmoe moe_experts_mma_acc_sweep: the 192 cases of moe_experts_mma_small_t_sweep, every INTARSIA_MOE_ACC mode
#             on the same reference and old-kernel outputs
#     bench   quantmoe moe_experts_mma_acc_bench: kernel time per mode, T = 64 .. 4096 at the real layer shape
set -u
T=$HOME/intarsia-decfuse/${TREE:-moe-mma-dev}
O=$T/out/${OUT:-w1}; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-9} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
cd $T || exit 1
t() { # t <name> <test filter> <max minutes>
  echo "== $1 ($(left) min left) $(date +%T)"
  CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=target/acc INTARSIA_REQUIRE_GPU=1 timeout ${3}m \
    cargo test --release -p intarsia-backend-cuda --test quantmoe $2 -- --include-ignored --nocapture --test-threads=1 > $O/$1.log 2>&1
  echo "exit $?"
}
for STAGE in ${STAGES:-sweep bench}; do
  case $STAGE in
    sweep) t sweep moe_experts_mma_acc_sweep 6; grep -E "ACC SUMMARY|test result|panicked" $O/sweep.log | cut -c1-200 ;;
    bench) t bench moe_experts_mma_acc_bench 3; grep -E "ACCBENCH|test result|panicked" $O/bench.log | cut -c1-200 ;;
  esac
done
echo "== w1 done $(date +%T), $(left) min left"
