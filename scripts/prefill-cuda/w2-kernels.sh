#!/usr/bin/env bash
# Window 2 (spark1, vLLM down, inside scripts/prefill-cuda-window.sh): the new kernels' tests and micro-benchmarks, then
# the pre-change 35b-long baseline + noise envelope. Tree under test: ~/intarsia-perf/$DEV (default dev).
set -u
P=$HOME/intarsia-perf
DEV=$P/${DEV:-dev}
BASE_BIN=$P/base/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/w2; mkdir -p $O $P/art
DEADLINE=$(( $(date +%s) + ${W2_BUDGET_MIN:-30} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
cd $DEV
export CARGO_TARGET_DIR=target/acc
T="cargo test --release -p intarsia-backend-cuda"

need 4 "kernel tests: gdn_lanes, quantmoe (all), attention" && {
  $T --test gdn_lanes --test quantmoe --test attention -- --nocapture --test-threads=1 > $O/tests.log 2>&1
  echo "exit $?"; grep -E "^test |test result|rel_l2|panicked|SKIPPED" $O/tests.log | cut -c1-230
}
need 2 "mutation checks: the tests must FAIL on a kernel with an injected bug" && {
  # mutg: gdn_delta_rule_bf16_lanes does not decay the .w component of the state; mutf: the e4m3 gemm drops its last k stage
  (cd $P/mutg && CARGO_TARGET_DIR=target/acc $T --test gdn_lanes -- --nocapture --test-threads=1 > $O/mut-gdn.log 2>&1; echo "mutg exit $? (expected non-zero)")
  grep -E "^test |test result|panicked" $O/mut-gdn.log | cut -c1-200
  (cd $P/mutf && CARGO_TARGET_DIR=target/acc $T --test quantmoe matmul_fp8_gemm_f8 -- --nocapture --test-threads=1 > $O/mut-f8.log 2>&1; echo "mutf exit $? (expected non-zero)")
  grep -E "^test |test result|panicked" $O/mut-f8.log | cut -c1-200
}
need 5 "admission (every registered kernel vs the cpu reference)" && {
  $T --test admission -- --nocapture --test-threads=1 > $O/admission.log 2>&1
  echo "exit $?"; grep -E "^test |test result|FAIL|gdn_delta_rule_bf16_lanes|matmul_fp8_gemm_f8|panicked" $O/admission.log | cut -c1-230 | head -30
}
need 3 "gdn timing, lanes per row 2/4/8" && {
  for l in 4 2 8; do echo "-- INTARSIA_GDN_LPR=$l"; INTARSIA_GDN_LPR=$l $T --test gdn_timing -- --nocapture 2>&1 | grep -E "us/launch"; done > $O/gdn_timing.log; cat $O/gdn_timing.log
}
need 4 "micro-benchmark: MoE (tile height 16/32/64), fp8 gemm (tile 64/128)" && {
  for bm in auto 16 32 64; do
    if [ $bm = auto ]; then unset INTARSIA_MOE_BM; else export INTARSIA_MOE_BM=$bm; fi
    echo "-- INTARSIA_MOE_BM=$bm"; $T --test quantmoe bench_real_shapes -- --ignored --nocapture 2>&1 | grep -E "moe_experts|fp8 (qkv|o_proj) M=(256|2048|4096) " | grep -E "T=(256|1024|4096)|M=(256|2048|4096)"
  done > $O/bench_moe_fp8.log
  unset INTARSIA_MOE_BM
  for b in 64 128; do echo "-- INTARSIA_F8_BM=$b"; INTARSIA_F8_BM=$b $T --test quantmoe bench_real_shapes -- --ignored --nocapture 2>&1 | grep -E "fp8_gemm_f8" ; done >> $O/bench_moe_fp8.log
  cat $O/bench_moe_fp8.log
}
acc() { # acc <name> <golden> <flags> <perturbations...>
  local name=$1 golden=$2 flags=$3; shift 3
  local A=$P/art/$name
  need $(( 3 + 5 * $# )) "accuracy $name baseline + $# envelope pass(es)" || return 0
  $BASE_BIN accuracy --model $M --backend cuda --golden $golden $flags --write-baseline $A.baseline.json --write-self-golden $A.self.safetensors > $O/$name.old.log 2>&1
  test -s $A.baseline.json || { echo "no baseline for $name"; return 0; }
  local p
  for p in "$@"; do
    need 5 "envelope $name $p" || continue
    $BASE_BIN accuracy --model $M --backend cuda --golden $golden $flags --noise-ref $A.baseline.json --noise-perturb $p --write-noise-envelope $A.envelope.json > $O/$name.envelope-${p//[^a-z0-9=]/_}.log 2>&1
    test -s $A.envelope.json && echo "envelope $p written"
  done
}
acc 35b-long $GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors "--mode both --decode-positions 64 --max-seq 16384 --mem-budget 70" chunk=1024
echo "== w2 done $(date +%T), $(left) min left"
