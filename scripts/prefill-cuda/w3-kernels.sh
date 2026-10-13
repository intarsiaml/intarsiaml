#!/usr/bin/env bash
# Window 3 (spark1, vLLM down, inside scripts/prefill-cuda-window.sh): tests of the new kernels (with mutation checks),
# kernel micro-benchmarks, and nsys profiles of the new build at prefill chunk 256 and 2048.
#   dev tree: ~/intarsia-perf/$DEV (default dev), mutants mutg (gdn lanes without the w decay) and mutf (f8 gemm drops its last k stage)
set -u
P=$HOME/intarsia-perf
DEV=$P/${DEV:-dev}
BIN=$DEV/target/acc/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
O=$P/out/w3; mkdir -p $O
DEADLINE=$(( $(date +%s) + ${W3_BUDGET_MIN:-27} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
export CARGO_TARGET_DIR=target/acc
T="cargo test --release -p intarsia-backend-cuda"

need 3 "kernel tests (dev): gdn_lanes quantmoe attention admission" && {
  cd $DEV
  $T --test gdn_lanes --test quantmoe --test attention --test admission -- --nocapture --test-threads=1 > $O/tests.log 2>&1
  echo "exit $?"; grep -E "^test |test result|panicked|rel_l2.*lanes kernel [0-9.e-]+$" $O/tests.log | grep -v "^dk=" | cut -c1-200
  grep -E "gdn_delta_rule_bf16_lanes +[0-9]|matmul_fp8_gemm_f8 +[0-9]|moe_experts_mma_bf16 +[0-9]" $O/tests.log | cut -c1-200
  grep -E "act_quant=.* rel_l2" $O/tests.log | cut -c1-200
}
need 2 "mutation checks (both must FAIL)" && {
  (cd $P/mutg && $T --test gdn_lanes -- --nocapture --test-threads=1 > $O/mut-gdn.log 2>&1; echo "mutg exit $? (expected non-zero)")
  grep -E "^test |test result|panicked" $O/mut-gdn.log | cut -c1-200
  (cd $P/mutf && $T --test quantmoe matmul_fp8_gemm_f8 -- --nocapture --test-threads=1 > $O/mut-f8.log 2>&1; echo "mutf exit $? (expected non-zero)")
  grep -E "^test |test result|panicked" $O/mut-f8.log | cut -c1-200
}
need 3 "gdn timing (lanes per row 4 and 2)" && {
  cd $DEV
  for l in 4 2; do echo "-- INTARSIA_GDN_LPR=$l"; INTARSIA_GDN_LPR=$l $T --test gdn_timing -- --nocapture 2>&1 | grep -E "us/launch" | grep -E "T=(16|512|2048)"; done | tee $O/gdn_timing.log
}
need 4 "micro-benchmark: fp8 gemm bf16 vs e4m3, MoE experts" && {
  cd $DEV
  $T --test quantmoe bench_real_shapes -- --ignored --nocapture 2>&1 | grep -E "fp8_gemm|moe_experts T=(256|1024|2048|4096)" | tee $O/bench.log
  for b in 64 128; do echo "-- INTARSIA_F8_BM=$b"; INTARSIA_F8_BM=$b $T --test quantmoe bench_real_shapes -- --ignored --nocapture 2>&1 | grep -E "fp8_gemm_f8"; done | tee -a $O/bench.log
  for bm in 32 64; do echo "-- INTARSIA_MOE_BM=$bm"; INTARSIA_MOE_BM=$bm $T --test quantmoe bench_real_shapes -- --ignored --nocapture 2>&1 | grep -E "moe_experts T=(1024|2048|4096) .*mma"; done | tee -a $O/bench.log
}
GEN="--model $M --backend cuda --max-seq 8192 --mem-budget 45 --max-tokens 1 --temperature 0"
if command -v nsys >/dev/null; then
  for ck in 2048 256; do
    need 3 "nsys new build, prompt~4096 chunk $ck" || continue
    nsys profile -t cuda --force-overwrite true -o $O/nsys4096-c$ck $BIN generate $GEN --prefill-chunk $ck --profile-every 0 -p "$(cat $P/prompts/p4096.txt)" > $O/nsys4096-c$ck.out 2> $O/nsys4096-c$ck.err
    grep -E "finish=" $O/nsys4096-c$ck.err | cut -c1-140
    nsys stats -r cuda_gpu_kern_sum -f csv -o $O/nsys4096-c$ck $O/nsys4096-c$ck.nsys-rep > $O/nsys4096-c$ck.stats.log 2>&1 || tail -3 $O/nsys4096-c$ck.stats.log
  done
fi
echo "== w3 done $(date +%T), $(left) min left"
