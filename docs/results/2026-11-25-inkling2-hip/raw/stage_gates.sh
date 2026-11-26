#!/usr/bin/env bash
# Stage: the gates of the real model on one R9700. Usage: stage_gates.sh [tiny] [gen] [g2short] [g2long] [admit]
#   tiny    G1 on the tiny fixture: cargo test -p intarsia-models --test inkling2 with INTARSIA_INKLING2_BACKEND=hip
#   gen     load + 32 greedy tokens (eager, no graphs)
#   g2short / g2long   accuracy --mode prefill against the oracle W4A16 goldens (budgets: KL21 <= 4e-3 / 6e-3, confident misses <= 1 / 2)
#   admit   kernels test for every Inkling-2 op
set -uo pipefail
R=$HOME/intarsiaml-ink-hip/docs/results/2026-11-25-inkling2-hip/raw
M=/models/Inkling-2-Mini
I=/target/release/intarsia
fail=0
for s in "$@"; do
  echo "==== $s ($(date '+%T'))"
  case $s in
    admit) bash "$R/stage_admit.sh" attn_res_mix loop_blend sconv_res conv_mix ink_in_proj ink_kv_prologue rel_sink_attn moe_experts moe_experts_mxfp8 moe_route; rc=$? ;;
    tiny) bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2 -- --test-threads 1 2>&1 | grep -E "^test |test result|panicked|error" -A2 | grep -v "^--" | tail -70; rc=${PIPESTATUS[0]} ;;
    gen) bash "$R/run.sh" $I generate -m $M --backend hip --prompt "The capital of France is" --max-tokens 32 ${GEN_FLAGS:-} 2>&1 | tail -25; rc=${PIPESTATUS[0]} ;;
    g2short) bash "$R/run.sh" $I accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors ${ACC_FLAGS:-} 2>&1 | tail -30; rc=${PIPESTATUS[0]} ;;
    g2long) bash "$R/run.sh" $I accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-long-top20.safetensors ${ACC_FLAGS:-} 2>&1 | tail -30; rc=${PIPESTATUS[0]} ;;
    *) echo "unknown stage $s"; rc=1 ;;
  esac
  echo "==== $s exit $rc"
  [ "$rc" -ne 0 ] && fail=1
done
exit $fail
