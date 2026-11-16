#!/usr/bin/env bash
# Lane hip-reuse, the radiance-down window (STEPS selects; default all):
#   reuse   tests/hybrid_reuse_real.rs on the real 35B (HIP, one R9700): the bit/KL shapes probe with the chunked GDN delta rule on
#           (default) and off (INTARSIA_GDN_CHUNK=0), engine_hybrid_reuse_equals_cold under the default rule (INTARSIA_PREFIX_REUSE=any,
#           INTARSIA_PROBE_ASSERT=1: the original failing assertion) with the knob on and off, and under the strict rule (pass expected)
#   rec     qualify's acc-4k stage with no HIP baseline installed -> MISSING-BASELINE; it writes the acc-4k baseline
#   chk     the recorded baseline installed in this tree's golden/baselines/, the stage again -> PASS expected (twice: determinism)
set -u
. "$(dirname "$0")/lib.sh"
cd "$TREE"
STEPS=${STEPS:-"reuse rec chk"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
W=/work/${O#"$TREE"/}
G=/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
OR=/work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
Q="$BIN qualify --backend hip --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 29 --bench-sizes 128 --reps 1 --gen 16"
SKA="--skip kernels --skip gate --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip envelope --skip bench-long --skip ab --skip tp"
B4K=qwen36-35b-a3b-nvfp4-vllm-4k-top20.hip.baseline.json
log "stages start; $STAMP; STEPS=$STEPS"
if has reuse; then
  P="INTARSIA_PROBE_DIR=$M35 INTARSIA_PROBE_BACKEND=hip INTARSIA_REQUIRE_GPU=1"
  T="cargo test --release --no-fail-fast -p intarsia-models --test hybrid_reuse_real -- --nocapture --test-threads=1"
  E="$P" run 0 900 shapes-gdn-on $T hybrid_reuse_shapes
  E="$P INTARSIA_GDN_CHUNK=0" run 0 900 shapes-gdn-off $T hybrid_reuse_shapes
  E="$P INTARSIA_PREFIX_REUSE=any INTARSIA_PROBE_ASSERT=1" run 0 900 engine-any $T engine_hybrid_reuse_equals_cold
  E="$P INTARSIA_PREFIX_REUSE=any INTARSIA_PROBE_ASSERT=1 INTARSIA_GDN_CHUNK=0" run 0 900 engine-any-gdn-off $T engine_hybrid_reuse_equals_cold
  E="$P INTARSIA_PREFIX_REUSE=any" run 0 900 engine-any-report $T engine_hybrid_reuse_equals_cold
  E="$P" run 0 900 engine-exact $T engine_hybrid_reuse_equals_cold
  E="$P INTARSIA_GDN_CHUNK=0" run 0 900 engine-exact-gdn-off $T engine_hybrid_reuse_equals_cold
  grep -ahE "^test |^test result|prefix reuse rule|B after A|A after B|A repeated|panicked" "$O"/engine-*.out "$O"/engine-*.err | sort | uniq -c | tee -a "$O/steps.log"
fi
if has rec; then
  mv "golden/baselines/$B4K" "$O/" 2>/dev/null
  run 0 1200 q-rec $Q --model $M35 --out $W/q-rec $SKA
  grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/q-rec.out" | cut -c1-260 | tee -a "$O/steps.log"
  cp "$O/q-rec/baselines/acc-4k.baseline.json" "$O/$B4K" && log "recorded: $(ls -la "$O/$B4K" | awk '{print $5}') $B4K"
fi
if has chk; then
  cp "$O/$B4K" golden/baselines/
  for i in 1 2; do
    run 0 1200 q-chk$i $Q --model $M35 --out $W/q-chk$i $SKA
    grep -aE "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" "$O/q-chk$i.out" | cut -c1-260 | tee -a "$O/steps.log"
  done
fi
log "stages done"
