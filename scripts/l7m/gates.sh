#!/usr/bin/env bash
# Lane l7m, ON tatooine inside ONE fleetq job that holds both tatooine hosts and takes radiance down (card $CARD, default 1). Needs build.sh both.
#   gates.sh [sections: "BIT GC IDS PLAN"]   logs: ~/intarsiaml-l7m-logs/gates/
#   BIT  : real_35b_macro_prefill_is_bitwise_the_plain_schedule (logits of the last prompt row + 16 decode steps, macro 4096/8192/16384 vs off, at
#          1.5k / 4k / 8.3k / 16.6k tokens) and real_35b_batched_macro_prefill_equals_the_plain_local_engine
#   GC   : 35B batched == solo (17,550 pairs), batched engine == local engine, long prompts, tiny hybrids (batch_recurrent), recipe test
#   IDS  : 256 greedy ids of `generate` after ~1.5k / 4k / 8.3k / 16.6k-token prompts: main binary vs this branch (default) vs this branch with INTARSIA_PREFILL_MACRO=0
#   PLAN : `plan show` text + json, main vs branch default vs branch macro off (+ --max-batch 16), the memory plan
set -uo pipefail
SECTIONS=${*:-BIT GC IDS PLAN}; CARD=${CARD:-1}
S=$HOME/intarsiaml-l7m-hip/scripts/l7m
O=$HOME/intarsiaml-l7m-logs/gates; mkdir -p "$O"
D=$HOME/intarsiaml-l7m-hip/deploy/tatooine/dev.sh
RD=$HOME/intarsiaml-l7m-hip; RB=$HOME/intarsiaml-l7m-base
VD=intarsia-cargo-l7m-dev; VB=intarsia-cargo-l7m-base
M35=/models/Qwen3.6-35B-A3B-NVFP4
export INTARSIA_REGISTRY_VOL=intarsia-cargo-l7m-registry
want() { [[ " $SECTIONS " == *" $1 "* ]]; }
trap '$S/win.sh restore' EXIT
$S/win.sh stop
dev() { r=$1; v=$2; shift 2; INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=$CARD $D "$@" < /dev/null; }
F='^test |^test result|FAILED|panicked|^error|pairs|bitwise|MACRO'
T35="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
if want BIT; then
  echo "[gates] BIT $(date +%T)"
  dev $RD $VD $T35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 \
     real_35b_macro_prefill_is_bitwise_the_plain_schedule real_35b_batched_macro_prefill_equals_the_plain_local_engine 2>&1 | grep -aE "$F" | tee "$O/bit.log"
fi
if want GC; then
  echo "[gates] GC $(date +%T)"
  dev $RD $VD $T35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real --test batch_recurrent --test recipe_macro -- --nocapture --test-threads=1 2>&1 | grep -aE "$F" | tee "$O/gc.log"
  dev $RD $VD $T35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 real_35b_batched_engine_long_prompts_equal_the_local_engine 2>&1 | grep -aE "$F" | tee "$O/gc-long.log"
fi
if want IDS; then
  echo "[gates] IDS $(date +%T)"
  for arm in base dev dev0; do
    case $arm in base) r=$RB; v=$VB; e="" ;; dev) r=$RD; v=$VD; e="" ;; dev0) r=$RD; v=$VD; e="INTARSIA_PREFILL_MACRO=0" ;; esac
    for chars in ${CHARS:-4950 13600 27300 56000}; do
      dev $r $v env $e /target/release/intarsia generate -m $M35 --backend hip --max-seq 20480 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos \
        --show-ids -p "$(head -c $chars $RD/tools/prompts/corpus_docs.txt)" --max-tokens 256 > /dev/null 2> "$O/ids-$arm-$chars.err"
      grep -a '^ids:' "$O/ids-$arm-$chars.err" > "$O/ids-$arm-$chars.ids"
      echo "$arm $chars: $(grep -aoE 'prompt_tokens=[0-9]+|prefill[^,]*' "$O/ids-$arm-$chars.err" | head -1) ids=$(tr ',' '\n' < "$O/ids-$arm-$chars.ids" | wc -l)" | tee -a "$O/ids.log"
    done
  done
  for chars in ${CHARS:-4950 13600 27300 56000}; do for arm in dev dev0; do
    if cmp -s "$O/ids-base-$chars.ids" "$O/ids-$arm-$chars.ids" && [ -s "$O/ids-base-$chars.ids" ]; then echo "IDS $chars base vs $arm: identical"; else echo "IDS $chars base vs $arm: DIFFER"; fi
  done; done | tee -a "$O/ids.log"
fi
if want PLAN; then
  echo "[gates] PLAN $(date +%T)"
  for arm in base dev dev0; do
    case $arm in base) r=$RB; v=$VB; e="" ;; dev) r=$RD; v=$VD; e="" ;; dev0) r=$RD; v=$VD; e="INTARSIA_PREFILL_MACRO=0" ;; esac
    for mb in 1 16; do
      dev $r $v env $e /target/release/intarsia plan show --backend hip --model $M35 --max-seq 20480 --max-batch $mb > "$O/plan-$arm-mb$mb.txt" 2> "$O/plan-$arm-mb$mb.err"
      dev $r $v env $e /target/release/intarsia plan show --backend hip --model $M35 --max-seq 20480 --max-batch $mb --json > "$O/plan-$arm-mb$mb.json" 2>> "$O/plan-$arm-mb$mb.err"
    done
  done
  for mb in 1 16; do for k in txt json; do
    if cmp -s "$O/plan-base-mb$mb.$k" "$O/plan-dev0-mb$mb.$k" && [ -s "$O/plan-base-mb$mb.$k" ]; then echo "plan mb$mb $k base vs macro-off: byte-identical"; else echo "plan mb$mb $k base vs macro-off: DIFFER"; fi
  done; done | tee "$O/plan.log"
  for mb in 1 16; do echo "-- mb$mb dev vs base (text diff)"; diff "$O/plan-base-mb$mb.txt" "$O/plan-dev-mb$mb.txt" | head -20; done | tee -a "$O/plan.log"
fi
echo "[gates] done $(date +%T)"
