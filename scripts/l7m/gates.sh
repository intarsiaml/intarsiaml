#!/usr/bin/env bash
# Lane l7m, ON tatooine inside ONE fleetq job that holds both tatooine hosts and takes radiance down (card $CARD, default 1). Needs build.sh both.
#   gates.sh [sections: "BIT GC IDS PLAN"]   logs: ~/intarsiaml-l7m-logs/gates/
#   BIT  : real_35b_macro_prefill_is_bitwise_the_plain_schedule (logits of the last prompt row + 16 decode steps, macro 4096/8192/16384 vs off, at
#          1.5k / 4k / 8.3k / 16.6k tokens) and real_35b_batched_macro_prefill_equals_the_plain_local_engine
#   GC   : 35B batched == solo (17,550 pairs), batched engine == local engine, long prompts, tiny hybrids (batch_recurrent), recipe test
#   IDS  : 256 greedy ids of `generate` after ~1.5k / 4k / 8.3k / 16.6k-token prompts: main binary vs this branch (default) vs this branch with INTARSIA_PREFILL_MACRO=0
#   CTX  : max context of the 35B server (serve loaded line) and plan show, macro off / 8192 / 16384, single-stream and --max-batch 16, --max-seq 17408
#   PLAN : `plan show` text + json, main vs branch default vs branch macro off (+ --max-batch 16), the memory plan
#   OLD  : the UNCHANGED real_35b_macro_prefill_is_bitwise_the_plain_schedule of origin/main, run on the base tree (expected: fails at len 8292 macro 4096, 0 macro passes)
#   NEG  : the new test on the negative-control tree (dev + scripts/l7m/neg.patch): no env (must pass), INTARSIA_L7M_NEG=nomacro (must fail "did not run"), =ulp (must fail "logits differ")
# NOWIN=1: the caller (window.sh) stopped radiance and restores it; this script neither stops nor restores it.
set -uo pipefail
SECTIONS=${*:-BIT GC IDS PLAN CTX}; CARD=${CARD:-1}
S=$HOME/intarsiaml-l7m-hip/scripts/l7m
O=$HOME/intarsiaml-l7m-logs/gates; mkdir -p "$O"
D=$HOME/intarsiaml-l7m-hip/deploy/tatooine/dev.sh
RD=$HOME/intarsiaml-l7m-hip; RB=$HOME/intarsiaml-l7m-base
VD=intarsia-cargo-l7m-dev; VB=intarsia-cargo-l7m-base
M35=/models/Qwen3.6-35B-A3B-NVFP4
export INTARSIA_REGISTRY_VOL=intarsia-cargo-l7m-registry
want() { [[ " $SECTIONS " == *" $1 "* ]]; }
RN=$HOME/intarsiaml-l7m-neg; VN=intarsia-cargo-l7m-neg
if [ -z "${NOWIN:-}" ]; then trap '$S/win.sh restore' EXIT; $S/win.sh stop; fi
dev() { r=$1; v=$2; shift 2; INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=$CARD $D "$@" < /dev/null; }
F='^test |^test result|FAILED|panicked|^error|pairs|bitwise|MACRO'
T35="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
if want OLD; then
  echo "[gates] OLD (unchanged test, origin/main tree) $(date +%T)"
  dev $RB $VB $T35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 \
     real_35b_macro_prefill_is_bitwise_the_plain_schedule 2>&1 | grep -aE "$F" | tee "$O/old-main.log"
fi
if want BIT; then
  echo "[gates] BIT $(date +%T)"
  dev $RD $VD $T35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 \
     real_35b_macro_prefill_is_bitwise_the_plain_schedule real_35b_batched_macro_prefill_equals_the_plain_local_engine 2>&1 | grep -aE "$F" | tee "$O/bit.log"
fi
if want NEG; then
  for neg in none nomacro ulp; do
    echo "[gates] NEG $neg $(date +%T)"
    dev $RN $VN $T35 INTARSIA_L7M_NEG=$neg cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 \
       real_35b_macro_prefill_is_bitwise_the_plain_schedule 2>&1 | grep -aE "$F" | tee "$O/neg-$neg.log"
  done
fi
if want GC; then
  echo "[gates] GC $(date +%T)"
  dev $RD $VD $T35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real --test batch_recurrent --test recipe_macro -- --nocapture --test-threads=1 2>&1 | grep -aE "$F" | tee "$O/gc.log"
  dev $RD $VD $T35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --ignored --nocapture --test-threads=1 real_35b_batched_engine_long_prompts_equal_the_local_engine 2>&1 | grep -aE "$F" | tee "$O/gc-long.log"
  # the same batched == solo and long prompts with the 8192-row macro (the alternative to the recipe's 16384), so no extra window is needed if it ships
  dev $RD $VD $T35 INTARSIA_PREFILL_MACRO=8192 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real -- --include-ignored --nocapture --test-threads=1 real_35b_batched_decode_is_bitwise_the_solo_step real_35b_batched_engine_long_prompts_equal_the_local_engine real_35b_batched_macro_prefill_equals_the_plain_local_engine 2>&1 | grep -aE "$F" | tee "$O/gc-m8192.log"
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
    for mb in 1; do  # `plan show` has no --max-batch (serve sets it): the batched plan is the serve "context" line, section CTX
      dev $r $v env $e /target/release/intarsia plan show --backend hip --model $M35 --max-seq 20480 > "$O/plan-$arm-mb$mb.txt" 2> "$O/plan-$arm-mb$mb.err"
      dev $r $v env $e /target/release/intarsia plan show --backend hip --model $M35 --max-seq 20480 --json > "$O/plan-$arm-mb$mb.json" 2>> "$O/plan-$arm-mb$mb.err"
    done
  done
  for mb in 1; do for k in txt json; do
    if cmp -s "$O/plan-base-mb$mb.$k" "$O/plan-dev0-mb$mb.$k" && [ -s "$O/plan-base-mb$mb.$k" ]; then echo "plan mb$mb $k base vs macro-off: byte-identical"; else echo "plan mb$mb $k base vs macro-off: DIFFER"; fi
  done; done | tee "$O/plan.log"
  for mb in 1; do echo "-- mb$mb dev vs base (text diff)"; diff "$O/plan-base-mb$mb.txt" "$O/plan-dev-mb$mb.txt" | head -20; done | tee -a "$O/plan.log"
fi
if want CTX; then
  echo "[gates] CTX $(date +%T)"
  for m in 0 8192 16384; do for mode in sb mb16; do
    mb=-; mba=""; [ $mode = mb16 ] && { mb=16; mba="--max-batch 16"; }
    lab="ctx-m$m-$mode"
    INTARSIA_PREFILL_MACRO=$m CARD=$CARD MAX_SEQ=17408 bash $S/serve.sh up dev $mb 2>&1 | tail -1
    bash $S/serve.sh down $lab | tail -1
    echo "macro $m $mode: $(grep -aoE 'context [0-9]+ tokens' $HOME/intarsiaml-l7m-logs/$lab.log)" | tee -a "$O/ctx.log"
    [ $mode = sb ] && dev $RD $VD env INTARSIA_PREFILL_MACRO=$m /target/release/intarsia plan show --backend hip --model $M35 --max-seq 17408 > "$O/ctxplan-m$m-$mode.txt" 2> "$O/ctxplan-m$m-$mode.err"
    [ $mode = sb ] && dev $RD $VD env INTARSIA_PREFILL_MACRO=$m /target/release/intarsia plan show --backend hip --model $M35 --max-seq 17408 --json > "$O/ctxplan-m$m-$mode.json" 2>> "$O/ctxplan-m$m-$mode.err"
  done; done
  python3 - <<'PY' 2>&1 | tee -a "$O/ctx.log"
import re,os
O=os.environ["HOME"]+"/intarsiaml-l7m-logs/gates"
ctx={}
for l in open(O+"/ctx.log"):
    m=re.match(r"macro (\d+) (\w+): context (\d+) tokens",l)
    if m: ctx[(int(m.group(1)),m.group(2))]=int(m.group(3))
for mode in ("sb","mb16"):
    b=ctx.get((0,mode))
    for m in (8192,16384):
        c=ctx.get((m,mode))
        if b and c: print(f"{mode} macro {m}: context {c} vs {b} off = {(c/b-1)*100:+.1f} %")
PY
fi
echo "[gates] done $(date +%T)"
