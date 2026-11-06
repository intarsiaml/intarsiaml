#!/usr/bin/env bash
# Lane dh window w1 (ON tatooine, radiance down; PRE=w1 of cat-window.sh, which then serves the suite smoke arms): items 0-2 and the gates.
#   1. INTARSIA_DFLASH_TP=1 plan show at serve's --max-seq 20480 FIRST: if it fails, every knob-on arm below is skipped (SPLIT_OK=0).
#   2. card 0: `kernels test --backend hip --quick` (dev, default env) in the background; card 1: the 3b kernel gates (mxfp4_a8, mxfp4_a8_fuse
#      with its FUSEHASH, rows_exact) at INTARSIA_HIP_A8_ILP unset / 2 / 4 and VRC_BENCH=1 verify_gemv_bench at 1 / 2 / 4 (inputs to w2).
#   3. plan show base (main 982ec8b27, dh-intarsia-base) vs dev: 27B tp2 off, mtp k2, dfl7; 35B tp2 off, mtp k2; 8B tp1 off.
#   4. generate, short-0 and 4k-1, INTARSIA_SPEC_TRACE=1, ids against off, order Z1 A1 B1 C1 Z2 A2 B2 C2 K:
#      Z = base binary dfl7 (item 0 speed control), A = dev knobs unset, B = INTARSIA_DFLASH_TP=1, C = B + INTARSIA_DFLASH_DEVICE_TOPK=1,
#      K = B + INTARSIA_DFLASH_DEVICE_TOPK=check (device vs host candidates, every differing row printed).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dh/.dh/lib.sh"
D=dev
clear "w1" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
# 1. the split draft loads and plans at serve's settings (memory lines per rank in the text plan)
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_TP=1" run "$D" 0,1 300 "plantext-dev-dfl7-tp-20k" plan show --backend hip --model "$M27" --mem-budget 29 --max-seq 20480 $TP2 $DFL
SPLIT_OK=1
if [[ ${DRY:-0} != 1 ]] && { grep -aqiE "no kernel|error" "$O/plantext-dev-dfl7-tp-20k.err" || [[ ! -s "$O/plantext-dev-dfl7-tp-20k.out" ]]; }; then
  SPLIT_OK=0; log "SPLIT NOT OK: the split-draft plan failed, knob-on arms skipped: $(grep -aiE 'no kernel|error' "$O/plantext-dev-dfl7-tp-20k.err" | head -3 | tr '\n' '|')"
fi
log "split plan at 20480: $(grep -aiE 'memory|weights|kv|budget|rank' "$O/plantext-dev-dfl7-tp-20k.out" 2>/dev/null | head -8 | tr '\n' '|')"
# 2. kernels test on card 0 (background) while the 3b gates run on card 1
if [[ ${DRY:-0} == 1 ]]; then log "DRY kernels test --backend hip --quick gpu 0"; else
  ( INTARSIA_REPO=$(T "$D") HIP_VISIBLE_DEVICES=0 timeout 1500 "$(T "$D")/deploy/tatooine/dev.sh" /work/dh-intarsia kernels test --backend hip --quick \
      > "$O/kernels-test.log" 2>&1 < /dev/null; echo "exit $?" >> "$O/kernels-test.log" ) &
  KT=$!
fi
for il in 1 2 4; do
  e=""; [[ $il != 1 ]] && e="INTARSIA_HIP_A8_ILP=$il"
  E="$e" htest "$D" "a8-ilp$il" 1 900 --test mxfp4_a8 --test mxfp4_a8_fuse --test rows_exact -- --nocapture
  log "a8 gate ilp $il: $(grep -aE '^FUSEHASH|^FUSEMODE' "$O/htest-a8-ilp$il.log" 2>/dev/null | sort -u | tr '\n' ' ')"
  E="VRC_BENCH=1 $e" htest "$D" "bench-ilp$il" 1 900 --test verify_gemv_bench -- --nocapture
done
if [[ -n ${KT:-} ]]; then wait "$KT"; log "kernels test: $(tail -4 "$O/kernels-test.log" | tr '\n' '|' | cut -c1-300)"; fi
# 3. plans: base vs dev, every other model and spec mode unchanged
for b in base dev; do
  BIN=dh-intarsia; [[ $b == base ]] && BIN=dh-intarsia-base
  BIN=$BIN plan "$D" 0,1 "$b-off" "$M27" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-mtp2" "$M27" $TP2 --spec mtp --spec-k 2
  # shellcheck disable=SC2086
  BIN=$BIN plan "$D" 0,1 "$b-dfl7" "$M27" $TP2 $DFL
  BIN=$BIN plan "$D" 0,1 "$b-35off" "$M35" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-35mtp2" "$M35" $TP2 --spec mtp --spec-k 2
  BIN=$BIN plan "$D" 0 "$b-8off" "$M8" --spec off
done
for t in off mtp2 35off 35mtp2 8off; do
  same "plan-base-$t.out" "plan-dev-$t.out" "plan json $t"
  same "plantext-base-$t.out" "plantext-dev-$t.out" "plan text $t"
done
log "item 0: window kernel in the dfl7 plan: base $(grep -c attn_decode_bf16_window "$O/plantext-base-dfl7.out" 2>/dev/null) dev $(grep -c attn_decode_bf16_window "$O/plantext-dev-dfl7.out" 2>/dev/null)"
# 4. traces
trace() { # trace <gen tag>
  log "trace $1: $(grep -a 'verify + rollback' "$O/gen-$1.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f pass %.2f ms n=%d", d/c, s/c, (d+s)/c, c}') | $(grep -a 'dflash2 draft flush' "$O/gen-$1.err" | tail -n +5 | awk '{f+=$5; b+=$7; h+=$9; sp+=$11; dl+=$13; se+=$15; t+=$17; c++} END{if(c)printf "dflash2 flush %.2f block %.2f head %.2f selproj %.2f dl %.2f select %.2f total %.2f n=%d", f/c,b/c,h/c,sp/c,dl/c,se/c,t/c,c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$1.err" | tail -1 | cut -c1-160) | $(grep -aE 'tokens per pass' "$O/gen-$1.err" | tail -1 | cut -c1-120)"
}
declare -A ENVS=([Z]="" [A]="" [B]="INTARSIA_DFLASH_TP=1" [C]="INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_DEVICE_TOPK=1" [K]="INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_DEVICE_TOPK=check")
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { # g <tag> <env> [args]
    local tag=$1 e=$2; shift 2
    E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids" 2>/dev/null
  }
  g "$n-off" "" --spec off
  for a in Z1 A1 B1 C1 Z2 A2 B2 C2 K; do
    x=${a:0:1}
    if [[ $SPLIT_OK == 0 && $x =~ [BCK] ]]; then log "SKIP $a $n (split plan failed)"; continue; fi
    BIN=dh-intarsia; [[ $x == Z ]] && BIN=dh-intarsia-base
    # shellcheck disable=SC2086
    BIN=$BIN g "$n-$a" "${ENVS[$x]}" $DFL
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    trace "$n-$a"
  done
  log "topk check $n: $(grep -a 'dflash2 topk check: rank .* rows' "$O/gen-$n-K.err" 2>/dev/null | awk '{r+=$7; d+=$9; c++} END{printf "%d calls, %d rows, %d differing", c, r, d}')"
done
log "w1 stages done"
