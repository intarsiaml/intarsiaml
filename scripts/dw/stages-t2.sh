#!/usr/bin/env bash
# Lane dw window t2 (ON tatooine, radiance down; scripts/dw/window.sh with W=t2): the default flip (INTARSIA_DFLASH_TP and INTARSIA_DFLASH_WIN_SPLIT unset = on).
#   Binaries: dw-intarsia-base = main (the CLI built for the t1 window, 3f46da4f7) and dw-intarsia = the flipped head (merged with main e45bd5e3c).
#   1. plan show, knobs unset: base vs dev for 27B tp2 off / mtp2, 35B tp2 off / mtp2, 8B tp1 off must be identical (nothing there touches a DFlash2 draft).
#   2. plan show, 27B DFlash k7 (JSON and text, --max-seq 4096 and 20480): dev with the knobs unset == dev with both =1 (the t1 knob-on plans); dev with both =0
#      lists no attn_decode_bf16_window_splitkv and differs.
#   3. generate short-0, knobs unset against --spec off: ids equal.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dh/.dw/lib.sh"
D=dev
clear "t2" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
for b in base dev; do
  BIN=dw-intarsia; [[ $b == base ]] && BIN=dw-intarsia-base
  BIN=$BIN plan "$D" 0,1 "$b-off" "$M27" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-mtp2" "$M27" $TP2 --spec mtp --spec-k 2
  BIN=$BIN plan "$D" 0,1 "$b-35off" "$M35" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-35mtp2" "$M35" $TP2 --spec mtp --spec-k 2
  BIN=$BIN plan "$D" 0 "$b-8off" "$M8" --spec off
done
for t in off mtp2 35off 35mtp2 8off; do
  same "plan-base-$t.out" "plan-dev-$t.out" "plan json $t, knobs unset"
  same "plantext-base-$t.out" "plantext-dev-$t.out" "plan text $t, knobs unset"
done
# shellcheck disable=SC2086
plan "$D" 0,1 "dev-dfl7" "$M27" $TP2 $DFL
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_WIN_SPLIT=1" plan "$D" 0,1 "dev-dfl7-on" "$M27" $TP2 $DFL
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_TP=0 INTARSIA_DFLASH_WIN_SPLIT=0" plan "$D" 0,1 "dev-dfl7-off" "$M27" $TP2 $DFL
same "plan-dev-dfl7.out" "plan-dev-dfl7-on.out" "plan json dfl7 4096: unset == both =1"
same "plantext-dev-dfl7.out" "plantext-dev-dfl7-on.out" "plan text dfl7 4096: unset == both =1"
same "plan-dev-dfl7.out" "plan-dev-dfl7-off.out" "plan json dfl7 4096: unset vs both =0 (expected DIFF)"
# shellcheck disable=SC2086
E="" run "$D" 0,1 300 "plantext-dev-dfl7-20k" plan show --backend hip --model "$M27" --mem-budget 29 --max-seq 20480 $TP2 $DFL
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_WIN_SPLIT=1" run "$D" 0,1 300 "plantext-dev-dfl7-on-20k" plan show --backend hip --model "$M27" --mem-budget 29 --max-seq 20480 $TP2 $DFL
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_TP=0 INTARSIA_DFLASH_WIN_SPLIT=0" run "$D" 0,1 300 "plantext-dev-dfl7-off-20k" plan show --backend hip --model "$M27" --mem-budget 29 --max-seq 20480 $TP2 $DFL
same "plantext-dev-dfl7-20k.out" "plantext-dev-dfl7-on-20k.out" "plan text dfl7 20480: unset == both =1"
log "plan lists the new kernel (20480): unset $(grep -ac attn_decode_bf16_window_splitkv "$O/plantext-dev-dfl7-20k.out") both=1 $(grep -ac attn_decode_bf16_window_splitkv "$O/plantext-dev-dfl7-on-20k.out") both=0 $(grep -ac attn_decode_bf16_window_splitkv "$O/plantext-dev-dfl7-off-20k.out")"
# 3. ids, knobs unset, against --spec off
P=$(cat "$TRK/short-0.txt")
g() { local tag=$1; shift
  E="INTARSIA_SPEC_TRACE=1" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids" 2>/dev/null; }
g "short-0-off" --spec off
# shellcheck disable=SC2086
g "short-0-unset" $DFL
if cmp -s "$O/gen-short-0-off.ids" "$O/gen-short-0-unset.ids" && [[ -s "$O/gen-short-0-off.ids" ]]; then log "PASS ids unset tp2 short-0"; else log "FAIL ids unset tp2 short-0"; fi
log "unset trace: $(grep -a 'dflash2 draft flush' "$O/gen-short-0-unset.err" | tail -n +5 | awk '{t+=$17; c++} END{if(c)printf "dflash2 total %.2f ms n=%d", t/c, c}') | $(grep -aE 'decode .*tok/s' "$O/gen-short-0-unset.err" | tail -1 | cut -c1-160)"
log "t2 done"
