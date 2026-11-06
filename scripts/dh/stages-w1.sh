#!/usr/bin/env bash
# Lane dh window w1 (ON tatooine, radiance down; PRE=w1 of cat-window.sh): items 0 and 1.
#   plan show, base (main 982ec8b27, dh-intarsia-base) vs dev (the branch, dh-intarsia), 27B tp2: --spec off, --spec mtp k2 (must be identical),
#   --spec dflash k7 (item 0: the window kernel planned); dev --spec dflash k7 with INTARSIA_DFLASH_TP=1 (recorded).
#   generate, short-0 and 4k-1: off; dfl7 A (knob unset) / B (INTARSIA_DFLASH_TP=1) in the order A B A B, INTARSIA_SPEC_TRACE=1; ids against off.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dh/.dh/lib.sh"
D=dev
clear "w1" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
for b in base dev; do
  BIN=dh-intarsia; [[ $b == base ]] && BIN=dh-intarsia-base
  BIN=$BIN plan "$D" 0,1 "$b-off" "$M27" $TP2 --spec off
  BIN=$BIN plan "$D" 0,1 "$b-mtp2" "$M27" $TP2 --spec mtp --spec-k 2
  # shellcheck disable=SC2086
  BIN=$BIN plan "$D" 0,1 "$b-dfl7" "$M27" $TP2 $DFL
done
same plan-base-off.out plan-dev-off.out "plan json 27B tp2 off"
same plan-base-mtp2.out plan-dev-mtp2.out "plan json 27B tp2 mtp k2"
same plantext-base-off.out plantext-dev-off.out "plan text 27B tp2 off"
same plantext-base-mtp2.out plantext-dev-mtp2.out "plan text 27B tp2 mtp k2"
log "item 0: window kernel in the dfl7 plan: base $(grep -c attn_decode_bf16_window "$O/plantext-base-dfl7.out") dev $(grep -c attn_decode_bf16_window "$O/plantext-dev-dfl7.out")"
# shellcheck disable=SC2086
E="INTARSIA_DFLASH_TP=1" plan "$D" 0,1 "dev-dfl7-tp" "$M27" $TP2 $DFL
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { # g <tag> <env> [args]
    local tag=$1 e=$2; shift 2
    E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  }
  g "$n-off" "" --spec off
  # shellcheck disable=SC2086
  { g "$n-A1" "" $DFL; g "$n-B1" "INTARSIA_DFLASH_TP=1" $DFL; g "$n-A2" "" $DFL; g "$n-B2" "INTARSIA_DFLASH_TP=1" $DFL; }
  for a in A1 B1 A2 B2; do
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    log "trace $a $n: $(grep -a 'verify + rollback' "$O/gen-$n-$a.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f ms n=%d", d/c, s/c, c}') | $(grep -a 'dflash2 draft flush' "$O/gen-$n-$a.err" | tail -n +5 | awk '{f+=$5; b+=$7; h+=$9; sp+=$11; dl+=$13; se+=$15; t+=$17; c++} END{if(c)printf "dflash2 flush %.2f block %.2f head %.2f selproj %.2f dl %.2f select %.2f total %.2f n=%d", f/c,b/c,h/c,sp/c,dl/c,se/c,t/c,c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$n-$a.err" | tail -1 | cut -c1-160)"
  done
done
log "w1 stages done"
