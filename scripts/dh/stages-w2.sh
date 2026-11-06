#!/usr/bin/env bash
# Lane dh window w2 (ON tatooine, radiance down; PRE=w2 of cat-window.sh): item 3 and the data for it.
#   generate short-0 and 4k-1, INTARSIA_SPEC_TRACE=1, ids against off, on top of the best of items 1-2 (BASEENV, set by the caller from w1):
#     X = BASEENV, D = X + INTARSIA_HIP_AR_LL_MAX=262144 (3a), E = X + INTARSIA_HIP_A8_ILP=2, F = X + INTARSIA_HIP_A8_ILP=4 (3b; ILPS="" drops
#     them when w1's kernel gates or bench did not pass), order X1 D1 E1 F1 X2 D2 E2 F2.
#   MTP k2 with and without 3a (the 3-row check), MTP k1..k6 at short-0 (verify + rollback against rows), profiled op tables of off and dfl7 X.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dh/.dh/lib.sh"
D=dev
clear "w2" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
X=${BASEENV:-"INTARSIA_DFLASH_TP=1 INTARSIA_DFLASH_DEVICE_TOPK=1"}
ILPS=${ILPS-"2 4"}
declare -A ENVS=([X]="$X" [D]="$X INTARSIA_HIP_AR_LL_MAX=262144" [E]="$X INTARSIA_HIP_A8_ILP=2" [F]="$X INTARSIA_HIP_A8_ILP=4")
ARMS="X1 D1"; [[ $ILPS == *2* ]] && ARMS+=" E1"; [[ $ILPS == *4* ]] && ARMS+=" F1"
ARMS="$ARMS ${ARMS//1/2}"
log "w2: X=[$X] arms $ARMS"
trace() {
  log "trace $1: $(grep -a 'verify + rollback' "$O/gen-$1.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f pass %.2f ms n=%d", d/c, s/c, (d+s)/c, c}') | $(grep -a 'dflash2 draft flush' "$O/gen-$1.err" | tail -n +5 | awk '{f+=$5; b+=$7; h+=$9; sp+=$11; dl+=$13; se+=$15; t+=$17; c++} END{if(c)printf "dflash2 flush %.2f block %.2f head %.2f selproj %.2f dl %.2f select %.2f total %.2f n=%d", f/c,b/c,h/c,sp/c,dl/c,se/c,t/c,c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$1.err" | tail -1 | cut -c1-160) | $(grep -aE 'tokens per pass' "$O/gen-$1.err" | tail -1 | cut -c1-120)"
}
g() { # g <tag> <prompt name> <env> [args]
  local tag=$1 n=$2 e=$3; shift 3
  E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$TRK/$n.txt")" --max-tokens 256 $TP2 "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids" 2>/dev/null
}
for n in short-0 4k-1; do
  g "$n-off" "$n" "" --spec off
  for a in $ARMS; do
    # shellcheck disable=SC2086
    g "$n-$a" "$n" "${ENVS[${a:0:1}]}" $DFL
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    trace "$n-$a"
  done
done
for a in mtp2-1 mtp2D-1 mtp2-2 mtp2D-2; do
  e=""; [[ $a == mtp2D* ]] && e="INTARSIA_HIP_AR_LL_MAX=262144"
  g "$a" short-0 "$e" --spec mtp --spec-k 2
  if cmp -s "$O/gen-short-0-off.ids" "$O/gen-$a.ids"; then log "PASS ids $a"; else log "FAIL ids $a"; fi
  trace "$a"
done
for kk in 1 3 4 5 6; do
  g "mtp$kk" short-0 "" --spec mtp --spec-k $kk
  log "vcurve mtp k$kk ($((kk + 1)) rows): $(grep -a 'verify + rollback' "$O/gen-mtp$kk.err" | tail -n +5 | awk '{s+=$(NF-1); c++} END{if(c)printf "verify+rollback %.2f ms n=%d", s/c, c}')"
done
P=$(cat "$TRK/short-0.txt")
for a in off dfl; do
  if [[ $a == off ]]; then e=""; args="--spec off"; else e="$X"; args="$DFL"; fi
  # shellcheck disable=SC2086
  E="$e" run "$D" 0,1 420 "optable-$a" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 1 --op-table --ignore-eos -p "$P" --max-tokens 128 $TP2 $args
done
log "w2 stages done"
