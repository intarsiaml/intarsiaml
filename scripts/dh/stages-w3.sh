#!/usr/bin/env bash
# Lane dh window w3 (ON tatooine, radiance down; PRE=w3 of cat-window.sh): item 2 with the whole draft, and the mxfp4_a8 tests on the reverted tree.
#   card 1: tests mxfp4_a8 + mxfp4_a8_fuse + rows_exact, --no-fail-fast, --nocapture (FUSEHASH).
#   generate short-0 and 4k-1, INTARSIA_SPEC_TRACE=1, ids against off: A T A T K (A = knobs unset, T = INTARSIA_DFLASH_DEVICE_TOPK=1,
#   K = INTARSIA_DFLASH_DEVICE_TOPK=check), whole draft on rank 0 (INTARSIA_DFLASH_TP unset).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-dh/.dh/lib.sh"
D=dev
clear "w3" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
htest "$D" "a8" 1 900 --no-fail-fast --test mxfp4_a8 --test mxfp4_a8_fuse --test rows_exact -- --nocapture
log "a8 gate: $(grep -aE '^FUSEHASH|^FUSEMODE|^test result|FAILED' "$O/htest-a8.log" 2>/dev/null | sort | uniq -c | tr '\n' '|' | cut -c1-400)"
trace() {
  log "trace $1: $(grep -a 'verify + rollback' "$O/gen-$1.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f pass %.2f ms n=%d", d/c, s/c, (d+s)/c, c}') | $(grep -a 'dflash2 draft flush' "$O/gen-$1.err" | tail -n +5 | awk '{f+=$5; b+=$7; h+=$9; sp+=$11; dl+=$13; se+=$15; t+=$17; c++} END{if(c)printf "dflash2 flush %.2f block %.2f head %.2f selproj %.2f dl %.2f select %.2f total %.2f n=%d", f/c,b/c,h/c,sp/c,dl/c,se/c,t/c,c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$1.err" | tail -1 | cut -c1-160) | $(grep -aE 'tokens per pass' "$O/gen-$1.err" | tail -1 | cut -c1-120)"
}
declare -A ENVS=([A]="" [T]="INTARSIA_DFLASH_DEVICE_TOPK=1" [K]="INTARSIA_DFLASH_DEVICE_TOPK=check")
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { local tag=$1 e=$2; shift 2
    E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids" 2>/dev/null; }
  g "$n-off" "" --spec off
  for a in A1 T1 A2 T2 K; do
    # shellcheck disable=SC2086
    g "$n-$a" "${ENVS[${a:0:1}]}" $DFL
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    trace "$n-$a"
  done
  log "topk check $n: $(grep -a 'dflash2 topk check: rank .* rows' "$O/gen-$n-K.err" 2>/dev/null | awk '{r+=$7; d+=$9; c++} END{printf "%d calls, %d rows, %d differing", c, r, d}')"
done
log "w3 stages done"
