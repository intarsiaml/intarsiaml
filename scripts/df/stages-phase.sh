#!/usr/bin/env bash
# Lane df (ON tatooine, radiance down; PRE=phase of cat-window.sh): per-phase wall time of speculative passes at tp=2 on current main
# (INTARSIA_SPEC_TRACE=1): off (graphs on and --no-graphs: launch/host cost of an eager one-row step), mtp k2, mtp k7 (an 8-row verify
# without the DFlash draft), dflash k7; short and 4k prompts; greedy ids against off. Then one profiled dflash k7, mtp k7 and off run
# (--profile-every 1 --op-table) for the per-op split of a pass.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/intarsiaml-df/.df/lib.sh"
D=dev
clear "phase" || exit 2
TRK="$(T "$D")/scripts/dfh/ids-prompts"
for n in short-0 4k-1; do
  P=$(cat "$TRK/$n.txt")
  g() { # g <tag> <env> [args]
    local tag=$1 e=$2; shift 2
    E="INTARSIA_SPEC_TRACE=1 $e" run "$D" 0,1 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
      --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $TP2 "$@"
    grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  }
  g "$n-off" "" --spec off
  g "$n-offng" "" --spec off --no-graphs
  g "$n-mtp2" "" --spec mtp --spec-k 2
  g "$n-mtp7" "" --spec mtp --spec-k 7
  g "$n-dfl7" "" --spec dflash --spec-draft "$DRAFT" --spec-k 7
  for a in offng mtp2 mtp7 dfl7; do
    if cmp -s "$O/gen-$n-off.ids" "$O/gen-$n-$a.ids" && [[ -s "$O/gen-$n-off.ids" ]]; then log "PASS ids $a tp2 $n"; else log "FAIL ids $a tp2 $n"; fi
    log "verify $a $n: $(grep -a 'verify + rollback' "$O/gen-$n-$a.err" | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f ms n=%d", d/c, s/c, c}') | $(grep -a 'dflash2 draft flush' "$O/gen-$n-$a.err" | tail -n +5 | awk '{f+=$5; b+=$7; h+=$9; sp+=$11; dl+=$13; se+=$15; t+=$17; c++} END{if(c)printf "dflash2 flush %.2f block %.2f head %.2f selproj %.2f dl %.2f select %.2f total %.2f n=%d", f/c,b/c,h/c,sp/c,dl/c,se/c,t/c,c}') | $(grep -aE 'decode .*tok/s' "$O/gen-$n-$a.err" | tail -1 | cut -c1-160)"
  done
done
P=$(cat "$TRK/short-0.txt")
for a in "dfl7:--spec dflash --spec-draft $DRAFT --spec-k 7" "mtp7:--spec mtp --spec-k 7" "off:--spec off"; do
  tag=${a%%:*}; args=${a#*:}
  # shellcheck disable=SC2086
  E="" run "$D" 0,1 420 "optable-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 1 --op-table --ignore-eos -p "$P" --max-tokens 128 $TP2 $args
done
log "phase done"
