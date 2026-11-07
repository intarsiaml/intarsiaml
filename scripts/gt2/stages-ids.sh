#!/usr/bin/env bash
# Lane gt2 (ON tatooine, radiance down, run by window.sh): (1) plan show (JSON) of main (gt2-intarsia-base) vs this branch (gt2-intarsia) with
# INTARSIA_SPEC_GRAPH_TP unset: 8B tp1 off, 27B and 35B tp1/tp2 off + mtp2; (2) rule (a): 256 greedy ids, tp2 --spec mtp --spec-k 2 with
# INTARSIA_SPEC_GRAPH_TP=1 (and INTARSIA_SPEC_TRACE=1: the "mtp graph step" lines prove the captured step ran) against tp2 --spec off, 35B and 27B,
# short-0 and 4k-1 (scripts/dfh/ids-prompts), `generate --show-ids`.
set -uo pipefail
export O=${1:?out dir}; mkdir -p "$O"
source "$HOME/intarsiaml-gt2/.gt2/lib.sh"
NIDS=${NIDS:-256}; P="$(T dev)/scripts/dfh/ids-prompts"
log "stages start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "stages" || exit 2
if [[ ${GATES:-1} == 1 ]]; then
  for cfg in "8b-tp1-off:$M8:0::--spec off" "27b-tp1-off:$M27:0::--spec off" "27b-tp1-mtp2:$M27:0::--spec mtp --spec-k 2" \
             "27b-tp2-off:$M27:0,1:$TP2:--spec off" "27b-tp2-mtp2:$M27:0,1:$TP2:--spec mtp --spec-k 2" \
             "35b-tp1-off:$M35:0::--spec off" "35b-tp1-mtp2:$M35:0::--spec mtp --spec-k 2" \
             "35b-tp2-off:$M35:0,1:$TP2:--spec off" "35b-tp2-mtp2:$M35:0,1:$TP2:--spec mtp --spec-k 2"; do
    tag=${cfg%%:*}; r=${cfg#*:}; m=${r%%:*}; r=${r#*:}; gpus=${r%%:*}; r=${r#*:}; ex=${r%%:*}; sa=${r#*:}
    for b in base head; do
      bin=gt2-intarsia; [[ $b == base ]] && bin=gt2-intarsia-base
      # shellcheck disable=SC2086
      BIN=$bin run dev "$gpus" 300 "plan-$tag-$b" plan show --backend hip --model "$m" --mem-budget 29 --max-seq 4096 --json $ex $sa
    done
    same "plan-$tag-base.out" "plan-$tag-head.out" "plan $tag"
  done
fi
g() { # g <gpus> <tag> <model> <prompt file> [args]
  local gpus=$1 tag=$2 m=$3 pf=$4; shift 4
  BIN=gt2-intarsia run dev "$gpus" 420 "gen-$tag" generate -m "$m" --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$pf")" --max-tokens "$NIDS" "$@"
  local rc=$?
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  return $rc
}
for mm in ${IDMODELS:-35b 27b}; do
  m=$M35; [[ $mm == 27b ]] && m=$M27
  for n in ${IDNAMES:-short-0 4k-1}; do
    # shellcheck disable=SC2086
    g 0,1 "$mm-tp2-$n-off" "$m" "$P/$n.txt" $TP2 --spec off
    # shellcheck disable=SC2086
    E="INTARSIA_SPEC_GRAPH_TP=1 INTARSIA_SPEC_TRACE=1" g 0,1 "$mm-tp2-$n-mtp2g" "$m" "$P/$n.txt" $TP2 --spec mtp --spec-k 2
    st=$(grep -ac 'mtp graph step' "$O/gen-$mm-tp2-$n-mtp2g.err"); cf=$(grep -ac 'capture failed' "$O/gen-$mm-tp2-$n-mtp2g.err")
    if [[ -s "$O/gen-$mm-tp2-$n-off.ids" ]] && cmp -s "$O/gen-$mm-tp2-$n-off.ids" "$O/gen-$mm-tp2-$n-mtp2g.ids"; then
      log "PASS ids $mm tp2 $n ($(grep -aoE 'prompt=[0-9]+' "$O/gen-$mm-tp2-$n-off.err" | tail -1); $st graph steps, $cf capture failures)"
    else log "FAIL ids $mm tp2 $n ($st graph steps, $cf capture failures)"; fi
    (( st > 0 )) || log "NOT-ENGAGED $mm tp2 $n: no graph step lines"
  done
done
log "ids summary: $(grep -c 'PASS ids' "$O/steps.log") PASS, $(grep -c 'FAIL ids' "$O/steps.log") FAIL, $(grep -c 'NOT-ENGAGED' "$O/steps.log") not engaged"
log "stages done"
