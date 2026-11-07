#!/usr/bin/env bash
# Lane gt2 (ON tatooine, radiance down, run by window.sh): the 35B gates. (1) plan show of the 35B at tp1 / tp2, --spec off and --spec mtp k2 (the plans of the
# main binary: this lane changes no code unless the run below fails), (2) the batching refusal of `--spec mtp --max-batch 16 --tp 2`, (3) rule (a): 256 greedy
# ids, tp2 --spec mtp --spec-k 2 against tp2 --spec off, on short-0, 4k-1 and 16k-0 (scripts/dfh/ids-prompts), `generate --show-ids`.
set -uo pipefail
export O=${1:?out dir}; mkdir -p "$O"
source "$HOME/intarsiaml-gt2/.gt2/lib.sh"
NIDS=${NIDS:-256}; P="$(T dev)/scripts/dfh/ids-prompts"; BIN=gt2-intarsia
log "stages start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "stages" || exit 2
if [[ ${GATES:-1} == 1 ]]; then
  for cfg in "tp1-off:0::--spec off" "tp1-mtp2:0::--spec mtp --spec-k 2" "tp2-off:0,1:$TP2:--spec off" "tp2-mtp2:0,1:$TP2:--spec mtp --spec-k 2"; do
    tag=${cfg%%:*}; r=${cfg#*:}; gpus=${r%%:*}; r=${r#*:}; ex=${r%%:*}; sa=${r#*:}
    # shellcheck disable=SC2086
    run dev "$gpus" 300 "plan-$tag" plan show --backend hip --model "$M35" --mem-budget 29 --max-seq 4096 --json $ex $sa
    # shellcheck disable=SC2086
    run dev "$gpus" 300 "plantext-$tag" plan show --backend hip --model "$M35" --mem-budget 29 --max-seq 4096 $ex $sa
  done
  # the batching refusal happens before any weight is read; 90 s cap in case it does not refuse (then the server would load: note it)
  # shellcheck disable=SC2086
  run dev 0,1 90 "batch-refusal" serve --backend hip --model "$M35" --tp 2 --devices 0,1 --spec mtp --spec-k 2 --max-batch 16 --mem-budget 29 --max-seq 4096 --port 8097
  log "batch-refusal: $(grep -aE 'Error|--max-batch|batching' "$O/batch-refusal.err" | head -3 | cut -c1-300 | tr '\n' '|')"
fi
g() { # g <gpus> <tag> <prompt file> [args]
  local gpus=$1 tag=$2 pf=$3; shift 3
  run dev "$gpus" 420 "gen-$tag" generate -m "$M35" --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$pf")" --max-tokens "$NIDS" "$@"
  local rc=$?
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  return $rc
}
for n in ${IDNAMES:-short-0 4k-1 16k-0}; do
  # shellcheck disable=SC2086
  g 0,1 "tp2-$n-off" "$P/$n.txt" $TP2 --spec off
  # shellcheck disable=SC2086
  g 0,1 "tp2-$n-mtp2" "$P/$n.txt" $TP2 --spec mtp --spec-k 2
  if [[ -s "$O/gen-tp2-$n-off.ids" ]] && cmp -s "$O/gen-tp2-$n-off.ids" "$O/gen-tp2-$n-mtp2.ids"; then
    log "PASS ids tp2 $n ($(grep -aoE 'prompt=[0-9]+' "$O/gen-tp2-$n-off.err" | tail -1), $(grep -ac . "$O/gen-tp2-$n-mtp2.ids") ids lines)"
  else log "FAIL ids tp2 $n"; fi
done
log "ids summary tp2: $(grep -c 'PASS ids tp2' "$O/steps.log") PASS, $(grep -c 'FAIL ids tp2' "$O/steps.log") FAIL"
log "stages done"
