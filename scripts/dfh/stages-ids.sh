#!/usr/bin/env bash
# Lane dfh (ON tatooine, radiance down): the 27B ids gate alone, as window 6's stages B / C (scripts/dfh/stages-w6.sh), for the arms in
# IDARMS (default "dfl mtp2") at the tensor-parallel sizes in TPS (default "tp1 tp2"); off is rerun beside them. Run by cat-window.sh
# (PRE=ids) at the start of a speed window, or by window.sh (W=<name> STAGES=ids).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}; K=${K:-7}; NIDS=${NIDS:-256}; IDARMS=${IDARMS:-"dfl mtp2"}; TPS=${TPS:-"tp1 tp2"}
P="$(T "$D")/scripts/dfh/ids-prompts"
log "ids start (arms $IDARMS, $TPS), VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "ids" || exit 2
# g <gpus> <tag> <prompt file> [args]
g() {
  local gpus=$1 tag=$2 pf=$3; shift 3
  run "$D" "$gpus" 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$pf")" --max-tokens "$NIDS" "$@"
  local rc=$?
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  return $rc
}
verdict() { # verdict <tp> <prompt> <arm>
  if [[ -s "$O/gen-$1-$2-off.ids" ]] && cmp -s "$O/gen-$1-$2-off.ids" "$O/gen-$1-$2-$3.ids"; then
    log "PASS ids $3 $1 $2 ($(grep -aoE 'prompt=[0-9]+' "$O/gen-$1-$2-off.err" | tail -1))"
  else log "FAIL ids $3 $1 $2"; fi
}
pair() { # pair <gpus> <tp tag> <prompt name> [tp args]
  local gpus=$1 tp=$2 n=$3; shift 3
  g "$gpus" "$tp-$n-off" "$P/$n.txt" "$@"
  # an arm that failed to run once (error, hang: the 420 s timeout) is not retried on the next prompts, so a broken arm cannot eat
  # the window
  if [[ " $IDARMS " != *" dfl "* ]]; then :; elif [[ -e "$O/broken-$tp-dfl" ]]; then log "SKIP dfl $tp $n (broken earlier)"; else
    E="INTARSIA_SPEC_TRACE=1" g "$gpus" "$tp-$n-dfl" "$P/$n.txt" "$@" --spec dflash --spec-draft "$DRAFT" --spec-k "$K" || touch "$O/broken-$tp-dfl"
    verdict "$tp" "$n" dfl
  fi
  if [[ " $IDARMS " != *" mtp2 "* ]]; then :; elif [[ -e "$O/broken-$tp-mtp2" ]]; then log "SKIP mtp2 $tp $n (broken earlier)"; else
    E="INTARSIA_SPEC_TRACE=1" g "$gpus" "$tp-$n-mtp2" "$P/$n.txt" "$@" --spec mtp --spec-k 2 || touch "$O/broken-$tp-mtp2"
    verdict "$tp" "$n" mtp2
  fi
}
names() { local k i; for k in short 4k 16k; do for i in 0 1 2 3 4 5; do echo "$k-$i"; done; done; }
mapfile -t ALL < <(names)
if [[ " $TPS " == *" tp1 "* ]]; then
  ( for i in "${!ALL[@]}"; do (( i % 2 == 0 )) && pair 0 tp1 "${ALL[$i]}"; done ) &
  ( for i in "${!ALL[@]}"; do (( i % 2 == 1 )) && pair 1 tp1 "${ALL[$i]}"; done ) &
  wait
fi
if [[ " $TPS " == *" tp2 "* ]]; then for n in "${ALL[@]}"; do pair 0,1 tp2 "$n" $TP2; done; fi
for a in $IDARMS; do for tp in $TPS; do
  log "ids summary $a $tp: $(grep -c "PASS ids $a $tp" "$O/steps.log") PASS, $(grep -c "FAIL ids $a $tp" "$O/steps.log") FAIL"
done; done
log "ids done"
