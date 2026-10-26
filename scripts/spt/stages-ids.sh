#!/usr/bin/env bash
# Lane spt (ON tatooine, radiance down): the 27B ids gate alone, as window 6's stages B / C (scripts/dfh/stages-w6.sh), for the arms in
# IDARMS (default "dfl mtp2") at the tensor-parallel sizes in TPS (default "tp1 tp2"); off is rerun beside them. Run by cat-window.sh
# (PRE=ids) at the start of a speed window, or by window.sh (W=<name> STAGES=ids).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/spt/lib.sh"
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
  g "$gpus" "$tp-$n-off" "$P/$n.txt" "$@" --spec off   # explicit: no --spec at tp=2 is now the MTP default
  # IDARMS entries: mtpN (--spec mtp --spec-k N), dflN (--spec dflash k N), dfl (k = K, default 7). An arm that failed to run once
  # (error, hang: the 420 s timeout) is not retried on the next prompts, so a broken arm cannot eat the window.
  local a args
  for a in $IDARMS; do
    case $a in
      mtp*) args=(--spec mtp --spec-k "${a#mtp}") ;;
      dfl) args=(--spec dflash --spec-draft "$DRAFT" --spec-k "$K") ;;
      dfl*) args=(--spec dflash --spec-draft "$DRAFT" --spec-k "${a#dfl}") ;;
      *) log "unknown arm $a"; continue ;;
    esac
    if [[ -e "$O/broken-$tp-$a" ]]; then log "SKIP $a $tp $n (broken earlier)"; continue; fi
    E="INTARSIA_SPEC_TRACE=1" g "$gpus" "$tp-$n-$a" "$P/$n.txt" "$@" "${args[@]}" || touch "$O/broken-$tp-$a"
    verdict "$tp" "$n" "$a"
  done
}
names() { local k i; for k in short 4k 16k; do for i in 0 1 2 3 4 5; do echo "$k-$i"; done; done; }
mapfile -t ALL < <(names)
# IDNAMES (default: all 18): a subset of prompt names, e.g. "4k-1" for a cost-only run (window 8: traced verify cost per pass)
[[ -n ${IDNAMES:-} ]] && read -r -a ALL <<< "$IDNAMES"
# VERIFY8=1: also the captured 8-row verify probe at tp=1 (verify_rows_real, card 0, 1024-token context, the window 3 measurement)
if [[ ${VERIFY8:-0} == 1 ]]; then
  E="INTARSIA_VERIFY_DIR=$M27 INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=1024" mtest "$D" verify8-tp1-captured 0 900 --test verify_rows_real -- --nocapture
fi
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
