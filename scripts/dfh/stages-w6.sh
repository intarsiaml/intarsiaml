#!/usr/bin/env bash
# Lane dfh window 6 (ON tatooine, radiance down, called by window.sh): the 27B ids gate. 18 prompts (scripts/dfh/ids-prompts:
# 6 short, 6 of ~4k, 6 of ~16k tokens; scripts/dfh/mkprompts.py), 256 greedy ids, --spec off vs --spec dflash k7, tp=1 then tp=2;
# at tp=1 also --spec mtp k2 (traced, acceptance only; its ids are compared too, as a side result).
#   tp=1: prompts split over card 0 (even) and card 1 (odd), both arms of a prompt on the same card
#   tp=2: both cards, every prompt, both arms
# A prompt passes when the two .ids files are byte-identical and non-empty.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}; K=${K:-7}; NIDS=${NIDS:-256}
P="$(T "$D")/scripts/dfh/ids-prompts"
log "w6 start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "w6" || exit 2
# g <gpus> <tag> <prompt file> [args]
g() {
  local gpus=$1 tag=$2 pf=$3; shift 3
  run "$D" "$gpus" 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$pf")" --max-tokens "$NIDS" "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
}
pair() { # pair <gpus> <tp tag> <prompt name> [tp args]
  local gpus=$1 tp=$2 n=$3; shift 3
  g "$gpus" "$tp-$n-off" "$P/$n.txt" "$@"
  # INTARSIA_SPEC_TRACE only prints (per-pass "cost sample pass N tokens": the acceptance histogram); it does not change ids
  E="INTARSIA_SPEC_TRACE=1" g "$gpus" "$tp-$n-dfl" "$P/$n.txt" "$@" --spec dflash --spec-draft "$DRAFT" --spec-k "$K"
  # tp=1 only: MTP k2 on the same prompt, traced, for the first-position acceptance against DFlash2's (MTP is refused under tp)
  [[ $tp == tp1 ]] && E="INTARSIA_SPEC_TRACE=1" g "$gpus" "$tp-$n-mtp2" "$P/$n.txt" "$@" --spec mtp --spec-k 2
  if [[ -s "$O/gen-$tp-$n-off.ids" ]] && cmp -s "$O/gen-$tp-$n-off.ids" "$O/gen-$tp-$n-dfl.ids"; then
    log "PASS ids $tp $n ($(grep -a 'prompt=' "$O/gen-$tp-$n-off.err" | grep -oE 'prompt=[0-9]+' | tail -1))"
  else log "FAIL ids $tp $n"; fi
  if [[ $tp == tp1 ]]; then if cmp -s "$O/gen-$tp-$n-off.ids" "$O/gen-$tp-$n-mtp2.ids"; then log "side: mtp2 ids same $n"; else log "side: mtp2 ids DIFFER $n"; fi; fi
}
names() { local k i; for k in short 4k 16k; do for i in 0 1 2 3 4 5; do echo "$k-$i"; done; done; }
mapfile -t ALL < <(names)
( for i in "${!ALL[@]}"; do (( i % 2 == 0 )) && pair 0 tp1 "${ALL[$i]}"; done ) &
( for i in "${!ALL[@]}"; do (( i % 2 == 1 )) && pair 1 tp1 "${ALL[$i]}"; done ) &
wait
for n in "${ALL[@]}"; do pair 0,1 tp2 "$n" $TP2; done
log "w6 summary: $(grep -c 'PASS ids' "$O/steps.log") PASS, $(grep -c 'FAIL ids' "$O/steps.log") FAIL"
log "w6 done"
