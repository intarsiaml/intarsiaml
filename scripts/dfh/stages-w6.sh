#!/usr/bin/env bash
# Lane dfh window 6 (ON tatooine, radiance down, called by window.sh): the gates at the lane head.
#   A. (cards in parallel) HIP G0 at the head: plan show (JSON, text) and 256 greedy ids of the 8B and the 35B, short and 4k,
#      dev (card 0) vs origin/main (card 1); GPU admission of every HIP kernel on both trees (the failure set must be main's).
#   B. the 27B ids gate at tp=1: 18 prompts (scripts/dfh/ids-prompts: 6 short, 6 of ~4k, 6 of ~16k tokens; scripts/dfh/mkprompts.py),
#      256 greedy ids, --spec off vs --spec dflash k7 and vs --spec mtp k2; prompts split over card 0 (even) and card 1 (odd).
#   C. the same at tp=2 (both cards): off vs dflash k7 and vs mtp k2 (MTP under tensor parallelism is new in this lane).
# The speculative arms run with INTARSIA_SPEC_TRACE=1, which only prints (per pass "cost sample pass N tokens": the acceptance
# histogram). A prompt passes when the arm's .ids file is byte-identical to off's and non-empty.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}; B=${B:-base}; K=${K:-7}; NIDS=${NIDS:-256}
P="$(T "$D")/scripts/dfh/ids-prompts"
log "w6 start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "w6" || exit 2

# --- A: G0 and admission ---
noop() { # noop <tree> <gpu> <tag>
  local t=$1 g=$2 tag=$3 m mm kind
  for m in 35 8; do
    mm=$M35; [[ $m == 8 ]] && mm=$M8
    for kind in short 4k; do gen $t $g "$tag-$m-$kind" $mm $kind; done
    plan $t $g "$tag-$m" $mm
  done
}
( noop $D 0 d; htest $D admission-d 0 1200 --test admission -- --nocapture --test-threads=1 ) &
( noop $B 1 b; htest $B admission-b 1 1200 --test admission -- --nocapture --test-threads=1 ) &
wait
for m in 35 8; do
  for kind in short 4k; do same "gen-d-$m-$kind.ids" "gen-b-$m-$kind.ids" "G0 tp=1 $m $kind 256 ids, dev vs main"; done
  for f in plan plantext; do
    if diff <(grep -av 'stamp\|head=\|code-tree' "$O/$f-d-$m.out") <(grep -av 'stamp\|head=\|code-tree' "$O/$f-b-$m.out") > "$O/$f-$m.diff"; then
      log "SAME G0 $f show $m"; else log "DIFF G0 $f show $m: $(wc -l < "$O/$f-$m.diff") lines in $f-$m.diff"; fi
  done
done
if [[ ${DRY:-0} != 1 ]]; then
  ks() { grep -aE 'ADMITTED|NO CASES|REJECT|FAIL' "$O/htest-$1.log" | awk '{print $2, $NF}' | sort; }
  diff <(ks admission-d) <(ks admission-b) > "$O/admission-dev-vs-main.diff"
  log "admission dev vs main (kernel, verdict) diff: $(tr '\n' '|' < "$O/admission-dev-vs-main.diff")"
fi

# --- B, C: the 27B ids gate ---
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
  if [[ -e "$O/broken-$tp-dfl" ]]; then log "SKIP dfl $tp $n (broken earlier)"; else
    E="INTARSIA_SPEC_TRACE=1" g "$gpus" "$tp-$n-dfl" "$P/$n.txt" "$@" --spec dflash --spec-draft "$DRAFT" --spec-k "$K" || touch "$O/broken-$tp-dfl"
    verdict "$tp" "$n" dfl
  fi
  if [[ -e "$O/broken-$tp-mtp2" ]]; then log "SKIP mtp2 $tp $n (broken earlier)"; else
    E="INTARSIA_SPEC_TRACE=1" g "$gpus" "$tp-$n-mtp2" "$P/$n.txt" "$@" --spec mtp --spec-k 2 || touch "$O/broken-$tp-mtp2"
    verdict "$tp" "$n" mtp2
  fi
}
names() { local k i; for k in short 4k 16k; do for i in 0 1 2 3 4 5; do echo "$k-$i"; done; done; }
mapfile -t ALL < <(names)
( for i in "${!ALL[@]}"; do (( i % 2 == 0 )) && pair 0 tp1 "${ALL[$i]}"; done ) &
( for i in "${!ALL[@]}"; do (( i % 2 == 1 )) && pair 1 tp1 "${ALL[$i]}"; done ) &
wait
for n in "${ALL[@]}"; do pair 0,1 tp2 "$n" $TP2; done
for a in dfl mtp2; do for tp in tp1 tp2; do
  log "w6 summary $a $tp: $(grep -c "PASS ids $a $tp" "$O/steps.log") PASS, $(grep -c "FAIL ids $a $tp" "$O/steps.log") FAIL"
done; done
log "w6 done"
