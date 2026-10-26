#!/usr/bin/env bash
# Lane dfh window 9 (ON tatooine, radiance down, called by window.sh): G0 at the final head (origin/main a3e60ce merged, the tp MTP default on).
#   A. (cards in parallel) HIP G0 at the head: plan show (JSON, text) and 256 greedy ids of the 8B and the 35B, short and 4k,
#      dev (card 0) vs origin/main (card 1); GPU admission of every HIP kernel on both trees (the failure set must be main's).
#   B. the 27B default: with no --spec at tp=2 the recipe default runs --spec mtp k2 (its log line must say so); ids vs --spec off on 6
#      prompts (2 short, 2 ~3k, 2 ~11k). At tp=1 with no --spec: no speculation (no default line), ids vs --spec off.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/dfh/lib.sh"
D=${D:-dev}; B=${B:-base}; K=${K:-7}; NIDS=${NIDS:-256}
P="$(T "$D")/scripts/dfh/ids-prompts"
log "w9 start, VRAM in use $(( $(vram) / 1048576 )) MiB"
clear "w9" || exit 2

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

# --- B: the 27B tp MTP default ---
g() { # g <gpus> <tag> <prompt file> [args]
  local gpus=$1 tag=$2 pf=$3; shift 3
  run "$D" "$gpus" 420 "gen-$tag" generate -m "$M27" --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat "$pf")" --max-tokens "$NIDS" "$@"
  local rc=$?
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  return $rc
}
P="$(T "$D")/scripts/dfh/ids-prompts"; NIDS=${NIDS:-256}
for n in short-0 short-3 4k-1 4k-4 16k-2 16k-5; do
  g 0,1 "tp2-$n-off" "$P/$n.txt" $TP2 --spec off
  g 0,1 "tp2-$n-def" "$P/$n.txt" $TP2
  d=$(grep -ac "the default of recipe qwen3_5_vl on hip at tp=2" "$O/gen-tp2-$n-def.err"); p=$(grep -aoE "verify passes=[0-9]+" "$O/gen-tp2-$n-def.err")
  if [[ -s "$O/gen-tp2-$n-off.ids" ]] && cmp -s "$O/gen-tp2-$n-off.ids" "$O/gen-tp2-$n-def.ids" && [[ $d == 1 && -n $p ]]; then
    log "PASS default tp2 $n (default line, $p, ids == --spec off)"; else log "FAIL default tp2 $n (default line $d, ${p:-no verify passes})"; fi
done
for n in short-0 4k-1; do
  g 0 "tp1-$n-off" "$P/$n.txt" --spec off
  g 0 "tp1-$n-def" "$P/$n.txt"
  d=$(grep -ac "the default of recipe" "$O/gen-tp1-$n-def.err"); p=$(grep -ac "verify passes=" "$O/gen-tp1-$n-def.err")
  if [[ -s "$O/gen-tp1-$n-off.ids" ]] && cmp -s "$O/gen-tp1-$n-off.ids" "$O/gen-tp1-$n-def.ids" && [[ $d == 0 && $p == 0 ]]; then
    log "PASS no default tp1 $n (no speculation, ids == --spec off)"; else log "FAIL tp1 $n (default line $d, verify lines $p)"; fi
done
log "w9 summary: $(grep -c 'PASS ' "$O/steps.log") PASS, $(grep -cE 'FAIL |DIFF ' "$O/steps.log") FAIL or DIFF"
log "w9 done"
