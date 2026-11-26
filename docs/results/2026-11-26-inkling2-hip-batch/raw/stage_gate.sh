#!/usr/bin/env bash
# Solo gate (lane ink-hip-batch): the solo path is unchanged by the row-table kernels. Greedy md5 at 64 tokens (short, 4k), then
# decode tok/s at 256 tokens, main (origin/main 40b85a391, after #526) vs the branch merged onto it, 4 processes per arm interleaved
# ABBA per prompt, each with its greedy md5 (= the 256-token md5 check). Default env (no INTARSIA_* knobs), graphs, --spec off.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
S=$REPO/.scratch
OUT=$REPO/logs/gate
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
MA=${MA:-main2}
BR=${BR:-b3}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
gen() {  # arm binary-name prompt-name prompt tokens rep
  local f=$OUT/$1-$3-$5-r$6
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$2 $G --max-tokens "$5" -p "$4" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "SPD $1 $3 $5 r$6 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1)"
}
echo "[stage] md5 64 $(now)"
for p in short 4k; do
  [ $p = short ] && PR="$PS" || PR="$P4"
  gen main $MA $p "$PR" 64 1
  gen branch $BR $p "$PR" 64 1
done
echo "[stage] speed + md5 256 $(now)"
for rep in $(seq 1 "${REPS:-4}"); do
  for p in short 4k; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    if [ $((rep % 2)) = 1 ]; then gen main $MA $p "$PR" 256 $rep; gen branch $BR $p "$PR" 256 $rep
    else gen branch $BR $p "$PR" 256 $rep; gen main $MA $p "$PR" 256 $rep; fi
  done
done
echo "[stage] done $(now)"
