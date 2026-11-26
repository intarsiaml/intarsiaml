#!/usr/bin/env bash
# Speed stage (lane ink-hip-p12): decode (and prefill) tok/s, main 3226af2b9 vs the branch at its default, 256 tokens, 4 processes
# per arm interleaved ABBA, each with its greedy md5. PROMPTS (subset of short 1k 4k 16k), SPEC=off (--spec off) or default (no
# --spec: the HIP default, DSpark k3). Card 0. Radiance down <= 5 min per window.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-p12
R=$REPO/docs/results/2026-11-26-inkling2-hip-p12/raw
OUT=$REPO/logs/speed
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-p12a}
MAIN=${MAIN:-main}
SPEC=${SPEC:-off}
PROMPTS=${PROMPTS:-short 4k}
REPS=${REPS:-1 2 3 4}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
P1="${P4:0:3300}"
P16="$P4
$P4
$P4
$P4"
SP=""
[ "$SPEC" = off ] && SP="--spec off"
now() { date '+%T'; }
gen() {  # arm binary prompt-name prompt tokens rep
  local f=$OUT/$SPEC-$1-$3-$5-r$6 ms=8192
  [ "$3" = 16k ] && ms=20480
  bash "$R/run.sh" /work/.scratch/bin/intarsia-$2 generate -m $M --backend hip --max-seq $ms $SP --ignore-eos --profile-every 0 --max-tokens "$5" -p "$4" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "SPD $1 $SPEC-$3 $5 r$6 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1) [$(now)]"
}
pr() { case $1 in short) echo "$PS" ;; 1k) echo "$P1" ;; 4k) echo "$P4" ;; 16k) echo "$P16" ;; esac; }
for rep in $REPS; do
  for p in $PROMPTS; do
    PR="$(pr $p)"
    if [ $((rep % 2)) = 1 ]; then gen main $MAIN $p "$PR" 256 $rep; gen branch $BR $p "$PR" 256 $rep
    else gen branch $BR $p "$PR" 256 $rep; gen main $MAIN $p "$PR" 256 $rep; fi
  done
done
echo "[stage] done $(now)"
