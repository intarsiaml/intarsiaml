#!/usr/bin/env bash
# Speed stage (lane ink-hip-dec3): decode tok/s, main f3a150646 vs the branch, 256 tokens, short and 4k prompts, 4 processes per
# arm interleaved ABBA, each with its greedy md5; then the branch with INTARSIA_HIP_INK_DEC3=0 once per prompt (md5 = main).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec3
S=$REPO/.scratch
OUT=$REPO/logs/speed
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-dec3c}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
gen() {  # arm binary-name fast prompt-name prompt tokens rep
  local f=$OUT/$1-$4-$6-r$7
  bash "$S/run.sh" env INTARSIA_HIP_INK_DEC3=$3 /work/.scratch/bin/intarsia-$2 $G --max-tokens "$6" -p "$5" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "SPD $1 $4 $6 r$7 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1)"
}
for rep in 1 2 3 4; do
  for p in short 4k; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    if [ $((rep % 2)) = 1 ]; then gen main main 7 $p "$PR" 256 $rep; gen branch $BR 7 $p "$PR" 256 $rep
    else gen branch $BR 7 $p "$PR" 256 $rep; gen main main 7 $p "$PR" 256 $rep; fi
  done
done
for p in short 4k; do
  [ $p = short ] && PR="$PS" || PR="$P4"
  gen branch0 $BR 0 $p "$PR" 256 1
done
echo "[stage] done $(now)"
