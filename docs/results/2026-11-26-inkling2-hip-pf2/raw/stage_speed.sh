#!/usr/bin/env bash
# Speed stage (lane ink-hip-pf2): prefill tok/s, main vs the branch, both with the new prefill settings
# (INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1 --prefill-chunk 2048), 4k and 16k prompts, 4 processes per arm
# interleaved ABBA (--max-tokens 2); then decode tok/s at the short prompt, 256 tokens, 2 + 2 ABBA. Card 0. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pf2
S=$REPO/.scratch
OUT=$REPO/logs/speed
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-pf2a}
MASK=${MASK:-}  # unset: the branch default
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
P16="$P4
$P4
$P4
$P4"
NEW="INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1"
G="generate -m $M --backend hip --max-seq 20480 --spec off --ignore-eos --profile-every 0 --prefill-chunk 2048"
now() { date '+%T'; }
gen() {  # arm binary prompt-name prompt tokens rep
  local f=$OUT/$1-$3-$5-r$6 extra=""
  [ "$1" = branch ] && [ -n "$MASK" ] && extra="INTARSIA_HIP_INK_PF2=$MASK"
  bash "$S/run.sh" env $NEW $extra /work/.scratch/bin/intarsia-$2 $G --max-tokens "$5" -p "$4" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "SPD $1 $3 $5 r$6 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
for rep in 1 2 3 4; do
  for p in 4k 16k; do
    [ $p = 4k ] && PR="$P4" || PR="$P16"
    if [ $((rep % 2)) = 1 ]; then gen main main $p "$PR" 2 $rep; gen branch $BR $p "$PR" 2 $rep
    else gen branch $BR $p "$PR" 2 $rep; gen main main $p "$PR" 2 $rep; fi
  done
done
for rep in 1 2; do
  if [ $rep = 1 ]; then gen main main short "$PS" 256 $rep; gen branch $BR short "$PS" 256 $rep
  else gen branch $BR short "$PS" 256 $rep; gen main main short "$PS" 256 $rep; fi
done
echo "[stage] done $(now)"
