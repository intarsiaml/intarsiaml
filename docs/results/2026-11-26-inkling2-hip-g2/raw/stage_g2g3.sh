#!/usr/bin/env bash
# One window stage (lane ink-hip-g2), radiance down. Two cards at once, same binary (origin/main 3c6ff64f0, no code changes).
#   card 0: G2 (oracle W4A16 short / long) for the five arms
#             A1 defaults | A2 INTARSIA_HIP_INK_MOE_WMMA=0 | A3 INTARSIA_HIP_INK_ATTN_FLASH=0 | A4 --prefill-chunk 256 | A5 all three off
#   card 1: G3 (vLLM short / long / 4k) for A1 and A5, plus G2 short on A1 and A5 as a card 0 <-> card 1 equality check
# Every run's full stdout+stderr goes to $OUT/<name>.log; the summary lines are echoed (and end up in the window log).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-g2
S=$REPO/.scratch
OUT=$REPO/logs/g2g3
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BIN=/work/.scratch/bin/intarsia-main
now() { date '+%T'; }
# acc CARD ARM KIND(g2|g3) GOLDEN(short|long|4k) "ENV..." "EXTRA FLAGS"
acc() {
  local card=$1 arm=$2 kind=$3 g=$4 envs=$5 extra=$6 gold maxseq=4096 name cont=ink-hip-g2-run
  [ "$card" = 1 ] && cont=ink-hip-g2-run2
  if [ "$kind" = g2 ]; then gold=golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors
  else gold=golden/inkling2-mini-vllm-$g-top20.safetensors; fi
  [ "$g" = 4k ] && maxseq=8192
  name=$kind-$arm-$g-card$card
  local t0=$SECONDS
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$cont bash "$S/run.sh" env $envs $BIN accuracy -m $M --backend hip --max-seq $maxseq \
    --mode prefill --allow-dirty $extra --golden "$gold" > "$OUT/$name.log" 2>&1
  echo "RUN $name [$(now), $((SECONDS - t0)) s, exit $?] | $(grep -aE 'prefill chunk|prompts spanning|prefill floor|prefill confident rows|floor verdict|panicked|^Error' "$OUT/$name.log" | cut -c1-420 | tr '\n' '|')"
}
card0() {
  for g in short long; do
    acc 0 A1 g2 $g "INK_ARM=A1" ""
    acc 0 A2 g2 $g "INTARSIA_HIP_INK_MOE_WMMA=0" ""
    acc 0 A3 g2 $g "INTARSIA_HIP_INK_ATTN_FLASH=0" ""
    acc 0 A4 g2 $g "INK_ARM=A4" "--prefill-chunk 256"
    acc 0 A5 g2 $g "INTARSIA_HIP_INK_MOE_WMMA=0 INTARSIA_HIP_INK_ATTN_FLASH=0" "--prefill-chunk 256"
  done
}
card1() {
  acc 1 A1 g2 short "INK_ARM=A1" ""
  acc 1 A5 g2 short "INTARSIA_HIP_INK_MOE_WMMA=0 INTARSIA_HIP_INK_ATTN_FLASH=0" "--prefill-chunk 256"
  for g in short long 4k; do
    acc 1 A1 g3 $g "INK_ARM=A1" ""
    acc 1 A5 g3 $g "INTARSIA_HIP_INK_MOE_WMMA=0 INTARSIA_HIP_INK_ATTN_FLASH=0" "--prefill-chunk 256"
  done
}
echo "[stage] g2g3 start $(now)"
card0 > "$OUT/c0.log" 2>&1 &
card1 > "$OUT/c1.log" 2>&1 &
wait
echo "---- card 0 (G2, five arms)"; cat "$OUT/c0.log"
echo "---- card 1 (G3 A1 / A5, G2 short cross-check)"; cat "$OUT/c1.log"
echo "[stage] g2g3 done $(now)"
