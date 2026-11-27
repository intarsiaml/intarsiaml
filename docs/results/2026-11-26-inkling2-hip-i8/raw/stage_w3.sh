#!/usr/bin/env bash
# Window 3 (lane ink-hip-i8): accuracy 4k (both arms, prefill + decode mode), then plain decode (spec off) short + 4k ABBA main vs i8.
# arms: main = no quant env (P12 default on HIP, what main runs); i8 = INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row
# (one binary: the branch with the env unset is main's code path; the md5 / accuracy lines say so).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-i8
R=$REPO/docs/results/2026-11-26-inkling2-hip-i8/raw
OUT=$REPO/logs/w3
mkdir -p "$OUT" "$REPO/logs/acc"
M=/models/Inkling-2-Mini
BIN=/work/.scratch/bin/intarsia-${BR:-a}
now() { date '+%T'; }
arm() { if [ "$1" = i8 ]; then echo "INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row"; else echo "INTARSIA_I8_NONE=1"; fi; }
for a in main i8; do
  echo "[stage] accuracy 4k $a $(now)"
  bash "$R/run.sh" env $(arm $a) $BIN accuracy -m $M --backend hip --max-seq 8192 --mode both --decode-positions 32 --golden golden/inkling2-mini-vllm-4k-top20.safetensors --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden logs/acc/$a-4k.safetensors > $OUT/acc-$a-4k.log 2>&1
  echo "ACC $a 4k exit $? $(grep -aE 'floor verdict|golden gate' $OUT/acc-$a-4k.log | tr '\n' ' ' | cut -c1-200)"
done
SPEC=${SPEC:-off} PROMPTS=${PROMPTS:-"short 4k"} REPS=${REPS:-"1 2 3 4"} bash "$R/stage_speed.sh"
echo "[stage] done $(now)"
