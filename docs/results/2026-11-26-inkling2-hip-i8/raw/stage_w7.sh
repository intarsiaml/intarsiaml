#!/usr/bin/env bash
# Window 7 (lane ink-hip-i8): after the verify-rows dispatch fix (i8 in_proj / wo on 1..=16 rows): the i8 arm's accuracy (short, 4k; self-goldens
# logs/acc/i8f-*.safetensors, against the vLLM goldens, prefill + decode mode), then the hip tiny suites with the i8 env.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-i8
R=$REPO/docs/results/2026-11-26-inkling2-hip-i8/raw
OUT=$REPO/logs/w7
mkdir -p "$OUT" "$REPO/logs/acc"
M=/models/Inkling-2-Mini
BIN=/work/.scratch/bin/intarsia-a
now() { date '+%T'; }
I8="INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row"
for g in short 4k; do
  ms=4096; [ "$g" = 4k ] && ms=8192
  dp=64; [ "$g" = 4k ] && dp=32
  echo "[stage] accuracy $g i8f $(now)"
  bash "$R/run.sh" env $I8 $BIN accuracy -m $M --backend hip --max-seq $ms --mode both --decode-positions $dp --golden golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden logs/acc/i8f-$g.safetensors > $OUT/acc-i8f-$g.log 2>&1
  echo "ACC i8f $g exit $? $(grep -aE 'floor verdict|golden gate' $OUT/acc-i8f-$g.log | tr '\n' ' ' | cut -c1-200)"
done
for t in inkling2 inkling2_batch; do
  echo "[stage] tiny $t i8 $(now)"
  bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip $I8 cargo test --release -p intarsia-models --test $t -- --test-threads 1 > "$OUT/$t-i8.log" 2>&1
  echo "TEST $t-i8 exit $? | $(grep -aE '^test result' "$OUT/$t-i8.log" | tr '\n' ' ') | FAILED: $(grep -aE '^test .* FAILED' "$OUT/$t-i8.log" | sed -e 's/^test //' -e 's/ \.\.\. FAILED//' | tr '\n' ' ')"
done
echo "[stage] done $(now)"
