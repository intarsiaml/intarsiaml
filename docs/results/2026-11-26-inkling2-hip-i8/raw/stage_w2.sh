#!/usr/bin/env bash
# Window 2 (lane ink-hip-i8): retuned defaults. Order (cheap first): i8 correctness tests; the kernel microbench at rows 1,2,4,8,16
# (i8 vs main's P12 kernels); greedy md5 + tok/s at 64 tokens (short, spec off) for both arms; accuracy short / long, both arms,
# prefill + decode mode against the vLLM goldens (self-goldens kept for KL of i8 against main).
# arms: main = no quant env (P12 default on HIP, what main runs); i8 = INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row
# (one binary: the branch with the env unset is main's code path; the md5 / accuracy lines say so).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-i8
R=$REPO/docs/results/2026-11-26-inkling2-hip-i8/raw
OUT=$REPO/logs/w2
mkdir -p "$OUT" "$REPO/logs/acc"
M=/models/Inkling-2-Mini
BIN=/work/.scratch/bin/intarsia-${BR:-a}
now() { date '+%T'; }
arm() { if [ "$1" = i8 ]; then echo "INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row"; else echo "INTARSIA_I8_NONE=1"; fi; }
echo "[stage] i8 correctness $(now)"
bash "$R/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_i8 -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|checked|panicked|differs|error|got " | head -30
echo "[stage] microbench $(now)"
bash "$R/run.sh" env I8_BENCH=1 I8_MS=1,2,4,8,16 INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_i8 i8_microbench -- --nocapture --test-threads=1 2>&1 | grep -aE "^BENCH|test result|panicked|error" | head -150
for a in main i8; do
  f=$OUT/md5-$a
  echo "[stage] md5 $a $(now)"
  bash "$R/run.sh" env $(arm $a) $BIN generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0 --max-tokens 64 -p "$(cat $REPO/scripts/cmx/ids-prompts/short-0.txt)" > $f.out 2> $f.err
  echo "MD5 $a short 64 $(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' $f.out | md5sum | cut -c1-32) | $(cat $f.out $f.err | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(grep -aE 'int8|lm_head' $f.err | head -3 | cut -c1-200 | tr '\n' ' ')"
done
for g in short long; do
  for a in main i8; do
    echo "[stage] accuracy $g $a $(now)"
    bash "$R/run.sh" env $(arm $a) $BIN accuracy -m $M --backend hip --max-seq 4096 --mode both --decode-positions 64 --golden golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden logs/acc/$a-$g.safetensors > $OUT/acc-$a-$g.log 2>&1
    echo "ACC $a $g exit $? $(grep -aE 'floor verdict|golden gate' $OUT/acc-$a-$g.log | tr '\n' ' ' | cut -c1-200)"
  done
done
echo "[stage] done $(now)"
