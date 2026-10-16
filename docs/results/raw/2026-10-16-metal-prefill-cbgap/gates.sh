#!/bin/zsh
# Job G: bitwise gates, base (70b15eb) vs new, chunk 256 and 2048. Runs inside an exclusive endor fleetq slot.
# gates.sh <new tag: binaries intarsia-<tag>, chunkgolden-<tag> in the scratch dir> <outdir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
TAG=$1; O=$2
W=/opt/code/intarsiaml-cbgap; R=$W/docs/results/raw/2026-10-16-metal-prefill-cbgap
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $O/serving $O/generate $O/accuracy $SC/dump
cd $W
LENS=2,3,255,256,257,258,511,512,513,514,1000,2047,2048,2049,2050,4095,4096,4097,4098,4200
for ARM in base $TAG; do
  BIN=$SC/intarsia-$ARM; H=$SC/chunkgolden-$ARM
  echo "== [$ARM] 1. serving path + Model::forward path, bitwise logits (harness)"
  for C in 256 2048; do
    mkdir -p $SC/dump/$ARM-c$C
    $H --model $M --backend metal --max-seq 8192 --profile-every 0 --prefill-chunk $C --docs tools/prompts/corpus_docs.txt --code tools/prompts/corpus_code.txt \
       --golden $R/golden_ids.jsonl --lens $LENS --n 32 --out $O/serving/$ARM-chunk$C.jsonl --dump-dir $SC/dump/$ARM-c$C > $O/serving/$ARM-chunk$C.log 2>&1
    echo "[$ARM] harness chunk=$C exit $?"
  done
  echo "== [$ARM] 2. intarsia generate (CLI) ids, 48 tokens"
  for f in p257s1 p2049s1 p2049s2 p4097s1 p4097s2; do for C in 256 2048; do
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
      --prefill-chunk $C --prompt "$(cat $R/prompts/$f.txt)" > $O/generate/$ARM-$f-c$C.out 2> $O/generate/$ARM-$f-c$C.err
    echo "[$ARM] generate $f chunk=$C exit $?"
  done; done
  echo "== [$ARM] 3. accuracy short/long/4k at chunk 256 and 2048"
  for C in 256 2048; do
    $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $O/accuracy/$ARM-c$C-short.log 2>&1; echo "[$ARM] acc c$C short exit $?"
    $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 > $O/accuracy/$ARM-c$C-long.log 2>&1; echo "[$ARM] acc c$C long exit $?"
    $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --min-top1 0 > $O/accuracy/$ARM-c$C-4k.log 2>&1; echo "[$ARM] acc c$C 4k exit $?"
  done
done
echo "== 4. intarsia kernels test (new)"
$SC/intarsia-$TAG kernels test --backend metal --device 0 --model $M -v > $O/kernels-test.txt 2>&1; echo "kernels test exit $?"
echo "== 5. cargo test -p intarsia-backend-metal --release"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $O/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
echo GATES_DONE
