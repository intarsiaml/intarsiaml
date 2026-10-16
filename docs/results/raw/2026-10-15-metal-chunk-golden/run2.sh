#!/bin/zsh
# Runs inside an exclusive endor fleetq slot. Qwen3-8B BF16, Metal, worktree /opt/code/intarsiaml-chunk-golden (28e5c5a).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
W=/opt/code/intarsiaml-chunk-golden; R=$W/docs/results/raw/2026-10-15-metal-chunk-golden
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
H=$W/target/release/chunkgolden; BIN=$W/target/release/intarsia
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/serving $R/generate $R/accuracy $SC/dump
cd $W
LENS=2,3,255,256,257,258,511,512,513,514,1000,2047,2048,2049,2050,4095,4096,4097,4098,4200
echo "== 2. intarsia generate (CLI, serving path) ids, 48 tokens"
P=$W/docs/results/raw/2026-10-15-metal-chunk-no1/prompts
for f in p257s1 p2049s1 p2049s2 p4097s1 p4097s2; do for C in 256 2048; do
  $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
    --prefill-chunk $C --prompt "$(cat $P/$f.txt)" > $R/generate/$f-c$C.out 2> $R/generate/$f-c$C.err
  echo "generate $f chunk=$C exit $?"
done; done
echo "== 3. accuracy short/long/4k at chunk 256 vs 2048"
for C in 256 2048; do
  $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $R/accuracy/c$C-short.log 2>&1; echo "acc c$C short exit $?"
  $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 > $R/accuracy/c$C-long.log 2>&1; echo "acc c$C long exit $?"
  $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --min-top1 0 > $R/accuracy/c$C-4k.log 2>&1; echo "acc c$C 4k exit $?"
done
echo RUN_DONE
