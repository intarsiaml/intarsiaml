#!/bin/zsh
# gates.sh <tag> <outdir>: runs inside an exclusive endor fleetq slot, binaries intarsia-<tag>, fwdrows-<tag>, chunkgolden-<tag>
# in the scratch dir. 1) per-row forward vs serving (fwdrows) 2) serving path + forward hashes (chunkgolden, 33 prompts)
# 3) accuracy on prefix-truncated golden (prompt lengths 1 mod 256), per-position KL dump 4) accuracy on the real goldens.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
TAG=$1; O=$2
W=/opt/code/intarsiaml-fwdhash; R=$W/docs/results/raw/2026-10-16-metal-forward-hash
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/fh
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
BIN=$SC/intarsia-$TAG; FR=$SC/fwdrows-$TAG; CG=$SC/chunkgolden-$TAG
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $O/fwdrows $O/serving $O/accuracy $O/trunc; cd $W
echo "== 1. fwdrows [$TAG]"
for C in 256 2048; do
  $FR --model $M --backend metal --max-seq 8192 --profile-every 0 --prefill-chunk $C --docs tools/prompts/corpus_docs.txt --code tools/prompts/corpus_code.txt \
     --lens 17,33,255,256,257,258,273,513,769,2049 --tail 24 --out $O/fwdrows/c$C.jsonl --dump-dir $SC/dump-$TAG-c$C > $O/fwdrows/c$C.log 2>&1
  echo "fwdrows chunk=$C exit $?"
done
echo "== 2. chunkgolden [$TAG]"
LENS=2,3,255,256,257,258,511,512,513,514,1000,2047,2048,2049,2050,4095,4096,4097,4098,4200
for C in 256 2048; do
  mkdir -p $SC/dumpcg-$TAG-c$C
  $CG --model $M --backend metal --max-seq 8192 --profile-every 0 --prefill-chunk $C --docs tools/prompts/corpus_docs.txt --code tools/prompts/corpus_code.txt \
     --golden $R/golden_ids.jsonl --lens $LENS --n 32 --out $O/serving/chunk$C.jsonl --dump-dir $SC/dumpcg-$TAG-c$C > $O/serving/chunk$C.log 2>&1
  echo "chunkgolden chunk=$C exit $?"
done
echo "== 3. accuracy on the truncated golden (lengths 257 513 258 769 2049) [$TAG]"
for C in 256 2048; do
  INTARSIA_ACC_POS_DUMP=$O/trunc/pos-c$C.tsv $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $SC/trunc-top20.safetensors --mode prefill \
     --max-seq 4096 --min-top1 0 > $O/trunc/c$C.log 2>&1; echo "acc trunc c$C exit $?"
done
echo "== 4. accuracy short/long/4k [$TAG]"
for C in 256 2048; do
  $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $O/accuracy/c$C-short.log 2>&1; echo "acc c$C short exit $?"
  $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 > $O/accuracy/c$C-long.log 2>&1; echo "acc c$C long exit $?"
  $BIN accuracy --model $M --backend metal --spec off --prefill-chunk $C --golden $G/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --min-top1 0 > $O/accuracy/c$C-4k.log 2>&1; echo "acc c$C 4k exit $?"
done
echo GATES_DONE
