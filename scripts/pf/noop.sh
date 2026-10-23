#!/usr/bin/env bash
# Lane pf, spark side: no-op check of a dev tree against a base tree with every new knob unset: `plan show` (8B json, 35B text)
# and greedy ids (256 after a ~1.2k-token prompt, 64 after a ~4k-token prompt) for Qwen3-8B and the 35B. Inside a fleetq slot.
# Usage: [NOOP_UNSET=<another knob to unset, e.g. the MoE branch's>] noop.sh <outname> <base tree> <dev tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
P=$HOME/intarsia-decfuse; O=$P/out/$1; B=$2; D=$3; mkdir -p $O
export PATH=/usr/local/cuda/bin:$PATH
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
CORPUS=$P/pf-base/tools/prompts/corpus_docs.txt
gen() { # gen <tag> <tree> <model> <chars> <tokens>
  env -u INTARSIA_CUDA_FP8_GEMM_LT -u INTARSIA_PREFILL_TAIL_MERGE ${NOOP_UNSET:+-u $NOOP_UNSET} timeout 900 $P/$2/target/acc/release/intarsia generate -m $3 --backend cuda --mem-budget 30 --max-seq 20480 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$(head -c $4 $CORPUS)" --max-tokens $5 < /dev/null > $O/gen-$1.out 2> $O/gen-$1.err
  grep -a '^ids:' $O/gen-$1.err > $O/gen-$1.ids
}
for t in $B $D; do
  gen 35s-$t $t $M35 5000 256; gen 35l-$t $t $M35 14000 64; gen 8s-$t $t $M8 5000 256; gen 8l-$t $t $M8 14000 64
  env -u INTARSIA_CUDA_FP8_GEMM_LT timeout 300 $P/$t/target/acc/release/intarsia plan show --backend cuda --model $M8 --max-seq 4096 --mem-budget 30 --json < /dev/null > $O/plan8-$t.json 2> $O/plan8-$t.err
  env -u INTARSIA_CUDA_FP8_GEMM_LT timeout 300 $P/$t/target/acc/release/intarsia plan show --backend cuda --model $M35 --max-seq 20480 --mem-budget 30 < /dev/null > $O/plan35-$t.txt 2> $O/plan35-$t.err
done
for p in 35s 35l 8s 8l; do
  if cmp -s $O/gen-$p-$B.ids $O/gen-$p-$D.ids && [[ -s $O/gen-$p-$B.ids ]]; then r=identical; else r=DIFFER; fi
  echo "ids $p $B vs $D: $r ($(tr ',' '\n' < $O/gen-$p-$D.ids | wc -l) ids, sha $(shasum $O/gen-$p-$D.ids | cut -c1-8))"
done
cmp -s $O/plan8-$B.json $O/plan8-$D.json && echo "8B plan: byte-identical ($(wc -c < $O/plan8-$D.json) B)" || echo "8B plan: DIFFER"
cmp -s $O/plan35-$B.txt $O/plan35-$D.txt && echo "35B plan: byte-identical ($(wc -c < $O/plan35-$D.txt) B)" || { echo "35B plan: DIFFER"; diff $O/plan35-$B.txt $O/plan35-$D.txt | head -10; }
echo "== done"
