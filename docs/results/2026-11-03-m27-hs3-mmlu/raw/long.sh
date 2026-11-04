#!/bin/zsh
export FLEETQ_OWNER=claude-m27-hs3
W=/opt/code/intarsiaml/.worktrees/m27-hs3; R=$W/docs/results/2026-11-03-m27-hs3-mmlu/raw/decode
while [ $(grep -c " done " $W/.scratch/gsm.log) -lt 2 ]; do sleep 10; done
for part in a b; do
  export QARGS="--dataset /opt/code/deepeval/datasets/m27-hs3-longctx-$part.jsonl --judge none --max-tokens 128"
  fleetq run --host endor --expect 25 --mem 28 --quiet gpu,cpu --label "m27-hs3: long-ctx 16K part $part n=50 arm off" -- zsh $W/.scratch/qarm.sh long-$part-off $W/.scratch/intarsia $R
  fleetq run --host endor --expect 25 --mem 28 --quiet gpu,cpu --label "m27-hs3: long-ctx 16K part $part n=50 arm on HS=3" -- env INTARSIA_METAL_ATTN_HS=3 zsh $W/.scratch/qarm.sh long-$part-on $W/.scratch/intarsia $R
done
