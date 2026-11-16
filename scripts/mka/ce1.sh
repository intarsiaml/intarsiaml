#!/bin/zsh
# Lane mka: per-dispatch decode attribution of the DEFAULT (MTP k2) and the plain path at 4k and 32k prompts
# (INTARSIA_METAL_PROFILE=cb + INTARSIA_METAL_COMMIT_EVERY=1: one command buffer per dispatch). Run under fleetq --quiet gpu.
# usage: ce1.sh <outdir> [extra env as VAR=val ...]
set -u
W=/opt/code/intarsiaml/.worktrees/mka; BIN=${BIN:-$W/target/release/intarsia}; R=$1; shift; mkdir -p $R
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for m in ${=MODELS:-Qwen3.6-35B-A3B-NVFP4 Qwen3.8-27B-Quark-AWQ-MXFP4}; do
  for ctx in ${=CTXS:-4096 32768}; do
    for spec in ${=SPECS:-default off}; do
      SA=(); [ $spec = off ] && SA=(--spec off)
      tag=$m-$ctx-$spec
      env "$@" INTARSIA_METAL_PROFILE=cb:$R/$tag.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $BIN generate --model $HOME/models/$m --backend metal --max-seq 40000 \
        --profile-every 0 --temperature 0 --max-tokens 97 --ignore-eos $SA --prompt "$(cat $PR/prompt-$ctx.txt)" > $R/$tag.out 2> $R/$tag.err
      echo "$tag $(grep -aoE '(prefill|decode) [0-9]+ ms \([0-9.]+ tok/s\)' $R/$tag.err | tr '\n' ' ') $(grep -a 'spec:' $R/$tag.err | tail -1)"
    done
  done
done
echo "## done"
