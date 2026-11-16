#!/bin/zsh
# Lane mfold: per-dispatch decode attribution (INTARSIA_METAL_PROFILE=cb + INTARSIA_METAL_COMMIT_EVERY=1: one command buffer per
# dispatch), MTP default and `--spec off`. Run under fleetq --quiet gpu,cpu.
# usage: ce1.sh <outdir> [extra env as VAR=val ...]   env: BIN, MODELS, CTXS, SPECS
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mfold; BIN=${BIN:-$W/target/release/intarsia}; R=$1; shift; mkdir -p $R
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for m in ${=MODELS:-Qwen3.6-35B-A3B-NVFP4}; do
  for ctx in ${=CTXS:-4096}; do
    for spec in ${=SPECS:-off default}; do
      SA=(); [ $spec = off ] && SA=(--spec off)
      tag=$m-$ctx-$spec
      env X=1 "$@" INTARSIA_METAL_PROFILE=cb:$R/$tag.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $BIN generate --model $HOME/models/$m --backend metal --max-seq 40000 \
        --profile-every 0 --temperature 0 --max-tokens 97 --ignore-eos $SA --prompt "$(cat $PR/prompt-$ctx.txt)" > $R/$tag.out 2> $R/$tag.err
      echo "$tag $(grep -aoE '(prefill|decode) [0-9]+ ms \([0-9.]+ tok/s\)' $R/$tag.err | tr '\n' ' ') $(grep -a 'spec:' $R/$tag.err | tail -1)"
      python3 $W/scripts/mfold/ce1.py $R/$tag.jsonl 10 40 > $R/$tag.table.txt 2>&1
    done
  done
done
echo "## done"
