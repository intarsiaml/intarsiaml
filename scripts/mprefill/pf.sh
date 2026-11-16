#!/bin/zsh
# Lane mprefill: per-dispatch PREFILL attribution (INTARSIA_METAL_PROFILE=cb + INTARSIA_METAL_COMMIT_EVERY=1: one command
# buffer per dispatch, so the three dispatches of a quantised NAX matmul show up as `<op>`, `<op>#1`, `<op>#2`).
# `generate`, --spec off, 2 forced tokens. Run under fleetq --quiet gpu,cpu.
# usage: pf.sh <outdir> [extra env as VAR=val ...]   env: BIN, MODELS, CTXS, TAG
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mprefill; BIN=${BIN:-$W/target/release/intarsia}; R=$1; shift; mkdir -p $R
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for m in ${=MODELS:-Qwen3.6-35B-A3B-NVFP4 Qwen3.8-27B-Quark-AWQ-MXFP4}; do
  for ctx in ${=CTXS:-4096 32764}; do
    tag=$m-$ctx${TAG:-}
    sleep 3
    env X=1 "$@" INTARSIA_METAL_PROFILE=cb:$R/$tag.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $BIN generate --model $HOME/models/$m --backend metal \
      --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 2 --ignore-eos --spec off --prompt "$(cat $PR/prompt-$ctx.txt)" > $R/$tag.out 2> $R/$tag.err
    echo "$tag $(grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' $R/$tag.err | tr '\n' ' ') thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}')"
    python3 $W/scripts/mprefill/pf.py $R/$tag.jsonl > $R/$tag.table.txt 2>&1
  done
done
echo "## done"
