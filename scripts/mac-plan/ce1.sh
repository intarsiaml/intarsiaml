#!/bin/zsh
# Per-dispatch decode attribution: INTARSIA_METAL_PROFILE=cb with INTARSIA_METAL_COMMIT_EVERY=1 (one command buffer per dispatch, MDEC's method), --spec off, 4k prompt, 65 tokens.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mac-plan; BIN=$W/target/release/intarsia; R=$W/scripts/mac-plan/raw/ce1; mkdir -p $R
PF=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
for m in Qwen3.6-35B-A3B-NVFP4 Qwen3.8-27B-Quark-AWQ-MXFP4; do
  INTARSIA_METAL_PROFILE=cb:$R/$m.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $BIN generate --model $HOME/models/$m --backend metal --max-seq 20000 --profile-every 0 --temperature 0 \
    --max-tokens 65 --ignore-eos --spec off --prompt "$(cat $PF)" > $R/$m.out 2> $R/$m.err
  echo "$m $(grep -aoE '(prefill|decode) [0-9]+ ms \([0-9.]+ tok/s\)' $R/$m.err | tr '\n' ' ')"
done
