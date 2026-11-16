#!/bin/zsh
# Lane mfold pilot A/B: `generate`, fresh process per visit, knob off (A: <KNOB>=0) vs on (B: unset), order ABBA BAAB ...
# AVAL / BVAL set the knob's value per arm (bisect). 257 forced greedy tokens; prints decode tok/s per visit and the ids md5. Run under fleetq --quiet gpu,cpu.
# usage: pilot.sh <outdir> <KNOB> [order]   env: BIN, MODEL, CTX, SPECS, COOL
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mfold; BIN=${BIN:-$W/target/release/intarsia}; R=$1; KNOB=$2; ORDER=${3:-ABBABAAB}; mkdir -p $R
M=${MODEL:-Qwen3.6-35B-A3B-NVFP4}; CTX=${CTX:-4096}
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for spec in ${=SPECS:-off default}; do
  SA=(); [ $spec = off ] && SA=(--spec off)
  i=0
  for arm in ${(s::)ORDER}; do
    i=$((i+1)); EV=(X=1); [ $arm = A ] && EV=($KNOB=${AVAL:-0}); [ $arm = B ] && [ -n "${BVAL:-}" ] && EV=($KNOB=$BVAL)
    sleep ${COOL:-3}
    o=$R/$M-$CTX-$spec-$i$arm
    env $EV $BIN generate --model $HOME/models/$M --backend metal --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 257 \
      --ignore-eos --show-ids $SA --prompt "$(cat $PR/prompt-$CTX.txt)" > $o.out 2> $o.err
    echo "$M $CTX $spec $i $arm $(grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' $o.err | tail -1) ids $(grep -a '^ids' $o.err | md5 -q | cut -c1-8) thermal $(notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}')"
  done
done
echo "## done"
