#!/bin/bash
# MMTP window 1: greedy-id gates (spec off / MTP knob 0 / MTP knob 1) + cb traces. Run inside a fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mmtp; BIN=$W/target/release/intarsia; O=$W/.lane/w1; mkdir -p $O
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
gen() { local tag=$1 p=$2 n=$3; shift 3
  echo "## $tag $(date +%T)"
  env "$@" $BIN generate --model $M --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens $n --ignore-eos --show-ids ${SPEC:-} --prompt "$(cat $W/.lane/prompts/$p.txt)" > $O/$tag.out 2> $O/$tag.err
  grep -iE 'decode|verify passes' $O/$tag.err | tail -2; }
for p in short 4ka 4kb 16k; do
  SPEC="--spec off" gen $p-off $p 256 X=1
  SPEC="--spec mtp" gen $p-mtp0 $p 256 INTARSIA_METAL_MTP_GPU_ACCEPT=0
  SPEC="--spec mtp" gen $p-mtp1 $p 256 INTARSIA_METAL_MTP_GPU_ACCEPT=1
done
for v in 0 1; do
  SPEC="--spec mtp" gen 4ka-cb$v 4ka 129 INTARSIA_METAL_MTP_GPU_ACCEPT=$v INTARSIA_METAL_PROFILE=cb:$O/4ka-cb$v.jsonl
done
echo "## done $(date +%T)"
