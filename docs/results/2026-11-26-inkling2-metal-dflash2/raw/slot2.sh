#!/bin/bash
# endor EXCLUSIVE fleetq slot 2, lane ink-metal-dflash2: (a) the Metal backend's dyn_conv / window-attention kernel tests (now with the Inkling
# draft's H 2304 / K 2 / G 144), (b) the one-process k sweep over speed.json: dspark (the Metal default, k3) then DFlash2 k4 / k7 / k3 / k5,
# 45 s GPU idle before each, (c) greedy ids at --max-tokens 64 (off, dspark, dfl4, dfl7) on md5.json.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
trap 'for c in $(pgrep -P $$ 2>/dev/null); do pkill -P $c 2>/dev/null; kill $c 2>/dev/null; done' EXIT
W=/opt/code/intarsiaml/.worktrees/ink-metal-dflash2; R=$W/docs/results/2026-11-26-inkling2-metal-dflash2/raw
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "slot2 start $(date +%T)"
mkdir -p .scratch/metal/s2
cargo test --release -p intarsia-backend-metal --test dflash2 > .scratch/metal/s2/metal-dflash2-tests.txt 2>&1
echo "== metal backend dflash2 tests rc $? $(date +%T) | $(grep -aE '^test result' .scratch/metal/s2/metal-dflash2-tests.txt | tr '\n' ' ')"
NAME=s2-sweep PROMPTS=speed TOK=256 IDLE=45 LIMIT=${LIMIT:-960} ARMS="dspark@r1 dfl4@r1 dfl7@r1 dfl3@r1 dfl5@r1" $R/slot.sh
NAME=s2-md5-64 PROMPTS=md5 TOK=64 IDLE=0 LIMIT=300 ARMS="off@r1 dspark@r1 dfl4@r1 dfl7@r1" $R/slot.sh
echo "slot2 end $(date +%T)"
