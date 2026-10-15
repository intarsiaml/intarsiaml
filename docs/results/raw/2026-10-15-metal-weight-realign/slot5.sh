#!/bin/zsh
# slot 5: golden accuracy numbers (no baseline gate: the unstamped scratch binaries cannot take --baseline) for BASE (c556008)
# and ON (this branch, INTARSIA_METAL_REALIGN=1); the two outputs are diffed digit for digit.
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
W=/opt/code/intarsiaml-weight-realign
R=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot5
BASE=$S/intarsia-c556008; BIN=$S/intarsia-realign
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden; G4=$G/qwen3-8b-4k-fp32-top20.safetensors
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 15 --label "metal-weight-realign: slot5 accuracy base vs realign" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
run() { local n=$1; shift; "$@" > $n.out 2> $n.err; echo "$n exit $?"; }
for cfg in base on; do
  if [ $cfg = base ]; then B=$BASE; F=0; else B=$BIN; F=1; fi
  export INTARSIA_METAL_REALIGN=$F
  run $R/short-$cfg $B accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 --mode both
  run $R/long-$cfg $B accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 --mode both --decode-positions 64
  run $R/4k-$cfg $B accuracy --model $M --backend metal --golden $G4 --max-seq 8192 --mode both --decode-positions 64 --min-top1 0
done
for n in short long 4k; do diff <(grep -vE "elapsed|ms|tok/s|[0-9]+\.[0-9]+ ?s\b" $R/$n-base.out) <(grep -vE "elapsed|ms|tok/s|[0-9]+\.[0-9]+ ?s\b" $R/$n-on.out) > $R/$n-diff.txt && echo "$n: outputs identical" || echo "$n: DIFFER ($(wc -l < $R/$n-diff.txt) diff lines)"; done
echo done
