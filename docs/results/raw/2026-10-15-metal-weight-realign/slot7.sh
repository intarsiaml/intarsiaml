#!/bin/zsh
# slot 7: repeat of the decode accuracy runs (short, 4k) for BASE and ON, to separate run-to-run variation from the
# aligned-vs-8-byte kernel difference seen in slot 5.
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
W=/opt/code/intarsiaml-weight-realign
R=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot7
BASE=$S/intarsia-c556008; BIN=$S/intarsia-realign
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden; G4=$G/qwen3-8b-4k-fp32-top20.safetensors
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 8 --label "metal-weight-realign: slot7 accuracy repeat (determinism)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
run() { local n=$1; shift; "$@" > $n.out 2> $n.err; echo "$n exit $?"; }
for cfg in base on; do
  if [ $cfg = base ]; then B=$BASE; F=0; else B=$BIN; F=1; fi
  export INTARSIA_METAL_REALIGN=$F
  run $R/short-$cfg-rep $B accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 --mode both
  run $R/4k-$cfg-rep $B accuracy --model $M --backend metal --golden $G4 --max-seq 8192 --mode both --decode-positions 64 --min-top1 0
done
for n in short 4k; do for c in base on; do echo "$n $c: $(grep -h 'decode (teacher' $R/$n-$c-rep.out | cut -c1-120) vs first run: $(grep -h 'decode (teacher' ../slot5/$n-$c.out | cut -c1-120)"; done; done
echo done
