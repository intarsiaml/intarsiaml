#!/bin/zsh
# GPU power with and without a keep-warm touch loop, idle model resident. power.sh <outdir> <reps>
# Segments of 60 s (macmon gpu_power, 200 ms samples): A idle box (no process), B wire_probe holding the shards with no touches,
# C keep every 500 ms, D keep every 1000 ms. Samples from the first 8 s of each segment (load, initial wiring) are dropped by powersum.py.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/resid
R=/opt/code/intarsiaml-residency/docs/results/raw/2026-10-16-metal-weight-residency
O=$1; N=$2; mkdir -p $O
source $R/waitidle.sh
for rep in $(seq 1 $N); do for seg in A B C D; do
  wait_idle || touch $O/$seg-$rep.contaminated
  macmon pipe -i 200 > $O/$seg-$rep.macmon.jsonl 2>/dev/null & MP=$!
  case $seg in
    A) sleep 60;;
    B) KEEP_MS=500 KEEP_WHAT=all $SC/wire_probe /Users/josef/models/Qwen3-8B touch 0 60000 > $O/$seg-$rep.txt 2>&1;;
    C) KEEP_MS=500 KEEP_WHAT=all $SC/wire_probe /Users/josef/models/Qwen3-8B keep 0 60000 > $O/$seg-$rep.txt 2>&1;;
    D) KEEP_MS=1000 KEEP_WHAT=all $SC/wire_probe /Users/josef/models/Qwen3-8B keep 0 60000 > $O/$seg-$rep.txt 2>&1;;
  esac
  kill $MP; wait $MP 2>/dev/null
done; done
echo done
