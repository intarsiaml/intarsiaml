#!/bin/zsh
# usage: ops_ab.sh <pairs> <outdir> <label=small_ops-binary> ...
# Interleaved (rotating order) runs of examples/small_ops: per-call GPU time of the small decode ops plus output
# fingerprints. One GPU idle sample before each run, waiting until at most 10 percent. Run inside a fleetq slot.
PAIRS=$1; O=$2; shift 2
mkdir -p $O
util() { ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
specs=("$@"); n=${#specs[@]}
for i in $(seq 1 $PAIRS); do
  for k in $(seq 0 $((n-1))); do
    spec=${specs[$(( (k + i - 1) % n + 1 ))]}
    label=${spec%%=*}; bin=${spec#*=}
    u=$(util); tries=0; while [ "${u:-99}" -gt 10 ] && [ $tries -lt 20 ]; do sleep 2; u=$(util); tries=$((tries+1)); done
    echo "== $label pair$i idle_util=$u(waits=$tries)"
    $bin > $O/$label-$i.out 2>&1
    grep -E '^(norm_add|qkrope|silu|argmax) ' $O/$label-$i.out
  done
done
