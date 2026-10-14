#!/bin/zsh
# usage: bw_ab.sh <pairs> <outdir> <preBin> <newBin> <ctx...>
# Interleaved microbench of the in-stream decode attention (examples/attn_bw): per context, `pairs` rounds of pre then new
# (new then pre on even rounds). One line per run: "<label> <ctx> <median us> <p10> <p90>". Run inside a fleetq slot.
PAIRS=$1; O=$2; PRE=$3; NEW=$4; shift 4
mkdir -p $O
util() { ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
one() {
  local label=$1 bin=$2 ctx=$3 rep=$4
  local u tries=0
  u=$(util); while [ "${u:-99}" -gt 10 ] && [ $tries -lt 20 ]; do sleep 2; u=$(util); tries=$((tries+1)); done
  $bin --reps ${REPS:-30} splitkv $ctx > $O/$label-$ctx-$rep.log 2>&1
  echo "$label $ctx rep$rep idle=$u $(tail -1 $O/$label-$ctx-$rep.log | awk '{print $3, $4, $5}')"
}
for ctx in "$@"; do
  for i in $(seq 1 $PAIRS); do
    if [ $((i % 2)) = 1 ]; then one pre $PRE $ctx $i; one new $NEW $ctx $i; else one new $NEW $ctx $i; one pre $PRE $ctx $i; fi
  done
done
