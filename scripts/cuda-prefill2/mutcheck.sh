#!/usr/bin/env bash
# Run the per-mutation `gdn_chunked` test binaries built by mutbuild.sh (GPU: INSIDE a vLLM-down window). Every mutant must FAIL
# (exit 101), the unmutated binary must pass. All binaries run at the same time (the tests are small and deterministic, GPU
# sharing does not change a verdict). Prints one KILLED / SURVIVED line each; logs to $O/mut-<name>.log.
#   mutcheck.sh <outdir> [mutation ...]
set -uo pipefail
O=${1:?outdir}; shift
B=$HOME/intarsia-decfuse/mutbins
ALL="no-interdecay no-kd mask-strict no-solve no-carry tail-decay no-intra-decay no-qe no-qscale"
MUTS=${*:-$ALL}
mkdir -p "$O"
for m in unmutated $MUTS; do
  [ -x "$B/gc-$m" ] || { echo "MISSING $m"; continue; }
  ( timeout 200 "$B/gc-$m" --test-threads=1 --nocapture > "$O/mut-$m.log" 2>&1; echo $? > "$O/mut-$m.rc" ) &
done
wait
bad=0
for m in unmutated $MUTS; do
  [ -f "$O/mut-$m.rc" ] || { bad=1; continue; }
  rc=$(cat "$O/mut-$m.rc")
  failed=$(grep -E '^test .*FAILED' "$O/mut-$m.log" | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')
  if [ "$m" = unmutated ]; then
    if [ "$rc" = 0 ]; then echo "unmutated: pass ($(grep -c '^test .* ok' "$O/mut-$m.log") tests)"; else echo "unmutated: FAILED rc=$rc: $failed"; bad=1; fi
  elif [ "$rc" != 0 ]; then
    echo "KILLED   $m (rc=$rc): $failed"
  else
    echo "SURVIVED $m: the tests passed on a deliberately broken kernel"; bad=1
  fi
done
exit $bad
