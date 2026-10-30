#!/usr/bin/env bash
# Lane wmr, FROM THE MAC under one fleetq job that holds both R9700 cards and takes radiance down: stop radiance (guard.sh), run a remote script on
# tatooine with its output to OUT/<name>.log, then restore radiance (trap, also if this script dies) and log the downtime.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 20 --mem 30 --label "WMR gates" -- \
#     env OUT=<raw dir> CAP=1080 scripts/wmr/guarded.sh gates.sh R,B
# args: <remote script in scripts/wmr (copied to ~/wmr/)> [its args]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
SCRIPT=${1:?remote script}; shift
CAP=${CAP:-1080}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:?set OUT}; mkdir -p "$OUT"
SSH="ssh -o BatchMode=yes tatooine"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
FIFO="$OUT/guard.fifo"; mkfifo "$FIFO"
$SSH 'mkdir -p ~/wmr' < /dev/null; scp -q "$HERE/guard.sh" "$HERE/$SCRIPT" tatooine:wmr/
$SSH 'bash ~/wmr/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() {
  $SSH 'docker rm -f wmr-job-0 wmr-job-1 >/dev/null 2>&1; true' < /dev/null > /dev/null 2>&1
  exec 9>&-; wait $GUARD; rm -f "$FIFO"
  say "guard: $(grep -h 'stopping radiance\|radiance answers\|DID NOT' "$OUT/guard.log" | tr '\n' '|')"
  echo "radiance down: $(grep -h 'stopping radiance' "$OUT/guard.log" | sed 's/.*radiance //') -> $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log" | sed 's/.*\] //')  (fleetq job $FLEETQ_JOB)" >> "$OUT/downtime.log"
}
trap cleanup EXIT
say "start $SCRIPT $*"
$SSH "timeout $CAP bash ~/wmr/$SCRIPT $*" < /dev/null > "$OUT/${SCRIPT%.sh}.log" 2>&1
say "$SCRIPT exit $?"
