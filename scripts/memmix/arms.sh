#!/usr/bin/env bash
# memmix: bisect arms by request kind, one fleetq job each, on spark2 (vLLM retro stays up). 25 min of load per arm, malloc_trim(0) every 5 min.
#   FLEETQ_OWNER=claude-overnight-memmix bash scripts/memmix/arms.sh [arm ...]      (default: aborts tools long rest)
set -uo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
weights() { case $1 in aborts) echo "abort_decode=10,abort_prefill=8";; tools) echo "tool=1";; long) echo "long=1";; rest) echo "chat=38,schema=5,longgen=10,multiturn=14";; *) echo "bad arm $1" >&2; exit 1;; esac; }
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(aborts tools long rest)
for a in "${ARMS[@]}"; do
  echo "[$(date +%T)] arm $a: $(weights "$a")"
  fleetq run --host spark2 --mem 52 --expect 45 --label "memmix arm $a (25 min, trim /5 min)" -- \
    env HOST=spark2 PROFILE=mixed WEIGHTS="$(weights "$a")" SOAKMIN=25 IDLEMIN=2 TRIMEVERY=5 SEED0=7 bash scripts/memmix/run.sh "arm-$a"
  echo "[$(date +%T)] arm $a exit $?"
done
