#!/usr/bin/env bash
# memmix: allocator-environment arms on a Spark (ENVHOST, default spark2; vLLM stays up): the same mixed load (seed 7), 25 min, malloc_trim(0) every 8 min, one fleetq job each.
#   FLEETQ_OWNER=claude-overnight-memmix bash scripts/memmix/envarms.sh [arm ...]      (default: base amax prefault trimthr)
set -uo pipefail
ENVHOST=${ENVHOST:-spark2}
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
senv() { case $1 in base) echo "";; amax) echo "MALLOC_ARENA_MAX=2";; prefault) echo "INTARSIA_PREFIX_STORE_PREFAULT=1";; trimthr) echo "MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_MMAP_THRESHOLD_=33554432";; *) echo "bad arm $1" >&2; exit 1;; esac; }
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(base amax prefault trimthr)
for a in "${ARMS[@]}"; do
  echo "[$(date +%T)] env arm $a: '$(senv "$a")'"
  fleetq run --host $ENVHOST --mem 52 --expect 40 --label "memmix env arm $a (25 min, trim /8 min)" -- \
    env HOST=$ENVHOST PROFILE=mixed SOAKMIN=25 IDLEMIN=1 TRIMEVERY=8 SEED0=7 BASESEC=45 SETTLESEC=60 AFTERSEC=30 SERVER_ENV="$(senv "$a")" bash scripts/memmix/run-env.sh "env-$a"
  echo "[$(date +%T)] env arm $a exit $?"
done
